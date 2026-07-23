"""
Multi-Prototype Routing: 1 prototype vs k>1 k-means centroids per product.

Tests whether using multiple prototypes per product improves routing accuracy
or detection AUROC, especially on harder cases.

Setting: MVTec AD, N=10, 3 seeds, 18 MB budget, uniform allocation
Variants: k_proto in {1, 2, 3, 5}

Output: output/continual/multi_prototype.csv
"""
from __future__ import annotations
import os, sys, csv
os.environ.setdefault("OMP_NUM_THREADS", "12")

import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from continual_utils import compute_metrics, KNN_K, FEATURE_DIM

N_PER_TASK = 10
SEEDS = [0, 1, 2]
BUDGET_MB = 18.0
K_PROTOS = [1, 2, 3, 5]
TASKS = [
    "mvtec_AD/bottle", "mvtec_AD/cable", "mvtec_AD/capsule", "mvtec_AD/carpet",
    "mvtec_AD/grid", "mvtec_AD/hazelnut", "mvtec_AD/leather", "mvtec_AD/metal_nut",
    "mvtec_AD/pill", "mvtec_AD/screw", "mvtec_AD/tile", "mvtec_AD/toothbrush",
    "mvtec_AD/transistor", "mvtec_AD/wood", "mvtec_AD/zipper",
]
CACHE = Path("output/feature_cache")
OUTPUT_DIR = Path("output/continual")
CSV_PATH = OUTPUT_DIR / "multi_prototype.csv"
COLUMNS = ["timestamp", "seed", "k_proto", "budget_mb", "task",
           "img_auroc_global", "img_auroc_routed", "routing_acc", "margin_mean"]


def cache_path(prefix, task, suffix=""):
    return CACHE / f"{prefix}_{task.replace('/', '_')}{suffix}.npy"


def kcenter_greedy(data, budget, seed):
    n = len(data)
    if budget >= n:
        return data
    pca_dim = min(64, data.shape[1], n)
    pca = PCA(n_components=pca_dim, random_state=seed)
    data_pca = pca.fit_transform(data)
    rng = np.random.RandomState(seed)
    first = rng.randint(0, n)
    selected = [first]
    min_d = np.full(n, np.inf)
    diffs = data_pca - data_pca[first]
    min_d = np.minimum(min_d, np.einsum('ij,ij->i', diffs, diffs))
    for _ in range(budget - 1):
        nxt = int(np.argmax(min_d))
        selected.append(nxt)
        diffs = data_pca - data_pca[nxt]
        min_d = np.minimum(min_d, np.einsum('ij,ij->i', diffs, diffs))
    return data[np.array(selected)]


def compute_prototypes(flat_train, k, seed):
    """Return k prototypes for a product. k=1 -> mean; k>1 -> k-means centroids."""
    if k == 1:
        return flat_train.mean(axis=0, keepdims=True)  # [1, D]
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(flat_train)
    return km.cluster_centers_  # [k, D]


def route_image(img_mean, all_prototypes, product_ranges):
    """Route by nearest prototype. Returns (predicted_product_idx, margin)."""
    # all_prototypes: [total_protos, D], product_ranges: list of (start, end) per product
    dists = np.linalg.norm(all_prototypes - img_mean, axis=1)  # [total_protos]
    # Min distance per product
    per_prod_min = []
    for start, end in product_ranges:
        per_prod_min.append(dists[start:end].min())
    per_prod_min = np.array(per_prod_min)
    sorted_d = np.sort(per_prod_min)
    margin = sorted_d[1] / max(sorted_d[0], 1e-10)
    return int(np.argmin(per_prod_min)), margin


def knn_score(bank, test_feats, k=KNN_K):
    n, p, d = test_feats.shape
    flat = test_feats.reshape(-1, d)
    k_actual = min(k, len(bank))
    nn = NearestNeighbors(n_neighbors=k_actual, metric='euclidean', algorithm='auto')
    nn.fit(bank)
    dists, _ = nn.kneighbors(flat)
    return np.percentile(dists.mean(axis=1).reshape(n, p), 95, axis=1)


def mb_to_embs(mb):
    return int(mb * 1e6 / (FEATURE_DIM * 4))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {CSV_PATH}")

    test_feats, test_labels = {}, {}
    for task in TASKS:
        test_feats[task] = np.load(cache_path("test_feat", task))
        test_labels[task] = np.load(cache_path("test_labels", task))

    per_prod_budget = mb_to_embs(BUDGET_MB) // len(TASKS)
    # snap to closest standard option
    options = [25, 50, 75, 100, 150, 200, 300, 500]
    per_prod_budget = min(options, key=lambda x: abs(x - per_prod_budget))
    print(f"Budget per product: {per_prod_budget} embeddings")

    for seed in SEEDS:
        print(f"\n[SEED {seed}]")
        train_feats = {}
        for task in TASKS:
            train_feats[task] = np.load(cache_path("train_feat", task, f"_s{seed}_n{N_PER_TASK}"))

        # Build banks (shared across k_proto variants)
        sub_banks = {}
        for task in TASKS:
            flat = train_feats[task].reshape(-1, train_feats[task].shape[-1])
            sub_banks[task] = kcenter_greedy(flat, per_prod_budget, seed)
        full_bank = np.concatenate([sub_banks[t] for t in TASKS])

        for k in K_PROTOS:
            print(f"  k_proto = {k}")

            # Compute prototypes for all products
            all_protos = []
            product_ranges = []
            for task in TASKS:
                flat = train_feats[task].reshape(-1, train_feats[task].shape[-1])
                protos = compute_prototypes(flat, k, seed)
                start = sum(p.shape[0] for p in all_protos) if all_protos else 0
                all_protos.append(protos)
                product_ranges.append((start, start + protos.shape[0]))
            all_protos = np.concatenate(all_protos, axis=0)

            # Evaluate per task: routing accuracy + AUROC (routed and global)
            for i, task in enumerate(TASKS):
                short = task.split("/")[-1]
                tf = test_feats[task]
                labels = test_labels[task]

                # Global score (no routing)
                global_scores = knn_score(full_bank, tf)
                m_global = compute_metrics(labels, global_scores)

                # Routed score
                # For each image, compute mean embedding, route, score against routed sub-bank
                routed_scores = np.zeros(len(tf))
                correct, margins = 0, []
                for j in range(len(tf)):
                    img_mean = tf[j].mean(axis=0)
                    pred, margin = route_image(img_mean, all_protos, product_ranges)
                    margins.append(margin)
                    if pred == i:
                        correct += 1
                    bank = sub_banks[TASKS[pred]]
                    scores = knn_score(bank, tf[j:j+1])
                    routed_scores[j] = scores[0]
                m_routed = compute_metrics(labels, routed_scores)

                routing_acc = correct / len(tf) * 100
                margin_mean = float(np.mean(margins))

                row = {
                    "timestamp": datetime.now().isoformat(),
                    "seed": seed, "k_proto": k, "budget_mb": BUDGET_MB,
                    "task": short,
                    "img_auroc_global": round(m_global["img_auroc"], 6),
                    "img_auroc_routed": round(m_routed["img_auroc"], 6),
                    "routing_acc": round(routing_acc, 2),
                    "margin_mean": round(margin_mean, 3),
                }
                exists = CSV_PATH.exists()
                with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
                    if not exists:
                        w.writeheader()
                    w.writerow(row)
                print(f"    {short:>15s}  routed AUROC={m_routed['img_auroc']:.4f}  acc={routing_acc:.1f}%  margin={margin_mean:.2f}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
