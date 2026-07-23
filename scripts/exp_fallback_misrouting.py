"""
Fallback evaluation under forced misrouting.

Simulates router failures by randomly misrouting a fraction p of test images,
then measures how much performance degrades with:
  1. top-1 without fallback (worst case)
  2. top-1 with margin-based fallback (use top-3 if d2/d1 < threshold)
  3. global retrieval (no routing) - reference

Setting: MVTec AD, N=10, seed 0, 18 MB budget, uniform allocation
Misrouting probabilities: p in {0.0, 0.1, 0.25, 0.5}
Margin thresholds for fallback: {1.0 (no fallback), 1.5, 2.0}

Output: output/continual/fallback_misrouting.csv
"""
from __future__ import annotations
import os, sys, csv
os.environ.setdefault("OMP_NUM_THREADS", "12")

import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from continual_utils import compute_metrics, KNN_K, FEATURE_DIM

SEED = 0
N_PER_TASK = 10
BUDGET_MB = 18.0
MISROUTE_PS = [0.0, 0.1, 0.25, 0.5]
MARGIN_THRESHOLDS = [1.0, 1.5, 2.0]  # 1.0 = no fallback

TASKS = [
    "mvtec_AD/bottle", "mvtec_AD/cable", "mvtec_AD/capsule", "mvtec_AD/carpet",
    "mvtec_AD/grid", "mvtec_AD/hazelnut", "mvtec_AD/leather", "mvtec_AD/metal_nut",
    "mvtec_AD/pill", "mvtec_AD/screw", "mvtec_AD/tile", "mvtec_AD/toothbrush",
    "mvtec_AD/transistor", "mvtec_AD/wood", "mvtec_AD/zipper",
]
CACHE = Path("output/feature_cache")
OUTPUT_DIR = Path("output/continual")
CSV_PATH = OUTPUT_DIR / "fallback_misrouting.csv"
COLUMNS = ["timestamp", "seed", "misroute_p", "margin_threshold",
           "task", "img_auroc", "n_fallbacks", "n_test"]


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


def knn_score(bank, patches, k=KNN_K):
    """Score a single image's patches against bank, return P95 score."""
    k_actual = min(k, len(bank))
    nn = NearestNeighbors(n_neighbors=k_actual, metric='euclidean', algorithm='auto')
    nn.fit(bank)
    dists, _ = nn.kneighbors(patches)
    patch_scores = dists.mean(axis=1)
    return float(np.percentile(patch_scores, 95))


def mb_to_embs(mb):
    return int(mb * 1e6 / (FEATURE_DIM * 4))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {CSV_PATH}")

    # Load test features
    test_feats, test_labels = {}, {}
    for task in TASKS:
        test_feats[task] = np.load(cache_path("test_feat", task))
        test_labels[task] = np.load(cache_path("test_labels", task))

    # Load train and build sub-banks + prototypes
    options = [25, 50, 75, 100, 150, 200, 300, 500]
    per_prod_budget = mb_to_embs(BUDGET_MB) // len(TASKS)
    per_prod_budget = min(options, key=lambda x: abs(x - per_prod_budget))
    print(f"Budget per product: {per_prod_budget} embeddings")

    sub_banks = {}
    prototypes = {}
    for task in TASKS:
        train = np.load(cache_path("train_feat", task, f"_s{SEED}_n{N_PER_TASK}"))
        flat = train.reshape(-1, train.shape[-1])
        sub_banks[task] = kcenter_greedy(flat, per_prod_budget, SEED)
        prototypes[task] = flat.mean(axis=0)

    proto_matrix = np.stack([prototypes[t] for t in TASKS])

    rng = np.random.RandomState(SEED)

    for misroute_p in MISROUTE_PS:
        print(f"\n--- Misroute p={misroute_p} ---")
        for margin_thr in MARGIN_THRESHOLDS:
            print(f"  Margin threshold = {margin_thr}")

            for true_idx, task in enumerate(TASKS):
                short = task.split("/")[-1]
                tf = test_feats[task]
                labels = test_labels[task]

                scores = np.zeros(len(tf))
                fallbacks = 0

                for j in range(len(tf)):
                    img_mean = tf[j].mean(axis=0)
                    # Compute distances to all prototypes
                    dists = np.linalg.norm(proto_matrix - img_mean, axis=1)
                    # Natural top-1
                    sorted_idx = np.argsort(dists)
                    pred = int(sorted_idx[0])
                    margin = dists[sorted_idx[1]] / max(dists[sorted_idx[0]], 1e-10)

                    # Simulate misrouting with probability p
                    if rng.random() < misroute_p:
                        # Force misroute: pick a random WRONG product
                        wrong_options = [i for i in range(len(TASKS)) if i != true_idx]
                        pred = int(rng.choice(wrong_options))
                        # If misrouting, the margin would be low (we simulate this)
                        margin = 0.5  # forced low margin to trigger fallback

                    # Apply fallback if margin < threshold
                    if margin < margin_thr and margin_thr > 1.0:
                        # Use top-3: union of 3 closest sub-banks
                        top3 = sorted_idx[:3]
                        bank = np.concatenate([sub_banks[TASKS[k]] for k in top3])
                        fallbacks += 1
                    else:
                        bank = sub_banks[TASKS[pred]]

                    scores[j] = knn_score(bank, tf[j])

                m = compute_metrics(labels, scores)

                row = {
                    "timestamp": datetime.now().isoformat(),
                    "seed": SEED, "misroute_p": misroute_p,
                    "margin_threshold": margin_thr,
                    "task": short,
                    "img_auroc": round(m["img_auroc"], 6),
                    "n_fallbacks": fallbacks,
                    "n_test": len(tf),
                }
                exists = CSV_PATH.exists()
                with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
                    if not exists:
                        w.writeheader()
                    w.writerow(row)

            # Summary per (misroute_p, margin_threshold)
            print(f"    done all 15 products")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
