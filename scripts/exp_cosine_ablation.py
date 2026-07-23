"""
Distance Metric Ablation: cosine vs euclidean.
MVTec AD, 1 seed, layer 18, 300 embs/product (uniform).
Uses cached features from output/feature_cache/.

Output: output/continual/cosine_ablation.csv
"""
from __future__ import annotations
import os, sys, csv, gc, time
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from continual_utils import (
    compute_metrics, RESOLUTION, KNN_K, FEATURE_DIM,
)
from sklearn.decomposition import PCA

SEED = 0
N_PER_TASK = 10
BUDGET_PER_PRODUCT = 300
TASKS = [
    "mvtec_AD/bottle", "mvtec_AD/cable", "mvtec_AD/capsule",
    "mvtec_AD/carpet", "mvtec_AD/grid", "mvtec_AD/hazelnut",
    "mvtec_AD/leather", "mvtec_AD/metal_nut", "mvtec_AD/pill",
    "mvtec_AD/screw", "mvtec_AD/tile", "mvtec_AD/toothbrush",
    "mvtec_AD/transistor", "mvtec_AD/wood", "mvtec_AD/zipper",
]
CACHE = Path("output/feature_cache")
OUTPUT_DIR = Path("output/continual")
CSV_PATH = OUTPUT_DIR / "cosine_ablation.csv"
COLUMNS = ["timestamp", "metric", "seed", "task", "budget_per_product",
           "img_auroc", "wnr"]


def cache_path(prefix: str, task: str, suffix: str = "") -> Path:
    return CACHE / f"{prefix}_{task.replace('/', '_')}{suffix}.npy"


def kcenter_greedy(data: np.ndarray, budget: int, seed: int) -> np.ndarray:
    n = len(data)
    if budget >= n:
        return data
    pca_dim = min(64, data.shape[1], n)
    pca = PCA(n_components=pca_dim, random_state=seed)
    data_pca = pca.fit_transform(data)
    rng = np.random.RandomState(seed)
    first_idx = rng.randint(0, n)
    selected = [first_idx]
    min_dists = np.full(n, np.inf)
    diffs = data_pca - data_pca[first_idx]
    min_dists = np.minimum(min_dists, np.einsum('ij,ij->i', diffs, diffs))
    for _ in range(budget - 1):
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        diffs = data_pca - data_pca[next_idx]
        min_dists = np.minimum(min_dists, np.einsum('ij,ij->i', diffs, diffs))
    return data[np.array(selected)]


def knn_score_images(bank: np.ndarray, test_feats: np.ndarray,
                     k: int = KNN_K, metric: str = "euclidean") -> np.ndarray:
    n, p, d = test_feats.shape
    flat = test_feats.reshape(-1, d)
    k_actual = min(k, len(bank))
    nn = NearestNeighbors(n_neighbors=k_actual, metric=metric, algorithm='brute')
    nn.fit(bank)
    distances, _ = nn.kneighbors(flat)
    patch_scores = distances.mean(axis=1)
    patch_scores = patch_scores.reshape(n, p)
    return np.percentile(patch_scores, 95, axis=1)


def compute_wnr(bank: np.ndarray, labels: np.ndarray,
                patches: np.ndarray, true_idx: int,
                k: int = KNN_K, metric: str = "euclidean") -> float:
    k_actual = min(k, len(bank))
    nn = NearestNeighbors(n_neighbors=k_actual, metric=metric, algorithm='brute')
    nn.fit(bank)
    _, indices = nn.kneighbors(patches)
    neighbor_labels = labels[indices]
    return float((neighbor_labels != true_idx).mean())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dist_metric in ["euclidean", "cosine"]:
        print(f"\n{'='*60}")
        print(f"METRIC: {dist_metric}")
        print(f"{'='*60}")

        # Build shared bank
        bank_parts, label_parts = [], []
        for idx, task in enumerate(TASKS):
            feat = np.load(cache_path("train_feat", task, f"_s{SEED}_n{N_PER_TASK}"))
            flat = feat.reshape(-1, feat.shape[-1])
            compressed = kcenter_greedy(flat, BUDGET_PER_PRODUCT, SEED)
            bank_parts.append(compressed)
            label_parts.append(np.full(len(compressed), idx, dtype=np.int32))

        full_bank = np.concatenate(bank_parts)
        product_labels = np.concatenate(label_parts)
        print(f"  Bank: {len(full_bank)} embeddings")

        for idx, task in enumerate(TASKS):
            short = task.split("/")[-1]
            test_feat = np.load(cache_path("test_feat", task))
            test_labels = np.load(cache_path("test_labels", task))

            scores = knn_score_images(full_bank, test_feat, metric=dist_metric)
            metrics = compute_metrics(test_labels, scores)

            wnr_vals = []
            for i in range(min(len(test_feat), 20)):
                wnr = compute_wnr(full_bank, product_labels, test_feat[i],
                                  idx, metric=dist_metric)
                wnr_vals.append(wnr)

            result = {
                "timestamp": datetime.now().isoformat(),
                "metric": dist_metric,
                "seed": SEED,
                "task": short,
                "budget_per_product": BUDGET_PER_PRODUCT,
                "img_auroc": round(metrics["img_auroc"], 6),
                "wnr": round(float(np.mean(wnr_vals)), 4),
            }

            exists = CSV_PATH.exists()
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
                if not exists:
                    w.writeheader()
                w.writerow(result)

            print(f"  {short:>15s}  AUROC={metrics['img_auroc']:.4f}  WNR={float(np.mean(wnr_vals)):.3f}")

    print(f"\n[DONE] {CSV_PATH}")


if __name__ == "__main__":
    main()
