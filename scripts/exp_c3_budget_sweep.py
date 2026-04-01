"""
C3 Budget Sweep: Per-product AUROC(budget) curves
===================================================
For each product individually, evaluate detection performance at various
memory budgets. This is NOT continual — each product is evaluated in isolation.

Two compressors:
  - coreset: budgets {25, 50, 75, 100, 150, 200, 300, 500, 784}
  - km_res: K centroids + R residuals, various K:R ratios

Also computes complexity_score (normal-only proxy for adaptive allocation).

Output: output/continual/c3_budget_sweep.csv
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from continual_utils import (
    ALL_TASKS_FULL,
    KNN_K, N_PER_TASK, FEATURE_DIM, OUTPUT_DIR,
    DINOv3Extractor,
    extract_train_features,
    preload_test_data_with_masks,
    knn_score_images, knn_score_maps,
    compute_metrics, compute_pixel_metrics,
    memory_mb, complexity_score,
    get_completed, save_result, log_error, cuda_cleanup,
)

# =============================================================================
# CONFIG
# =============================================================================

TASKS = ALL_TASKS_FULL  # Start with MVTec, add VisA/BTAD later
SEEDS = [0, 1, 2]
PCA_DIM = 64

CSV_PATH = OUTPUT_DIR / "c3_budget_sweep.csv"
ERROR_PATH = OUTPUT_DIR / "c3_budget_sweep_errors.log"

COLUMNS = [
    "timestamp", "seed", "compressor", "budget_embs", "task_name",
    "n_test", "n_train_patches", "memory_n_embeddings", "memory_mb",
    "img_auroc", "img_aupr", "tpr_fpr1",
    "pix_auroc", "pix_aupr", "spro", "auspro",
    "complexity_score", "query_time_ms",
]

# Coreset budgets (embeddings per product)
CORESET_BUDGETS = [25, 50, 75, 100, 150, 200, 300, 500, 784]

# KM+Res budgets: (n_centroids, n_residuals) per product
KMRES_CONFIGS = [
    (10, 15),    # 25 total
    (20, 30),    # 50 total
    (30, 45),    # 75 total
    (40, 60),    # 100 total
    (60, 90),    # 150 total
    (80, 120),   # 200 total
    (100, 200),  # 300 total
    (150, 350),  # 500 total
    (200, 584),  # 784 total (full)
]


# =============================================================================
# COMPRESSION FUNCTIONS
# =============================================================================

def kcenter_greedy(data: np.ndarray, budget: int, seed: int = 42) -> np.ndarray:
    """K-center greedy coreset selection."""
    n = len(data)
    if n <= budget:
        return data.copy()

    pca_dim = min(PCA_DIM, data.shape[1], n)
    pca = PCA(n_components=pca_dim, random_state=seed)
    data_pca = pca.fit_transform(data)

    rng = np.random.RandomState(seed)
    first_idx = rng.randint(0, n)
    selected = [first_idx]
    min_dists = np.full(n, np.inf, dtype=np.float32)

    diffs = data_pca - data_pca[first_idx]
    min_dists = np.minimum(min_dists, np.einsum('ij,ij->i', diffs, diffs))

    for _ in range(budget - 1):
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        diffs = data_pca - data_pca[next_idx]
        min_dists = np.minimum(min_dists, np.einsum('ij,ij->i', diffs, diffs))

    return data[np.array(selected)]


def compress_kmeans_residuals(
    features_flat: np.ndarray,
    n_centroids: int,
    n_residuals: int,
) -> np.ndarray:
    """K centroids + R farthest-from-centroid patches."""
    total = n_centroids + n_residuals
    if len(features_flat) <= total:
        return features_flat.copy()

    k = min(n_centroids, len(features_flat))
    kmeans = MiniBatchKMeans(
        n_clusters=k, batch_size=min(4096, len(features_flat)),
        n_init=3, max_iter=100, random_state=42,
    )
    kmeans.fit(features_flat)
    centroids = kmeans.cluster_centers_.astype(np.float32)

    if n_residuals == 0:
        return centroids

    nn = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='auto')
    nn.fit(centroids)
    distances, _ = nn.kneighbors(features_flat)
    r = min(n_residuals, len(features_flat))
    residual_indices = np.argsort(distances.ravel())[-r:]

    return np.concatenate([centroids, features_flat[residual_indices]], axis=0)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = get_completed(CSV_PATH, ["seed", "compressor", "budget_embs", "task_name"])
    print(f"[INIT] {len(completed)} completed experiments found")

    # Build experiment list
    experiments: list[tuple[int, str, int, str]] = []  # (seed, compressor, budget, task)
    for seed in SEEDS:
        for task in TASKS:
            for budget in CORESET_BUDGETS:
                experiments.append((seed, "coreset", budget, task))
            for k, r in KMRES_CONFIGS:
                experiments.append((seed, f"km{k}_res{r}", k + r, task))

    remaining = [e for e in experiments if (str(e[0]), e[1], str(e[2]), e[3]) not in completed]
    print(f"[INIT] {len(remaining)}/{len(experiments)} remaining")
    if not remaining:
        print("[DONE] All experiments completed!")
        return

    extractor = DINOv3Extractor()

    # Pre-load test data with masks
    test_data = preload_test_data_with_masks(extractor, TASKS)

    # Cache complexity scores per (task, seed)
    complexity_cache: dict[tuple[str, int], float] = {}

    for seed in SEEDS:
        for task_name in TASKS:
            # Load train features
            train_feat, _ = extract_train_features(extractor, task_name, seed, N_PER_TASK)
            flat = train_feat.reshape(-1, train_feat.shape[-1])
            n_train_patches = len(flat)

            # Compute complexity (once per task/seed)
            cache_key = (task_name, seed)
            if cache_key not in complexity_cache:
                complexity_cache[cache_key] = complexity_score(train_feat)

            cs = complexity_cache[cache_key]
            test_features, test_labels, test_info = test_data[task_name]
            short = task_name.split("/")[-1]

            # Coreset budgets
            for budget in CORESET_BUDGETS:
                key = (str(seed), "coreset", str(budget), task_name)
                if key in completed:
                    continue

                try:
                    bank = kcenter_greedy(flat, budget, seed)
                    n_embs = len(bank)
                    mem_mb = memory_mb(n_embs)

                    t0 = time.perf_counter()
                    image_scores, score_maps = knn_score_maps(bank, test_features, k=KNN_K, grid_size=28)
                    query_ms = (time.perf_counter() - t0) * 1000 / len(test_labels)

                    img_m = compute_metrics(test_labels, image_scores)
                    pix_m = compute_pixel_metrics(score_maps, test_info)

                    result = {
                        "timestamp": datetime.now().isoformat(),
                        "seed": seed,
                        "compressor": "coreset",
                        "budget_embs": budget,
                        "task_name": task_name,
                        "n_test": len(test_labels),
                        "n_train_patches": n_train_patches,
                        "memory_n_embeddings": n_embs,
                        "memory_mb": round(mem_mb, 3),
                        "img_auroc": img_m["img_auroc"],
                        "img_aupr": img_m["img_aupr"],
                        "tpr_fpr1": img_m["tpr_fpr1"],
                        "pix_auroc": pix_m["pix_auroc"],
                        "pix_aupr": pix_m["pix_aupr"],
                        "spro": pix_m["spro"],
                        "auspro": pix_m["auspro"],
                        "complexity_score": round(cs, 4),
                        "query_time_ms": round(query_ms, 2),
                    }
                    save_result(CSV_PATH, COLUMNS, result)
                    completed.add(key)
                    print(f"  s{seed} {short:15s} coreset_{budget:3d} img={img_m['img_auroc']:.3f} pix={pix_m['pix_auroc'] or 0:.3f}")

                except Exception as e:
                    log_error(ERROR_PATH, f"coreset s={seed} b={budget} t={task_name}", e)
                    print(f"  ERROR: {e}")

            # KM+Res budgets
            for k_c, r_c in KMRES_CONFIGS:
                budget = k_c + r_c
                compressor = f"km{k_c}_res{r_c}"
                key = (str(seed), compressor, str(budget), task_name)
                if key in completed:
                    continue

                try:
                    bank = compress_kmeans_residuals(flat, k_c, r_c)
                    n_embs = len(bank)
                    mem_mb = memory_mb(n_embs)

                    t0 = time.perf_counter()
                    image_scores, score_maps = knn_score_maps(bank, test_features, k=KNN_K, grid_size=28)
                    query_ms = (time.perf_counter() - t0) * 1000 / len(test_labels)

                    img_m = compute_metrics(test_labels, image_scores)
                    pix_m = compute_pixel_metrics(score_maps, test_info)

                    result = {
                        "timestamp": datetime.now().isoformat(),
                        "seed": seed,
                        "compressor": compressor,
                        "budget_embs": budget,
                        "task_name": task_name,
                        "n_test": len(test_labels),
                        "n_train_patches": n_train_patches,
                        "memory_n_embeddings": n_embs,
                        "memory_mb": round(mem_mb, 3),
                        "img_auroc": img_m["img_auroc"],
                        "img_aupr": img_m["img_aupr"],
                        "tpr_fpr1": img_m["tpr_fpr1"],
                        "pix_auroc": pix_m["pix_auroc"],
                        "pix_aupr": pix_m["pix_aupr"],
                        "spro": pix_m["spro"],
                        "auspro": pix_m["auspro"],
                        "complexity_score": round(cs, 4),
                        "query_time_ms": round(query_ms, 2),
                    }
                    save_result(CSV_PATH, COLUMNS, result)
                    completed.add(key)
                    print(f"  s{seed} {short:15s} {compressor:15s} img={img_m['img_auroc']:.3f} pix={pix_m['pix_auroc'] or 0:.3f}")

                except Exception as e:
                    log_error(ERROR_PATH, f"km_res s={seed} k={k_c} r={r_c} t={task_name}", e)
                    print(f"  ERROR: {e}")

        cuda_cleanup()

    print("\n[DONE] Budget sweep complete!")


if __name__ == "__main__":
    main()
