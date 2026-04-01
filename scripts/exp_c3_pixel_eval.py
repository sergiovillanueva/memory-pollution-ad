"""
C3 Pixel Evaluation: Pixel-AUROC for all compression configs
=============================================================
Evaluates pixel-level anomaly localization for the final memory bank
of each compression configuration. NOT continual protocol — just build
the final bank from all 15 tasks and evaluate per-task.

Configs tested (10 total):
- baseline (full 117600 embs)
- coreset_500, coreset_200, coreset_100
- km100_res400, km100_res100
- k500, k200
- avg2x2 (grid_size=14), avg4x4 (grid_size=7)

Output: output/continual/c3_pixel_eval.csv
Resumable via get_completed().
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from continual_utils import (
    ALL_TASKS_FULL, RESOLUTION, KNN_K, N_PER_TASK, FEATURE_DIM,
    OUTPUT_DIR, CACHE_DIR,
    DINOv3Extractor,
    load_dataset_with_masks,
    extract_train_features,
    preload_test_data_with_masks,
    knn_score_maps,
    compute_metrics,
    compute_pixel_metrics,
    memory_mb,
    get_completed, save_result, log_error, cuda_cleanup,
    _cache_path,
)

# =============================================================================
# CONFIG
# =============================================================================

TASKS = ALL_TASKS_FULL
SEEDS = [0, 1, 2]
CSV_PATH = OUTPUT_DIR / "c3_pixel_eval.csv"
ERROR_PATH = OUTPUT_DIR / "c3_pixel_eval_errors.log"
PCA_DIM = 64

COLUMNS = [
    "timestamp", "seed", "config", "task_name", "smooth_sigma",
    "n_test", "memory_n_embeddings", "memory_mb", "grid_size",
    "img_auroc", "img_aupr", "tpr_fpr1",
    "pix_auroc", "pix_aupr", "spro", "auspro",
    "query_time_ms",
]

CONFIGS = [
    "baseline",
    "coreset_500", "coreset_200", "coreset_100",
    "km100_res400", "km100_res100",
    "k500", "k200",
    "avg2x2", "avg4x4",
]

# Smoothing ablation: sigma values per config
# Only test multiple sigmas for 3 representative configs (saves time)
SMOOTH_ABLATION_CONFIGS = {"baseline", "coreset_500", "avg4x4"}
SIGMAS_ABLATION = [0.0, 1.0, 4.0]
SIGMAS_DEFAULT = [0.0]


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
    total_budget = n_centroids + n_residuals
    if len(features_flat) <= total_budget:
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
    distances = distances.ravel()

    r = min(n_residuals, len(features_flat))
    residual_indices = np.argsort(distances)[-r:]
    residuals = features_flat[residual_indices]

    return np.concatenate([centroids, residuals], axis=0)


def apply_spatial(features: np.ndarray, config: str) -> np.ndarray:
    """Apply spatial pooling: [N, P, D] -> [N, P', D]."""
    if config == "full":
        return features
    N, P, D = features.shape
    grid_size = int(P**0.5)
    grid = features.reshape(N, grid_size, grid_size, D)

    if config == "avg2x2":
        ng = grid_size // 2
        out = grid[:, :ng*2, :ng*2, :].reshape(N, ng, 2, ng, 2, D).mean(axis=(2, 4))
    elif config == "avg4x4":
        ng = grid_size // 4
        out = grid[:, :ng*4, :ng*4, :].reshape(N, ng, 4, ng, 4, D).mean(axis=(2, 4))
    else:
        raise ValueError(f"Unknown spatial config: {config}")
    return out.reshape(N, -1, D)


# =============================================================================
# BUILD FINAL MEMORY BANK
# =============================================================================

def build_final_bank(
    config: str,
    tasks: list[str],
    seed: int,
    extractor: DINOv3Extractor,
) -> tuple[np.ndarray, int]:
    """Build final compressed memory bank from all tasks.

    Returns:
        (memory_bank [M, D], grid_size for scoring)
    """
    # Collect all train features
    all_features: list[np.ndarray] = []
    for task_name in tasks:
        feat, _ = extract_train_features(extractor, task_name, seed, N_PER_TASK)
        # feat shape: [N_images, 784, 1024]

        if config in ("avg2x2", "avg4x4"):
            feat = apply_spatial(feat, config)

        all_features.append(feat)

    # Stack: [total_images, P, D]
    all_feat = np.concatenate(all_features, axis=0)
    n_patches = all_feat.shape[1]
    grid_size = int(n_patches**0.5)

    # Flatten to [total_patches, D] for compression
    flat = all_feat.reshape(-1, all_feat.shape[-1])

    if config == "baseline":
        bank = flat
    elif config.startswith("coreset_"):
        budget = int(config.split("_")[1]) * len(tasks)
        bank = kcenter_greedy(flat, budget, seed)
    elif config.startswith("km") and "_res" in config:
        parts = config.split("_")
        k = int(parts[0][2:])
        r = int(parts[1][3:])
        # Per-task compression then concatenate
        bank_parts = []
        for feat in all_features:
            f = feat.reshape(-1, feat.shape[-1])
            bank_parts.append(compress_kmeans_residuals(f, k, r))
        bank = np.concatenate(bank_parts, axis=0)
    elif config.startswith("k") and not config.startswith("km"):
        budget_per_task = int(config[1:])
        bank_parts = []
        for feat in all_features:
            f = feat.reshape(-1, feat.shape[-1])
            k = min(budget_per_task, len(f))
            kmeans = MiniBatchKMeans(
                n_clusters=k, batch_size=min(4096, len(f)),
                n_init=3, max_iter=100, random_state=42,
            )
            kmeans.fit(f)
            bank_parts.append(kmeans.cluster_centers_.astype(np.float32))
        bank = np.concatenate(bank_parts, axis=0)
    elif config in ("avg2x2", "avg4x4"):
        bank = flat  # Already pooled above
    else:
        raise ValueError(f"Unknown config: {config}")

    return bank, grid_size


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = get_completed(CSV_PATH, ["seed", "config", "task_name", "smooth_sigma"])
    print(f"[INIT] {len(completed)} completed experiments found")

    # Count remaining
    total = 0
    for config in CONFIGS:
        sigmas = SIGMAS_ABLATION if config in SMOOTH_ABLATION_CONFIGS else SIGMAS_DEFAULT
        total += len(SEEDS) * len(sigmas) * len(TASKS)
    remaining = total - len(completed)
    print(f"[INIT] {remaining}/{total} experiments remaining")
    if remaining == 0:
        print("[DONE] All experiments completed!")
        return

    # Load extractor (for train features if not cached)
    extractor = DINOv3Extractor()

    # Pre-load all test data with masks
    test_data = preload_test_data_with_masks(extractor, TASKS)

    for seed in SEEDS:
        for config in CONFIGS:
            sigmas = SIGMAS_ABLATION if config in SMOOTH_ABLATION_CONFIGS else SIGMAS_DEFAULT

            # Check if ALL (task, sigma) for this (seed, config) are done
            all_done = all(
                (str(seed), config, task, str(sigma)) in completed
                for task in TASKS for sigma in sigmas
            )
            if all_done:
                print(f"\n[SKIP] seed={seed}, config={config} — all done")
                continue

            print(f"\n[BUILD] seed={seed}, config={config} (sigmas={sigmas})")
            t0 = time.perf_counter()

            try:
                bank, grid_size = build_final_bank(config, TASKS, seed, extractor)
                build_time = time.perf_counter() - t0
                n_embs = len(bank)
                mem_mb = memory_mb(n_embs)
                print(f"  Bank: {n_embs} embs, {mem_mb:.1f} MB, grid={grid_size} ({build_time:.1f}s)")
            except Exception as e:
                log_error(ERROR_PATH, f"build_bank seed={seed} config={config}", e)
                print(f"  ERROR building bank: {e}")
                continue

            # Evaluate each task — compute score maps once, then iterate sigmas
            for task_name in TASKS:
                # Check if all sigmas done for this task
                all_sigmas_done = all(
                    (str(seed), config, task_name, str(sigma)) in completed
                    for sigma in sigmas
                )
                if all_sigmas_done:
                    continue

                try:
                    test_features, test_labels, test_info = test_data[task_name]

                    # Apply spatial pooling to test features if needed
                    if config in ("avg2x2", "avg4x4"):
                        test_feat_eval = apply_spatial(test_features, config)
                    else:
                        test_feat_eval = test_features

                    # Score with maps (compute ONCE, reuse for all sigmas)
                    t0 = time.perf_counter()
                    image_scores, score_maps = knn_score_maps(
                        bank, test_feat_eval, k=KNN_K, grid_size=grid_size
                    )
                    query_time_total = (time.perf_counter() - t0) * 1000
                    query_time_per_img = query_time_total / len(test_labels)

                    # Image metrics (same for all sigmas)
                    img_metrics = compute_metrics(test_labels, image_scores)

                    for sigma in sigmas:
                        key = (str(seed), config, task_name, str(sigma))
                        if key in completed:
                            continue

                        # Pixel metrics with this sigma
                        pix_metrics = compute_pixel_metrics(score_maps, test_info, smooth_sigma=sigma)

                        result = {
                            "timestamp": datetime.now().isoformat(),
                            "seed": seed,
                            "config": config,
                            "task_name": task_name,
                            "smooth_sigma": sigma,
                            "n_test": len(test_labels),
                            "memory_n_embeddings": n_embs,
                            "memory_mb": round(mem_mb, 3),
                            "grid_size": grid_size,
                            "img_auroc": img_metrics["img_auroc"],
                            "img_aupr": img_metrics["img_aupr"],
                            "tpr_fpr1": img_metrics["tpr_fpr1"],
                            "pix_auroc": pix_metrics["pix_auroc"],
                            "pix_aupr": pix_metrics["pix_aupr"],
                            "spro": pix_metrics["spro"],
                            "auspro": pix_metrics["auspro"],
                            "query_time_ms": round(query_time_per_img, 2),
                        }

                        save_result(CSV_PATH, COLUMNS, result)
                        completed.add(key)

                    short = task_name.split("/")[-1]
                    pix_str = f"pix={pix_metrics['pix_auroc']:.3f}" if pix_metrics['pix_auroc'] else "pix=N/A"
                    sigma_str = f" σ={sigmas}" if len(sigmas) > 1 else ""
                    print(f"    {short:15s} img={img_metrics['img_auroc']:.3f} {pix_str}{sigma_str} q={query_time_per_img:.0f}ms")

                except Exception as e:
                    log_error(ERROR_PATH, f"eval seed={seed} config={config} task={task_name}", e)
                    print(f"    {task_name}: ERROR - {e}")

            cuda_cleanup()

    print("\n[DONE] All pixel evaluation experiments completed!")


if __name__ == "__main__":
    main()
