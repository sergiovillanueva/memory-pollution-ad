"""
C3 Extra Benchmarks: Run key configs on VisA and BTAD
======================================================
Repeats a subset of experiments on additional benchmarks:
  - VisA: 12 categories, N=10, 3 seeds
  - BTAD: 3 categories, N=10, 3 seeds

Configs: baseline, coreset_200, coreset_500, km100_res400
Full continual protocol + pixel-AUROC on final task.

Requires: precache_features_extra.py to have run first.
Output: output/continual/c3_extra_benchmarks.csv
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
    ALL_TASKS_VISA, ALL_TASKS_BTAD,
    KNN_K, N_PER_TASK, FEATURE_DIM, OUTPUT_DIR,
    DINOv3Extractor,
    extract_train_features,
    preload_test_data_with_masks,
    knn_score_images, knn_score_maps,
    compute_metrics, compute_pixel_metrics,
    memory_mb, complexity_score,
    get_completed, save_result, log_error, cuda_cleanup,
    _cache_path,
)

# =============================================================================
# CONFIG
# =============================================================================

SEEDS = [0, 1, 2]
PCA_DIM = 64

BENCHMARKS = {
    "VisA": ALL_TASKS_VISA,
    "BTAD": ALL_TASKS_BTAD,
}

CONFIGS = {
    "baseline": {},
    "coreset_200": {"budget_per_task": 200},
    "coreset_500": {"budget_per_task": 500},
    "km100_res400": {"n_centroids": 100, "n_residuals": 400},
}

CSV_PATH = OUTPUT_DIR / "c3_extra_benchmarks.csv"
ERROR_PATH = OUTPUT_DIR / "c3_extra_benchmarks_errors.log"

COLUMNS = [
    "timestamp", "seed", "benchmark", "config", "task_idx", "task_name", "eval_task",
    "n_test", "memory_n_embeddings", "memory_mb",
    "img_auroc", "img_aupr", "tpr_fpr1",
    "pix_auroc", "pix_aupr", "spro", "auspro",
    "complexity_score", "query_time_ms",
]


# =============================================================================
# COMPRESSION FUNCTIONS
# =============================================================================

def kcenter_greedy(data: np.ndarray, budget: int, seed: int = 42) -> np.ndarray:
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


def compress_kmeans_residuals(features_flat: np.ndarray, n_centroids: int, n_residuals: int) -> np.ndarray:
    total = n_centroids + n_residuals
    if len(features_flat) <= total:
        return features_flat.copy()
    k = min(n_centroids, len(features_flat))
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=min(4096, len(features_flat)), n_init=3, max_iter=100, random_state=42)
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


def compress_task(features_3d: np.ndarray, config_name: str, config_params: dict, seed: int) -> np.ndarray:
    """Compress one task's features according to config."""
    flat = features_3d.reshape(-1, features_3d.shape[-1])
    if config_name == "baseline":
        return flat
    elif config_name.startswith("coreset_"):
        budget = config_params["budget_per_task"]
        return kcenter_greedy(flat, budget, seed)
    elif config_name.startswith("km"):
        return compress_kmeans_residuals(flat, config_params["n_centroids"], config_params["n_residuals"])
    else:
        raise ValueError(f"Unknown config: {config_name}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = get_completed(CSV_PATH, ["seed", "benchmark", "config", "task_idx", "eval_task"])
    print(f"[INIT] {len(completed)} completed experiments found")

    extractor = DINOv3Extractor()

    for bench_name, tasks in BENCHMARKS.items():
        # Check if features are cached
        missing = [t for t in tasks if not _cache_path("test_feat", t).exists()]
        if missing:
            print(f"\n[WARN] Missing caches for {bench_name}: {missing}")
            print("       Run precache_features_extra.py first!")
            continue

        test_data = preload_test_data_with_masks(extractor, tasks)

        for seed in SEEDS:
            for config_name, config_params in CONFIGS.items():
                print(f"\n{'='*60}")
                print(f"[RUN] {bench_name} seed={seed} config={config_name}")

                memory_bank = np.empty((0, FEATURE_DIM), dtype=np.float32)

                for task_idx, task_name in enumerate(tasks):
                    train_feat, _ = extract_train_features(extractor, task_name, seed, N_PER_TASK)
                    task_bank = compress_task(train_feat, config_name, config_params, seed)
                    memory_bank = np.concatenate([memory_bank, task_bank], axis=0)

                    n_embs = len(memory_bank)
                    mem_mb = memory_mb(n_embs)
                    is_final = (task_idx == len(tasks) - 1)

                    cs = complexity_score(train_feat)
                    img_m = None

                    for eval_idx in range(task_idx + 1):
                        eval_task = tasks[eval_idx]
                        key = (str(seed), bench_name, config_name, str(task_idx), eval_task)
                        if key in completed:
                            continue

                        try:
                            test_features, test_labels, test_info = test_data[eval_task]

                            t0 = time.perf_counter()
                            image_scores = knn_score_images(memory_bank, test_features, k=KNN_K)
                            query_ms = (time.perf_counter() - t0) * 1000 / len(test_labels)

                            img_m = compute_metrics(test_labels, image_scores)

                            pix_m = {"pix_auroc": None, "pix_aupr": None, "spro": None, "auspro": None}
                            if is_final:
                                _, score_maps = knn_score_maps(memory_bank, test_features, k=KNN_K, grid_size=28)
                                pix_m = compute_pixel_metrics(score_maps, test_info)

                            result = {
                                "timestamp": datetime.now().isoformat(),
                                "seed": seed,
                                "benchmark": bench_name,
                                "config": config_name,
                                "task_idx": task_idx,
                                "task_name": task_name,
                                "eval_task": eval_task,
                                "n_test": len(test_labels),
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

                        except Exception as e:
                            log_error(ERROR_PATH, f"{bench_name} s={seed} c={config_name} t={task_idx} e={eval_task}", e)
                            print(f"  ERROR: {e}")

                    short = task_name.split("/")[-1]
                    if img_m is not None:
                        print(f"  Task {task_idx:2d} {short:15s} bank={n_embs:6d} auroc={img_m['img_auroc']:.3f}")
                    else:
                        print(f"  Task {task_idx:2d} {short:15s} bank={n_embs:6d} (skipped - already completed)")

                cuda_cleanup()

    print("\n[DONE] Extra benchmarks complete!")


if __name__ == "__main__":
    main()
