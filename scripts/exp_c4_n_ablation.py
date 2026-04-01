"""
C4 N-Ablation: Effect of training set size on detection performance.
=====================================================================
Evaluates how the number of training images per product (N) affects
AUROC under both global and routed retrieval.

Tests N in {5, 10, 20, 50} with 3 seeds. Uses coreset compression at
18 MB total budget with uniform allocation.

This requires extracting new features for N != 10 (which need new caches).

Output: output/continual/c4_n_ablation.csv
"""
from __future__ import annotations

import sys
import os
os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
os.environ.setdefault("KMP_BLOCKTIME", "0")

import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from continual_utils import (
    ALL_TASKS_FULL, KNN_K, FEATURE_DIM,
    OUTPUT_DIR,
    DINOv3Extractor,
    extract_train_features,
    preload_test_data,
    knn_score_images, knn_score_images_gated,
    compute_metrics, compute_prototypes,
    memory_mb,
    get_completed, save_result, log_error, cuda_cleanup,
)

# =============================================================================
# CONFIG
# =============================================================================

TASKS = ALL_TASKS_FULL
SEEDS = [0, 1, 2]
N_VALUES = [5, 10, 20, 50]
PCA_DIM = 64

BUDGET_TOTAL_MB = 18.0
ROUTING_MODES = ["global", "top_1"]
BUDGET_OPTIONS = [25, 50, 75, 100, 150, 200, 300, 500]

CSV_PATH = OUTPUT_DIR / "c4_n_ablation.csv"
ERROR_PATH = OUTPUT_DIR / "c4_n_ablation_errors.log"

COLUMNS = [
    "timestamp", "seed", "n_per_task", "routing_mode",
    "budget_total_mb", "eval_task",
    "budget_per_product", "n_test",
    "memory_n_embeddings", "memory_mb",
    "img_auroc", "img_aupr", "tpr_fpr1",
    "query_time_ms",
]


# =============================================================================
# CORESET COMPRESSION
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


def mb_to_embs(mb: float, dim: int = FEATURE_DIM, bytes_per_val: int = 4) -> int:
    return int(mb * 1e6 / (dim * bytes_per_val))

def snap_to_options(budget: int) -> int:
    return min(BUDGET_OPTIONS, key=lambda x: abs(x - budget))


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = get_completed(CSV_PATH, [
        "seed", "n_per_task", "routing_mode", "eval_task"
    ])
    print(f"[INIT] {len(completed)} completed experiments found")

    extractor = DINOv3Extractor()
    test_data = preload_test_data(extractor, TASKS)

    for n_per_task in N_VALUES:
        for seed in SEEDS:
            combo = f"N={n_per_task} seed={seed}"
            print(f"\n{'='*60}")
            print(f"[RUN] {combo}")
            print("=" * 60)

            # Extract train features for this N
            train_feat_cache: dict[tuple[str, int], np.ndarray] = {}
            for task_name in TASKS:
                feat, _ = extract_train_features(
                    extractor, task_name, seed, n_per_task
                )
                train_feat_cache[(task_name, seed)] = feat
                short = task_name.split("/")[-1]
                print(f"  {short}: {feat.shape}")

            # Compute prototypes
            prototypes = compute_prototypes(train_feat_cache, TASKS, seed)

            # Allocate uniform budget
            per_product_mb = BUDGET_TOTAL_MB / len(TASKS)
            per_product_embs = snap_to_options(mb_to_embs(per_product_mb))

            # Build banks
            sub_banks: dict[str, np.ndarray] = {}
            full_bank = np.empty((0, FEATURE_DIM), dtype=np.float32)

            for task_name in TASKS:
                feat = train_feat_cache[(task_name, seed)]
                flat = feat.reshape(-1, feat.shape[-1])
                compressed = kcenter_greedy(flat, per_product_embs, seed)
                sub_banks[task_name] = compressed
                full_bank = np.concatenate([full_bank, compressed], axis=0)

            n_embs = len(full_bank)
            mem_mb = memory_mb(n_embs)
            print(f"  Bank: {n_embs} embs = {mem_mb:.1f} MB")

            # Evaluate each routing mode
            for routing_mode in ROUTING_MODES:
                for eval_task in TASKS:
                    key = (str(seed), str(n_per_task), routing_mode, eval_task)
                    if key in completed:
                        continue

                    try:
                        test_features, test_labels = test_data[eval_task]

                        t0 = time.perf_counter()
                        if routing_mode == "global":
                            scores = knn_score_images(full_bank, test_features)
                        else:
                            scores = knn_score_images_gated(
                                sub_banks, test_features, prototypes, top_m=1
                            )
                        query_ms = (time.perf_counter() - t0) * 1000 / len(test_labels)

                        metrics = compute_metrics(test_labels, scores)

                        result = {
                            "timestamp": datetime.now().isoformat(),
                            "seed": seed,
                            "n_per_task": n_per_task,
                            "routing_mode": routing_mode,
                            "budget_total_mb": BUDGET_TOTAL_MB,
                            "eval_task": eval_task,
                            "budget_per_product": per_product_embs,
                            "n_test": len(test_labels),
                            "memory_n_embeddings": n_embs,
                            "memory_mb": round(mem_mb, 3),
                            "img_auroc": metrics["img_auroc"],
                            "img_aupr": metrics["img_aupr"],
                            "tpr_fpr1": metrics["tpr_fpr1"],
                            "query_time_ms": round(query_ms, 2),
                        }
                        save_result(CSV_PATH, COLUMNS, result)
                        completed.add(key)

                        short = eval_task.split("/")[-1]
                        print(f"  {routing_mode:6s} {short:15s} AUROC={metrics['img_auroc']:.3f}")

                    except Exception as e:
                        log_error(ERROR_PATH,
                            f"N={n_per_task} s={seed} r={routing_mode} e={eval_task}", e)
                        print(f"  ERROR: {e}")

            cuda_cleanup()

    print(f"\n[DONE] N-ablation complete! Results: {CSV_PATH}")


if __name__ == "__main__":
    main()
