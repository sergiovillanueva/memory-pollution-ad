"""
C2_003: Coreset k-Center Greedy (Optimized)
============================================
Select most representative embeddings using k-center greedy.
Optimized: incremental min_dists (O(M*budget) not O(M^2)), PCA64 previa.
Tests: full (baseline), coreset_N for N in {50, 100, 200, 500} per task.

Key question: Does geometric coverage (coreset) outperform random (reservoir)
or clustering (prototypes) at the same memory budget?

Usage: uv run exp_c2_003_coreset.py [--debug] [--device 0]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
args, _ = parser.parse_known_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

import numpy as np
from sklearn.decomposition import PCA
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from continual_utils import (
    ALL_TASKS_FULL, OUTPUT_DIR, N_PER_TASK, KNN_K,
    DINOv3Extractor, load_dataset, compute_metrics,
    knn_score_images, knn_score_single_image,
    get_completed, save_result, log_error, cuda_cleanup,
    preload_test_data, extract_train_features,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEBUG = "--debug" in sys.argv
DEVICE = "cuda:0"

ALL_TASKS = ALL_TASKS_FULL[:4] if DEBUG else ALL_TASKS_FULL
SEEDS = [0] if DEBUG else [0, 1, 2]

EXPERIMENT = "c2_003"
CSV_PATH = OUTPUT_DIR / "c2_003_coreset.csv"
ERROR_PATH = OUTPUT_DIR / "c2_003_errors.txt"

COLUMNS = [
    "timestamp", "seed", "config", "task_idx", "task_name", "n_tasks_seen",
    "eval_task_idx", "eval_task_name",
    "n_train_task", "n_test", "memory_n_embeddings", "memory_mb",
    "img_auroc", "img_aupr", "img_f1max", "tpr_fpr1", "tpr_fpr5",
    "update_time_s", "query_time_ms", "eval_time_s",
]

# full baseline already in C2_001. Only test compressed variants.
CONFIGS = ["coreset_50", "coreset_100", "coreset_200", "coreset_500"]
PCA_DIM = 64  # PCA for faster distance computation during selection


# =============================================================================
# CORESET FUNCTIONS
# =============================================================================

def kcenter_greedy(data: np.ndarray, budget: int, seed: int = 42) -> np.ndarray:
    """K-center greedy coreset selection (incremental, O(M*budget)).

    Args:
        data: [M, D] embeddings
        budget: number of points to select

    Returns:
        [budget, D_original] selected embeddings from original data
    """
    n = len(data)
    if n <= budget:
        return data.copy()

    # PCA for fast distance computation
    pca_dim = min(PCA_DIM, data.shape[1], n)
    pca = PCA(n_components=pca_dim, random_state=seed)
    data_pca = pca.fit_transform(data)

    # Initialize: pick random first point
    rng = np.random.RandomState(seed)
    first_idx = rng.randint(0, n)
    selected = [first_idx]

    # min_dists[i] = min distance from point i to any selected point
    min_dists = np.full(n, np.inf, dtype=np.float32)

    # Update distances from first point
    diffs = data_pca - data_pca[first_idx]
    dists_to_first = np.einsum('ij,ij->i', diffs, diffs)  # squared L2
    min_dists = np.minimum(min_dists, dists_to_first)

    for _ in range(budget - 1):
        # Pick the farthest point from selected set
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)

        # Update min_dists incrementally
        diffs = data_pca - data_pca[next_idx]
        dists_to_new = np.einsum('ij,ij->i', diffs, diffs)
        min_dists = np.minimum(min_dists, dists_to_new)

    # Return original (non-PCA) embeddings
    return data[np.array(selected)]


def parse_config(config: str) -> int | None:
    """Parse config -> budget per task. None for full."""
    if config == "full":
        return None
    return int(config.split("_")[1])


# =============================================================================
# CONTINUAL PROTOCOL
# =============================================================================

def run_protocol(
    seed: int,
    config: str,
    extractor: DINOv3Extractor,
    test_data: dict[str, tuple[np.ndarray, np.ndarray]],
    completed: set,
) -> None:
    """Run continual protocol for one seed and config."""
    budget = parse_config(config)

    print(f"\n{'='*70}")
    budget_str = f"budget={budget}/task" if budget else "unbounded"
    print(f"SEED {seed} | CONFIG: {config} ({budget_str})")
    print(f"{'='*70}")

    memory_banks: dict[str, np.ndarray] = {}

    for task_idx, task_name in enumerate(ALL_TASKS):
        n_tasks_seen = task_idx + 1
        short = task_name.split("/")[-1]

        # Check what needs evaluation
        evals_needed = []
        for eval_idx in range(n_tasks_seen):
            key = (str(seed), config, str(task_idx), str(eval_idx))
            if key not in completed:
                evals_needed.append(eval_idx)

        # --- UPDATE PHASE ---
        t_update = time.time()
        train_features, n_train = extract_train_features(extractor, task_name, seed)
        flat_train = train_features.reshape(-1, train_features.shape[-1])

        if budget is not None:
            compressed = kcenter_greedy(flat_train, budget, seed=seed)
        else:
            compressed = flat_train

        memory_banks[task_name] = compressed
        update_time = time.time() - t_update

        if not evals_needed:
            if DEBUG:
                print(f"  [Task {task_idx}] {short} - all done ({len(compressed)} embs)")
            continue

        total_embs = sum(len(v) for v in memory_banks.values())
        memory_mb = total_embs * compressed.shape[-1] * 4 / (1024 * 1024)

        if DEBUG:
            print(f"  [Task {task_idx}] {short}: {len(flat_train)}->{len(compressed)} embs, "
                  f"total: {total_embs} ({memory_mb:.1f} MB)")

        # --- EVALUATION PHASE ---
        global_bank = np.concatenate(list(memory_banks.values()), axis=0)

        from sklearn.neighbors import NearestNeighbors as NN
        k_actual = min(KNN_K, len(global_bank))
        nn_index = NN(n_neighbors=k_actual, metric='euclidean', algorithm='auto')
        nn_index.fit(global_bank)

        for eval_idx in evals_needed:
            eval_task = ALL_TASKS[eval_idx]
            eval_short = eval_task.split("/")[-1]

            try:
                test_features_raw, test_labels = test_data[eval_task]

                t_eval = time.time()
                scores = knn_score_images(global_bank, test_features_raw, k=KNN_K)
                eval_time = time.time() - t_eval

                _, query_ms = knn_score_single_image(
                    global_bank, test_features_raw[0], nn_index=nn_index, k=KNN_K
                )

                metrics = compute_metrics(test_labels, scores)

                result = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "seed": seed,
                    "config": config,
                    "task_idx": task_idx,
                    "task_name": task_name,
                    "n_tasks_seen": n_tasks_seen,
                    "eval_task_idx": eval_idx,
                    "eval_task_name": eval_task,
                    "n_train_task": n_train,
                    "n_test": len(test_labels),
                    "memory_n_embeddings": total_embs,
                    "memory_mb": round(memory_mb, 2),
                    "update_time_s": round(update_time, 2),
                    "query_time_ms": round(query_ms, 2),
                    "eval_time_s": round(eval_time, 2),
                    **metrics,
                }

                save_result(CSV_PATH, COLUMNS, result)

                if DEBUG:
                    print(f"    eval {eval_short}: AUROC={metrics['img_auroc']:.3f} "
                          f"(query: {query_ms:.1f}ms, eval: {eval_time:.1f}s)")
                else:
                    print(f"    eval {eval_short}: AUROC={metrics['img_auroc']:.3f}", flush=True)

            except Exception as e:
                print(f"    eval {eval_short}: ERROR - {e}")
                log_error(ERROR_PATH, f"seed={seed} cfg={config} task={task_idx} eval={eval_idx}", e)

        cuda_cleanup(f"after task {task_idx}", DEBUG)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("C2_003: Coreset k-Center Greedy (Optimized)")
    print(f"  Tasks: {len(ALL_TASKS)}, Seeds: {SEEDS}")
    print(f"  Configs: {CONFIGS}")
    print(f"  Debug: {DEBUG}")
    print("=" * 70)

    completed = get_completed(CSV_PATH, ["seed", "config", "task_idx", "eval_task_idx"])
    total = sum(i + 1 for i in range(len(ALL_TASKS))) * len(SEEDS) * len(CONFIGS)
    print(f"[INFO] Completed: {len(completed)}/{total}")

    extractor = DINOv3Extractor(DEVICE)
    test_data = preload_test_data(extractor, ALL_TASKS)

    for seed in SEEDS:
        for config in CONFIGS:
            run_protocol(seed, config, extractor, test_data, completed)

    print("\n" + "=" * 70)
    print(f"C2_003 COMPLETE. Results: {CSV_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
