"""
C3 Adaptive Allocation: Adaptive vs Uniform budget distribution
================================================================
THE main contribution of the paper.

3 allocators:
  1. uniform:     each product gets B_total / N
  2. adaptive:    proportional to complexity_score (normal-only, NO data leakage)
  3. oracle:      greedy marginal gain using test AUROC curves (upper bound only)

Budget total in MB: {6, 12, 18, 24, 30} MB.
Budget per product in embeddings, converted from MB target.

Full continual protocol: 15 tasks, 3 seeds.
Static allocation (decided once, never re-allocated).

Output: output/continual/c3_adaptive.csv
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
    ALL_TASKS_FULL, KNN_K, N_PER_TASK, FEATURE_DIM,
    OUTPUT_DIR,
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

TASKS = ALL_TASKS_FULL
SEEDS = [0, 1, 2]
PCA_DIM = 64

# Budget total in MB
BUDGET_LEVELS_MB = [6.0, 12.0, 18.0, 24.0, 30.0]
ALLOCATORS = ["uniform", "adaptive", "oracle"]

# Discrete per-product budgets (embeddings)
BUDGET_OPTIONS = [25, 50, 75, 100, 150, 200, 300, 500]

CSV_PATH = OUTPUT_DIR / "c3_adaptive.csv"
ERROR_PATH = OUTPUT_DIR / "c3_adaptive_errors.log"

COLUMNS = [
    "timestamp", "seed", "allocator", "budget_total_mb",
    "task_idx", "task_name", "eval_task",
    "budget_per_product", "n_test", "memory_n_embeddings", "memory_mb",
    "img_auroc", "img_aupr", "tpr_fpr1",
    "pix_auroc", "pix_aupr", "spro", "auspro",
    "query_time_ms",
]


# =============================================================================
# CORESET COMPRESSION
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


# =============================================================================
# ALLOCATION STRATEGIES
# =============================================================================

def mb_to_embs(mb: float, dim: int = FEATURE_DIM, bytes_per_val: int = 4) -> int:
    """Convert MB budget to number of embeddings."""
    return int(mb * 1e6 / (dim * bytes_per_val))


def snap_to_options(budget: int) -> int:
    """Snap to nearest discrete budget option."""
    return min(BUDGET_OPTIONS, key=lambda x: abs(x - budget))


def allocate_uniform(budget_total_mb: float, n_products: int) -> list[int]:
    """Equal allocation to all products."""
    per_product_mb = budget_total_mb / n_products
    per_product_embs = mb_to_embs(per_product_mb)
    budget = snap_to_options(per_product_embs)
    return [budget] * n_products


def allocate_adaptive(
    budget_total_mb: float,
    complexity_scores: list[float],
    alpha: float = 1.0,
) -> list[int]:
    """Allocate proportional to complexity^alpha, snapped to discrete options.

    No data leakage: uses only normal-training-based complexity scores.
    """
    n = len(complexity_scores)
    total_embs = mb_to_embs(budget_total_mb)
    cs = np.array(complexity_scores)

    # Weights proportional to complexity^alpha
    weights = cs ** alpha
    weights /= weights.sum()

    # Initial allocation (continuous)
    raw_alloc = weights * total_embs

    # Snap to discrete budgets
    allocation = [snap_to_options(int(b)) for b in raw_alloc]

    # Adjust to stay within budget: reduce largest allocations if over budget
    while sum(allocation) > total_embs and max(allocation) > min(BUDGET_OPTIONS):
        idx = np.argmax(allocation)
        current = allocation[idx]
        lower = [b for b in BUDGET_OPTIONS if b < current]
        if lower:
            allocation[idx] = max(lower)
        else:
            break

    # If under budget, increase smallest allocations
    while sum(allocation) < total_embs * 0.95:
        idx = np.argmin(allocation)
        current = allocation[idx]
        higher = [b for b in BUDGET_OPTIONS if b > current]
        if higher:
            candidate = min(higher)
            if sum(allocation) - current + candidate <= total_embs:
                allocation[idx] = candidate
            else:
                break
        else:
            break

    return allocation


def allocate_oracle(
    budget_total_mb: float,
    auroc_curves: dict[str, dict[int, float]],
) -> list[int]:
    """Oracle: greedy marginal gain using test AUROC curves.

    This uses test-set information and is reported as UPPER BOUND only.

    Args:
        auroc_curves: dict[task_name -> {budget_embs -> auroc}]
    """
    tasks = list(auroc_curves.keys())
    n = len(tasks)
    total_embs = mb_to_embs(budget_total_mb)

    # Start with minimum budget for each product
    allocation = {t: min(BUDGET_OPTIONS) for t in tasks}

    remaining = total_embs - sum(allocation.values())

    while remaining > 0:
        best_task = None
        best_gain = -1
        best_new_budget = None

        for t in tasks:
            current = allocation[t]
            higher = [b for b in BUDGET_OPTIONS if b > current]
            if not higher:
                continue
            next_budget = min(higher)
            cost = next_budget - current
            if cost > remaining:
                continue
            current_auroc = auroc_curves[t].get(current, 0.0)
            next_auroc = auroc_curves[t].get(next_budget, current_auroc)
            gain = (next_auroc - current_auroc) / cost
            if gain > best_gain:
                best_gain = gain
                best_task = t
                best_new_budget = next_budget

        if best_task is None:
            break

        remaining -= (best_new_budget - allocation[best_task])
        allocation[best_task] = best_new_budget

    return [allocation[t] for t in tasks]


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = get_completed(CSV_PATH, ["seed", "allocator", "budget_total_mb", "task_idx", "eval_task"])
    print(f"[INIT] {len(completed)} completed experiments found")

    extractor = DINOv3Extractor()
    test_data = preload_test_data_with_masks(extractor, TASKS)

    # Pre-compute complexity scores and cache train features
    print("\n[PREP] Computing complexity scores...")
    complexity_cache: dict[tuple[str, int], float] = {}
    train_feat_cache: dict[tuple[str, int], np.ndarray] = {}

    for seed in SEEDS:
        for task_name in TASKS:
            feat, _ = extract_train_features(extractor, task_name, seed, N_PER_TASK)
            train_feat_cache[(task_name, seed)] = feat
            cs = complexity_score(feat)
            complexity_cache[(task_name, seed)] = cs
            short = task_name.split("/")[-1]
            print(f"  s{seed} {short:15s} complexity={cs:.2f}")

    # Pre-compute oracle AUROC curves from budget_sweep CSV if available
    oracle_curves: dict[int, dict[str, dict[int, float]]] = {}
    sweep_csv = OUTPUT_DIR / "c3_budget_sweep.csv"
    if sweep_csv.exists():
        import csv
        print("\n[PREP] Loading oracle curves from budget_sweep...")
        with open(sweep_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                seed = int(row["seed"])
                task = row["task_name"]
                budget = int(row["budget_embs"])
                auroc = float(row["img_auroc"])
                if seed not in oracle_curves:
                    oracle_curves[seed] = {}
                if task not in oracle_curves[seed]:
                    oracle_curves[seed][task] = {}
                # Keep best across compressors
                if budget not in oracle_curves[seed][task] or auroc > oracle_curves[seed][task][budget]:
                    oracle_curves[seed][task][budget] = auroc
        print(f"  Loaded curves for {len(oracle_curves)} seeds")
    else:
        print("\n[WARN] No budget_sweep CSV found — oracle allocator will be skipped")

    for seed in SEEDS:
        for budget_mb in BUDGET_LEVELS_MB:
            for allocator in ALLOCATORS:
                # Skip oracle if no curves
                if allocator == "oracle" and seed not in oracle_curves:
                    continue

                print(f"\n{'='*60}")
                print(f"[RUN] seed={seed}, budget={budget_mb}MB, allocator={allocator}")

                # Compute allocation
                if allocator == "uniform":
                    alloc = allocate_uniform(budget_mb, len(TASKS))
                elif allocator == "adaptive":
                    cs_list = [complexity_cache[(t, seed)] for t in TASKS]
                    alloc = allocate_adaptive(budget_mb, cs_list)
                elif allocator == "oracle":
                    alloc = allocate_oracle(budget_mb, oracle_curves[seed])
                else:
                    raise ValueError(f"Unknown allocator: {allocator}")

                alloc_dict = dict(zip(TASKS, alloc))
                total_embs = sum(alloc)
                actual_mb = memory_mb(total_embs)
                print(f"  Allocation: {alloc}")
                print(f"  Total: {total_embs} embs = {actual_mb:.1f} MB")

                # Build memory bank incrementally
                memory_bank = np.empty((0, FEATURE_DIM), dtype=np.float32)

                for task_idx, task_name in enumerate(TASKS):
                    budget_this = alloc_dict[task_name]
                    train_feat = train_feat_cache[(task_name, seed)]
                    flat = train_feat.reshape(-1, train_feat.shape[-1])

                    # Compress to budget
                    task_bank = kcenter_greedy(flat, budget_this, seed)
                    memory_bank = np.concatenate([memory_bank, task_bank], axis=0)

                    n_embs = len(memory_bank)
                    mem_mb_now = memory_mb(n_embs)
                    is_final = (task_idx == len(TASKS) - 1)

                    # Fit NN index once for this memory bank (reused across eval tasks)
                    k_actual = min(KNN_K, len(memory_bank))
                    nn_index = NearestNeighbors(n_neighbors=k_actual, metric="euclidean", algorithm="auto")
                    nn_index.fit(memory_bank)

                        # Evaluate all tasks seen so far
                    last_auroc = None
                    for eval_idx in range(task_idx + 1):
                        eval_task = TASKS[eval_idx]
                        key = (str(seed), allocator, str(budget_mb), str(task_idx), eval_task)
                        if key in completed:
                            continue

                        try:
                            test_features, test_labels, test_info = test_data[eval_task]

                            t0 = time.perf_counter()
                            if is_final:
                                image_scores, score_maps = knn_score_maps(
                                    memory_bank, test_features, nn_index=nn_index, k=KNN_K, grid_size=28
                                )
                            else:
                                image_scores = knn_score_images(
                                    memory_bank, test_features, nn_index=nn_index, k=KNN_K
                                )
                                score_maps = None
                            query_ms = (time.perf_counter() - t0) * 1000 / len(test_labels)

                            img_m = compute_metrics(test_labels, image_scores)

                            pix_m = {"pix_auroc": None, "pix_aupr": None, "spro": None, "auspro": None}
                            if is_final:
                                pix_m = compute_pixel_metrics(score_maps, test_info)

                            result = {
                                "timestamp": datetime.now().isoformat(),
                                "seed": seed,
                                "allocator": allocator,
                                "budget_total_mb": budget_mb,
                                "task_idx": task_idx,
                                "task_name": task_name,
                                "eval_task": eval_task,
                                "budget_per_product": alloc_dict[eval_task],
                                "n_test": len(test_labels),
                                "memory_n_embeddings": n_embs,
                                "memory_mb": round(mem_mb_now, 3),
                                "img_auroc": img_m["img_auroc"],
                                "img_aupr": img_m["img_aupr"],
                                "tpr_fpr1": img_m["tpr_fpr1"],
                                "pix_auroc": pix_m["pix_auroc"],
                                "pix_aupr": pix_m["pix_aupr"],
                                "spro": pix_m["spro"],
                                "auspro": pix_m["auspro"],
                                "query_time_ms": round(query_ms, 2),
                            }
                            save_result(CSV_PATH, COLUMNS, result)
                            completed.add(key)
                            last_auroc = img_m["img_auroc"]

                        except Exception as e:
                            log_error(ERROR_PATH, f"s={seed} a={allocator} b={budget_mb} t={task_idx} e={eval_task}", e)
                            print(f"  ERROR: {e}")

                    short = task_name.split("/")[-1]
                    auroc_str = f"{last_auroc:.3f}" if last_auroc is not None else "cached"
                    print(f"  Task {task_idx:2d} {short:15s} b={budget_this:3d} bank={n_embs:5d} auroc={auroc_str}")

                cuda_cleanup()

    print("\n[DONE] Adaptive allocation experiments complete!")


if __name__ == "__main__":
    main()
