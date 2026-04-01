"""
C3 Routing: Product-Conditioned Retrieval (Gating)
====================================================
Diagnoses and resolves inter-product interference (memory pollution)
in shared KNN memory banks.

Experiments:
  1. Router accuracy: how often does centroid routing pick the correct product?
  2. Perfect router (known-ID): upper bound, uses true labels to filter bank
  3. Centroid router top-M: M in {1, 2, 3, 5, N} with N=all (global bank)
  4. Re-evaluate allocators UNDER routing:
     - uniform + routing vs adaptive + routing vs oracle + routing
  5. Full continual protocol: 15 tasks, 3 seeds

Key insight: if perfect_router(oracle) > uniform(global), then the problem
was interference, not the allocator.

Output: output/continual/c3_routing.csv
        output/continual/c3_routing_accuracy.csv  (router accuracy stats)
"""
from __future__ import annotations

import sys
import os
os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
os.environ.setdefault("KMP_BLOCKTIME", "0")

import csv
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
    knn_score_images, knn_score_images_gated, knn_score_maps_gated,
    compute_metrics, compute_pixel_metrics,
    memory_mb, complexity_score,
    compute_prototypes, route_batch, router_accuracy,
    get_completed, save_result, log_error, cuda_cleanup,
)

# =============================================================================
# CONFIG
# =============================================================================

TASKS = ALL_TASKS_FULL
SEEDS = [0, 1, 2]
PCA_DIM = 64

# Routing modes to evaluate
# "global" = no routing (baseline, same as current shared bank)
# "known_id" = perfect routing, test label used (industrial upper bound)
# "top_1", "top_2", "top_3", "top_5" = centroid-based routing
#
# PAPER-ESSENTIAL subset (24 combos vs 90 full → 73% faster):
#   - known_id removed: identical to top_1 (router accuracy=100%)
#   - top_2, top_5 removed: intermediates with no narrative value
#     (top_1 vs top_3 already shows the M-gating degradation pattern)
#   - adaptive removed: ≈uniform (±0.001), uniform vs oracle is the key contrast
#   - 24MB removed: ≈18MB in all conditions
#
# FULL set (uncomment to run all, e.g. for appendix/supplementary):
# ROUTING_MODES = ["global", "known_id", "top_1", "top_2", "top_3", "top_5"]
# BUDGET_LEVELS_MB = [6.0, 12.0, 18.0, 24.0, 30.0]
# ALLOCATORS = ["uniform", "adaptive", "oracle"]

ROUTING_MODES = ["global", "top_1", "top_3"]
BUDGET_LEVELS_MB = [6.0, 12.0, 18.0, 30.0]
ALLOCATORS = ["uniform", "oracle"]

# Discrete per-product budgets (embeddings)
BUDGET_OPTIONS = [25, 50, 75, 100, 150, 200, 300, 500]

CSV_PATH = OUTPUT_DIR / "c3_routing.csv"
ACCURACY_CSV_PATH = OUTPUT_DIR / "c3_routing_accuracy.csv"
ERROR_PATH = OUTPUT_DIR / "c3_routing_errors.log"

COLUMNS = [
    "timestamp", "seed", "routing_mode", "allocator", "budget_total_mb",
    "task_idx", "task_name", "eval_task",
    "budget_per_product", "n_test", "memory_n_embeddings", "memory_mb",
    "img_auroc", "img_aupr", "tpr_fpr1",
    "pix_auroc", "pix_aupr", "spro", "auspro",
    "query_time_ms",
]

ACCURACY_COLUMNS = [
    "timestamp", "seed", "task_name", "routing_mode",
    "n_test", "n_correct", "accuracy",
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
# ALLOCATION
# =============================================================================

def mb_to_embs(mb: float, dim: int = FEATURE_DIM, bytes_per_val: int = 4) -> int:
    return int(mb * 1e6 / (dim * bytes_per_val))

def snap_to_options(budget: int) -> int:
    return min(BUDGET_OPTIONS, key=lambda x: abs(x - budget))

def allocate_uniform(budget_total_mb: float, n_products: int) -> list[int]:
    per_product_mb = budget_total_mb / n_products
    per_product_embs = mb_to_embs(per_product_mb)
    budget = snap_to_options(per_product_embs)
    return [budget] * n_products

def allocate_adaptive(
    budget_total_mb: float,
    complexity_scores: list[float],
    alpha: float = 1.0,
) -> list[int]:
    n = len(complexity_scores)
    total_embs = mb_to_embs(budget_total_mb)
    cs = np.array(complexity_scores)
    weights = cs ** alpha
    weights /= weights.sum()
    raw_alloc = weights * total_embs
    allocation = [snap_to_options(int(b)) for b in raw_alloc]
    while sum(allocation) > total_embs and max(allocation) > min(BUDGET_OPTIONS):
        idx = np.argmax(allocation)
        current = allocation[idx]
        lower = [b for b in BUDGET_OPTIONS if b < current]
        if lower:
            allocation[idx] = max(lower)
        else:
            break
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
    tasks = list(auroc_curves.keys())
    total_embs = mb_to_embs(budget_total_mb)
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
# ROUTING HELPERS
# =============================================================================

def parse_top_m(mode: str) -> int | None:
    """Parse routing mode to top_m value. Returns None for non-gated modes."""
    if mode == "global":
        return None  # No routing, use full bank
    if mode == "known_id":
        return None  # Special: perfect routing
    if mode.startswith("top_"):
        return int(mode.split("_")[1])
    raise ValueError(f"Unknown routing mode: {mode}")


def score_with_routing(
    mode: str,
    sub_banks: dict[str, np.ndarray],
    full_bank: np.ndarray,
    test_features: np.ndarray,
    prototypes: dict[str, np.ndarray],
    eval_task: str,
    k: int = KNN_K,
) -> np.ndarray:
    """Score test images using the specified routing mode."""
    if mode == "global":
        return knn_score_images(full_bank, test_features, k=k)
    elif mode == "known_id":
        return knn_score_images_gated(
            sub_banks, test_features, prototypes,
            top_m=1, k=k, known_task=eval_task,
        )
    else:
        top_m = parse_top_m(mode)
        return knn_score_images_gated(
            sub_banks, test_features, prototypes,
            top_m=top_m, k=k,
        )


def score_maps_with_routing(
    mode: str,
    sub_banks: dict[str, np.ndarray],
    full_bank: np.ndarray,
    test_features: np.ndarray,
    prototypes: dict[str, np.ndarray],
    eval_task: str,
    k: int = KNN_K,
    grid_size: int = 28,
):
    """Score maps for pixel metrics using the specified routing mode."""
    from continual_utils import knn_score_maps
    if mode == "global":
        return knn_score_maps(full_bank, test_features, k=k, grid_size=grid_size)
    elif mode == "known_id":
        return knn_score_maps_gated(
            sub_banks, test_features, prototypes,
            top_m=1, k=k, grid_size=grid_size, known_task=eval_task,
        )
    else:
        top_m = parse_top_m(mode)
        return knn_score_maps_gated(
            sub_banks, test_features, prototypes,
            top_m=top_m, k=k, grid_size=grid_size,
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = get_completed(CSV_PATH, [
        "seed", "routing_mode", "allocator", "budget_total_mb", "task_idx", "eval_task"
    ])
    print(f"[INIT] {len(completed)} completed experiments found")

    extractor = DINOv3Extractor()
    test_data = preload_test_data_with_masks(extractor, TASKS)

    # Pre-compute complexity scores and cache train features
    print("\n[PREP] Computing complexity scores and train features...")
    complexity_cache: dict[tuple[str, int], float] = {}
    train_feat_cache: dict[tuple[str, int], np.ndarray] = {}

    for seed in SEEDS:
        for task_name in TASKS:
            feat, _ = extract_train_features(extractor, task_name, seed, N_PER_TASK)
            train_feat_cache[(task_name, seed)] = feat
            cs = complexity_score(feat)
            complexity_cache[(task_name, seed)] = cs

    # Load oracle curves from budget_sweep
    oracle_curves: dict[int, dict[str, dict[int, float]]] = {}
    sweep_csv = OUTPUT_DIR / "c3_budget_sweep.csv"
    if sweep_csv.exists():
        print("\n[PREP] Loading oracle curves from budget_sweep...")
        with open(sweep_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                s = int(row["seed"])
                task = row["task_name"]
                budget = int(row["budget_embs"])
                auroc = float(row["img_auroc"])
                if s not in oracle_curves:
                    oracle_curves[s] = {}
                if task not in oracle_curves[s]:
                    oracle_curves[s][task] = {}
                if budget not in oracle_curves[s][task] or auroc > oracle_curves[s][task][budget]:
                    oracle_curves[s][task][budget] = auroc
        print(f"  Loaded curves for {len(oracle_curves)} seeds")

    # =========================================================================
    # PHASE A: Router accuracy measurement
    # =========================================================================
    print("\n" + "=" * 60)
    print("[PHASE A] Measuring router accuracy...")
    print("=" * 60)

    accuracy_completed = set()
    if ACCURACY_CSV_PATH.exists():
        accuracy_completed = get_completed(
            ACCURACY_CSV_PATH, ["seed", "task_name", "routing_mode"]
        )

    for seed in SEEDS:
        prototypes = compute_prototypes(train_feat_cache, TASKS, seed)
        for task_name in TASKS:
            for mode in ["top_1", "top_2", "top_3", "top_5"]:
                key = (str(seed), task_name, mode)
                if key in accuracy_completed:
                    continue

                test_features, test_labels, _ = test_data[task_name]
                top_m = parse_top_m(mode)
                acc = router_accuracy(test_features, task_name, prototypes, top_m)
                n_test = len(test_labels)
                n_correct = int(round(acc * n_test))

                result = {
                    "timestamp": datetime.now().isoformat(),
                    "seed": seed,
                    "task_name": task_name,
                    "routing_mode": mode,
                    "n_test": n_test,
                    "n_correct": n_correct,
                    "accuracy": round(acc, 6),
                }
                save_result(ACCURACY_CSV_PATH, ACCURACY_COLUMNS, result)
                accuracy_completed.add(key)

                short = task_name.split("/")[-1]
                print(f"  s{seed} {short:15s} {mode}: {acc:.3f} ({n_correct}/{n_test})")

    # =========================================================================
    # PHASE B: Routing + Allocation experiments (continual protocol)
    # =========================================================================
    print("\n" + "=" * 60)
    print("[PHASE B] Routing + Allocation continual experiments...")
    print("=" * 60)

    for seed in SEEDS:
        prototypes = compute_prototypes(train_feat_cache, TASKS, seed)

        for budget_mb in BUDGET_LEVELS_MB:
            for allocator in ALLOCATORS:
                if allocator == "oracle" and seed not in oracle_curves:
                    continue

                for routing_mode in ROUTING_MODES:
                    combo_name = f"s={seed} b={budget_mb}MB a={allocator} r={routing_mode}"
                    print(f"\n{'='*60}")
                    print(f"[RUN] {combo_name}")

                    # Compute allocation
                    if allocator == "uniform":
                        alloc = allocate_uniform(budget_mb, len(TASKS))
                    elif allocator == "adaptive":
                        cs_list = [complexity_cache[(t, seed)] for t in TASKS]
                        alloc = allocate_adaptive(budget_mb, cs_list)
                    elif allocator == "oracle":
                        alloc = allocate_oracle(budget_mb, oracle_curves[seed])

                    alloc_dict = dict(zip(TASKS, alloc))
                    total_embs = sum(alloc)
                    actual_mb = memory_mb(total_embs)
                    print(f"  Alloc total: {total_embs} embs = {actual_mb:.1f} MB")

                    # Build sub-banks and full bank incrementally
                    sub_banks: dict[str, np.ndarray] = {}
                    full_bank = np.empty((0, FEATURE_DIM), dtype=np.float32)

                    for task_idx, task_name in enumerate(TASKS):
                        budget_this = alloc_dict[task_name]
                        train_feat = train_feat_cache[(task_name, seed)]
                        flat = train_feat.reshape(-1, train_feat.shape[-1])

                        task_bank = kcenter_greedy(flat, budget_this, seed)
                        sub_banks[task_name] = task_bank
                        full_bank = np.concatenate([full_bank, task_bank], axis=0)

                        n_embs = len(full_bank)
                        mem_mb_now = memory_mb(n_embs)
                        is_final = (task_idx == len(TASKS) - 1)

                        last_auroc = None
                        for eval_idx in range(task_idx + 1):
                            eval_task = TASKS[eval_idx]
                            key = (str(seed), routing_mode, allocator,
                                   str(budget_mb), str(task_idx), eval_task)
                            if key in completed:
                                continue

                            try:
                                test_features, test_labels, test_info = test_data[eval_task]

                                t0 = time.perf_counter()
                                image_scores = score_with_routing(
                                    routing_mode, sub_banks, full_bank,
                                    test_features, prototypes, eval_task,
                                )
                                query_ms = (time.perf_counter() - t0) * 1000 / len(test_labels)

                                img_m = compute_metrics(test_labels, image_scores)

                                pix_m = {"pix_auroc": None, "pix_aupr": None,
                                         "spro": None, "auspro": None}
                                if is_final:
                                    _, score_maps = score_maps_with_routing(
                                        routing_mode, sub_banks, full_bank,
                                        test_features, prototypes, eval_task,
                                    )
                                    pix_m = compute_pixel_metrics(score_maps, test_info)

                                result = {
                                    "timestamp": datetime.now().isoformat(),
                                    "seed": seed,
                                    "routing_mode": routing_mode,
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
                                log_error(ERROR_PATH,
                                    f"s={seed} r={routing_mode} a={allocator} "
                                    f"b={budget_mb} t={task_idx} e={eval_task}", e)
                                print(f"  ERROR: {e}")

                        short = task_name.split("/")[-1]
                        auroc_str = f"{last_auroc:.3f}" if last_auroc is not None else "cached"
                        print(f"  Task {task_idx:2d} {short:15s} b={budget_this:3d} "
                              f"bank={n_embs:5d} auroc={auroc_str}")

                    cuda_cleanup()

    print("\n[DONE] Routing experiments complete!")
    print(f"  Results: {CSV_PATH}")
    print(f"  Accuracy: {ACCURACY_CSV_PATH}")


if __name__ == "__main__":
    main()
