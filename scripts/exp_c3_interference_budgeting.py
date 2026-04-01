"""
C3 Interference-Aware Budgeting
================================
Tests whether overlap-aware allocation (based on inter-product distance)
outperforms complexity-based and uniform allocation UNDER routing.

Allocators tested:
  1. uniform:         B_total / N for each product
  2. complexity:      proportional to intra-normal complexity (existing)
  3. overlap:         proportional to inter-product overlap score (NEW)
  4. inverse_overlap: INVERSE of overlap (more budget to isolated products)
  5. oracle:          greedy marginal gain (test-set upper bound)

All experiments run WITH routing (top-3 centroid gating) since we proved
that shared-bank budgeting is dominated by memory pollution.

Output: output/continual/c3_interference_budgeting.csv
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
    knn_score_images_gated, knn_score_maps_gated,
    compute_metrics, compute_pixel_metrics,
    memory_mb, complexity_score,
    compute_prototypes, compute_overlap_scores,
    get_completed, save_result, log_error, cuda_cleanup,
)

# =============================================================================
# CONFIG
# =============================================================================

TASKS = ALL_TASKS_FULL
SEEDS = [0, 1, 2]
PCA_DIM = 64

BUDGET_LEVELS_MB = [6.0, 12.0, 18.0, 24.0, 30.0]
ALLOCATORS = ["uniform", "complexity", "overlap", "inverse_overlap", "oracle"]

# Routing mode for all experiments here
ROUTING_TOP_M = 3  # top-3 centroid gating

BUDGET_OPTIONS = [25, 50, 75, 100, 150, 200, 300, 500]

CSV_PATH = OUTPUT_DIR / "c3_interference_budgeting.csv"
ERROR_PATH = OUTPUT_DIR / "c3_interference_budgeting_errors.log"

COLUMNS = [
    "timestamp", "seed", "allocator", "budget_total_mb",
    "task_idx", "task_name", "eval_task",
    "budget_per_product", "overlap_score", "complexity_score_val",
    "n_test", "memory_n_embeddings", "memory_mb",
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
    return int(mb * 1e6 / (dim * bytes_per_val))

def snap_to_options(budget: int) -> int:
    return min(BUDGET_OPTIONS, key=lambda x: abs(x - budget))

def allocate_uniform(budget_total_mb: float, n_products: int) -> list[int]:
    per_product_mb = budget_total_mb / n_products
    per_product_embs = mb_to_embs(per_product_mb)
    budget = snap_to_options(per_product_embs)
    return [budget] * n_products

def allocate_proportional(
    budget_total_mb: float,
    scores: list[float],
    alpha: float = 1.0,
) -> list[int]:
    """Generic proportional allocation: budget ∝ score^alpha."""
    total_embs = mb_to_embs(budget_total_mb)
    cs = np.array(scores, dtype=np.float64)

    # Handle edge case: all scores equal → uniform
    if cs.max() - cs.min() < 1e-10:
        budget = snap_to_options(total_embs // len(scores))
        return [budget] * len(scores)

    weights = cs ** alpha
    weights /= weights.sum()
    raw_alloc = weights * total_embs
    allocation = [snap_to_options(int(b)) for b in raw_alloc]

    # Adjust to stay within budget
    while sum(allocation) > total_embs and max(allocation) > min(BUDGET_OPTIONS):
        idx = int(np.argmax(allocation))
        current = allocation[idx]
        lower = [b for b in BUDGET_OPTIONS if b < current]
        if lower:
            allocation[idx] = max(lower)
        else:
            break

    while sum(allocation) < total_embs * 0.95:
        idx = int(np.argmin(allocation))
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
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = get_completed(CSV_PATH, [
        "seed", "allocator", "budget_total_mb", "task_idx", "eval_task"
    ])
    print(f"[INIT] {len(completed)} completed experiments found")

    extractor = DINOv3Extractor()
    test_data = preload_test_data_with_masks(extractor, TASKS)

    # Pre-compute all scores
    print("\n[PREP] Computing complexity + overlap scores...")
    complexity_cache: dict[tuple[str, int], float] = {}
    overlap_cache: dict[tuple[str, int], float] = {}
    train_feat_cache: dict[tuple[str, int], np.ndarray] = {}

    for seed in SEEDS:
        for task_name in TASKS:
            feat, _ = extract_train_features(extractor, task_name, seed, N_PER_TASK)
            train_feat_cache[(task_name, seed)] = feat
            cs = complexity_score(feat)
            complexity_cache[(task_name, seed)] = cs

        # Overlap scores (need all tasks loaded for this seed)
        overlap_scores = compute_overlap_scores(train_feat_cache, TASKS, seed)
        for task_name in TASKS:
            overlap_cache[(task_name, seed)] = overlap_scores[task_name]
            short = task_name.split("/")[-1]
            cs_val = complexity_cache[(task_name, seed)]
            ov_val = overlap_scores[task_name]
            print(f"  s{seed} {short:15s} complexity={cs_val:.2f}  overlap={ov_val:.2f}")

    # Load oracle curves
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

    for seed in SEEDS:
        prototypes = compute_prototypes(train_feat_cache, TASKS, seed)

        for budget_mb in BUDGET_LEVELS_MB:
            for allocator in ALLOCATORS:
                if allocator == "oracle" and seed not in oracle_curves:
                    continue

                combo_name = f"s={seed} b={budget_mb}MB a={allocator}"
                print(f"\n{'='*60}")
                print(f"[RUN] {combo_name} (routing=top_{ROUTING_TOP_M})")

                # Compute allocation based on allocator type
                if allocator == "uniform":
                    alloc = allocate_uniform(budget_mb, len(TASKS))
                elif allocator == "complexity":
                    cs_list = [complexity_cache[(t, seed)] for t in TASKS]
                    alloc = allocate_proportional(budget_mb, cs_list, alpha=1.0)
                elif allocator == "overlap":
                    # INVERSE of overlap: products with MORE overlap → MORE budget
                    # because they need more embeddings to avoid being confused
                    ov_list = [1.0 / (overlap_cache[(t, seed)] + 1e-6) for t in TASKS]
                    alloc = allocate_proportional(budget_mb, ov_list, alpha=1.0)
                elif allocator == "inverse_overlap":
                    # Products with LESS overlap (more isolated) → MORE budget
                    ov_list = [overlap_cache[(t, seed)] for t in TASKS]
                    alloc = allocate_proportional(budget_mb, ov_list, alpha=1.0)
                elif allocator == "oracle":
                    alloc = allocate_oracle(budget_mb, oracle_curves[seed])

                alloc_dict = dict(zip(TASKS, alloc))
                total_embs = sum(alloc)
                actual_mb = memory_mb(total_embs)
                print(f"  Alloc total: {total_embs} embs = {actual_mb:.1f} MB")

                # Build sub-banks incrementally (for routed KNN)
                sub_banks: dict[str, np.ndarray] = {}

                for task_idx, task_name in enumerate(TASKS):
                    budget_this = alloc_dict[task_name]
                    train_feat = train_feat_cache[(task_name, seed)]
                    flat = train_feat.reshape(-1, train_feat.shape[-1])

                    task_bank = kcenter_greedy(flat, budget_this, seed)
                    sub_banks[task_name] = task_bank

                    n_embs = sum(len(v) for v in sub_banks.values())
                    mem_mb_now = memory_mb(n_embs)
                    is_final = (task_idx == len(TASKS) - 1)

                    last_auroc = None
                    for eval_idx in range(task_idx + 1):
                        eval_task = TASKS[eval_idx]
                        key = (str(seed), allocator, str(budget_mb),
                               str(task_idx), eval_task)
                        if key in completed:
                            continue

                        try:
                            test_features, test_labels, test_info = test_data[eval_task]

                            t0 = time.perf_counter()
                            image_scores = knn_score_images_gated(
                                sub_banks, test_features, prototypes,
                                top_m=ROUTING_TOP_M, k=KNN_K,
                            )
                            query_ms = (time.perf_counter() - t0) * 1000 / len(test_labels)

                            img_m = compute_metrics(test_labels, image_scores)

                            pix_m = {"pix_auroc": None, "pix_aupr": None,
                                     "spro": None, "auspro": None}
                            if is_final:
                                _, score_maps = knn_score_maps_gated(
                                    sub_banks, test_features, prototypes,
                                    top_m=ROUTING_TOP_M, k=KNN_K, grid_size=28,
                                )
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
                                "overlap_score": round(overlap_cache.get(
                                    (eval_task, seed), 0.0), 4),
                                "complexity_score_val": round(complexity_cache.get(
                                    (eval_task, seed), 0.0), 4),
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
                                f"s={seed} a={allocator} b={budget_mb} "
                                f"t={task_idx} e={eval_task}", e)
                            print(f"  ERROR: {e}")

                    short = task_name.split("/")[-1]
                    auroc_str = f"{last_auroc:.3f}" if last_auroc is not None else "cached"
                    print(f"  Task {task_idx:2d} {short:15s} b={budget_this:3d} "
                          f"bank={n_embs:5d} auroc={auroc_str}")

                cuda_cleanup()

    print("\n[DONE] Interference-aware budgeting experiments complete!")
    print(f"  Results: {CSV_PATH}")


if __name__ == "__main__":
    main()
