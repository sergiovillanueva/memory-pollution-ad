"""
N=50 Oracle Validation: Verify oracle inversion persists with more training data
=================================================================================
Replicates the key finding (oracle inversion under global, reversal under routing)
using N=50 training images per product instead of N=10.

This strengthens the paper against the reviewer objection:
  "With more training data, the oracle might work correctly under global retrieval."

Pipeline:
  1. Budget sweep: per-product AUROC(budget) curves with N=50 (for oracle allocation)
  2. Continual evaluation: uniform vs oracle, global vs top_1, 3 seeds

Output: output/continual/n50_budget_sweep.csv
        output/continual/n50_oracle_validation.csv

Estimated runtime: ~30-45 min (budget sweep ~15 min, continual eval ~15-25 min).
Feature extraction for N=50 is cached on first run.
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
    ALL_TASKS_FULL, KNN_K, FEATURE_DIM,
    OUTPUT_DIR,
    DINOv3Extractor,
    extract_train_features,
    preload_test_data_with_masks,
    knn_score_images,
    knn_score_maps,
    compute_metrics, compute_pixel_metrics,
    memory_mb, complexity_score,
    compute_prototypes,
    get_completed, save_result, log_error, cuda_cleanup,
)

# =============================================================================
# CONFIG
# =============================================================================

TASKS = ALL_TASKS_FULL
SEEDS = [0, 1, 2]
N_PER_TASK = 50  # <--- KEY CHANGE: 50 images per product
PCA_DIM = 64

# Budget sweep: per-product individual evaluation (image-level only)
CORESET_BUDGETS = [25, 50, 75, 100, 150, 200, 300, 500]

# Continual evaluation: only the key conditions for the paper
BUDGET_LEVELS_MB = [6.0, 12.0, 18.0, 30.0]
ALLOCATORS = ["uniform", "oracle"]
ROUTING_MODES = ["global", "top_1"]

BUDGET_OPTIONS = [25, 50, 75, 100, 150, 200, 300, 500]

# Output paths (separate from N=10 data)
SWEEP_CSV = OUTPUT_DIR / "n50_budget_sweep.csv"
SWEEP_ERROR = OUTPUT_DIR / "n50_budget_sweep_errors.log"
EVAL_CSV = OUTPUT_DIR / "n50_oracle_validation.csv"
EVAL_ERROR = OUTPUT_DIR / "n50_oracle_validation_errors.log"

SWEEP_COLUMNS = [
    "timestamp", "seed", "compressor", "budget_embs", "task_name",
    "n_test", "n_train_patches", "memory_n_embeddings", "memory_mb",
    "img_auroc", "img_aupr", "tpr_fpr1",
    "complexity_score", "query_time_ms",
]

EVAL_COLUMNS = [
    "timestamp", "seed", "routing_mode", "allocator", "budget_total_mb",
    "task_idx", "task_name", "eval_task",
    "budget_per_product", "n_test", "memory_n_embeddings", "memory_mb",
    "img_auroc", "img_aupr", "tpr_fpr1",
    "pix_auroc", "pix_aupr", "spro", "auspro",
    "query_time_ms",
]


# =============================================================================
# CORESET COMPRESSION (with PCA caching for repeated calls on same data)
# =============================================================================

def kcenter_greedy(
    data: np.ndarray,
    budget: int,
    seed: int = 42,
    data_pca: np.ndarray | None = None,
) -> np.ndarray:
    """K-center greedy coreset selection.

    Args:
        data: [N, D] original features
        budget: number of points to select
        seed: random seed
        data_pca: pre-computed PCA projection (optional, avoids recomputation)
    """
    n = len(data)
    if n <= budget:
        return data.copy()

    if data_pca is None:
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

def snap_to_options_nearest(budget: int) -> int:
    return min(BUDGET_OPTIONS, key=lambda x: abs(x - budget))


def snap_to_options_floor(budget: int) -> int:
    """Snap to the largest option <= budget (best-effort to not exceed)."""
    lower = [b for b in BUDGET_OPTIONS if b <= budget]
    if lower:
        return max(lower)
    # If budget is below the minimum option, we cannot respect it.
    return min(BUDGET_OPTIONS)

def allocate_uniform(budget_total_mb: float, n_products: int) -> list[int]:
    per_product_mb = budget_total_mb / n_products
    per_product_embs = mb_to_embs(per_product_mb)
    budget = snap_to_options_floor(per_product_embs)
    return [budget] * n_products

def allocate_oracle(
    budget_total_mb: float,
    auroc_curves: dict[str, dict[int, float]],
    tasks: list[str],
) -> list[int]:
    total_embs = mb_to_embs(budget_total_mb)
    allocation = {t: min(BUDGET_OPTIONS) for t in tasks}
    remaining = total_embs - sum(allocation.values())
    if remaining < 0:
        # Extremely small total budget relative to min coreset option.
        # Best-effort: keep mins and proceed (will slightly exceed the nominal MB).
        return [allocation[t] for t in tasks]
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
            curve_t = auroc_curves.get(t, {})
            current_auroc = curve_t.get(current, 0.0)
            next_auroc = curve_t.get(next_budget, current_auroc)
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
# FAST TOP-1 ROUTING (batch KNN per routed product)
# =============================================================================

def _route_top1_batch(
    test_means: np.ndarray,
    prototypes: dict[str, np.ndarray],
) -> list[str]:
    """Return routed task per test image (top-1 by prototype distance)."""
    tasks = list(prototypes.keys())
    proto_matrix = np.stack([prototypes[t] for t in tasks], axis=0)  # [T, D]
    dists = np.linalg.norm(test_means[:, None, :] - proto_matrix[None, :, :], axis=2)  # [N, T]
    top_idx = np.argmin(dists, axis=1)
    return [tasks[int(i)] for i in top_idx]


def knn_score_images_top1_fast(
    sub_banks: dict[str, np.ndarray],
    test_features: np.ndarray,
    prototypes: dict[str, np.ndarray],
    k: int = KNN_K,
) -> np.ndarray:
    """Fast top-1 gated scoring: routes once, then does batched KNN per product."""
    n_test, n_patches, feat_dim = test_features.shape
    flat_test = test_features.reshape(n_test, n_patches, feat_dim)
    test_means = flat_test.mean(axis=1)
    routed = _route_top1_batch(test_means, prototypes)

    image_scores = np.zeros(n_test, dtype=np.float32)

    # Group image indices by routed task
    by_task: dict[str, list[int]] = {}
    for i, t in enumerate(routed):
        if t in sub_banks:
            by_task.setdefault(t, []).append(i)

    for task_name, indices in by_task.items():
        bank = sub_banks[task_name]
        k_actual = min(k, len(bank))
        nn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean")
        nn.fit(bank)

        patches = flat_test[np.array(indices)]  # [B, P, D]
        distances, _ = nn.kneighbors(patches.reshape(-1, feat_dim))  # [B*P, k]
        patch_scores = distances.mean(axis=1).reshape(len(indices), n_patches)
        image_scores[np.array(indices)] = np.percentile(patch_scores, 95, axis=1).astype(np.float32)

    return image_scores


def knn_score_maps_top1_fast(
    sub_banks: dict[str, np.ndarray],
    test_features: np.ndarray,
    prototypes: dict[str, np.ndarray],
    k: int = KNN_K,
    grid_size: int = 28,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast top-1 gated scoring returning maps (batched per routed product)."""
    n_test, n_patches, feat_dim = test_features.shape
    assert n_patches == grid_size * grid_size, (
        f"Mismatch: n_patches={n_patches} but grid_size={grid_size} -> {grid_size**2}"
    )

    flat_test = test_features.reshape(n_test, n_patches, feat_dim)
    test_means = flat_test.mean(axis=1)
    routed = _route_top1_batch(test_means, prototypes)

    image_scores = np.zeros(n_test, dtype=np.float32)
    score_maps = np.zeros((n_test, grid_size, grid_size), dtype=np.float32)

    by_task: dict[str, list[int]] = {}
    for i, t in enumerate(routed):
        if t in sub_banks:
            by_task.setdefault(t, []).append(i)

    for task_name, indices in by_task.items():
        bank = sub_banks[task_name]
        k_actual = min(k, len(bank))
        nn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean")
        nn.fit(bank)

        patches = flat_test[np.array(indices)]
        distances, _ = nn.kneighbors(patches.reshape(-1, feat_dim))
        patch_scores = distances.mean(axis=1).reshape(len(indices), n_patches)
        image_scores[np.array(indices)] = np.percentile(patch_scores, 95, axis=1).astype(np.float32)
        score_maps[np.array(indices)] = patch_scores.reshape(len(indices), grid_size, grid_size).astype(np.float32)

    return image_scores, score_maps


# =============================================================================
# STEP 1: BUDGET SWEEP (per-product, image-level only)
# =============================================================================

def run_budget_sweep_for_seed(
    seed: int,
    test_data: dict,
    train_feat_cache: dict,
) -> None:
    """Run per-product budget sweep for ONE seed.

    Only computes image-level metrics (no pixel maps) — that's all the oracle
    allocator needs, and it's ~3x faster.
    """
    completed = get_completed(SWEEP_CSV, ["seed", "compressor", "budget_embs", "task_name"])
    print(f"\n[SWEEP] {len(completed)} completed, running budget sweep with N={N_PER_TASK} (seed={seed})...")

    for task_name in TASKS:
        train_feat = train_feat_cache[(task_name, seed)]
        flat = train_feat.reshape(-1, train_feat.shape[-1])
        cs = complexity_score(train_feat)
        test_features, test_labels, _ = test_data[task_name]
        short = task_name.split("/")[-1]

        # PCA once per (task, seed), reuse across all budgets
        pca_dim = min(PCA_DIM, flat.shape[1], len(flat))
        pca = PCA(n_components=pca_dim, random_state=seed, svd_solver="randomized")
        flat_pca = pca.fit_transform(flat).astype(np.float32, copy=False)

        for budget in CORESET_BUDGETS:
            key = (str(seed), "coreset", str(budget), task_name)
            if key in completed:
                continue

            try:
                bank = kcenter_greedy(flat, budget, seed, data_pca=flat_pca)
                n_embs = len(bank)

                t0 = time.perf_counter()
                image_scores = knn_score_images(bank, test_features, k=KNN_K)
                query_ms = (time.perf_counter() - t0) * 1000 / len(test_labels)

                img_m = compute_metrics(test_labels, image_scores)

                result = {
                    "timestamp": datetime.now().isoformat(),
                    "seed": seed,
                    "compressor": "coreset",
                    "budget_embs": budget,
                    "task_name": task_name,
                    "n_test": len(test_labels),
                    "n_train_patches": len(flat),
                    "memory_n_embeddings": n_embs,
                    "memory_mb": round(memory_mb(n_embs), 3),
                    "img_auroc": img_m["img_auroc"],
                    "img_aupr": img_m["img_aupr"],
                    "tpr_fpr1": img_m["tpr_fpr1"],
                    "complexity_score": round(cs, 4),
                    "query_time_ms": round(query_ms, 2),
                }
                save_result(SWEEP_CSV, SWEEP_COLUMNS, result)
                completed.add(key)
                print(f"  s{seed} {short:15s} cs_{budget:3d} "
                      f"auroc={img_m['img_auroc']:.3f}")

            except Exception as e:
                log_error(SWEEP_ERROR, f"s={seed} b={budget} t={task_name}", e)
                print(f"  ERROR: {e}")

        del flat_pca  # free PCA projection

    cuda_cleanup()


def load_oracle_curves_for_seed(seed: int) -> dict[str, dict[int, float]]:
    """Load per-task AUROC(budget) curves from the sweep CSV for a given seed."""
    curves: dict[str, dict[int, float]] = {}
    if not SWEEP_CSV.exists():
        return curves

    with open(SWEEP_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            s = int(row["seed"])
            if s != seed:
                continue
            task = row["task_name"]
            budget = int(row["budget_embs"])
            auroc = float(row["img_auroc"])
            curves.setdefault(task, {})
            if budget not in curves[task] or auroc > curves[task][budget]:
                curves[task][budget] = auroc
    return curves


# =============================================================================
# STEP 2: CONTINUAL EVALUATION (uniform vs oracle, global vs top_1)
# =============================================================================

def run_continual_eval_for_seed(
    seed: int,
    test_data: dict,
    train_feat_cache: dict,
    oracle_curves_seed: dict[str, dict[int, float]],
) -> None:
    """Run continual protocol: 15 tasks sequentially, evaluate after each.

    Optimization: coreset banks are computed once per (seed, allocator, budget)
    and reused across routing modes (global + top_1).
    """
    completed = get_completed(EVAL_CSV, [
        "seed", "routing_mode", "allocator", "budget_total_mb", "task_idx", "eval_task"
    ])
    print(f"\n[EVAL] {len(completed)} completed, running continual evaluation...")

    prototypes = compute_prototypes(train_feat_cache, TASKS, seed)

    # Pre-compute PCA projections per task for coreset reuse
    pca_cache: dict[str, np.ndarray] = {}
    for task_name in TASKS:
        train_feat = train_feat_cache[(task_name, seed)]
        flat = train_feat.reshape(-1, train_feat.shape[-1])
        pca_dim = min(PCA_DIM, flat.shape[1], len(flat))
        pca = PCA(n_components=pca_dim, random_state=seed, svd_solver="randomized")
        pca_cache[task_name] = pca.fit_transform(flat).astype(np.float32, copy=False)

    for budget_mb in BUDGET_LEVELS_MB:
        for allocator in ALLOCATORS:
            if allocator == "oracle" and not oracle_curves_seed:
                print(f"  SKIP oracle for seed {seed}: no curves")
                continue

            # Compute allocation once (MUST be in TASKS order)
            if allocator == "uniform":
                alloc = allocate_uniform(budget_mb, len(TASKS))
            elif allocator == "oracle":
                alloc = allocate_oracle(budget_mb, oracle_curves_seed, TASKS)

            alloc_dict = dict(zip(TASKS, alloc))

            # Check if ALL routing modes are already completed
            all_done = True
            for routing_mode in ROUTING_MODES:
                for task_idx in range(len(TASKS)):
                    for eval_idx in range(task_idx + 1):
                        key = (str(seed), routing_mode, allocator,
                               str(budget_mb), str(task_idx), TASKS[eval_idx])
                        if key not in completed:
                            all_done = False
                            break
                    if not all_done:
                        break
                if not all_done:
                    break
            if all_done:
                continue

            combo = f"s={seed} b={budget_mb}MB a={allocator}"
            print(f"\n{'='*60}")
            print(f"[RUN] {combo} (global + top_1)")

            # Build banks ONCE, reuse for both routing modes
            sub_banks: dict[str, np.ndarray] = {}
            full_bank = np.empty((0, FEATURE_DIM), dtype=np.float32)

            for task_idx, task_name in enumerate(TASKS):
                budget_this = alloc_dict[task_name]
                train_feat = train_feat_cache[(task_name, seed)]
                flat = train_feat.reshape(-1, train_feat.shape[-1])

                task_bank = kcenter_greedy(
                    flat, budget_this, seed, data_pca=pca_cache[task_name],
                )
                sub_banks[task_name] = task_bank
                full_bank = np.concatenate([full_bank, task_bank], axis=0)

                n_embs = len(full_bank)
                mem_mb_now = memory_mb(n_embs)
                is_final = (task_idx == len(TASKS) - 1)

                # Pre-fit KNN index for global mode (reuse across eval_tasks)
                k_actual = min(KNN_K, n_embs)
                nn_global = NearestNeighbors(
                    n_neighbors=k_actual, metric='euclidean', algorithm='auto',
                )
                nn_global.fit(full_bank)

                for routing_mode in ROUTING_MODES:
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
                            if routing_mode == "global":
                                image_scores = knn_score_images(
                                    full_bank, test_features,
                                    nn_index=nn_global, k=KNN_K,
                                )
                            else:
                                # Only top_1 is used in this experiment; use fast batched path.
                                image_scores = knn_score_images_top1_fast(
                                    sub_banks, test_features, prototypes, k=KNN_K,
                                )
                            query_ms = (time.perf_counter() - t0) * 1000 / len(test_labels)

                            img_m = compute_metrics(test_labels, image_scores)

                            pix_m = {"pix_auroc": None, "pix_aupr": None,
                                     "spro": None, "auspro": None}
                            if is_final:
                                if routing_mode == "global":
                                    _, score_maps = knn_score_maps(
                                        full_bank, test_features,
                                        nn_index=nn_global, k=KNN_K,
                                        grid_size=28,
                                    )
                                else:
                                    _, score_maps = knn_score_maps_top1_fast(
                                        sub_banks, test_features, prototypes,
                                        k=KNN_K, grid_size=28,
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
                            save_result(EVAL_CSV, EVAL_COLUMNS, result)
                            completed.add(key)
                            last_auroc = img_m["img_auroc"]

                        except Exception as e:
                            log_error(EVAL_ERROR,
                                f"s={seed} r={routing_mode} a={allocator} "
                                f"b={budget_mb} t={task_idx} e={eval_task}", e)
                            print(f"  ERROR: {e}")

                    if routing_mode == "global":
                        short = task_name.split("/")[-1]
                        auroc_str = f"{last_auroc:.3f}" if last_auroc is not None else "cached"
                        print(f"  Task {task_idx:2d} {short:15s} b={budget_this:3d} "
                              f"bank={n_embs:5d} auroc={auroc_str}")

            cuda_cleanup()

    del pca_cache  # free PCA projections for this seed


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_global_start = time.time()

    print("=" * 60)
    print(f"N=50 Oracle Validation Experiment")
    print(f"  Training images per product: {N_PER_TASK}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Budgets: {BUDGET_LEVELS_MB} MB")
    print(f"  Allocators: {ALLOCATORS}")
    print(f"  Routing: {ROUTING_MODES}")
    print("=" * 60)

    extractor = DINOv3Extractor()
    test_data = preload_test_data_with_masks(extractor, TASKS)

    # Process per seed to keep RAM bounded (important for overnight stability)
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"[SEED] Starting seed={seed}")

        print(f"\n[PREP] Extracting train features (N={N_PER_TASK}, seed={seed})...")
        train_feat_cache: dict[tuple[str, int], np.ndarray] = {}
        for task_name in TASKS:
            feat, n_img = extract_train_features(
                extractor, task_name, seed, N_PER_TASK,
            )
            train_feat_cache[(task_name, seed)] = feat
            short = task_name.split("/")[-1]
            n_patches = feat.shape[0] * feat.shape[1]
            print(f"  s{seed} {short:15s} {n_img} imgs, {n_patches} patches")

        # Step 1: Budget sweep (for oracle curves) — image-level only
        run_budget_sweep_for_seed(seed, test_data, train_feat_cache)
        oracle_curves_seed = load_oracle_curves_for_seed(seed)
        print(f"[SWEEP] Oracle curves loaded for seed {seed}: {len(oracle_curves_seed)} tasks")

        # Continual evaluation
        run_continual_eval_for_seed(seed, test_data, train_feat_cache, oracle_curves_seed)

        del train_feat_cache
        cuda_cleanup(f"after seed={seed}")

    elapsed = time.time() - t_global_start
    print(f"\n{'='*60}")
    print(f"[DONE] N=50 oracle validation complete in {elapsed/60:.1f} min")
    print(f"  Budget sweep: {SWEEP_CSV}")
    print(f"  Continual eval: {EVAL_CSV}")

    # Quick summary
    print(f"\n[SUMMARY] Key results at 18 MB:")
    if EVAL_CSV.exists():
        try:
            import pandas as pd

            df = pd.read_csv(EVAL_CSV, sep=";")
            final = df[df["task_idx"] == 14]
            for rm in ROUTING_MODES:
                for alloc in ALLOCATORS:
                    subset = final[(final["routing_mode"] == rm) &
                                   (final["allocator"] == alloc) &
                                   (final["budget_total_mb"] == 18.0)]
                    if not subset.empty:
                        seed_means = subset.groupby("seed")["img_auroc"].mean()
                        mean_val = seed_means.mean()
                        std_val = seed_means.std()
                        print(f"  {rm:8s} {alloc:8s}: {mean_val:.4f} ± {std_val:.4f}")
        except Exception as e:
            print(f"  (summary skipped: {e})")


if __name__ == "__main__":
    main()
