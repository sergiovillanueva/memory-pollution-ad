"""
Compute Wrong-Neighbor Rate (WNR) for Memory Pollution analysis.
================================================================
WNR = fraction of KNN neighbors that belong to a different product than the query.

Quantifies inter-product interference in shared memory banks.
- High WNR = high pollution (neighbors from wrong products)
- WNR under routing = 0 by definition (only searches own sub-bank)

Usage: uv run compute_wnr.py
Output: output/paper_tables/table7_wnr.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent))
from continual_utils import (
    ALL_TASKS_FULL, FEATURE_DIM, KNN_K, N_PER_TASK,
    CACHE_DIR, _cache_path,
)

OUT_DIR = Path("output/paper_tables")
SEEDS = [0, 1, 2]
BUDGET_LEVELS = [50, 100, 200, 500]  # embeddings per product (not MB)
PCA_DIM = 64


def kcenter_greedy(data: np.ndarray, budget: int, seed: int = 42) -> np.ndarray:
    """K-center greedy coreset selection (same as experiments)."""
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


def compute_wnr(
    full_bank: np.ndarray,
    product_labels: np.ndarray,
    test_patches: np.ndarray,
    true_product_idx: int,
    k: int = KNN_K,
) -> float:
    """Compute Wrong-Neighbor Rate for a set of test patches.

    Args:
        full_bank: [M, D] shared memory bank
        product_labels: [M] product index for each embedding
        test_patches: [P, D] test patches from one image
        true_product_idx: which product the test image belongs to
        k: number of KNN neighbors

    Returns:
        WNR: fraction of neighbors from wrong products
    """
    k_actual = min(k, len(full_bank))
    nn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean", algorithm="auto")
    nn.fit(full_bank)
    _, indices = nn.kneighbors(test_patches)  # [P, K]
    neighbor_labels = product_labels[indices]  # [P, K]
    wrong = (neighbor_labels != true_product_idx)
    return float(wrong.mean())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("COMPUTING WRONG-NEIGHBOR RATE (WNR)")
    print("=" * 60)

    tasks = ALL_TASKS_FULL  # 15 MVTec products
    results = []

    for seed in SEEDS:
        print(f"\n[SEED {seed}]")

        # Load all train features from cache
        train_feats: dict[str, np.ndarray] = {}
        for task_name in tasks:
            cache = _cache_path("train_feat", task_name, f"_s{seed}_n{N_PER_TASK}")
            if not cache.exists():
                print(f"  [WARN] Missing cache: {cache}")
                continue
            train_feats[task_name] = np.load(cache)  # [N, P, D]

        # Load test features from cache
        test_feats: dict[str, np.ndarray] = {}
        for task_name in tasks:
            cache = _cache_path("test_feat", task_name)
            if not cache.exists():
                print(f"  [WARN] Missing test cache: {cache}")
                continue
            test_feats[task_name] = np.load(cache)  # [N_test, P, D]

        if not train_feats or not test_feats:
            print("  [SKIP] Missing feature caches")
            continue

        for budget_per_product in BUDGET_LEVELS:
            t0 = time.perf_counter()

            # Build shared memory bank with product labels
            bank_parts = []
            label_parts = []
            for idx, task_name in enumerate(tasks):
                if task_name not in train_feats:
                    continue
                feat = train_feats[task_name]  # [N, P, D]
                flat = feat.reshape(-1, feat.shape[-1])  # [N*P, D]
                compressed = kcenter_greedy(flat, budget_per_product, seed)
                bank_parts.append(compressed)
                label_parts.append(np.full(len(compressed), idx, dtype=np.int32))

            full_bank = np.concatenate(bank_parts, axis=0)
            product_labels = np.concatenate(label_parts, axis=0)
            total_mb = full_bank.nbytes / (1024 * 1024)

            # Compute WNR per product (using test images)
            for idx, task_name in enumerate(tasks):
                if task_name not in test_feats:
                    continue
                test_feat = test_feats[task_name]  # [N_test, P, D]
                # Use only normal test images (first half typically)
                # For WNR we want normal queries to check pollution
                n_test = len(test_feat)
                # Compute WNR across all test images
                wnr_per_image = []
                for img_idx in range(min(n_test, 20)):  # Cap at 20 images for speed
                    patches = test_feat[img_idx]  # [P, D]
                    wnr = compute_wnr(full_bank, product_labels, patches, idx)
                    wnr_per_image.append(wnr)

                mean_wnr = float(np.mean(wnr_per_image))
                short = task_name.split("/")[-1]

                results.append({
                    "seed": seed,
                    "budget_per_product": budget_per_product,
                    "total_memory_mb": round(total_mb, 1),
                    "product": short,
                    "product_idx": idx,
                    "wnr": round(mean_wnr, 4),
                    "n_images_evaluated": len(wnr_per_image),
                })

            elapsed = time.perf_counter() - t0
            # Compute global WNR for this budget
            budget_results = [r for r in results if r["seed"] == seed and r["budget_per_product"] == budget_per_product]
            global_wnr = np.mean([r["wnr"] for r in budget_results])
            print(f"  budget={budget_per_product:4d} embs/product ({total_mb:.1f} MB) "
                  f"-> WNR={global_wnr:.4f} ({elapsed:.1f}s)")

    # Save results
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        out_path = OUT_DIR / "table7_wnr.csv"
        df.to_csv(out_path, index=False)
        print(f"\n[SAVED] {out_path} ({len(df)} rows)")

        # Print summary table
        print("\n" + "=" * 60)
        print("WNR SUMMARY (mean across seeds)")
        print("=" * 60)
        summary = df.groupby(["budget_per_product", "product"]).agg(
            wnr_mean=("wnr", "mean"),
            wnr_std=("wnr", "std"),
        ).reset_index()

        global_summary = df.groupby("budget_per_product").agg(
            wnr_mean=("wnr", "mean"),
            wnr_std=("wnr", "std"),
            total_mb=("total_memory_mb", "first"),
        ).reset_index()

        print("\nGlobal WNR by budget:")
        for _, row in global_summary.iterrows():
            print(f"  {int(row['budget_per_product']):4d} embs ({row['total_mb']:.1f} MB): "
                  f"WNR = {row['wnr_mean']:.4f} +/- {row['wnr_std']:.4f}")

        # Per-product at highest budget
        max_budget = df["budget_per_product"].max()
        product_summary = summary[summary["budget_per_product"] == max_budget].sort_values("wnr_mean", ascending=False)
        print(f"\nPer-product WNR (budget={max_budget}):")
        for _, row in product_summary.iterrows():
            print(f"  {row['product']:15s}: WNR = {row['wnr_mean']:.4f}")
    else:
        print("[WARN] No results generated")


if __name__ == "__main__":
    main()
