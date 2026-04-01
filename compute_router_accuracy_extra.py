"""
Compute Router Accuracy for VisA and BTAD benchmarks.
=====================================================
Uses cached features (no GPU needed). Measures whether the nearest-centroid
router correctly identifies the product for each test image.

Usage: uv run compute_router_accuracy_extra.py
Output: output/paper_tables/router_accuracy_extra.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from continual_utils import (
    ALL_TASKS_VISA, ALL_TASKS_BTAD, N_PER_TASK,
    CACHE_DIR, _cache_path, router_accuracy,
    compute_prototypes,
)

OUT_DIR = Path("output/paper_tables")
SEEDS = [0, 1, 2]
ROUTING_MODES = [1, 2, 3, 5]  # top_M values


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("ROUTER ACCURACY: VisA + BTAD")
    print("=" * 60)

    results = []

    for benchmark, tasks in [("VisA", ALL_TASKS_VISA), ("BTAD", ALL_TASKS_BTAD)]:
        print(f"\n[{benchmark}] {len(tasks)} categories")

        for seed in SEEDS:
            # Load train features
            train_feat_cache: dict[tuple[str, int], np.ndarray] = {}
            missing = False
            for task_name in tasks:
                cache = _cache_path("train_feat", task_name, f"_s{seed}_n{N_PER_TASK}")
                if not cache.exists():
                    print(f"  [WARN] Missing: {cache}")
                    missing = True
                    break
                train_feat_cache[(task_name, seed)] = np.load(cache)

            if missing:
                continue

            # Load test features
            test_data: dict[str, np.ndarray] = {}
            for task_name in tasks:
                cache = _cache_path("test_feat", task_name)
                if not cache.exists():
                    print(f"  [WARN] Missing test: {cache}")
                    missing = True
                    break
                test_data[task_name] = np.load(cache)

            if missing:
                continue

            # Compute prototypes
            prototypes = compute_prototypes(train_feat_cache, tasks, seed)

            # Measure accuracy per product per routing mode
            for task_name in tasks:
                test_features = test_data[task_name]
                short = task_name.split("/")[-1]

                for top_m in ROUTING_MODES:
                    acc = router_accuracy(test_features, task_name, prototypes, top_m)
                    n_test = len(test_features)
                    n_correct = int(round(acc * n_test))
                    mode_str = f"top_{top_m}"

                    results.append({
                        "benchmark": benchmark,
                        "seed": seed,
                        "task_name": task_name,
                        "product": short,
                        "routing_mode": mode_str,
                        "n_test": n_test,
                        "n_correct": n_correct,
                        "accuracy": round(acc, 6),
                    })

                    if top_m == 1:
                        status = "OK" if acc == 1.0 else f"FAIL {acc:.3f}"
                        print(f"  s{seed} {short:15s} top_1: {status} ({n_correct}/{n_test})")

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        out_path = OUT_DIR / "router_accuracy_extra.csv"
        df.to_csv(out_path, index=False)
        print(f"\n[SAVED] {out_path} ({len(df)} rows)")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for benchmark in df["benchmark"].unique():
            bdf = df[df["benchmark"] == benchmark]
            for mode in sorted(bdf["routing_mode"].unique()):
                mdf = bdf[bdf["routing_mode"] == mode]
                mean_acc = mdf["accuracy"].mean()
                min_acc = mdf["accuracy"].min()
                n_perfect = (mdf["accuracy"] == 1.0).sum()
                total = len(mdf)
                print(f"  {benchmark:5s} {mode:6s}: mean={mean_acc:.4f}, min={min_acc:.4f}, "
                      f"perfect={n_perfect}/{total}")

        # Per-product detail for top_1 failures
        failures = df[(df["routing_mode"] == "top_1") & (df["accuracy"] < 1.0)]
        if len(failures) > 0:
            print(f"\n[IMPORTANT] top_1 routing failures:")
            for _, row in failures.iterrows():
                print(f"  {row['benchmark']}/{row['product']} s{row['seed']}: "
                      f"{row['accuracy']:.3f} ({row['n_correct']}/{row['n_test']})")
        else:
            print(f"\n[RESULT] 100% top_1 accuracy across ALL benchmarks and seeds!")
    else:
        print("[WARN] No results generated")


if __name__ == "__main__":
    main()
