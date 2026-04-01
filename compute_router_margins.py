"""Compute router margins: d2/d1 ratio for all test images across benchmarks.

Reports min, mean, and std of the margin ratio (distance to 2nd closest
prototype / distance to 1st closest prototype). A ratio >> 1 means the
router has a large confidence margin.

No GPU needed: uses cached .npy features only.
"""

import csv
from pathlib import Path

import numpy as np

# ── paths ──
CACHE_DIR = Path("output/feature_cache")
OUT_PATH = Path("output/paper_tables/router_margins.csv")

ALL_TASKS_FULL = [
    "mvtec_AD/bottle", "mvtec_AD/cable", "mvtec_AD/capsule",
    "mvtec_AD/carpet", "mvtec_AD/grid", "mvtec_AD/hazelnut",
    "mvtec_AD/leather", "mvtec_AD/metal_nut", "mvtec_AD/pill",
    "mvtec_AD/screw", "mvtec_AD/tile", "mvtec_AD/toothbrush",
    "mvtec_AD/transistor", "mvtec_AD/wood", "mvtec_AD/zipper",
]
ALL_TASKS_VISA = [
    "VisA/candle", "VisA/capsules", "VisA/cashew", "VisA/chewinggum",
    "VisA/fryum", "VisA/macaroni1", "VisA/macaroni2", "VisA/pcb1",
    "VisA/pcb2", "VisA/pcb3", "VisA/pcb4", "VisA/pipe_fryum",
]
ALL_TASKS_BTAD = ["btad/01", "btad/02", "btad/03"]

SEEDS = [0, 1, 2]
N_PER_TASK = 10


def _cache_path(prefix: str, task_name: str, suffix: str = "") -> Path:
    safe_name = task_name.replace("/", "_")
    return CACHE_DIR / f"{prefix}_{safe_name}{suffix}.npy"


def compute_margins_for_benchmark(
    tasks: list[str], benchmark: str
) -> list[dict]:
    """Compute margin ratios for all products, seeds in a benchmark."""
    rows: list[dict] = []

    for seed in SEEDS:
        # Load train features and build prototypes
        train_cache: dict[str, np.ndarray] = {}
        for task_name in tasks:
            path = _cache_path("train_feat", task_name, f"_s{seed}_n{N_PER_TASK}")
            if not path.exists():
                print(f"  SKIP {task_name} seed={seed} (no cache)")
                continue
            feat = np.load(path)  # [N, P, D]
            flat = feat.reshape(-1, feat.shape[-1])  # [N*P, D]
            train_cache[task_name] = flat.mean(axis=0)  # prototype [D]

        available_tasks = [t for t in tasks if t in train_cache]
        if len(available_tasks) < 2:
            continue

        proto_matrix = np.stack([train_cache[t] for t in available_tasks])
        task_names = available_tasks

        # For each product, load test features and compute margins
        for task_name in available_tasks:
            test_path = _cache_path("test_feat", task_name)
            if not test_path.exists():
                print(f"  SKIP test {task_name} (no cache)")
                continue

            test_feat = np.load(test_path)  # [N_test, P, D]
            n_test = test_feat.shape[0]
            test_means = test_feat.reshape(n_test, -1, test_feat.shape[-1]).mean(axis=1)  # [N_test, D]

            # Distances to all prototypes: [N_test, N_tasks]
            dists = np.linalg.norm(
                test_means[:, np.newaxis, :] - proto_matrix[np.newaxis, :, :],
                axis=2,
            )

            # Sort distances per image
            sorted_dists = np.sort(dists, axis=1)
            d1 = sorted_dists[:, 0]  # closest
            d2 = sorted_dists[:, 1]  # 2nd closest

            # Margin ratio: d2/d1 (>1 means well separated)
            ratios = d2 / np.clip(d1, 1e-10, None)

            # Check routing correctness
            closest_idx = np.argmin(dists, axis=1)
            true_idx = task_names.index(task_name)
            n_correct = int(np.sum(closest_idx == true_idx))

            product_short = task_name.split("/")[-1]
            rows.append({
                "benchmark": benchmark,
                "seed": seed,
                "product": product_short,
                "n_test": n_test,
                "n_correct": n_correct,
                "accuracy": n_correct / n_test,
                "margin_ratio_min": float(np.min(ratios)),
                "margin_ratio_mean": float(np.mean(ratios)),
                "margin_ratio_std": float(np.std(ratios)),
                "margin_ratio_p5": float(np.percentile(ratios, 5)),
                "d1_mean": float(np.mean(d1)),
                "d2_mean": float(np.mean(d2)),
            })

    return rows


def main() -> None:
    all_rows: list[dict] = []

    print("[Router Margins] MVTec AD...")
    all_rows.extend(compute_margins_for_benchmark(ALL_TASKS_FULL, "MVTec_AD"))

    print("[Router Margins] VisA...")
    all_rows.extend(compute_margins_for_benchmark(ALL_TASKS_VISA, "VisA"))

    print("[Router Margins] BTAD...")
    all_rows.extend(compute_margins_for_benchmark(ALL_TASKS_BTAD, "BTAD"))

    # Write CSV
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark", "seed", "product", "n_test", "n_correct", "accuracy",
        "margin_ratio_min", "margin_ratio_mean", "margin_ratio_std",
        "margin_ratio_p5", "d1_mean", "d2_mean",
    ]
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[Done] Wrote {len(all_rows)} rows to {OUT_PATH}")

    # Summary stats
    if all_rows:
        all_min_ratios = [r["margin_ratio_min"] for r in all_rows]
        all_mean_ratios = [r["margin_ratio_mean"] for r in all_rows]
        all_p5_ratios = [r["margin_ratio_p5"] for r in all_rows]

        print(f"\n=== SUMMARY ===")
        print(f"Global min margin ratio:  {min(all_min_ratios):.3f}")
        print(f"Global mean margin ratio: {np.mean(all_mean_ratios):.3f} ± {np.std(all_mean_ratios):.3f}")
        print(f"Global p5 margin ratio:   {np.mean(all_p5_ratios):.3f}")
        print(f"All accuracies 100%: {all(r['accuracy'] == 1.0 for r in all_rows)}")

        # Per-benchmark summary
        for bm in ["MVTec_AD", "VisA", "BTAD"]:
            bm_rows = [r for r in all_rows if r["benchmark"] == bm]
            if bm_rows:
                mins = [r["margin_ratio_min"] for r in bm_rows]
                means = [r["margin_ratio_mean"] for r in bm_rows]
                print(f"\n{bm}: min_of_mins={min(mins):.3f}, mean_of_means={np.mean(means):.3f}")

        # Worst-case products (lowest min margin)
        print(f"\n=== WORST-CASE PRODUCTS (lowest min margin ratio) ===")
        sorted_rows = sorted(all_rows, key=lambda r: r["margin_ratio_min"])
        for r in sorted_rows[:10]:
            print(f"  {r['benchmark']}/{r['product']} s{r['seed']}: "
                  f"min={r['margin_ratio_min']:.3f} mean={r['margin_ratio_mean']:.3f} "
                  f"p5={r['margin_ratio_p5']:.3f}")


if __name__ == "__main__":
    main()
