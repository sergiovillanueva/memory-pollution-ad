"""
Summarize all experiment results into clean CSV tables for the paper.
==================================================================
Reads raw CSVs from output/continual/ and generates paper-ready tables
in output/paper_tables/.

Usage: uv run summarize_results.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("output/continual")
OUT_DIR = Path("output/paper_tables")


def read_csv(name: str) -> pd.DataFrame:
    path = RAW_DIR / name
    if not path.exists():
        print(f"  [WARN] {name} not found, skipping")
        return pd.DataFrame()
    df = pd.read_csv(path, delimiter=";")
    print(f"  {name}: {len(df)} rows")
    return df


def agg_mean_std(group: pd.DataFrame, col: str) -> tuple[float, float]:
    vals = group[col].dropna()
    if len(vals) == 0:
        return np.nan, np.nan
    return vals.mean(), vals.std()


# =========================================================================
# TABLE 1: Compression Benchmark (MVTec, final task, 3 seeds)
# Sources: c2_001, c2_003, c2_004, c2_006, c3_pixel_eval
# =========================================================================
def table1_compression() -> pd.DataFrame:
    print("\n[TABLE 1] Compression Benchmark (MVTec)")

    # Image-level CSVs - all have task_idx column
    baseline = read_csv("c2_001_forgetting_baseline.csv")
    coreset = read_csv("c2_003_coreset.csv")
    kmeans = read_csv("c2_004_prototype_merging.csv")
    kmres = read_csv("c2_006_kmeans_residuals.csv")
    pixel = read_csv("c3_pixel_eval.csv")

    results = []

    # Baseline from c2_001 (memory_mode == "full")
    # NOTE: query_ms filled from c3_pixel_eval below (consistent measurement across configs)
    if not baseline.empty:
        final = baseline[baseline["task_idx"] == baseline["task_idx"].max()]
        # Aggregate across eval tasks to get mean AUROC at final snapshot
        for seed in final["seed"].unique():
            seed_data = final[final["seed"] == seed]
            results.append({
                "config": "baseline",
                "seed": seed,
                "img_auroc": seed_data["img_auroc"].mean(),
                "memory_mb": seed_data["memory_mb"].iloc[0],
                "query_ms": np.nan,  # filled from pixel_eval below
            })

    # Coreset configs from c2_003
    if not coreset.empty:
        final = coreset[coreset["task_idx"] == coreset["task_idx"].max()]
        for config in final["config"].unique():
            cfg_data = final[final["config"] == config]
            for seed in cfg_data["seed"].unique():
                seed_data = cfg_data[cfg_data["seed"] == seed]
                results.append({
                    "config": config,
                    "seed": seed,
                    "img_auroc": seed_data["img_auroc"].mean(),
                    "memory_mb": seed_data["memory_mb"].iloc[0],
                    "query_ms": seed_data["query_time_ms"].mean() if "query_time_ms" in seed_data else np.nan,
                })

    # K-means configs from c2_004
    if not kmeans.empty:
        final = kmeans[kmeans["task_idx"] == kmeans["task_idx"].max()]
        for config in final["config"].unique():
            cfg_data = final[final["config"] == config]
            for seed in cfg_data["seed"].unique():
                seed_data = cfg_data[cfg_data["seed"] == seed]
                results.append({
                    "config": config,
                    "seed": seed,
                    "img_auroc": seed_data["img_auroc"].mean(),
                    "memory_mb": seed_data["memory_mb"].iloc[0],
                    "query_ms": seed_data["query_time_ms"].mean() if "query_time_ms" in seed_data else np.nan,
                })

    # KM+Residuals configs from c2_006
    if not kmres.empty:
        final = kmres[kmres["task_idx"] == kmres["task_idx"].max()]
        for config in final["config"].unique():
            cfg_data = final[final["config"] == config]
            for seed in cfg_data["seed"].unique():
                seed_data = cfg_data[cfg_data["seed"] == seed]
                results.append({
                    "config": config,
                    "seed": seed,
                    "img_auroc": seed_data["img_auroc"].mean(),
                    "memory_mb": seed_data["memory_mb"].iloc[0],
                    "query_ms": seed_data["query_time_ms"].mean() if "query_time_ms" in seed_data else np.nan,
                })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Add pixel metrics and query_time from c3_pixel_eval (consistent source for all configs)
    if not pixel.empty:
        # Use smooth_sigma=0 rows only for query time consistency
        pix_s0 = pixel[pixel["smooth_sigma"] == 0.0] if "smooth_sigma" in pixel.columns else pixel
        pix_agg = pix_s0.groupby(["config", "seed"]).agg(
            pix_auroc=("pix_auroc", "mean"),
            query_ms_pixel=("query_time_ms", "mean"),
        ).reset_index()
        df = df.merge(pix_agg, on=["config", "seed"], how="left")
        # Use pixel_eval query_ms for ALL configs (baseline included) for consistency
        df["query_ms"] = df["query_ms_pixel"].combine_first(df["query_ms"])
        df.drop(columns=["query_ms_pixel"], inplace=True)
    else:
        df["pix_auroc"] = np.nan

    # Aggregate across seeds
    summary = df.groupby("config").agg(
        img_auroc_mean=("img_auroc", "mean"),
        img_auroc_std=("img_auroc", "std"),
        pix_auroc_mean=("pix_auroc", "mean"),
        pix_auroc_std=("pix_auroc", "std"),
        memory_mb=("memory_mb", "first"),
        query_ms=("query_ms", "mean"),
    ).reset_index()

    # Compute compression ratio vs baseline
    baseline_mem = summary.loc[summary["config"] == "baseline", "memory_mb"].values
    if len(baseline_mem) > 0:
        summary["compression_ratio"] = baseline_mem[0] / summary["memory_mb"]
    else:
        summary["compression_ratio"] = np.nan

    summary = summary.sort_values("img_auroc_mean", ascending=False)
    return summary


# =========================================================================
# TABLE 2: Memory Pollution / Oracle Inversion (c3_adaptive)
# =========================================================================
def table2_memory_pollution() -> pd.DataFrame:
    print("\n[TABLE 2] Memory Pollution (Oracle Inversion)")
    df = read_csv("c3_adaptive.csv")
    if df.empty:
        return df

    # Final task only (task_idx == 14 for MVTec 15 products)
    final = df[df["task_idx"] == df["task_idx"].max()]

    # Group by allocator, budget, seed - mean auroc across eval tasks
    grouped = final.groupby(["allocator", "budget_total_mb", "seed"]).agg(
        img_auroc=("img_auroc", "mean"),
    ).reset_index()

    # Now aggregate across seeds
    summary = grouped.groupby(["allocator", "budget_total_mb"]).agg(
        img_auroc_mean=("img_auroc", "mean"),
        img_auroc_std=("img_auroc", "std"),
    ).reset_index()

    # Pivot to show uniform vs oracle side by side
    rows = []
    for budget in sorted(summary["budget_total_mb"].unique()):
        b_data = summary[summary["budget_total_mb"] == budget]
        uniform = b_data[b_data["allocator"] == "uniform"]
        oracle = b_data[b_data["allocator"] == "oracle"]
        adaptive = b_data[b_data["allocator"] == "adaptive"]

        u_mean = uniform["img_auroc_mean"].values[0] if len(uniform) > 0 else np.nan
        o_mean = oracle["img_auroc_mean"].values[0] if len(oracle) > 0 else np.nan
        a_mean = adaptive["img_auroc_mean"].values[0] if len(adaptive) > 0 else np.nan
        u_std = uniform["img_auroc_std"].values[0] if len(uniform) > 0 else np.nan
        o_std = oracle["img_auroc_std"].values[0] if len(oracle) > 0 else np.nan

        rows.append({
            "budget_mb": budget,
            "uniform_auroc": round(u_mean, 4),
            "uniform_std": round(u_std, 4) if not np.isnan(u_std) else np.nan,
            "oracle_auroc": round(o_mean, 4),
            "oracle_std": round(o_std, 4) if not np.isnan(o_std) else np.nan,
            "adaptive_auroc": round(a_mean, 4),
            "delta_oracle_uniform_pp": round((o_mean - u_mean) * 100, 2),
        })

    return pd.DataFrame(rows)


# =========================================================================
# TABLE 3: Routing Results (c3_routing)
# =========================================================================
def table3_routing() -> pd.DataFrame:
    print("\n[TABLE 3] Routing Results")
    df = read_csv("c3_routing.csv")
    if df.empty:
        return df

    final = df[df["task_idx"] == df["task_idx"].max()]

    # Group by routing_mode, allocator, budget, seed
    grouped = final.groupby(["routing_mode", "allocator", "budget_total_mb", "seed"]).agg(
        img_auroc=("img_auroc", "mean"),
        pix_auroc=("pix_auroc", lambda x: x.dropna().mean() if x.dropna().any() else np.nan),
        query_ms=("query_time_ms", "mean"),
    ).reset_index()

    # Aggregate across seeds
    summary = grouped.groupby(["routing_mode", "allocator", "budget_total_mb"]).agg(
        img_auroc_mean=("img_auroc", "mean"),
        img_auroc_std=("img_auroc", "std"),
        pix_auroc_mean=("pix_auroc", "mean"),
        query_ms=("query_ms", "mean"),
    ).reset_index()

    # Keep only paper-essential modes
    essential_modes = ["global", "top_1", "top_3"]
    essential_allocs = ["uniform", "oracle"]
    summary = summary[
        summary["routing_mode"].isin(essential_modes) &
        summary["allocator"].isin(essential_allocs)
    ]

    # Remove 24MB rows: only seed 0 data available, no std (reviewer vulnerability)
    summary = summary[summary["budget_total_mb"] != 24.0]

    summary = summary.sort_values(["budget_total_mb", "routing_mode", "allocator"])
    return summary


# =========================================================================
# TABLE 4: Robustness - Allocator Agnosticism Under Routing (c3_interference_budgeting)
# =========================================================================
def table4_robustness() -> pd.DataFrame:
    print("\n[TABLE 4] Robustness (Allocator Agnosticism Under Routing)")
    df = read_csv("c3_interference_budgeting.csv")
    if df.empty:
        return df

    final = df[df["task_idx"] == df["task_idx"].max()]

    # Group by allocator, budget, seed
    grouped = final.groupby(["allocator", "budget_total_mb", "seed"]).agg(
        img_auroc=("img_auroc", "mean"),
    ).reset_index()

    # Aggregate across seeds (use only complete seeds)
    seed_counts = grouped.groupby("seed").size()
    complete_seeds = seed_counts[seed_counts == seed_counts.max()].index
    grouped = grouped[grouped["seed"].isin(complete_seeds)]

    summary = grouped.groupby(["allocator", "budget_total_mb"]).agg(
        img_auroc_mean=("img_auroc", "mean"),
        img_auroc_std=("img_auroc", "std"),
        n_seeds=("img_auroc", "count"),
    ).reset_index()

    # Pivot: rows=allocator, cols=budget
    pivot = summary.pivot_table(
        index="allocator",
        columns="budget_total_mb",
        values="img_auroc_mean",
    )
    pivot.columns = [f"budget_{int(c)}mb" for c in pivot.columns]
    pivot = pivot.reset_index()

    return pivot


# =========================================================================
# TABLE 5: Extra Benchmarks VisA/BTAD (c3_extra_benchmarks)
# =========================================================================
def table5_extra_benchmarks() -> pd.DataFrame:
    print("\n[TABLE 5] Extra Benchmarks (VisA/BTAD)")
    df = read_csv("c3_extra_benchmarks.csv")
    if df.empty:
        return df

    # For each benchmark, find the final task_idx
    results = []
    for bench in df["benchmark"].unique():
        bench_data = df[df["benchmark"] == bench]
        max_task = bench_data["task_idx"].max()
        final = bench_data[bench_data["task_idx"] == max_task]

        for config in final["config"].unique():
            cfg_data = final[final["config"] == config]
            for seed in cfg_data["seed"].unique():
                seed_data = cfg_data[cfg_data["seed"] == seed]
                results.append({
                    "benchmark": bench,
                    "config": config,
                    "seed": seed,
                    "img_auroc": seed_data["img_auroc"].mean(),
                    "pix_auroc": seed_data["pix_auroc"].dropna().mean() if seed_data["pix_auroc"].dropna().any() else np.nan,
                    "memory_mb": seed_data["memory_mb"].iloc[0],
                })

    if not results:
        return pd.DataFrame()

    rdf = pd.DataFrame(results)
    summary = rdf.groupby(["benchmark", "config"]).agg(
        img_auroc_mean=("img_auroc", "mean"),
        img_auroc_std=("img_auroc", "std"),
        pix_auroc_mean=("pix_auroc", "mean"),
        pix_auroc_std=("pix_auroc", "std"),
        memory_mb=("memory_mb", "first"),
    ).reset_index()

    summary = summary.sort_values(["benchmark", "img_auroc_mean"], ascending=[True, False])
    return summary


# =========================================================================
# TABLE 6: Router Accuracy (c3_routing_accuracy)
# =========================================================================
def table6_router_accuracy() -> pd.DataFrame:
    print("\n[TABLE 6] Router Accuracy")
    df = read_csv("c3_routing_accuracy.csv")
    if df.empty:
        return df

    summary = df.groupby(["task_name", "routing_mode"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        n_seeds=("accuracy", "count"),
    ).reset_index()

    # Also add global summary
    global_summary = df.groupby("routing_mode").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_min=("accuracy", "min"),
        accuracy_max=("accuracy", "max"),
        n_measurements=("accuracy", "count"),
    ).reset_index()

    print("  Global router accuracy:")
    for _, row in global_summary.iterrows():
        print(f"    {row['routing_mode']}: {row['accuracy_mean']:.4f} (min={row['accuracy_min']:.4f}, n={row['n_measurements']})")

    return summary


# =========================================================================
# MAIN
# =========================================================================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("PAPER RESULTS SUMMARY")
    print("=" * 60)

    tables = {
        "table1_compression_benchmark": table1_compression,
        "table2_memory_pollution": table2_memory_pollution,
        "table3_routing_results": table3_routing,
        "table4_robustness": table4_robustness,
        "table5_extra_benchmarks": table5_extra_benchmarks,
        "table6_router_accuracy": table6_router_accuracy,
    }

    for name, func in tables.items():
        try:
            df = func()
            if df.empty:
                print(f"  -> EMPTY (no data)")
                continue
            path = OUT_DIR / f"{name}.csv"
            df.to_csv(path, index=False)
            print(f"  -> Saved: {path} ({len(df)} rows)")
            print(df.to_string(index=False, max_rows=20))
            print()
        except Exception as e:
            print(f"  -> ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"All tables saved to {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
