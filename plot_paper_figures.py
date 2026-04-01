"""
Plot Paper Figures
==================
Generates publication-quality figures for the paper.

Fig 2: Pareto AUROC vs Memory (compression benchmark)
Fig 3: Oracle Inversion Reversal (global vs routed, 18 MB)
Fig 4: WNR vs AUROC degradation scatter (per-product)
Fig 5: Router margin ratio histogram

Usage: uv run plot_paper_figures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_DIR = Path("output/figures")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def get_table_path(name: str) -> Path:
    """Get path to a table, falling back to pre-computed results/ if needed."""
    p_out = Path("output/paper_tables") / name
    if p_out.exists():
        return p_out
    return Path("results") / name

# Publication style — Elsevier single-column width ~3.5in
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
})


# =============================================================================
# FIG 2: Pareto AUROC vs Memory
# =============================================================================

def plot_fig2_pareto():
    """Pareto front: Image AUROC vs Memory (MB), log-x scale.

    Shows all compression configs from table1. Highlights the 16x knee.
    Configs in Table 1 of the paper are emphasized; others are faded.
    """
    df = pd.read_csv(get_table_path("table1_compression_benchmark.csv"))

    # Configs shown in Table 1 of the paper
    table1_configs = [
        "baseline", "km100_res400", "coreset_500", "coreset_200",
        "k500", "coreset_100", "k50",
    ]

    # Visual grouping by compression family
    family = {
        "baseline": ("Baseline", "k", "s", 70),
        "coreset_500": ("Coreset", "C0", "o", 50),
        "coreset_200": ("Coreset", "C0", "o", 50),
        "coreset_100": ("Coreset", "C0", "o", 50),
        "coreset_50": ("Coreset", "C0", "o", 30),
        "k500": ("K-means", "C2", "D", 45),
        "k200": ("K-means", "C2", "D", 35),
        "k100": ("K-means", "C2", "D", 30),
        "k50": ("K-means", "C2", "D", 30),
        "km100_res400": ("KM+Residual", "C1", "^", 55),
        "km100_res100": ("KM+Residual", "C1", "^", 35),
        "km200_res300": ("KM+Residual", "C1", "^", 35),
        "km50_res150": ("KM+Residual", "C1", "^", 30),
        "km50_res50": ("KM+Residual", "C1", "^", 30),
        "avg2x2": ("Spatial pool", "C3", "v", 40),
        "avg4x4": ("Spatial pool", "C3", "v", 40),
    }

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Track legend entries (one per family)
    legend_handles: dict[str, object] = {}

    for _, row in df.iterrows():
        cfg = row["config"]
        mem = row["memory_mb"]
        auroc = row["img_auroc_mean"]
        in_table = cfg in table1_configs
        fam_name, color, marker, size = family.get(cfg, ("Other", "gray", "x", 25))

        alpha = 1.0 if in_table else 0.35
        zorder = 4 if in_table else 2
        edgecolor = "black" if in_table else "none"
        lw = 0.4 if in_table else 0

        h = ax.scatter(
            mem, auroc, marker=marker, color=color,
            s=size, alpha=alpha, zorder=zorder,
            edgecolors=edgecolor, linewidths=lw,
        )
        if fam_name not in legend_handles:
            legend_handles[fam_name] = h

        # Label only Table 1 configs
        if in_table:
            label = cfg.replace("coreset_", "cs").replace("km100_res", "km+r")
            label = label.replace("k50", "km50") if cfg == "k50" else label
            offset = (4, 3)
            if cfg == "baseline":
                offset = (-8, -10)
            elif cfg == "coreset_500":
                offset = (4, -8)
            elif cfg == "k500":
                offset = (4, -8)
            ax.annotate(
                label, (mem, auroc), fontsize=6.5,
                textcoords="offset points", xytext=offset,
                color=color, alpha=0.9,
            )

    # 16x compression zone (knee point: ~29 MB cluster)
    ax.axvspan(20, 40, alpha=0.06, color="green", zorder=0)
    ax.text(
        26, 0.878, r"16$\times$", fontsize=7,
        color="green", alpha=0.6, style="italic",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Total memory (MB)")
    ax.set_ylabel("Image AUROC")
    ax.set_xlim(2, 600)
    ax.set_ylim(0.875, 0.918)
    ax.grid(True, alpha=0.2, linewidth=0.4)

    # Legend
    ordered = ["Baseline", "Coreset", "K-means", "KM+Residual"]
    handles = [legend_handles[k] for k in ordered if k in legend_handles]
    labels = [k for k in ordered if k in legend_handles]
    ax.legend(handles, labels, loc="lower right", framealpha=0.8, borderpad=0.3)

    fig.savefig(PLOT_DIR / "fig2_pareto.pdf")
    fig.savefig(PLOT_DIR / "fig2_pareto.png")
    plt.close(fig)
    print("[FIG2] Pareto AUROC vs Memory saved")


# =============================================================================
# FIG 3: Oracle Inversion Reversal (bar chart)
# =============================================================================

def plot_fig3_oracle_inversion():
    """Grouped bar chart showing oracle inversion under global retrieval
    and its reversal under top-1 routing. Budget = 18 MB.

    Two groups: "Global" and "Top-1 Routed"
    Two bars per group: Uniform (blue) and Oracle (orange)
    Annotated with delta values.
    """
    df = pd.read_csv(get_table_path("table3_routing_results.csv"))

    # Filter to 18 MB budget
    d18 = df[df["budget_total_mb"] == 18.0]

    global_uni = d18[(d18["routing_mode"] == "global") & (d18["allocator"] == "uniform")]["img_auroc_mean"].values[0]
    global_ora = d18[(d18["routing_mode"] == "global") & (d18["allocator"] == "oracle")]["img_auroc_mean"].values[0]
    top1_uni = d18[(d18["routing_mode"] == "top_1") & (d18["allocator"] == "uniform")]["img_auroc_mean"].values[0]
    top1_ora = d18[(d18["routing_mode"] == "top_1") & (d18["allocator"] == "oracle")]["img_auroc_mean"].values[0]

    global_uni_std = d18[(d18["routing_mode"] == "global") & (d18["allocator"] == "uniform")]["img_auroc_std"].values[0]
    global_ora_std = d18[(d18["routing_mode"] == "global") & (d18["allocator"] == "oracle")]["img_auroc_std"].values[0]
    top1_uni_std = d18[(d18["routing_mode"] == "top_1") & (d18["allocator"] == "uniform")]["img_auroc_std"].values[0]
    top1_ora_std = d18[(d18["routing_mode"] == "top_1") & (d18["allocator"] == "oracle")]["img_auroc_std"].values[0]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    x = np.array([0, 1.2])
    width = 0.35

    bars_uni = ax.bar(
        x - width / 2,
        [global_uni, top1_uni],
        width, label="Uniform",
        color="#4878CF", edgecolor="black", linewidth=0.4,
        yerr=[global_uni_std, top1_uni_std],
        capsize=3, error_kw={"linewidth": 0.7},
    )
    bars_ora = ax.bar(
        x + width / 2,
        [global_ora, top1_ora],
        width, label="Per-product oracle",
        color="#D65F5F", edgecolor="black", linewidth=0.4,
        yerr=[global_ora_std, top1_ora_std],
        capsize=3, error_kw={"linewidth": 0.7},
    )

    # Value labels on bars
    for bar_group in [bars_uni, bars_ora]:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.002,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7,
            )

    # Delta annotations
    delta_global = (global_ora - global_uni) * 100
    delta_top1 = (top1_ora - top1_uni) * 100

    mid_global = (global_uni + global_ora) / 2
    mid_top1 = (top1_uni + top1_ora) / 2

    ax.annotate(
        f"$\\Delta$ = {delta_global:+.1f} pp",
        xy=(0, mid_global - 0.003), fontsize=8, fontweight="bold",
        ha="center", color="#8B0000",
    )
    ax.annotate(
        f"$\\Delta$ = {delta_top1:+.1f} pp",
        xy=(1.2, mid_top1 + 0.006), fontsize=8, fontweight="bold",
        ha="center", color="#006400",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["Global retrieval", "Top-1 routed"], fontsize=9)
    ax.set_ylabel("Image AUROC")
    ax.set_ylim(0.885, 0.922)
    ax.legend(loc="upper left", framealpha=0.8, borderpad=0.3)
    ax.grid(axis="y", alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)

    # Title as text (cleaner for journal)
    ax.set_title("Budget = 18 MB, MVTec AD (15 products, 3 seeds)", fontsize=8, style="italic")

    fig.savefig(PLOT_DIR / "fig3_oracle_inversion.pdf")
    fig.savefig(PLOT_DIR / "fig3_oracle_inversion.png")
    plt.close(fig)
    print("[FIG3] Oracle Inversion Reversal saved")


# =============================================================================
# FIG 4: WNR vs AUROC degradation scatter
# =============================================================================

def plot_fig4_wnr_scatter():
    """Scatter: per-product WNR vs AUROC difference (global_uniform - top1_uniform).

    Shows that products with higher WNR suffer more from memory pollution.
    Uses 18 MB budget, averaged across seeds.
    """
    CONTINUAL_DIR = Path("output/continual")

    # Load WNR data (budget=200 embs/product ~ 18 MB for 15 products)
    wnr_df = pd.read_csv(get_table_path("table7_wnr.csv"))
    wnr_18 = wnr_df[wnr_df["budget_per_product"] == 200]
    wnr_mean = wnr_18.groupby("product")["wnr"].mean()

    # Load per-product AUROC from routing CSV
    routing_df = pd.read_csv(CONTINUAL_DIR / "c3_routing.csv", sep=";")
    final = routing_df[routing_df["task_idx"] == 14]  # final step (all 15 products)

    # Global uniform 18 MB
    g_uni = final[
        (final["routing_mode"] == "global") &
        (final["allocator"] == "uniform") &
        (final["budget_total_mb"] == 18.0)
    ]
    g_uni_auroc = g_uni.groupby("eval_task")["img_auroc"].mean()

    # Top-1 uniform 18 MB
    t1_uni = final[
        (final["routing_mode"] == "top_1") &
        (final["allocator"] == "uniform") &
        (final["budget_total_mb"] == 18.0)
    ]
    t1_uni_auroc = t1_uni.groupby("eval_task")["img_auroc"].mean()

    # Compute delta = routed - global (positive means routing helped)
    products = []
    wnr_vals = []
    delta_vals = []
    for task in g_uni_auroc.index:
        short = task.split("/")[-1]
        if short not in wnr_mean.index:
            continue
        if task not in t1_uni_auroc.index:
            continue
        products.append(short)
        wnr_vals.append(wnr_mean[short])
        delta_vals.append((t1_uni_auroc[task] - g_uni_auroc[task]) * 100)  # in pp

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.scatter(wnr_vals, delta_vals, s=30, color="#4878CF", edgecolors="black",
               linewidths=0.4, zorder=3)

    for i, p in enumerate(products):
        ax.annotate(p, (wnr_vals[i], delta_vals[i]), fontsize=5.5,
                    textcoords="offset points", xytext=(3, 3), alpha=0.8)

    # Trend line
    if len(wnr_vals) > 2:
        z = np.polyfit(wnr_vals, delta_vals, 1)
        x_line = np.linspace(min(wnr_vals) - 0.02, max(wnr_vals) + 0.02, 50)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="gray", alpha=0.5, linewidth=0.8)

        # Pearson correlation
        corr = np.corrcoef(wnr_vals, delta_vals)[0, 1]
        ax.text(0.95, 0.05, f"r = {corr:.2f}", transform=ax.transAxes,
                fontsize=7, ha="right", va="bottom", style="italic", color="gray")

    ax.axhline(0, color="black", linewidth=0.3, alpha=0.3)
    ax.set_xlabel("Wrong-Neighbor Rate (WNR)")
    ax.set_ylabel(r"$\Delta$AUROC: routed $-$ global (pp)")
    ax.set_title("Per-product routing gain vs. pollution\n(18 MB, uniform, 3 seeds)", fontsize=8, style="italic")
    ax.grid(True, alpha=0.2, linewidth=0.4)

    fig.savefig(PLOT_DIR / "fig4_wnr_scatter.pdf")
    fig.savefig(PLOT_DIR / "fig4_wnr_scatter.png")
    plt.close(fig)
    print("[FIG4] WNR vs AUROC scatter saved")


# =============================================================================
# FIG 5: Router margin ratio histogram
# =============================================================================

def plot_fig5_margin_histogram():
    """Histogram of router margin ratios (d2/d1) across all test images.

    Shows the distribution of routing confidence. All values > 1 means
    the router always picks the correct product with a margin.
    """
    margins_df = pd.read_csv(get_table_path("router_margins.csv"))

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Plot per-benchmark histograms using mean margin per (product, seed)
    colors = {"MVTec_AD": "#4878CF", "VisA": "#D65F5F", "BTAD": "#6ACC65"}
    labels = {"MVTec_AD": "MVTec AD", "VisA": "VisA", "BTAD": "BTAD"}

    all_means = []
    for bm in ["MVTec_AD", "VisA", "BTAD"]:
        bm_data = margins_df[margins_df["benchmark"] == bm]["margin_ratio_mean"].values
        all_means.extend(bm_data)
        ax.hist(bm_data, bins=15, alpha=0.5, color=colors[bm],
                edgecolor=colors[bm], linewidth=0.5, label=labels[bm])

    # Vertical line at minimum
    global_min = margins_df["margin_ratio_min"].min()
    ax.axvline(global_min, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.text(global_min + 0.1, ax.get_ylim()[1] * 0.85,
            f"min = {global_min:.2f}", fontsize=7, color="red", alpha=0.8)

    # Vertical line at x=1 (decision boundary)
    ax.axvline(1.0, color="black", linewidth=0.5, linestyle=":", alpha=0.4)

    ax.set_xlabel("Mean margin ratio ($d_2 / d_1$)")
    ax.set_ylabel("Count (product, seed)")
    ax.set_title("Router confidence distribution\n(all benchmarks, 3 seeds)", fontsize=8, style="italic")
    ax.legend(loc="upper right", framealpha=0.8, borderpad=0.3)
    ax.grid(axis="y", alpha=0.2, linewidth=0.4)
    ax.set_axisbelow(True)

    fig.savefig(PLOT_DIR / "fig5_margin_histogram.pdf")
    fig.savefig(PLOT_DIR / "fig5_margin_histogram.png")
    plt.close(fig)
    print("[FIG5] Margin ratio histogram saved")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("[PLOT] Generating paper figures...")
    print(f"[PLOT] Output directory: {PLOT_DIR}")

    plot_fig2_pareto()
    plot_fig3_oracle_inversion()
    plot_fig4_wnr_scatter()
    plot_fig5_margin_histogram()

    print(f"\n[DONE] Figures saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
