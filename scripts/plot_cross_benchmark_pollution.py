"""Bar chart: oracle inversion delta across MVTec AD, VisA, BTAD."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load cross-benchmark data
df_cb = pd.read_csv("output/continual/cross_benchmark_pollution.csv", sep=";")

# Load MVTec data from routing CSV
df_mvtec = pd.read_csv("output/continual/c3_routing.csv", sep=";")
df_mvtec = df_mvtec[
    (df_mvtec["routing_mode"] == "global")
    & (df_mvtec["task_idx"] == 14)
    & (df_mvtec["allocator"].isin(["uniform", "oracle"]))
]

BUDGETS = [6, 12, 18, 24, 30]


def compute_delta(df, budget):
    u = df[df["allocator"] == "uniform"]
    o = df[df["allocator"] == "oracle"]
    u_avg = u[u["budget_total_mb" if "budget_total_mb" in df.columns else "budget_mb"] == budget][
        "img_auroc"
    ].mean()
    o_avg = o[o["budget_total_mb" if "budget_total_mb" in df.columns else "budget_mb"] == budget][
        "img_auroc"
    ].mean()
    return (o_avg - u_avg) * 100


# MVTec deltas
mvtec_deltas = []
for b in BUDGETS:
    u = df_mvtec[
        (df_mvtec["allocator"] == "uniform") & (df_mvtec["budget_total_mb"] == float(b))
    ]["img_auroc"].mean()
    o = df_mvtec[
        (df_mvtec["allocator"] == "oracle") & (df_mvtec["budget_total_mb"] == float(b))
    ]["img_auroc"].mean()
    mvtec_deltas.append((o - u) * 100)

# VisA deltas
visa_deltas = []
for b in BUDGETS:
    sub = df_cb[df_cb["dataset"] == "VisA"]
    u = sub[(sub["allocator"] == "uniform") & (sub["budget_mb"] == float(b))]["img_auroc"].mean()
    o = sub[(sub["allocator"] == "oracle") & (sub["budget_mb"] == float(b))]["img_auroc"].mean()
    visa_deltas.append((o - u) * 100)

# BTAD deltas
btad_deltas = []
for b in BUDGETS:
    sub = df_cb[df_cb["dataset"] == "BTAD"]
    u = sub[(sub["allocator"] == "uniform") & (sub["budget_mb"] == float(b))]["img_auroc"].mean()
    o = sub[(sub["allocator"] == "oracle") & (sub["budget_mb"] == float(b))]["img_auroc"].mean()
    btad_deltas.append((o - u) * 100)

# Plot
fig, ax = plt.subplots(figsize=(4.8, 3.2))

x = np.arange(len(BUDGETS))
width = 0.28

bars1 = ax.bar(
    x - width, mvtec_deltas, width, label="MVTec AD (15)", color="#4878CF",
    edgecolor="black", linewidth=0.4,
)
bars2 = ax.bar(
    x, visa_deltas, width, label="VisA (12)", color="#D65F5F",
    edgecolor="black", linewidth=0.4,
)
bars3 = ax.bar(
    x + width, btad_deltas, width, label="BTAD (3)", color="#6ACC65",
    edgecolor="black", linewidth=0.4,
)

ax.axhline(0, color="black", linewidth=0.4)
ax.set_xticks(x)
ax.set_xticklabels([f"{b}" for b in BUDGETS])
ax.set_xlabel("Memory budget (MB)")
ax.set_ylabel(r"$\Delta$AUROC: oracle $-$ uniform (pp)")
ax.set_title(
    r"Oracle inversion across benchmarks (global retrieval, $N=10$)",
    fontsize=8.5,
    style="italic",
    pad=8,
)
ax.legend(loc="lower right", framealpha=0.8, fontsize=7.5, borderpad=0.3)
ax.grid(axis="y", alpha=0.2, linewidth=0.4)
ax.set_axisbelow(True)

fig.tight_layout()
out_pdf = Path("output/figures/fig6_cross_benchmark_pollution.pdf")
out_png = Path("output/figures/fig6_cross_benchmark_pollution.png")
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=300)
plt.close(fig)
print(f"Saved: {out_pdf} and {out_png}")
