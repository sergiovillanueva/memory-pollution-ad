# Memory Pollution in Multi-Product Anomaly Detection

Official code for the paper *"Memory Pollution in Multi-Product Anomaly Detection: Diagnosis and Resolution"*.

## Overview

This repository provides the code to reproduce the experiments in the paper.
We study inter-product interference ("memory pollution") in shared KNN memory
banks for visual anomaly detection and propose a training-free routing mechanism
to eliminate it.

**Key contributions:**

1. Compression benchmark: 16x memory reduction with <1 pp AUROC loss using coreset selection and prototype clustering on frozen DINOv3 features.
2. Memory pollution diagnosis: per-product oracle allocation underperforms uniform by up to 1.6 pp due to cross-product KNN interference, quantified via Wrong-Neighbor Rate (WNR).
3. Latent routing: mean-embedding prototype router (0.06 MB overhead, 100% accuracy across 30 products) eliminates pollution and restores oracle behavior.
4. Allocator robustness: under routing, five allocation strategies converge to within 1 pp, making simple uniform allocation sufficient.

## Requirements

- Python 3.11
- NVIDIA GPU with >= 16 GB VRAM (tested on RTX 4080)
- ~20 GB disk for DINOv3 feature caches

Install dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

Download and place datasets under `data/`:

```
data/
  mvtec_AD/       # MVTec AD (15 categories)
  VisA/           # VisA (12 categories)
  btad/           # BTAD (3 categories)
```

- **MVTec AD**: https://www.mvtec.com/company/research/datasets/mvtec-ad
- **VisA**: https://github.com/amazon-science/spot-diff
- **BTAD**: https://avires.dimi.uniud.it/papers/btad/btad.zip

## Reproducing the experiments

### Reproducibility levels (reviewer-friendly)

- **Level A (verify reported numbers, no datasets/GPU):** regenerate paper tables from the included raw CSV logs in `output/continual/` and compare against `results/`.
- **Level B (full re-run):** requires downloading datasets to `data/` and extracting DINOv3 feature caches.

### Step 0: Extract features

Pre-extract and cache DINOv3-ViT-L/16 features (layer -6, 448x448 resolution).
This step takes a few hours but only needs to run once:

```bash
python scripts/precache_features.py        # MVTec AD
python scripts/precache_features_extra.py  # VisA + BTAD
```

Cached features are saved as `.npy` files under `output/feature_cache/`.

### Step 1: Run experiments

Each script corresponds to a set of experiments in the paper. All scripts are
resumable (completed runs are skipped on restart) and append results to CSV files
under `output/continual/`.

| Script | Paper section | Description |
|--------|--------------|-------------|
| `scripts/exp_c2_001_forgetting_baseline.py` | Table 1 | Baseline (no compression) |
| `scripts/exp_c2_003_coreset.py` | Table 1 | Coreset selection |
| `scripts/exp_c2_004_prototype_merging.py` | Table 1 | K-means clustering |
| `scripts/exp_c2_006_kmeans_residuals.py` | Table 1 | K-means + density-aware residuals |
| `scripts/exp_c3_pixel_eval.py` | Table 1 | Pixel-level metrics |
| `scripts/exp_c3_budget_sweep.py` | (oracle curves) | Per-product AUROC vs budget |
| `scripts/exp_c3_adaptive.py` | Table 2 | Memory pollution (oracle inversion) |
| `scripts/exp_c3_routing.py` | Table 3, Fig 3-4 | Routing evaluation (main contribution) |
| `scripts/exp_c3_interference_budgeting.py` | Table 4 | Allocator robustness under routing |
| `scripts/exp_c3_extra_benchmarks.py` | Table 5 | VisA and BTAD validation |
| `scripts/exp_c4_megabank.py` | Sec 4.5 | 30-product mega-bank |
| `scripts/exp_c4_n_ablation.py` | Sec 4.6 | Training set size ablation |
| `scripts/exp_n50_oracle_validation.py` | Appendix | N=50 oracle validation |

### Step 2: Generate paper tables

Aggregate raw CSVs into the clean tables reported in the paper:

```bash
python summarize_results.py
python compute_wnr.py
python compute_router_margins.py
python compute_router_accuracy_extra.py
```

Output tables are saved to `output/paper_tables/` (also provided in `results/`
for reference).

### Step 3: Generate figures

```bash
python plot_paper_figures.py
```

Figures are saved to `output/figures/`.

## Pre-computed results

The `results/` directory contains the aggregated CSV tables used in the paper,
so that reviewers can verify the reported numbers without re-running all
experiments:

| File | Paper table |
|------|-------------|
| `table1_compression_benchmark.csv` | Table 1 |
| `table2_memory_pollution.csv` | Table 2 |
| `table3_routing_results.csv` | Table 3 |
| `table4_robustness.csv` | Table 4 |
| `table5_extra_benchmarks.csv` | Table 5 |
| `table6_router_accuracy.csv` | Table 6 |
| `table7_wnr.csv` | WNR analysis |
| `router_margins.csv` | Fig 5 |
| `router_accuracy_extra.csv` | VisA/BTAD router accuracy |

## Quick verification (no GPU, no datasets)

This repo includes the raw experiment logs in `output/continual/` (CSV, a few MB).
You can regenerate Tables 1–6 from those logs and verify they match the
reference tables in `results/`:

```bash
python verify_tables.py
```

## Notes on model/version stability

Feature extraction uses HuggingFace model `facebook/dinov3-vitl16-pretrain-lvd1689m` via `transformers`.
If you want fully deterministic weight resolution over time, set `DINO_REVISION` (see `continual_utils.DINOv3Extractor`).

## Project structure

```
memory-pollution-ad/
├── continual_utils.py          # Core module: features, KNN, routing, metrics
├── summarize_results.py        # Aggregate raw CSVs -> paper tables
├── compute_wnr.py              # Wrong-Neighbor Rate computation
├── compute_router_margins.py   # Router confidence margins
├── compute_router_accuracy_extra.py  # Router accuracy on VisA/BTAD
├── plot_paper_figures.py       # Generate all paper figures
├── requirements.txt            # Python dependencies
├── LICENSE
├── scripts/
│   ├── precache_features.py    # Feature extraction (MVTec AD)
│   ├── precache_features_extra.py  # Feature extraction (VisA, BTAD)
│   ├── exp_c2_*.py             # Compression experiments
│   ├── exp_c3_*.py             # Routing and allocation experiments
│   ├── exp_c4_*.py             # Ablation experiments
│   └── exp_n50_*.py            # N=50 validation
└── results/                    # Pre-computed paper tables (CSV)
```

## Hardware

All experiments were run on a single workstation:

- CPU: Intel Core i9
- RAM: 32 GB
- GPU: NVIDIA GPU, 16 GB VRAM
- OS: Windows 11

## Citation

```
@article{villanueva2025memorypollution,
  title={Memory Pollution in Multi-Product Anomaly Detection: Diagnosis and Resolution},
  author={Villanueva, Sergio},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
