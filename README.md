# Memory Pollution in Multi-Product Visual Anomaly Detection: Diagnosis and Mitigation

[![Paper](https://img.shields.io/badge/Paper-MAKE%202026-1F3864)](https://doi.org/10.3390/make8070219)
[![License: non-commercial, Eines](https://img.shields.io/badge/License-non--commercial%20%C2%B7%20Eines-red.svg)](LICENSE)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)

Official code of the paper

> S. Villanueva López, E. Soria-Olivas, M. Sánchez-Montañés. *Memory Pollution in Multi-Product Visual Anomaly
> Detection: Diagnosis and Mitigation.* Machine Learning and Knowledge Extraction **8**(7), 219, 2026.
> [doi:10.3390/make8070219](https://doi.org/10.3390/make8070219)

Memory-bank detectors such as PatchCore are popular in industrial quality control because they need no training and
are quick to deploy. A production line usually inspects several products or uses several cameras, and keeping one
shared nearest-neighbor memory bank for all of them makes deployment simpler. The catch: features from different
products interfere during retrieval. The paper calls this **memory pollution**, measures it and removes it.

## Main contributions

1. **Wrong-neighbor rate (WNR)**, a metric of how often a query's nearest neighbor belongs to a different product.
2. **Oracle inversion**, an effect validated on real benchmarks: under a fixed memory budget, querying only the
   product's own data can score worse than the shared bank.
3. **A first-order analytical model of the WNR** that predicts how pollution grows with the number of products and
   the memory budget.
4. **A minimal training-free router** (mean-embedding prototypes on frozen DINOv3 features) that removes the effect
   of memory pollution, with up to 25x memory reduction across datasets and backbones.

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

```bibtex
@article{villanueva2026memorypollution,
  title   = {Memory Pollution in Multi-Product Visual Anomaly Detection: Diagnosis and Mitigation},
  author  = {Villanueva L{\'o}pez, Sergio and Soria-Olivas, Emilio and S{\'a}nchez-Monta{\~n}{\'e}s, Manuel},
  journal = {Machine Learning and Knowledge Extraction},
  volume  = {8},
  number  = {7},
  pages   = {219},
  year    = {2026},
  doi     = {10.3390/make8070219}
}
```

## License

Copyright (c) 2026 Eines Systems S.L.U. (Eines Vision Systems). The code was developed at
Eines Vision Systems within an industrial PhD with the Universitat de València. It may be
used for non-commercial research and teaching and to reproduce the results of the paper.
Commercial use requires prior written permission from Eines. See [LICENSE](LICENSE).
The datasets keep their own licenses.

## Related repositories

This repository is one of five from my industrial PhD on anomaly detection for visual inspection (Universitat de València and Eines Vision Systems, 2026). All five work on frozen foundation-model features and do not train a network for each product.

| Repository | Paper | Question |
|---|---|---|
| [rgb-ensembles-industrial-uad](https://github.com/sergiovillanueva/rgb-ensembles-industrial-uad) | Electronics, 2026 | Does combining several detectors improve defect localization on a real automotive benchmark? |
| [GaLAD_Gaussian_layered_AD](https://github.com/sergiovillanueva/GaLAD_Gaussian_layered_AD) | under review | How do you set up a new product from one to five good images and still find small defects? |
| **memory-pollution-ad** (this one) | MAKE, 2026 | What goes wrong when many products share one memory bank, and how do you fix it? |
| [patch-purification-fewshot](https://github.com/sergiovillanueva/patch-purification-fewshot) | Electronics, 2026 | What happens if a defective part slips into the reference images at commissioning? |
| [pc-reg](https://github.com/sergiovillanueva/pc-reg) | Electronics, 2026 | How do you detect parts that look fine but are missing or in the wrong place? |

Sergio Villanueva López · [LinkedIn](https://www.linkedin.com/in/sergiovillanuevalopez)
