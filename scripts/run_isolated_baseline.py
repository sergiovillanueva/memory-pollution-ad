"""
Isolated baseline: per-product banks without shared retrieval.

Step 1: Pre-cache DINOv3 features (if not cached).
Step 2: For each product, build its own bank (7840 embeddings, uncompressed)
        and evaluate ONLY against that bank. No sharing, no compression.

This answers Emilio's question: how much is lost from sharing vs from few-shot.

Usage: uv run scripts/run_isolated_baseline.py
"""
from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FEATURE_DIM: int = 1024
KNN_K: int = 5
N_PER_TASK: int = 10
SEEDS: list[int] = [0, 1, 2]

TASKS_MVTEC: list[str] = [
    "mvtec_AD/bottle", "mvtec_AD/cable", "mvtec_AD/capsule",
    "mvtec_AD/carpet", "mvtec_AD/grid", "mvtec_AD/hazelnut",
    "mvtec_AD/leather", "mvtec_AD/metal_nut", "mvtec_AD/pill",
    "mvtec_AD/screw", "mvtec_AD/tile", "mvtec_AD/toothbrush",
    "mvtec_AD/transistor", "mvtec_AD/wood", "mvtec_AD/zipper",
]

OUTPUT_DIR = Path("output/continual")
CSV_PATH = OUTPUT_DIR / "isolated_baseline.csv"
CACHE_DIR = Path("output/feature_cache")

COLUMNS = [
    "timestamp", "seed", "task_name", "n_test", "n_train_patches",
    "memory_mb", "img_auroc", "img_aupr", "query_time_ms",
]


def knn_score_images(bank: np.ndarray, test_feat: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """Score test images using KNN on a single bank."""
    n_test = test_feat.shape[0]
    flat = test_feat.reshape(n_test, -1, FEATURE_DIM)
    scores = np.zeros(n_test, dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=min(k, len(bank)), metric="euclidean")
    nn.fit(bank)
    for i in range(n_test):
        dists, _ = nn.kneighbors(flat[i])
        patch_scores = dists.mean(axis=1)
        scores[i] = np.percentile(patch_scores, 95)
    return scores


def get_completed() -> set[tuple[str, ...]]:
    """Load completed experiments from CSV."""
    completed: set[tuple[str, ...]] = set()
    if CSV_PATH.exists():
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                completed.add((row["seed"], row["task_name"]))
    return completed


def save_result(result: dict) -> None:
    """Append one result row to CSV."""
    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def _cache_path(prefix: str, task_name: str, suffix: str = "") -> Path:
    safe_name = task_name.replace("/", "_")
    return CACHE_DIR / f"{prefix}_{safe_name}{suffix}.npy"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    completed = get_completed()
    print(f"[INIT] {len(completed)} completed experiments")

    # Step 1: Check if features are cached, if not extract them
    need_extraction = False
    for task_name in TASKS_MVTEC:
        test_path = _cache_path("test_feat", task_name)
        if not test_path.exists():
            need_extraction = True
            break
    for seed in SEEDS:
        for task_name in TASKS_MVTEC:
            train_path = _cache_path("train_feat", task_name, f"_s{seed}_n{N_PER_TASK}")
            if not train_path.exists():
                need_extraction = True
                break

    if need_extraction:
        print("\n[STEP 1] Feature cache missing. Extracting DINOv3 features...")
        from continual_utils import (
            DINOv3Extractor, preload_test_data, extract_train_features,
            ALL_TASKS_FULL,
        )
        extractor = DINOv3Extractor("cuda:0")
        # Only cache MVTec AD tasks
        preload_test_data(extractor, TASKS_MVTEC)
        for seed in SEEDS:
            for task_name in TASKS_MVTEC:
                short = task_name.split("/")[-1]
                features, n = extract_train_features(extractor, task_name, seed, N_PER_TASK)
                print(f"  seed={seed} {short}: {features.shape}")
        del extractor
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        print("[STEP 1] Feature extraction complete.")
    else:
        print("[STEP 1] Feature cache found. Skipping extraction.")

    # Step 2: Evaluate isolated baseline
    print("\n[STEP 2] Evaluating isolated per-product baseline...")

    for seed in SEEDS:
        aurocs = []
        for task_name in TASKS_MVTEC:
            key = (str(seed), task_name)
            if key in completed:
                print(f"  [SKIP] seed={seed} {task_name}")
                continue

            train_path = _cache_path("train_feat", task_name, f"_s{seed}_n{N_PER_TASK}")
            test_path = _cache_path("test_feat", task_name)
            label_path = _cache_path("test_labels", task_name)

            train_feat = np.load(train_path)   # [N, 784, 1024]
            test_feat = np.load(test_path)     # [N_test, 784, 1024]
            test_labels = np.load(label_path)  # [N_test]

            # Build isolated bank: ALL embeddings, no compression
            bank = train_feat.reshape(-1, FEATURE_DIM)  # [N*784, 1024]
            memory_mb = bank.nbytes / (1024 * 1024)

            t0 = time.perf_counter()
            scores = knn_score_images(bank, test_feat, k=KNN_K)
            query_ms = (time.perf_counter() - t0) * 1000 / len(test_labels)

            auroc = float(roc_auc_score(test_labels, scores))
            # AUPR
            from sklearn.metrics import average_precision_score
            aupr = float(average_precision_score(test_labels, scores))

            short = task_name.split("/")[-1]
            print(f"  seed={seed} {short:15s}: AUROC={auroc:.4f}  "
                  f"bank={len(bank)} embs ({memory_mb:.1f} MB)  "
                  f"query={query_ms:.1f} ms")

            save_result({
                "timestamp": datetime.now().isoformat(),
                "seed": seed,
                "task_name": task_name,
                "n_test": len(test_labels),
                "n_train_patches": len(bank),
                "memory_mb": round(memory_mb, 2),
                "img_auroc": round(auroc, 6),
                "img_aupr": round(aupr, 6),
                "query_time_ms": round(query_ms, 2),
            })
            aurocs.append(auroc)

        if aurocs:
            print(f"  Seed {seed} mean AUROC: {np.mean(aurocs)*100:.1f}%")

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Isolated per-product baseline (uncompressed)")
    print("=" * 60)

    if CSV_PATH.exists():
        import pandas as pd
        df = pd.read_csv(CSV_PATH, sep=";")
        per_seed = df.groupby("seed")["img_auroc"].mean()
        total_mb = df.groupby("seed")["memory_mb"].sum()
        print(f"  AUROC: {per_seed.mean()*100:.1f} ± {per_seed.std()*100:.1f}%")
        print(f"  Total memory: {total_mb.mean():.1f} MB (15 products)")
        print(f"  Per-product: {total_mb.mean()/15:.1f} MB")
        print()
        print("  Comparison:")
        print(f"    Isolated uncompressed:    {per_seed.mean()*100:.1f}% ({total_mb.mean():.1f} MB)")
        print(f"    Shared uncompressed:      91.2% (459.4 MB)")
        print(f"    Shared + routing (18 MB): 91.1% (18.0 MB)")


if __name__ == "__main__":
    main()
