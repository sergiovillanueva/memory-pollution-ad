"""
Pre-cache DINOv3 features for VisA and BTAD datasets.
======================================================
Must run BEFORE any VisA/BTAD experiments.
Extracts test + train features and saves to output/feature_cache/.

Runtime: ~40 min on GPU (one-time cost).
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from continual_utils import (
    ALL_TASKS_VISA, ALL_TASKS_BTAD,
    N_PER_TASK, CACHE_DIR,
    DINOv3Extractor,
    load_dataset_with_masks,
    preload_test_data_with_masks,
    extract_train_features,
    cuda_cleanup,
    _cache_path,
)

SEEDS = [0, 1, 2]


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_tasks = ALL_TASKS_VISA + ALL_TASKS_BTAD

    # Check what's missing
    missing_test = [t for t in all_tasks if not _cache_path("test_feat", t).exists()]
    missing_train: list[tuple[str, int]] = []
    for seed in SEEDS:
        for t in all_tasks:
            if not _cache_path("train_feat", t, f"_s{seed}_n{N_PER_TASK}").exists():
                missing_train.append((t, seed))

    print(f"[INIT] Missing test caches: {len(missing_test)}/{len(all_tasks)}")
    print(f"[INIT] Missing train caches: {len(missing_train)}/{len(all_tasks)*len(SEEDS)}")

    if not missing_test and not missing_train:
        print("[DONE] All features already cached!")
        return

    extractor = DINOv3Extractor()

    # Cache test features
    if missing_test:
        print(f"\n[TEST] Extracting test features for {len(missing_test)} tasks...")
        preload_test_data_with_masks(extractor, missing_test)

    # Cache train features
    if missing_train:
        print(f"\n[TRAIN] Extracting train features for {len(missing_train)} (task, seed) pairs...")
        for task_name, seed in missing_train:
            feat, n = extract_train_features(extractor, task_name, seed, N_PER_TASK)
            short = task_name.split("/")[-1]
            print(f"  s{seed} {short:15s}: {feat.shape} ({n} images)")

    cuda_cleanup("done")
    print("\n[DONE] All extra features cached!")


if __name__ == "__main__":
    main()
