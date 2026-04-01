"""
Pre-cache all DINOv3 features to disk (.npy files).
Run once, then all experiment scripts load from cache instantly.

Usage: uv run precache_features.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from continual_utils import (
    ALL_TASKS_FULL, DINOv3Extractor, N_PER_TASK,
    preload_test_data, extract_train_features, CACHE_DIR,
)

SEEDS = [0, 1, 2]


def main() -> None:
    print("=" * 70)
    print("Pre-caching DINOv3 features to disk")
    print(f"  Tasks: {len(ALL_TASKS_FULL)}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Cache dir: {CACHE_DIR}")
    print("=" * 70)

    t0 = time.time()
    extractor = DINOv3Extractor("cuda:0")

    # Cache test features
    preload_test_data(extractor, ALL_TASKS_FULL)

    # Cache train features for all seeds
    print("\n[STEP 2] Pre-extracting train features for all tasks and seeds...")
    for seed in SEEDS:
        for task_name in ALL_TASKS_FULL:
            short = task_name.split("/")[-1]
            features, n = extract_train_features(extractor, task_name, seed, N_PER_TASK)
            print(f"  seed={seed} {short}: {features.shape} ({n} images)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} minutes")
    print(f"Cache files: {len(list(CACHE_DIR.glob('*.npy')))} .npy files")


if __name__ == "__main__":
    main()
