"""
C2_001: Continual AD Forgetting Baseline
=========================================
Measure how much forgetting occurs in KNN-5 when tasks are added sequentially.

Protocol:
- 15 MVTec AD categories in alphabetical order
- N=10 normal images per task
- After each task t, evaluate ALL tasks [1..t]
- Memory bank: unlimited (naive concatenation)

Metrics:
- ACC[t][j] = AUROC on task j after training task t
- Avg Accuracy = mean(ACC[T][j]) for j=1..T
- BWT = (1/(T-1)) * sum(ACC[T][j] - ACC[j][j]) for j=1..T-1
- Memory MB, update time, inference time

Backbone: DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m)
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import sys
import gc
import csv
import time
import warnings
import traceback
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModel

# =============================================================================
# CONFIGURATION
# =============================================================================

DEBUG = "--debug" in sys.argv
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path("data")
OUTPUT_DIR = Path("output/continual")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESOLUTION = 448
LAYER_IDX = -6
BATCH_SIZE = 16
KNN_K = 5
N_PER_TASK = 10

# Continual task sequence: MVTec AD alphabetical order
ALL_TASKS = [
    "mvtec_AD/bottle", "mvtec_AD/cable", "mvtec_AD/capsule",
    "mvtec_AD/carpet", "mvtec_AD/grid", "mvtec_AD/hazelnut",
    "mvtec_AD/leather", "mvtec_AD/metal_nut", "mvtec_AD/pill",
    "mvtec_AD/screw", "mvtec_AD/tile", "mvtec_AD/toothbrush",
    "mvtec_AD/transistor", "mvtec_AD/wood", "mvtec_AD/zipper",
]

SEEDS = [0, 1, 2] if not DEBUG else [0]

if DEBUG:
    ALL_TASKS = ALL_TASKS[:4]  # bottle, cable, capsule, carpet

CSV_PATH = OUTPUT_DIR / "c2_001_forgetting_baseline.csv"
ERROR_PATH = OUTPUT_DIR / "c2_001_errors.txt"

COLUMNS = [
    "timestamp", "seed", "memory_mode", "task_idx", "task_name", "n_tasks_seen",
    "eval_task_idx", "eval_task_name",
    "n_train_task", "n_test", "memory_n_embeddings", "memory_mb",
    "img_auroc", "img_aupr", "img_f1max", "tpr_fpr1", "tpr_fpr5",
    "update_time_s", "inference_time_s",
]

# Two memory modes:
# "per_task": each task has its own memory bank (no cross-task interference)
# "global":  all embeddings in a single shared bank (realistic continual scenario)
MEMORY_MODES = ["per_task", "global"]


# =============================================================================
# DINOv3 FEATURE EXTRACTOR
# =============================================================================

class DINOv3Extractor:
    def __init__(self, device: str = "cuda"):
        self.device = device
        model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        print(f"[FEAT] Loading DINOv3 on {device}...")
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

        self.patch_size = 16
        self.grid_size = RESOLUTION // self.patch_size  # 28
        self.num_patches = self.grid_size * self.grid_size  # 784
        self.num_register = getattr(self.model.config, "num_register_tokens", 4)
        self.feature_dim = 1024

        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((RESOLUTION, RESOLUTION)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print(f"[FEAT] Grid: {self.grid_size}x{self.grid_size}, patches: {self.num_patches}, dim: {self.feature_dim}")

    @torch.no_grad()
    def extract_batch(self, image_paths: list[str], batch_size: int = 16) -> np.ndarray:
        """Extract features -> [N, num_patches, feature_dim]."""
        all_features = []
        start_idx = 1 + self.num_register

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = [self.transform(Image.open(p).convert("RGB")) for p in batch_paths]
            batch = torch.stack(images).to(self.device)

            with torch.autocast(device_type='cuda', enabled=self.device.startswith('cuda')):
                outputs = self.model(batch, output_hidden_states=True)

            features = outputs.hidden_states[LAYER_IDX][:, start_idx:start_idx + self.num_patches, :]
            all_features.append(features.float().cpu().numpy())

            del batch, outputs, features

        return np.concatenate(all_features, axis=0)


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_standard_dataset(dataset_path: Path) -> tuple[list[str], list[str], list[int]]:
    """Load MVTec-style dataset (train/good, test/good+defect)."""
    train_dir = dataset_path / "train" / "good"
    if not train_dir.exists():
        train_dir = dataset_path / "train"

    train_images = sorted(
        [str(p) for p in train_dir.rglob("*.png")] +
        [str(p) for p in train_dir.rglob("*.jpg")] +
        [str(p) for p in train_dir.rglob("*.JPG")]
    )

    test_dir = dataset_path / "test"
    test_images, test_labels = [], []
    for subdir in sorted(test_dir.iterdir()):
        if not subdir.is_dir():
            continue
        label = 0 if subdir.name == "good" else 1
        for img_path in sorted(
            list(subdir.rglob("*.png")) +
            list(subdir.rglob("*.jpg")) +
            list(subdir.rglob("*.JPG"))
        ):
            test_images.append(str(img_path))
            test_labels.append(label)

    return train_images, test_images, test_labels


def load_dataset(dataset_name: str, train_limit: int | None = None, seed: int = 0) -> tuple:
    """Load train/test images and labels."""
    dataset_path = DATA_ROOT / dataset_name
    train_images, test_images, test_labels = load_standard_dataset(dataset_path)

    if train_limit and len(train_images) > train_limit:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(train_images), train_limit, replace=False)
        train_images = [train_images[i] for i in sorted(indices)]

    return train_images, test_images, np.array(test_labels)


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Compute image-level anomaly detection metrics."""
    if len(np.unique(labels)) < 2:
        return {"img_auroc": 0.0, "img_aupr": 0.0, "img_f1max": 0.0, "tpr_fpr1": 0.0, "tpr_fpr5": 0.0}

    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    # F1-max
    thresholds = np.percentile(scores, np.arange(0, 100, 1))
    f1_max = max(
        f1_score(labels, (scores >= t).astype(int), zero_division=0)
        for t in thresholds
    )

    # TPR at fixed FPR
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    if len(anomaly_scores) > 0 and len(normal_scores) > 0:
        fpr1_thresh = np.percentile(normal_scores, 99)
        fpr5_thresh = np.percentile(normal_scores, 95)
        tpr_fpr1 = float((anomaly_scores >= fpr1_thresh).mean())
        tpr_fpr5 = float((anomaly_scores >= fpr5_thresh).mean())
    else:
        tpr_fpr1 = tpr_fpr5 = 0.0

    return {
        "img_auroc": round(auroc, 6),
        "img_aupr": round(aupr, 6),
        "img_f1max": round(f1_max, 6),
        "tpr_fpr1": round(tpr_fpr1, 6),
        "tpr_fpr5": round(tpr_fpr5, 6),
    }


# =============================================================================
# KNN SCORER
# =============================================================================

def knn_score_images(
    memory_bank: np.ndarray,
    test_features: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Score test images using KNN distance to memory bank.

    Args:
        memory_bank: [M, D] all stored patch embeddings
        test_features: [N, P, D] test image features
        k: number of neighbors

    Returns:
        [N] image-level anomaly scores (higher = more anomalous)
    """
    n_samples, n_patches, feat_dim = test_features.shape
    flat_test = test_features.reshape(-1, feat_dim)

    k_actual = min(k, len(memory_bank))
    nn = NearestNeighbors(n_neighbors=k_actual, metric='euclidean', algorithm='auto')
    nn.fit(memory_bank)

    distances, _ = nn.kneighbors(flat_test)
    patch_scores = distances.mean(axis=1)  # Mean dist to k neighbors

    # Aggregate per image: percentile 95 (robust to outlier patches)
    patch_scores = patch_scores.reshape(n_samples, n_patches)
    image_scores = np.percentile(patch_scores, 95, axis=1)

    return image_scores


# =============================================================================
# RESUMABILITY
# =============================================================================

def get_completed() -> set[tuple]:
    """Load completed (seed, memory_mode, task_idx, eval_task_idx) from CSV."""
    completed = set()
    if CSV_PATH.exists():
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                key = (int(row["seed"]), row["memory_mode"], int(row["task_idx"]), int(row["eval_task_idx"]))
                completed.add(key)
    return completed


def save_result(result: dict) -> None:
    """Append one result row to CSV."""
    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def log_error(msg: str, exc: Exception) -> None:
    """Log error with traceback."""
    with open(ERROR_PATH, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | {msg}\n")
        f.write(traceback.format_exc() + "\n")


def cuda_cleanup(reason: str = "") -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if DEBUG and reason:
        print(f"  [CLEANUP] {reason}")


# =============================================================================
# MAIN CONTINUAL PROTOCOL
# =============================================================================

def run_continual_protocol(
    seed: int,
    memory_mode: str,
    extractor: DINOv3Extractor,
    test_data: dict,
    completed: set,
) -> None:
    """Run continual protocol for one seed and memory mode."""
    print(f"\n{'='*70}")
    print(f"SEED {seed} | MODE: {memory_mode}")
    print(f"{'='*70}")

    # Memory bank: dict task_name -> flat embeddings [M_i, D]
    memory_banks = {}

    for task_idx, task_name in enumerate(ALL_TASKS):
        n_tasks_seen = task_idx + 1
        short_name = task_name.split("/")[-1]

        # Check if all evals for this task_idx are done
        evals_needed = []
        for eval_idx in range(n_tasks_seen):
            if (seed, memory_mode, task_idx, eval_idx) not in completed:
                evals_needed.append(eval_idx)

        if not evals_needed:
            # Still need to load train features for memory bank
            print(f"\n  [Task {task_idx}] {short_name} - all evals done, loading memory...")
            train_images, _, _ = load_dataset(task_name, N_PER_TASK, seed)
            train_features = extractor.extract_batch(train_images)
            memory_banks[task_name] = train_features.reshape(-1, train_features.shape[-1])
            if DEBUG:
                print(f"    Memory bank: {sum(len(v) for v in memory_banks.values())} embeddings")
            continue

        print(f"\n  [Task {task_idx}] Adding: {short_name}")

        # --- UPDATE PHASE ---
        t_update = time.time()
        train_images, _, _ = load_dataset(task_name, N_PER_TASK, seed)
        train_features = extractor.extract_batch(train_images)
        flat_train = train_features.reshape(-1, train_features.shape[-1])
        memory_banks[task_name] = flat_train
        update_time = time.time() - t_update

        # Compute total memory
        total_embeddings = sum(len(v) for v in memory_banks.values())
        memory_mb = total_embeddings * 1024 * 4 / (1024 * 1024)

        if DEBUG:
            print(f"    Train images: {len(train_images)}, New embs: {len(flat_train)}")
            print(f"    Total memory: {total_embeddings} embs ({memory_mb:.1f} MB)")

        # --- EVALUATION PHASE ---
        # Build global bank once for this step
        global_bank = np.concatenate(list(memory_banks.values()), axis=0)

        if DEBUG:
            print(f"    Evaluating {len(evals_needed)} tasks ({memory_mode} mode)...")

        for eval_idx in evals_needed:
            eval_task = ALL_TASKS[eval_idx]
            eval_short = eval_task.split("/")[-1]

            try:
                test_features, test_labels = test_data[eval_task]
                t_infer = time.time()

                if memory_mode == "per_task":
                    # Each task scored against its own memory bank
                    bank = memory_banks[eval_task]
                else:
                    # Scored against ALL embeddings (shared memory)
                    bank = global_bank

                scores = knn_score_images(bank, test_features, k=KNN_K)
                inference_time = time.time() - t_infer

                metrics = compute_metrics(test_labels, scores)

                result = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "seed": seed,
                    "memory_mode": memory_mode,
                    "task_idx": task_idx,
                    "task_name": task_name,
                    "n_tasks_seen": n_tasks_seen,
                    "eval_task_idx": eval_idx,
                    "eval_task_name": eval_task,
                    "n_train_task": len(train_images),
                    "n_test": len(test_labels),
                    "memory_n_embeddings": total_embeddings,
                    "memory_mb": round(memory_mb, 2),
                    "update_time_s": round(update_time, 2),
                    "inference_time_s": round(inference_time, 2),
                    **metrics,
                }

                save_result(result)

                if DEBUG:
                    print(f"      eval {eval_short}: AUROC={metrics['img_auroc']:.3f} "
                          f"(infer: {inference_time:.1f}s)")
                else:
                    print(f"    eval {eval_short}: AUROC={metrics['img_auroc']:.3f}", flush=True)

            except Exception as e:
                print(f"    eval {eval_short}: ERROR - {e}")
                log_error(f"seed={seed} mode={memory_mode} task={task_idx} eval={eval_idx}", e)

        cuda_cleanup(f"after task {task_idx}")

    # --- SUMMARY ---
    print_summary(seed, memory_mode)


def print_summary(seed: int, memory_mode: str) -> None:
    """Print summary for one seed/mode combination."""
    T = len(ALL_TASKS)
    acc_matrix = {}
    if CSV_PATH.exists():
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                if int(row["seed"]) != seed or row["memory_mode"] != memory_mode:
                    continue
                key = (int(row["task_idx"]), int(row["eval_task_idx"]))
                acc_matrix[key] = float(row["img_auroc"])

    final_aurocs = [acc_matrix.get((T - 1, j), 0) for j in range(T)]
    avg_acc = np.mean(final_aurocs) if final_aurocs else 0

    bwt_values = []
    for j in range(T - 1):
        acc_T_j = acc_matrix.get((T - 1, j), 0)
        acc_j_j = acc_matrix.get((j, j), 0)
        bwt_values.append(acc_T_j - acc_j_j)
    bwt = np.mean(bwt_values) if bwt_values else 0

    print(f"\n  --- Summary seed={seed} mode={memory_mode} ---")
    print(f"  Avg Accuracy (final): {avg_acc:.4f}")
    print(f"  BWT (forgetting):     {bwt:.4f}")
    print(f"  Per-task final AUROC:")
    for j, task in enumerate(ALL_TASKS):
        short = task.split("/")[-1]
        auroc_j_j = acc_matrix.get((j, j), 0)
        auroc_T_j = acc_matrix.get((T - 1, j), 0)
        delta = auroc_T_j - auroc_j_j
        marker = "  " if abs(delta) < 0.02 else ("--" if delta < 0 else "++")
        print(f"    {short:15s}: first={auroc_j_j:.3f}  final={auroc_T_j:.3f}  delta={delta:+.3f} {marker}")


def main():
    print("=" * 70)
    print("C2_001: Continual AD - Forgetting Baseline (KNN Naive)")
    print(f"  Tasks: {len(ALL_TASKS)}")
    print(f"  N per task: {N_PER_TASK}")
    print(f"  KNN k: {KNN_K}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Memory modes: {MEMORY_MODES}")
    print(f"  Debug: {DEBUG}")
    print("=" * 70)

    completed = get_completed()
    total_evals = sum(i + 1 for i in range(len(ALL_TASKS))) * len(SEEDS) * len(MEMORY_MODES)
    print(f"[INFO] Completed: {len(completed)}/{total_evals} evaluations")

    # Load DINOv3
    extractor = DINOv3Extractor(DEVICE)

    # Pre-extract ALL test features
    print("\n[STEP 1] Pre-extracting test features for all tasks...")
    test_data = {}
    for task_name in ALL_TASKS:
        print(f"  Loading test data: {task_name}...", end=" ", flush=True)
        _, test_images, test_labels = load_dataset(task_name)
        test_features = extractor.extract_batch(test_images)
        test_data[task_name] = (test_features, np.array(test_labels))
        print(f"OK ({len(test_images)} images, {np.sum(test_labels)} anomalies)")

    cuda_cleanup("after test feature extraction")

    # Run for each seed x memory_mode
    for seed in SEEDS:
        for memory_mode in MEMORY_MODES:
            run_continual_protocol(seed, memory_mode, extractor, test_data, completed)

    print("\n" + "=" * 70)
    print("C2_001 COMPLETE")
    print(f"Results: {CSV_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
