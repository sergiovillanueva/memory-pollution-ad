"""
Shared utilities for multi-product anomaly detection experiments.

Provides feature extraction (DINOv3), KNN scoring, memory compression,
pixel-level metrics, budget allocation, and prototype-based routing.
All experiment scripts import from this module.
"""
from __future__ import annotations

import csv
import gc
import os
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.neighbors import NearestNeighbors

# =============================================================================
# CONSTANTS
# =============================================================================

RESOLUTION = 448
LAYER_IDX = -6
BATCH_SIZE = 16
KNN_K = 5
N_PER_TASK = 10
DATA_ROOT = Path("data").resolve()
OUTPUT_DIR = Path("output/continual")
FEATURE_DIM = 1024

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

ALL_TASKS_BTAD = [
    "btad/01", "btad/02", "btad/03",
]


# =============================================================================
# DINOv3 FEATURE EXTRACTOR
# =============================================================================

class DINOv3Extractor:
    def __init__(self, device: str = "cuda"):
        from transformers import AutoModel
        self.device = device
        model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        revision = os.environ.get("DINO_REVISION")
        print(f"[FEAT] Loading DINOv3 on {device}...")
        if revision:
            self.model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()
        else:
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
    def extract_batch(self, image_paths: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
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
# DATASET LOADING (with mask support)
# =============================================================================

def _load_mvtec_dataset(data_root: Path) -> tuple[list[str], list[dict]]:
    """Load MVTec/BTAD-style dataset with mask paths.

    Returns:
        train_paths: list of train image paths
        test_info: list of dicts {path, label, mask_path}
    """
    train_dir = data_root / "train" / "good"
    if not train_dir.exists():
        train_dir = data_root / "train"

    train_paths = sorted(
        [str(p) for p in train_dir.glob("*.*")
         if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')]
    )

    test_dir = data_root / "test"
    gt_dir = data_root / "ground_truth"
    test_info: list[dict] = []

    for cat_dir in sorted(test_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        is_normal = cat_dir.name == "good"
        label = 0 if is_normal else 1
        for img_path in sorted(cat_dir.glob("*.*")):
            if img_path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.bmp'):
                continue
            entry: dict = {"path": str(img_path), "label": label, "mask_path": None}
            if not is_normal and gt_dir.exists():
                mask_dir = gt_dir / cat_dir.name
                if mask_dir.exists():
                    stem = img_path.stem
                    for ext in (".png", "_mask.png"):
                        mp = mask_dir / f"{stem}{ext}"
                        if mp.exists():
                            entry["mask_path"] = str(mp)
                            break
            test_info.append(entry)

    return train_paths, test_info


def _load_visa_dataset(data_root: Path) -> tuple[list[str], list[dict]]:
    """Load VisA dataset using split_csv/1cls.csv.

    Returns:
        train_paths: list of train image paths
        test_info: list of dicts {path, label, mask_path}
    """
    import csv as csv_mod
    category = data_root.name
    visa_root = data_root.parent
    split_csv = visa_root / "split_csv" / "1cls.csv"

    if not split_csv.exists():
        raise FileNotFoundError(f"VisA split CSV not found: {split_csv}")

    train_paths: list[str] = []
    test_info: list[dict] = []

    with open(split_csv, "r") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            if row["object"] != category:
                continue
            img_path = visa_root / row["image"]
            if not img_path.exists():
                continue
            if row["split"] == "train":
                train_paths.append(str(img_path))
            elif row["split"] == "test":
                label = 0 if row["label"] == "normal" else 1
                mask_path = None
                if row.get("mask") and row["mask"].strip():
                    mf = visa_root / row["mask"]
                    if mf.exists():
                        mask_path = str(mf)
                test_info.append({"path": str(img_path), "label": label, "mask_path": mask_path})

    return sorted(train_paths), test_info


def load_dataset_with_masks(
    dataset_name: str,
    train_limit: int | None = None,
    seed: int = 0,
) -> tuple[list[str], list[dict]]:
    """Load dataset with mask info. Routes by prefix.

    Args:
        dataset_name: e.g. "mvtec_AD/bottle", "VisA/pcb1", "btad/01"
        train_limit: max train images (subsampled with seed)
        seed: random seed for subsampling

    Returns:
        train_paths: list of train image paths
        test_info: list of dicts {path, label, mask_path}
    """
    dataset_path = DATA_ROOT / dataset_name

    if dataset_name.startswith("VisA/"):
        train_paths, test_info = _load_visa_dataset(dataset_path)
    else:
        # MVTec AD, BTAD, and any other MVTec-style dataset
        train_paths, test_info = _load_mvtec_dataset(dataset_path)

    if train_limit and len(train_paths) > train_limit:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(train_paths), train_limit, replace=False)
        train_paths = [train_paths[i] for i in sorted(indices)]

    return train_paths, test_info


def load_standard_dataset(dataset_path: Path) -> tuple[list[str], list[str], list[int]]:
    """Load MVTec-style dataset (train/good, test/good+defect). Legacy compatibility."""
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
    """Load train/test images and labels. Legacy compatibility."""
    dataset_path = DATA_ROOT / dataset_name
    train_images, test_images, test_labels = load_standard_dataset(dataset_path)

    if train_limit and len(train_images) > train_limit:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(train_images), train_limit, replace=False)
        train_images = [train_images[i] for i in sorted(indices)]

    return train_images, test_images, np.array(test_labels)


def load_mask(entry: dict) -> np.ndarray | None:
    """Load ground truth mask from test_info entry.

    Returns:
        Grayscale mask as np.ndarray (H,W) or None for normal images.
    """
    if not entry.get("mask_path"):
        return None
    return np.array(Image.open(entry["mask_path"]).convert("L"))


# =============================================================================
# METRICS (image-level)
# =============================================================================

def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Compute image-level anomaly detection metrics."""
    if len(np.unique(labels)) < 2:
        return {"img_auroc": 0.0, "img_aupr": 0.0, "img_f1max": 0.0, "tpr_fpr1": 0.0, "tpr_fpr5": 0.0}

    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)

    thresholds = np.percentile(scores, np.arange(0, 100, 1))
    f1_max = max(
        f1_score(labels, (scores >= t).astype(int), zero_division=0)
        for t in thresholds
    )

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
    nn_index: NearestNeighbors | None = None,
    k: int = KNN_K,
) -> np.ndarray:
    """Score test images using KNN distance to memory bank.

    Args:
        memory_bank: [M, D] all stored patch embeddings
        test_features: [N, P, D] test image features
        k: number of neighbors

    Returns:
        [N] image-level anomaly scores
    """
    n_samples, n_patches, feat_dim = test_features.shape
    flat_test = test_features.reshape(-1, feat_dim)

    k_actual = min(k, len(memory_bank))
    if nn_index is None:
        nn_index = NearestNeighbors(n_neighbors=k_actual, metric='euclidean', algorithm='auto')
        nn_index.fit(memory_bank)

    distances, _ = nn_index.kneighbors(flat_test)
    patch_scores = distances.mean(axis=1)

    patch_scores = patch_scores.reshape(n_samples, n_patches)
    image_scores = np.percentile(patch_scores, 95, axis=1)

    return image_scores


def knn_score_maps(
    memory_bank: np.ndarray,
    test_features: np.ndarray,
    nn_index: NearestNeighbors | None = None,
    k: int = KNN_K,
    grid_size: int = 28,
) -> tuple[np.ndarray, np.ndarray]:
    """Score test images and return both image scores and spatial score maps.

    Args:
        memory_bank: [M, D] all stored patch embeddings
        test_features: [N, P, D] test image features (P = grid_size^2)
        k: number of neighbors
        grid_size: spatial grid size (28 for full, 14 for avg2x2, 7 for avg4x4)

    Returns:
        image_scores: [N] image-level anomaly scores (percentile 95)
        score_maps: [N, grid_size, grid_size] spatial anomaly maps (float32)
    """
    n_samples, n_patches, feat_dim = test_features.shape
    assert n_patches == grid_size * grid_size, \
        f"Mismatch: n_patches={n_patches} but grid_size={grid_size} -> {grid_size**2}"

    flat_test = test_features.reshape(-1, feat_dim)

    k_actual = min(k, len(memory_bank))
    if nn_index is None:
        nn_index = NearestNeighbors(n_neighbors=k_actual, metric='euclidean', algorithm='auto')
        nn_index.fit(memory_bank)

    distances, _ = nn_index.kneighbors(flat_test)
    patch_scores = distances.mean(axis=1)  # [N*P]

    patch_scores = patch_scores.reshape(n_samples, n_patches)
    image_scores = np.percentile(patch_scores, 95, axis=1)  # [N]
    score_maps = patch_scores.reshape(n_samples, grid_size, grid_size).astype(np.float32)

    return image_scores, score_maps


def knn_score_single_image(
    memory_bank: np.ndarray,
    test_features_single: np.ndarray,
    nn_index: NearestNeighbors | None = None,
    k: int = KNN_K,
) -> tuple[float, float]:
    """Score ONE image, return (score, query_time_ms).

    Args:
        memory_bank: [M, D] stored embeddings
        test_features_single: [P, D] one image's patches
        nn_index: pre-fitted NearestNeighbors (optional, avoids refit)
        k: number of neighbors

    Returns:
        (anomaly_score, query_time_ms)
    """
    t0 = time.perf_counter()

    k_actual = min(k, len(memory_bank))
    if nn_index is None:
        nn_index = NearestNeighbors(n_neighbors=k_actual, metric='euclidean', algorithm='auto')
        nn_index.fit(memory_bank)

    distances, _ = nn_index.kneighbors(test_features_single)
    patch_scores = distances.mean(axis=1)
    score = float(np.percentile(patch_scores, 95))

    query_ms = (time.perf_counter() - t0) * 1000
    return score, query_ms


# =============================================================================
# PIXEL-LEVEL METRICS
# =============================================================================

def compute_pixel_metrics(
    score_maps: np.ndarray,
    test_info: list[dict],
    eval_shape: tuple[int, int] = (RESOLUTION, RESOLUTION),
    smooth_sigma: float = 0.0,
) -> dict:
    """Compute pixel-level anomaly detection metrics.

    Upsamples score_maps from GxG to eval_shape using bilinear interpolation.
    GT masks are upsampled with nearest-neighbor (order=0) to preserve binary values.
    Optional Gaussian smoothing after upsample (default: off).

    Args:
        score_maps: [N, G, G] spatial anomaly score maps
        test_info: list of dicts with {path, label, mask_path}
        eval_shape: target resolution for evaluation (default 448x448)
        smooth_sigma: Gaussian smoothing sigma after upsample (0 = no smoothing)

    Returns:
        dict with pix_auroc, pix_aupr, spro, auspro (None if not computable)
    """
    from scipy.ndimage import zoom, gaussian_filter

    result = {"pix_auroc": None, "pix_aupr": None, "spro": None, "auspro": None}

    # Check if any masks exist
    has_pixel_gt = any(e.get("mask_path") is not None for e in test_info)
    if not has_pixel_gt:
        return result

    all_pix_scores: list[np.ndarray] = []
    all_pix_labels: list[np.ndarray] = []
    maps_2d: list[np.ndarray] = []
    masks_2d: list[np.ndarray] = []
    labels_pro: list[int] = []

    for i, entry in enumerate(test_info):
        am = score_maps[i].astype(np.float64)
        mask = load_mask(entry)

        # Ground truth mask
        if mask is not None:
            gt = (mask > 0).astype(np.uint8)
            if gt.shape != eval_shape:
                zy = eval_shape[0] / gt.shape[0]
                zx = eval_shape[1] / gt.shape[1]
                gt = (zoom(gt.astype(float), (zy, zx), order=0) > 0.5).astype(np.uint8)
        else:
            gt = np.zeros(eval_shape, dtype=np.uint8)

        # Upsample anomaly map to eval_shape (bilinear)
        if am.shape != eval_shape:
            zy = eval_shape[0] / am.shape[0]
            zx = eval_shape[1] / am.shape[1]
            am = zoom(am, (zy, zx), order=1)

        # Optional Gaussian smoothing
        if smooth_sigma > 0:
            am = gaussian_filter(am, sigma=smooth_sigma)

        all_pix_scores.append(am.ravel())
        all_pix_labels.append(gt.ravel())
        maps_2d.append(am)
        masks_2d.append(gt)
        labels_pro.append(int(entry.get("label", 0)))

    # Pixel AUROC / AUPR
    if all_pix_scores:
        pix_s = np.concatenate(all_pix_scores)
        pix_l = np.concatenate(all_pix_labels)
        if len(np.unique(pix_l)) > 1:
            result["pix_auroc"] = round(float(roc_auc_score(pix_l, pix_s)), 6)
            result["pix_aupr"] = round(float(average_precision_score(pix_l, pix_s)), 6)

    # SPRO / AuSPRO
    if maps_2d:
        spro, auspro = compute_auspro(maps_2d, masks_2d, labels_pro)
        if spro is not None:
            result["spro"] = round(float(spro), 6)
        if auspro is not None:
            result["auspro"] = round(float(auspro), 6)

    return result


def compute_auspro(
    anomaly_maps: list[np.ndarray],
    gt_masks: list[np.ndarray],
    image_labels: list[int],
    alpha: float = 0.3,
) -> tuple[float | None, float | None]:
    """Compute SPRO and AuSPRO (Area under SPRO curve up to FPR=alpha).

    Uses connected components for per-region overlap.

    Args:
        anomaly_maps: list of [H,W] anomaly score maps
        gt_masks: list of [H,W] binary GT masks
        image_labels: list of int (0=normal, 1=anomaly)
        alpha: max FPR for integration

    Returns:
        (spro_at_alpha, auspro) or (None, None) if not computable
    """
    from scipy.ndimage import label as cc_label

    labels_arr = np.array(image_labels)
    normal_idx = np.where(labels_arr == 0)[0]
    abnormal_idx = np.where(labels_arr == 1)[0]
    if len(normal_idx) == 0 or len(abnormal_idx) == 0:
        return None, None

    # Collect all normal pixels for FPR thresholds
    all_normal_pixels: list[np.ndarray] = []
    for idx in normal_idx:
        if idx < len(anomaly_maps):
            all_normal_pixels.append(anomaly_maps[idx].ravel())
    if not all_normal_pixels:
        return None, None
    all_normal_pixels_flat = np.concatenate(all_normal_pixels)

    thresholds = np.linspace(
        np.max(all_normal_pixels_flat), np.min(all_normal_pixels_flat), 100
    )
    total_normal = len(all_normal_pixels_flat)
    fprs: list[float] = []
    pros: list[float] = []

    for thr in thresholds:
        fpr = float((all_normal_pixels_flat >= thr).sum() / total_normal)
        fprs.append(fpr)

        pro_vals: list[float] = []
        for idx in abnormal_idx:
            if idx >= len(anomaly_maps):
                continue
            amap = anomaly_maps[idx]
            gtm = gt_masks[idx]
            labeled, num_cc = cc_label(gtm)
            for cc_id in range(1, num_cc + 1):
                cc_mask = labeled == cc_id
                cc_size = cc_mask.sum()
                if cc_size > 0:
                    pro_vals.append(float(((amap >= thr) & cc_mask).sum() / cc_size))
        pros.append(np.mean(pro_vals) if pro_vals else 0.0)

    fprs_arr = np.array(fprs)
    pros_arr = np.array(pros)
    valid = fprs_arr <= alpha
    if not np.any(valid):
        return None, None

    order = np.argsort(fprs_arr[valid])
    xf = fprs_arr[valid][order]
    yf = pros_arr[valid][order]
    auspro = float(np.trapezoid(yf, xf) / alpha)
    return float(yf[-1]), auspro


# =============================================================================
# MEMORY BUDGET HELPERS
# =============================================================================

def memory_mb(n_embeddings: int, dim: int = FEATURE_DIM, bytes_per_val: int = 4) -> float:
    """Compute memory in MB for a given number of embeddings."""
    return n_embeddings * dim * bytes_per_val / 1e6


def complexity_score(
    train_features: np.ndarray,
    k: int = KNN_K,
) -> float:
    """Compute normal-only complexity score for a product.

    Uses percentile 95 of intra-normal 5-NN distances as proxy for product difficulty.
    Higher = more complex normal distribution = needs more memory.

    Args:
        train_features: [N, P, D] features from normal training images
        k: number of neighbors for density estimation

    Returns:
        complexity score (float)
    """
    flat = train_features.reshape(-1, train_features.shape[-1])
    k_actual = min(k + 1, len(flat))  # +1 because self is included
    nn = NearestNeighbors(n_neighbors=k_actual, metric='euclidean', algorithm='auto')
    nn.fit(flat)
    distances, _ = nn.kneighbors(flat)
    # Skip self-distance (column 0), take mean of k neighbors
    knn_dists = distances[:, 1:].mean(axis=1)
    return float(np.percentile(knn_dists, 95))


# =============================================================================
# CSV / LOGGING HELPERS
# =============================================================================

def get_completed(csv_path: Path, key_fields: list[str]) -> set[tuple]:
    """Load completed experiment keys from CSV."""
    completed = set()
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                key = tuple(row[k] for k in key_fields)
                completed.add(key)
    return completed


def save_result(csv_path: Path, columns: list[str], result: dict) -> None:
    """Append one result row to CSV."""
    file_exists = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def log_error(error_path: Path, msg: str, exc: Exception) -> None:
    """Log error with traceback."""
    error_path.parent.mkdir(parents=True, exist_ok=True)
    with open(error_path, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | {msg}\n")
        f.write(traceback.format_exc() + "\n")


def cuda_cleanup(reason: str = "", debug: bool = False) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if debug and reason:
        print(f"  [CLEANUP] {reason}")


# =============================================================================
# FEATURE CACHING
# =============================================================================

CACHE_DIR = Path("output/feature_cache")


def _cache_path(prefix: str, task_name: str, suffix: str = "") -> Path:
    """Build cache file path."""
    safe_name = task_name.replace("/", "_")
    return CACHE_DIR / f"{prefix}_{safe_name}{suffix}.npy"


def preload_test_data(
    extractor: DINOv3Extractor,
    tasks: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pre-extract test features for all tasks (with disk caching)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[STEP 1] Loading test features for all tasks...")
    test_data = {}
    for task_name in tasks:
        feat_path = _cache_path("test_feat", task_name)
        label_path = _cache_path("test_labels", task_name)

        if feat_path.exists() and label_path.exists():
            print(f"  {task_name}: cached", end=" ", flush=True)
            test_features = np.load(feat_path)
            test_labels = np.load(label_path)
        else:
            print(f"  {task_name}: extracting...", end=" ", flush=True)
            _, test_images, test_labels = load_dataset(task_name)
            test_labels = np.array(test_labels)
            test_features = extractor.extract_batch(test_images)
            np.save(feat_path, test_features)
            np.save(label_path, test_labels)

        test_data[task_name] = (test_features, test_labels)
        print(f"OK ({len(test_labels)} images, {np.sum(test_labels)} anomalies)")

    cuda_cleanup("after test feature loading")
    return test_data


def preload_test_data_with_masks(
    extractor: DINOv3Extractor,
    tasks: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray, list[dict]]]:
    """Pre-extract test features for all tasks AND load test_info with mask paths.

    Returns:
        dict[task_name -> (test_features [N,P,D], test_labels [N], test_info [list of dicts])]
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[STEP 1] Loading test features + mask info for all tasks...")
    test_data = {}

    for task_name in tasks:
        feat_path = _cache_path("test_feat", task_name)
        label_path = _cache_path("test_labels", task_name)

        # Load test_info (masks) -- always from disk, it's instant
        _, test_info = load_dataset_with_masks(task_name)
        test_labels = np.array([e["label"] for e in test_info])

        if feat_path.exists() and label_path.exists():
            print(f"  {task_name}: cached", end=" ", flush=True)
            test_features = np.load(feat_path)
            cached_labels = np.load(label_path)
            # Sanity check: cached labels should match test_info
            if len(cached_labels) != len(test_labels):
                print(f"  WARNING: cache mismatch for {task_name}, re-extracting...")
                test_paths = [e["path"] for e in test_info]
                test_features = extractor.extract_batch(test_paths)
                np.save(feat_path, test_features)
                np.save(label_path, test_labels)
            else:
                test_labels = cached_labels
        else:
            print(f"  {task_name}: extracting...", end=" ", flush=True)
            test_paths = [e["path"] for e in test_info]
            test_features = extractor.extract_batch(test_paths)
            np.save(feat_path, test_features)
            np.save(label_path, test_labels)

        test_data[task_name] = (test_features, test_labels, test_info)
        n_anom = int(np.sum(test_labels))
        n_masks = sum(1 for e in test_info if e.get("mask_path"))
        print(f"OK ({len(test_labels)} imgs, {n_anom} anom, {n_masks} masks)")

    cuda_cleanup("after test feature loading")
    return test_data


def extract_train_features(
    extractor: DINOv3Extractor,
    task_name: str,
    seed: int,
    n_per_task: int = N_PER_TASK,
) -> tuple[np.ndarray, int]:
    """Extract train features for one task (with disk caching). Returns (features [N,P,D], n_images)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path("train_feat", task_name, f"_s{seed}_n{n_per_task}")

    if cache_path.exists():
        train_features = np.load(cache_path)
        n_images = train_features.shape[0]
    else:
        train_images, _ = load_dataset_with_masks(task_name, n_per_task, seed)
        train_features = extractor.extract_batch(train_images)
        n_images = len(train_images)
        np.save(cache_path, train_features)

    return train_features, n_images


# =============================================================================
# ROUTING FUNCTIONS (Product-Conditioned Retrieval)
# =============================================================================

def compute_prototypes(
    train_feat_cache: dict[tuple[str, int], np.ndarray],
    tasks: list[str],
    seed: int,
) -> dict[str, np.ndarray]:
    """Compute mean-embedding prototype per product. Returns {task_name: prototype [D]}."""
    prototypes = {}
    for task_name in tasks:
        feat = train_feat_cache[(task_name, seed)]  # [N, P, D]
        flat = feat.reshape(-1, feat.shape[-1])  # [N*P, D]
        prototypes[task_name] = flat.mean(axis=0)  # [D]
    return prototypes


def route_image(
    test_embedding: np.ndarray,
    prototypes: dict[str, np.ndarray],
    top_m: int = 1,
) -> list[str]:
    """Route a test image to the top-M closest products by prototype distance.

    Args:
        test_embedding: mean patch embedding of the test image [D]
        prototypes: {task_name: prototype [D]}
        top_m: number of candidate products to return

    Returns:
        list of top-M task_names, sorted by distance (closest first)
    """
    tasks = list(prototypes.keys())
    proto_matrix = np.stack([prototypes[t] for t in tasks])  # [N_tasks, D]
    dists = np.linalg.norm(proto_matrix - test_embedding, axis=1)
    top_indices = np.argsort(dists)[:top_m]
    return [tasks[i] for i in top_indices]


def route_batch(
    test_embeddings: np.ndarray,
    prototypes: dict[str, np.ndarray],
    top_m: int = 1,
) -> list[list[str]]:
    """Route a batch of test images. test_embeddings: [N, D] mean patch embeddings."""
    tasks = list(prototypes.keys())
    proto_matrix = np.stack([prototypes[t] for t in tasks])  # [N_tasks, D]
    # Compute all distances: [N_test, N_tasks]
    dists = np.linalg.norm(
        test_embeddings[:, np.newaxis, :] - proto_matrix[np.newaxis, :, :],
        axis=2,
    )
    top_indices = np.argsort(dists, axis=1)[:, :top_m]
    return [[tasks[i] for i in row] for row in top_indices]


def knn_score_images_gated(
    sub_banks: dict[str, np.ndarray],
    test_features: np.ndarray,
    prototypes: dict[str, np.ndarray],
    top_m: int = 1,
    k: int = KNN_K,
    known_task: str | None = None,
) -> np.ndarray:
    """KNN scoring with routing (gated retrieval).

    If known_task is provided (Known-ID mode), uses that sub-bank directly.
    Otherwise routes each test image to top-M products and searches their union.

    Args:
        sub_banks: {task_name: embeddings [N_t, D]} per-product compressed banks
        test_features: [N_test, P, D] test patch features
        prototypes: {task_name: prototype [D]}
        top_m: number of candidate products for routing
        k: KNN neighbors
        known_task: if provided, skip routing and use this task's sub-bank

    Returns:
        image_scores: [N_test] anomaly scores
    """
    n_test = test_features.shape[0]
    flat_test = test_features.reshape(n_test, -1, test_features.shape[-1])  # [N, P, D]

    # Compute mean embedding per test image for routing
    test_means = flat_test.mean(axis=1)  # [N, D]

    image_scores = np.zeros(n_test, dtype=np.float32)

    for i in range(n_test):
        if known_task is not None:
            candidates = [known_task]
        else:
            candidates = route_image(test_means[i], prototypes, top_m)

        # Union of sub-banks for candidates
        bank_parts = [sub_banks[c] for c in candidates if c in sub_banks]
        if not bank_parts:
            image_scores[i] = 0.0
            continue
        local_bank = np.concatenate(bank_parts, axis=0)

        # KNN on local bank
        patches = flat_test[i]  # [P, D]
        k_actual = min(k, len(local_bank))
        nn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean")
        nn.fit(local_bank)
        dists, _ = nn.kneighbors(patches)
        patch_scores = dists.mean(axis=1)
        image_scores[i] = np.percentile(patch_scores, 95)

    return image_scores


def knn_score_maps_gated(
    sub_banks: dict[str, np.ndarray],
    test_features: np.ndarray,
    prototypes: dict[str, np.ndarray],
    top_m: int = 1,
    k: int = KNN_K,
    grid_size: int = 28,
    known_task: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """KNN scoring with routing, returning score maps for pixel-level evaluation.

    Returns:
        image_scores: [N_test]
        score_maps: [N_test, grid_size, grid_size]
    """
    n_test = test_features.shape[0]
    flat_test = test_features.reshape(n_test, -1, test_features.shape[-1])
    test_means = flat_test.mean(axis=1)

    image_scores = np.zeros(n_test, dtype=np.float32)
    score_maps = np.zeros((n_test, grid_size, grid_size), dtype=np.float32)

    for i in range(n_test):
        if known_task is not None:
            candidates = [known_task]
        else:
            candidates = route_image(test_means[i], prototypes, top_m)

        bank_parts = [sub_banks[c] for c in candidates if c in sub_banks]
        if not bank_parts:
            continue
        local_bank = np.concatenate(bank_parts, axis=0)

        patches = flat_test[i]  # [P, D]
        k_actual = min(k, len(local_bank))
        nn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean")
        nn.fit(local_bank)
        dists, _ = nn.kneighbors(patches)
        patch_scores = dists.mean(axis=1)

        image_scores[i] = np.percentile(patch_scores, 95)
        score_maps[i] = patch_scores.reshape(grid_size, grid_size)

    return image_scores, score_maps


def compute_overlap_scores(
    train_feat_cache: dict[tuple[str, int], np.ndarray],
    tasks: list[str],
    seed: int,
    percentile: int = 5,
) -> dict[str, float]:
    """Compute inter-product overlap score per product.

    For each product p, measures how close its embeddings are to centroids of
    OTHER products. High overlap = embeddings from p are in "crowded" zones.

    No data leakage: uses only normal training embeddings.

    Args:
        train_feat_cache: {(task_name, seed): features [N, P, D]}
        tasks: list of task names
        seed: random seed
        percentile: lower percentile of cross-product distances (default 5)

    Returns:
        {task_name: overlap_score} (higher = more overlap with neighbors)
    """
    # Compute centroids for each product
    centroids = {}
    for task_name in tasks:
        feat = train_feat_cache[(task_name, seed)]
        flat = feat.reshape(-1, feat.shape[-1])
        centroids[task_name] = flat.mean(axis=0)  # [D]

    centroid_matrix = np.stack([centroids[t] for t in tasks])  # [N_tasks, D]

    overlap_scores = {}
    for idx, task_name in enumerate(tasks):
        feat = train_feat_cache[(task_name, seed)]
        flat = feat.reshape(-1, feat.shape[-1])  # [N*P, D]

        # Distances to ALL other centroids (exclude own)
        other_centroids = np.delete(centroid_matrix, idx, axis=0)  # [N_tasks-1, D]

        # Min distance from each embedding to nearest OTHER centroid
        # Shape: [N*P, N_tasks-1]
        dists_to_others = np.linalg.norm(
            flat[:, np.newaxis, :] - other_centroids[np.newaxis, :, :],
            axis=2,
        )
        min_dists = dists_to_others.min(axis=1)  # [N*P] - dist to closest alien centroid

        # Low percentile = some embeddings are VERY close to other products
        overlap_scores[task_name] = float(np.percentile(min_dists, percentile))

    return overlap_scores


def router_accuracy(
    test_features: np.ndarray,
    test_true_task: str,
    prototypes: dict[str, np.ndarray],
    top_m: int = 1,
) -> float:
    """Compute routing accuracy: fraction of test images correctly routed.

    Args:
        test_features: [N, P, D] test patch features
        test_true_task: the correct task name for these test images
        prototypes: {task_name: prototype [D]}
        top_m: number of candidates (correct if true_task is in top-M)

    Returns:
        accuracy in [0, 1]
    """
    n_test = test_features.shape[0]
    test_means = test_features.reshape(n_test, -1, test_features.shape[-1]).mean(axis=1)
    routes = route_batch(test_means, prototypes, top_m)
    correct = sum(1 for r in routes if test_true_task in r)
    return correct / n_test
