"""
Layer Ablation: verify memory pollution persists across feature layers.
Tests layer 12 (-12) and layer 21 (-3) vs default layer 18 (-6).
MVTec AD, 1 seed, 18 MB budget, coreset compression.

Output: output/continual/layer_ablation.csv
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "12")

import sys
import csv
import time
import gc
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from continual_utils import (
    DINOv3Extractor, load_dataset_with_masks,
    knn_score_images, compute_metrics,
    RESOLUTION, KNN_K, FEATURE_DIM,
)

# Settings
SEED = 0
BUDGET_MB = 18.0
N_PER_TASK = 10
LAYERS_TO_TEST = [-12, -6, -3]  # layers 12, 18, 21
TASKS = [
    "mvtec_AD/bottle", "mvtec_AD/cable", "mvtec_AD/capsule",
    "mvtec_AD/carpet", "mvtec_AD/grid", "mvtec_AD/hazelnut",
    "mvtec_AD/leather", "mvtec_AD/metal_nut", "mvtec_AD/pill",
    "mvtec_AD/screw", "mvtec_AD/tile", "mvtec_AD/toothbrush",
    "mvtec_AD/transistor", "mvtec_AD/wood", "mvtec_AD/zipper",
]
BUDGET_OPTIONS = [25, 50, 75, 100, 150, 200, 300, 500]
OUTPUT_DIR = Path("output/continual")
CSV_PATH = OUTPUT_DIR / "layer_ablation.csv"
COLUMNS = [
    "timestamp", "layer_idx", "layer_num", "seed", "task",
    "allocator", "budget_per_product", "n_embeddings",
    "img_auroc", "wnr",
]

from sklearn.decomposition import PCA

def kcenter_greedy(data: np.ndarray, budget: int, seed: int) -> np.ndarray:
    n = len(data)
    if budget >= n:
        return data
    pca_dim = min(64, data.shape[1], n)
    pca = PCA(n_components=pca_dim, random_state=seed)
    data_pca = pca.fit_transform(data)
    rng = np.random.RandomState(seed)
    first_idx = rng.randint(0, n)
    selected = [first_idx]
    min_dists = np.full(n, np.inf)
    diffs = data_pca - data_pca[first_idx]
    min_dists = np.minimum(min_dists, np.einsum('ij,ij->i', diffs, diffs))
    for _ in range(budget - 1):
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        diffs = data_pca - data_pca[next_idx]
        min_dists = np.minimum(min_dists, np.einsum('ij,ij->i', diffs, diffs))
    return data[np.array(selected)]


def mb_to_embs(mb: float, dim: int = FEATURE_DIM) -> int:
    return int(mb * 1e6 / (dim * 4))


def snap_to_options(budget: int) -> int:
    return min(BUDGET_OPTIONS, key=lambda x: abs(x - budget))


def compute_wnr(full_bank: np.ndarray, product_labels: np.ndarray,
                test_patches: np.ndarray, true_idx: int, k: int = KNN_K) -> float:
    k_actual = min(k, len(full_bank))
    nn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean")
    nn.fit(full_bank)
    _, indices = nn.kneighbors(test_patches)
    neighbor_labels = product_labels[indices]
    wrong = (neighbor_labels != true_idx)
    return float(wrong.mean())


class LayerExtractor:
    """DINOv3 extractor with configurable layer."""
    def __init__(self, layer_idx: int, device: str = "cuda"):
        from transformers import AutoModel
        self.device = device
        self.layer_idx = layer_idx
        model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        revision = os.environ.get("DINO_REVISION")
        print(f"[FEAT] Loading DINOv3, layer_idx={layer_idx}...")
        if revision:
            self.model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()
        else:
            self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.patch_size = 16
        self.grid_size = RESOLUTION // self.patch_size
        self.num_patches = self.grid_size ** 2
        self.num_register = getattr(self.model.config, "num_register_tokens", 4)
        self.feature_dim = 1024
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((RESOLUTION, RESOLUTION)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract_batch(self, image_paths: list[str], batch_size: int = 16) -> np.ndarray:
        from PIL import Image
        all_features = []
        start_idx = 1 + self.num_register
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = [self.transform(Image.open(p).convert("RGB")) for p in batch_paths]
            batch = torch.stack(images).to(self.device)
            with torch.autocast(device_type='cuda', enabled=True):
                outputs = self.model(batch, output_hidden_states=True)
            features = outputs.hidden_states[self.layer_idx][:, start_idx:start_idx + self.num_patches, :]
            all_features.append(features.float().cpu().numpy())
            del batch, outputs, features
        return np.concatenate(all_features, axis=0)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    # Load model once, reuse across layers
    extractor = None
    budget_per_product = snap_to_options(mb_to_embs(BUDGET_MB) // len(TASKS))
    print(f"Budget per product: {budget_per_product} embeddings")

    for layer_idx in LAYERS_TO_TEST:
        layer_num = 24 + layer_idx  # -6 -> 18, -3 -> 21, -12 -> 12
        print(f"\n{'='*60}")
        print(f"LAYER {layer_num} (idx={layer_idx})")
        print(f"{'='*60}")

        # Extract features with this layer
        del extractor
        gc.collect()
        torch.cuda.empty_cache()
        extractor = LayerExtractor(layer_idx)

        # Extract train and test features
        train_feats: dict[str, np.ndarray] = {}
        test_feats: dict[str, np.ndarray] = {}
        test_labels_dict: dict[str, np.ndarray] = {}

        for task in TASKS:
            short = task.split("/")[-1]
            print(f"  Extracting {short}...", end=" ", flush=True)

            # Train
            train_paths, test_info = load_dataset_with_masks(task, N_PER_TASK, SEED)
            train_features = extractor.extract_batch(train_paths)
            train_feats[task] = train_features

            # Test
            test_paths = [e["path"] for e in test_info]
            test_labels = np.array([e["label"] for e in test_info])
            test_features = extractor.extract_batch(test_paths)
            test_feats[task] = test_features
            test_labels_dict[task] = test_labels
            print(f"OK ({len(train_paths)} train, {len(test_paths)} test)")

        torch.cuda.empty_cache()

        # Build shared bank with uniform allocation
        bank_parts = []
        label_parts = []
        for idx, task in enumerate(TASKS):
            flat = train_feats[task].reshape(-1, train_feats[task].shape[-1])
            compressed = kcenter_greedy(flat, budget_per_product, SEED)
            bank_parts.append(compressed)
            label_parts.append(np.full(len(compressed), idx, dtype=np.int32))

        full_bank = np.concatenate(bank_parts)
        product_labels = np.concatenate(label_parts)
        print(f"\n  Shared bank: {len(full_bank)} embeddings, {full_bank.nbytes/1e6:.1f} MB")

        # Evaluate: AUROC + WNR per product
        for idx, task in enumerate(TASKS):
            short = task.split("/")[-1]
            test_features = test_feats[task]
            labels = test_labels_dict[task]

            # Image AUROC
            scores = knn_score_images(full_bank, test_features, k=KNN_K)
            metrics = compute_metrics(labels, scores)

            # WNR (first 20 test images)
            wnr_vals = []
            for img_idx in range(min(len(test_features), 20)):
                patches = test_features[img_idx]
                wnr = compute_wnr(full_bank, product_labels, patches, idx)
                wnr_vals.append(wnr)
            mean_wnr = float(np.mean(wnr_vals))

            result = {
                "timestamp": datetime.now().isoformat(),
                "layer_idx": layer_idx,
                "layer_num": layer_num,
                "seed": SEED,
                "task": short,
                "allocator": "uniform",
                "budget_per_product": budget_per_product,
                "n_embeddings": len(full_bank),
                "img_auroc": round(metrics["img_auroc"], 6),
                "wnr": round(mean_wnr, 4),
            }

            # Append to CSV
            file_exists = CSV_PATH.exists()
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
                if not file_exists:
                    writer.writeheader()
                writer.writerow(result)

            print(f"  {short:>15s}  AUROC={metrics['img_auroc']:.4f}  WNR={mean_wnr:.3f}")

        # Cleanup
        del train_feats, test_feats, test_labels_dict
        gc.collect()

    print(f"\n[DONE] Results saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
