"""
Backbone Ablation: DINOv2-ViT-B/14 vs DINOv3-ViT-L/16.
MVTec AD, 1 seed, 300 embs/product (uniform), layer -6.
Re-extracts features with DINOv2-ViT-B/14.

Output: output/continual/backbone_ablation.csv
"""
from __future__ import annotations
import os, sys, csv, gc, time
os.environ.setdefault("OMP_NUM_THREADS", "12")

import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from continual_utils import (
    load_dataset_with_masks, compute_metrics, KNN_K, FEATURE_DIM,
)

SEED = 0
N_PER_TASK = 10
BUDGET_PER_PRODUCT = 300
RESOLUTION = 448
LAYER_IDX = -6
TASKS = [
    "mvtec_AD/bottle", "mvtec_AD/cable", "mvtec_AD/capsule",
    "mvtec_AD/carpet", "mvtec_AD/grid", "mvtec_AD/hazelnut",
    "mvtec_AD/leather", "mvtec_AD/metal_nut", "mvtec_AD/pill",
    "mvtec_AD/screw", "mvtec_AD/tile", "mvtec_AD/toothbrush",
    "mvtec_AD/transistor", "mvtec_AD/wood", "mvtec_AD/zipper",
]
OUTPUT_DIR = Path("output/continual")
CSV_PATH = OUTPUT_DIR / "backbone_ablation.csv"
COLUMNS = ["timestamp", "backbone", "feature_dim", "num_patches", "seed", "task",
           "budget_per_product", "img_auroc", "wnr"]


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


class DINOv2Extractor:
    """DINOv2-ViT-B/14 feature extractor."""
    def __init__(self, device: str = "cuda"):
        self.device = device
        model_name = "facebook/dinov2-base"
        print(f"[FEAT] Loading DINOv2-ViT-B/14 on {device}...")
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.patch_size = 14
        self.grid_size = RESOLUTION // self.patch_size  # 32
        self.num_patches = self.grid_size ** 2  # 1024
        self.num_register = 0  # DINOv2 base has no register tokens
        self.feature_dim = 768  # ViT-B
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((RESOLUTION, RESOLUTION)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"[FEAT] Grid: {self.grid_size}x{self.grid_size}, patches: {self.num_patches}, dim: {self.feature_dim}")

    @torch.no_grad()
    def extract_batch(self, image_paths: list[str], batch_size: int = 16) -> np.ndarray:
        all_features = []
        start_idx = 1 + self.num_register  # skip CLS token
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            images = [self.transform(Image.open(p).convert("RGB")) for p in batch_paths]
            batch = torch.stack(images).to(self.device)
            with torch.autocast(device_type='cuda', enabled=True):
                outputs = self.model(batch, output_hidden_states=True)
            features = outputs.hidden_states[LAYER_IDX][:, start_idx:start_idx + self.num_patches, :]
            all_features.append(features.float().cpu().numpy())
            del batch, outputs, features
        return np.concatenate(all_features, axis=0)


def compute_wnr(bank, labels, patches, true_idx, k=KNN_K):
    k_actual = min(k, len(bank))
    nn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean")
    nn.fit(bank)
    _, indices = nn.kneighbors(patches)
    neighbor_labels = labels[indices]
    return float((neighbor_labels != true_idx).mean())


def knn_score_images(bank, test_feats, k=KNN_K):
    n, p, d = test_feats.shape
    flat = test_feats.reshape(-1, d)
    k_actual = min(k, len(bank))
    nn = NearestNeighbors(n_neighbors=k_actual, metric='euclidean', algorithm='auto')
    nn.fit(bank)
    dists, _ = nn.kneighbors(flat)
    patch_scores = dists.mean(axis=1).reshape(n, p)
    return np.percentile(patch_scores, 95, axis=1)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extractor = DINOv2Extractor()

    # Extract features
    train_feats, test_feats, test_labels_dict = {}, {}, {}
    for task in TASKS:
        short = task.split("/")[-1]
        print(f"  Extracting {short}...", end=" ", flush=True)
        train_paths, test_info = load_dataset_with_masks(task, N_PER_TASK, SEED)
        train_feats[task] = extractor.extract_batch(train_paths)
        test_paths = [e["path"] for e in test_info]
        test_feats[task] = extractor.extract_batch(test_paths)
        test_labels_dict[task] = np.array([e["label"] for e in test_info])
        print(f"OK ({len(train_paths)} train, {len(test_paths)} test)")

    torch.cuda.empty_cache()

    # Build shared bank
    bank_parts, label_parts = [], []
    for idx, task in enumerate(TASKS):
        flat = train_feats[task].reshape(-1, train_feats[task].shape[-1])
        compressed = kcenter_greedy(flat, BUDGET_PER_PRODUCT, SEED)
        bank_parts.append(compressed)
        label_parts.append(np.full(len(compressed), idx, dtype=np.int32))

    full_bank = np.concatenate(bank_parts)
    product_labels = np.concatenate(label_parts)
    print(f"\nBank: {len(full_bank)} embeddings, dim={full_bank.shape[1]}, {full_bank.nbytes/1e6:.1f} MB")

    # Evaluate
    for idx, task in enumerate(TASKS):
        short = task.split("/")[-1]
        scores = knn_score_images(full_bank, test_feats[task])
        metrics = compute_metrics(test_labels_dict[task], scores)
        wnr_vals = [compute_wnr(full_bank, product_labels, test_feats[task][i], idx)
                     for i in range(min(len(test_feats[task]), 20))]

        result = {
            "timestamp": datetime.now().isoformat(),
            "backbone": "dinov2-vit-b14",
            "feature_dim": extractor.feature_dim,
            "num_patches": extractor.num_patches,
            "seed": SEED,
            "task": short,
            "budget_per_product": BUDGET_PER_PRODUCT,
            "img_auroc": round(metrics["img_auroc"], 6),
            "wnr": round(float(np.mean(wnr_vals)), 4),
        }

        exists = CSV_PATH.exists()
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
            if not exists:
                w.writeheader()
            w.writerow(result)
        print(f"  {short:>15s}  AUROC={metrics['img_auroc']:.4f}  WNR={float(np.mean(wnr_vals)):.3f}")

    print(f"\n[DONE] {CSV_PATH}")


if __name__ == "__main__":
    main()
