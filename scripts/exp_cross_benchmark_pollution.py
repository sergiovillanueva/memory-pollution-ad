"""
Cross-Benchmark Pollution: oracle inversion + WNR in VisA and BTAD.
Tests whether memory pollution is universal across datasets.

Setting: N=10, 3 seeds, budgets {6, 12, 18, 24, 30} MB, global retrieval
Allocators: uniform vs per-product oracle (greedy marginal gain)

Output: output/continual/cross_benchmark_pollution.csv
"""
from __future__ import annotations
import os, sys, csv, time
os.environ.setdefault("OMP_NUM_THREADS", "12")

import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from continual_utils import compute_metrics, KNN_K, FEATURE_DIM

N_PER_TASK = 10
SEEDS = [0, 1, 2]
BUDGETS_MB = [6.0, 12.0, 18.0, 24.0, 30.0]
BUDGET_OPTIONS = [25, 50, 75, 100, 150, 200, 300, 500]

VISA_TASKS = [
    "VisA/candle", "VisA/capsules", "VisA/cashew", "VisA/chewinggum",
    "VisA/fryum", "VisA/macaroni1", "VisA/macaroni2", "VisA/pcb1",
    "VisA/pcb2", "VisA/pcb3", "VisA/pcb4", "VisA/pipe_fryum",
]
BTAD_TASKS = ["btad/01", "btad/02", "btad/03"]

CACHE = Path("output/feature_cache")
OUTPUT_DIR = Path("output/continual")
CSV_PATH = OUTPUT_DIR / "cross_benchmark_pollution.csv"
COLUMNS = ["timestamp", "dataset", "seed", "budget_mb", "allocator",
           "task", "img_auroc", "wnr"]


def cache_path(prefix, task, suffix=""):
    return CACHE / f"{prefix}_{task.replace('/', '_')}{suffix}.npy"


def mb_to_embs(mb, dim=FEATURE_DIM):
    return int(mb * 1e6 / (dim * 4))


def snap(b):
    return min(BUDGET_OPTIONS, key=lambda x: abs(x - b))


def kcenter_greedy(data, budget, seed):
    n = len(data)
    if budget >= n:
        return data
    pca_dim = min(64, data.shape[1], n)
    pca = PCA(n_components=pca_dim, random_state=seed)
    data_pca = pca.fit_transform(data)
    rng = np.random.RandomState(seed)
    first = rng.randint(0, n)
    selected = [first]
    min_d = np.full(n, np.inf)
    diffs = data_pca - data_pca[first]
    min_d = np.minimum(min_d, np.einsum('ij,ij->i', diffs, diffs))
    for _ in range(budget - 1):
        nxt = int(np.argmax(min_d))
        selected.append(nxt)
        diffs = data_pca - data_pca[nxt]
        min_d = np.minimum(min_d, np.einsum('ij,ij->i', diffs, diffs))
    return data[np.array(selected)]


def allocate_uniform(total_mb, n_products):
    per_emb = mb_to_embs(total_mb / n_products)
    return [snap(per_emb)] * n_products


def allocate_oracle(total_mb, tasks, train_feats, test_feats, test_labels, seed):
    """Greedy marginal gain using per-product AUROC curves on isolated banks."""
    # Build AUROC curves per product (isolated)
    curves = {}
    for task in tasks:
        flat = train_feats[task].reshape(-1, train_feats[task].shape[-1])
        curves[task] = {}
        for b in BUDGET_OPTIONS:
            if b > len(flat):
                curves[task][b] = curves[task][BUDGET_OPTIONS[BUDGET_OPTIONS.index(b)-1]]
                continue
            bank = kcenter_greedy(flat, b, seed)
            scores = knn_score(bank, test_feats[task])
            m = compute_metrics(test_labels[task], scores)
            curves[task][b] = m["img_auroc"]

    total_embs = mb_to_embs(total_mb)
    alloc = {t: BUDGET_OPTIONS[0] for t in tasks}
    remaining = total_embs - sum(alloc.values())

    while remaining > 0:
        best_task, best_gain, best_b = None, -1, None
        for t in tasks:
            cur = alloc[t]
            higher = [b for b in BUDGET_OPTIONS if b > cur]
            if not higher:
                continue
            nxt = min(higher)
            cost = nxt - cur
            if cost > remaining:
                continue
            gain = (curves[t].get(nxt, 0) - curves[t].get(cur, 0)) / cost
            if gain > best_gain:
                best_gain, best_task, best_b = gain, t, nxt
        if best_task is None:
            break
        remaining -= (best_b - alloc[best_task])
        alloc[best_task] = best_b
    return [alloc[t] for t in tasks]


def knn_score(bank, test_feats, k=KNN_K):
    n, p, d = test_feats.shape
    flat = test_feats.reshape(-1, d)
    k_actual = min(k, len(bank))
    nn = NearestNeighbors(n_neighbors=k_actual, metric='euclidean', algorithm='auto')
    nn.fit(bank)
    dists, _ = nn.kneighbors(flat)
    patch_scores = dists.mean(axis=1).reshape(n, p)
    return np.percentile(patch_scores, 95, axis=1)


def compute_wnr(bank, labels, patches, true_idx, k=KNN_K):
    k_actual = min(k, len(bank))
    nn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean")
    nn.fit(bank)
    _, indices = nn.kneighbors(patches)
    return float((labels[indices] != true_idx).mean())


def run_dataset(dataset_name, tasks):
    print(f"\n{'='*70}\nDATASET: {dataset_name} ({len(tasks)} products)\n{'='*70}")

    # Load test features and labels (cached) - same for all seeds
    test_feats, test_labels = {}, {}
    for task in tasks:
        test_feats[task] = np.load(cache_path("test_feat", task))
        test_labels[task] = np.load(cache_path("test_labels", task))

    for seed in SEEDS:
        print(f"\n[SEED {seed}]")
        train_feats = {}
        for task in tasks:
            train_feats[task] = np.load(cache_path("train_feat", task, f"_s{seed}_n{N_PER_TASK}"))

        for budget_mb in BUDGETS_MB:
            print(f"  Budget {budget_mb} MB:")

            # UNIFORM allocation
            unif_alloc = allocate_uniform(budget_mb, len(tasks))

            # Build banks
            for allocator_name, alloc in [("uniform", unif_alloc)]:
                bank_parts, label_parts = [], []
                for i, task in enumerate(tasks):
                    flat = train_feats[task].reshape(-1, train_feats[task].shape[-1])
                    comp = kcenter_greedy(flat, alloc[i], seed)
                    bank_parts.append(comp)
                    label_parts.append(np.full(len(comp), i, dtype=np.int32))
                full_bank = np.concatenate(bank_parts)
                product_labels = np.concatenate(label_parts)

                for i, task in enumerate(tasks):
                    short = task.split("/")[-1]
                    scores = knn_score(full_bank, test_feats[task])
                    m = compute_metrics(test_labels[task], scores)
                    wnr_vals = [compute_wnr(full_bank, product_labels, test_feats[task][j], i)
                                 for j in range(min(len(test_feats[task]), 20))]
                    write_row(dataset_name, seed, budget_mb, allocator_name, short,
                              m["img_auroc"], float(np.mean(wnr_vals)))

            # ORACLE allocation
            oracle_alloc = allocate_oracle(budget_mb, tasks, train_feats, test_feats, test_labels, seed)
            bank_parts, label_parts = [], []
            for i, task in enumerate(tasks):
                flat = train_feats[task].reshape(-1, train_feats[task].shape[-1])
                comp = kcenter_greedy(flat, oracle_alloc[i], seed)
                bank_parts.append(comp)
                label_parts.append(np.full(len(comp), i, dtype=np.int32))
            full_bank = np.concatenate(bank_parts)
            product_labels = np.concatenate(label_parts)

            for i, task in enumerate(tasks):
                short = task.split("/")[-1]
                scores = knn_score(full_bank, test_feats[task])
                m = compute_metrics(test_labels[task], scores)
                wnr_vals = [compute_wnr(full_bank, product_labels, test_feats[task][j], i)
                             for j in range(min(len(test_feats[task]), 20))]
                write_row(dataset_name, seed, budget_mb, "oracle", short,
                          m["img_auroc"], float(np.mean(wnr_vals)))

            # Print summary
            print(f"    uniform vs oracle done")


def write_row(dataset, seed, budget, allocator, task, auroc, wnr):
    row = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset, "seed": seed, "budget_mb": budget,
        "allocator": allocator, "task": task,
        "img_auroc": round(auroc, 6), "wnr": round(wnr, 4),
    }
    exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {CSV_PATH}")
    t0 = time.time()
    run_dataset("VisA", VISA_TASKS)
    run_dataset("BTAD", BTAD_TASKS)
    print(f"\n[DONE] Total: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
