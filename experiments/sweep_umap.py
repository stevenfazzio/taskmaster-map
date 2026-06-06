"""UMAP parameter sweep — clustering only, NO LLM naming (free, fast).

For each (n_neighbors, min_dist), re-run UMAP on the fixed task embeddings, then run
ONLY the Toponymy clusterer on the resulting 2D coords (no Opus naming). Report:

  task-objective metrics
    n_layers          len(cluster_layers_) — the map's hierarchy depth
    n_clusters        distinct finest-layer regions (excl. noise)
    noise%            fraction of tasks in no finest-layer region
    prize_regions     # distinct finest regions the 134 prize-synonym tasks land in
                      (target: 6 -> 1-2)  + their noise%
  UMAP-faithfulness guardrails (NOT the objective; see notes)
    trust@k, cont@k   trustworthiness / continuity at fixed k (cosine high-D, eucl 2D)
  stability
    ARI               mean pairwise adjusted-Rand of finest-layer labels over S seeds

The prize task_ids come from the production labels (the 6 find-or-acquire synonyms).
Clustering membership is what we vary; naming is deferred to the winner only.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "pipeline")

import numpy as np
import pandas as pd
import umap
from config import TASK_EMB_NPZ, TOPONYMY_LABELS_PARQUET, UMAP_COORDS_NPZ
from sklearn.manifold import trustworthiness as sk_trust
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from toponymy import ToponymyClusterer

# ── sweep config ──────────────────────────────────────────────────────────────
CONFIGS = [(15, 0.05), (30, 0.05), (50, 0.05), (100, 0.05)]  # (n_neighbors, min_dist)
TC_KS = [15, 30]  # neighborhood sizes for trust/continuity (fixed across configs)
N_SEEDS = 5  # seeds for stability ARI
CANONICAL_SEED = 42

SYNONYMS = [
    "Most [Superlative] Item Challenges",
    "Best Object Bring-In Tasks",
    "Best Thing Challenges",
    "Bring the Most Superlative Object",
    "Superlative Object Showcase Tasks",
    "Best/Nicest Object Superlative Challenges",
]


def trust_continuity(high, low, k, high_metric="cosine"):
    """Trustworthiness & continuity at neighborhood size k.
    high-D neighbors by `high_metric` (cosine, matching the UMAP graph); 2D by euclidean.
    """
    n = high.shape[0]
    rank_h = pairwise_distances(high, metric=high_metric).argsort(1).argsort(1)
    rank_l = pairwise_distances(low, metric="euclidean").argsort(1).argsort(1)
    knn_h = (rank_h >= 1) & (rank_h <= k)
    knn_l = (rank_l >= 1) & (rank_l <= k)
    norm = 2.0 / (n * k * (2 * n - 3 * k - 1))
    trust = 1 - norm * np.where(knn_l & ~knn_h, rank_h - k, 0).sum()  # false neighbors
    cont = 1 - norm * np.where(knn_h & ~knn_l, rank_l - k, 0).sum()  # missing neighbors
    return float(trust), float(cont)


def cluster_finest(coords2d, embeddings):
    """Run clusterer only (no naming). Returns (n_layers, finest per-point int labels)."""
    cl = ToponymyClusterer(min_clusters=6)
    cl.fit(clusterable_vectors=coords2d.astype(np.float32), embedding_vectors=embeddings, show_progress_bar=False)
    return len(cl.cluster_layers_), np.asarray(cl.cluster_layers_[0].cluster_labels)


def run_umap(embeddings, n_neighbors, min_dist, seed):
    return umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    ).fit_transform(embeddings)


def main():
    crd = np.load(UMAP_COORDS_NPZ, allow_pickle=True)
    task_id = crd["task_id"]
    pos = {t: i for i, t in enumerate(task_id)}

    ed = np.load(TASK_EMB_NPZ, allow_pickle=True)
    row = {c: i for i, c in enumerate(ed["task_id"])}
    embeddings = ed["emb"][np.array([row[c] for c in task_id])].astype(np.float32)

    prod = pd.read_parquet(TOPONYMY_LABELS_PARQUET)
    prize_ids = prod.loc[prod.label_layer_0.isin(SYNONYMS), "task_id"].tolist()
    prize_idx = np.array([pos[t] for t in prize_ids])
    print(f"{len(task_id)} tasks; {len(prize_idx)} prize-district tasks; embeddings {embeddings.shape}\n")

    rows = []
    for nn, md in CONFIGS:
        coords = run_umap(embeddings, nn, md, CANONICAL_SEED)

        n_layers, finest = cluster_finest(coords, embeddings)
        n_clusters = len(set(finest[finest >= 0].tolist()))
        noise_pct = 100 * (finest < 0).mean()

        pf = finest[prize_idx]
        prize_regions = len(set(pf[pf >= 0].tolist()))
        prize_noise_pct = 100 * (pf < 0).mean()

        tc = {k: trust_continuity(embeddings, coords, k) for k in TC_KS}

        # stability: ARI of finest labels across seeds
        labelings = [cluster_finest(run_umap(embeddings, nn, md, s), embeddings)[1] for s in range(1, N_SEEDS + 1)]
        aris = [
            adjusted_rand_score(labelings[a], labelings[b])
            for a in range(len(labelings))
            for b in range(a + 1, len(labelings))
        ]
        stability = float(np.mean(aris))

        rec = {
            "n_neighbors": nn,
            "min_dist": md,
            "n_layers": n_layers,
            "n_clusters": n_clusters,
            "noise%": round(noise_pct, 1),
            "prize_regions": prize_regions,
            "prize_noise%": round(prize_noise_pct, 1),
            "stability_ARI": round(stability, 3),
        }
        for k in TC_KS:
            rec[f"trust@{k}"] = round(tc[k][0], 3)
            rec[f"cont@{k}"] = round(tc[k][1], 3)
        rows.append(rec)

        # validation on the baseline: our trust@15 should match sklearn's
        if (nn, md) == CONFIGS[0]:
            skt = sk_trust(embeddings, coords, n_neighbors=15, metric="cosine")
            print(f"[validation] baseline trust@15: ours={tc[15][0]:.3f}  sklearn={skt:.3f}\n")

        print(f"done: n_neighbors={nn} min_dist={md} -> {rec}")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
