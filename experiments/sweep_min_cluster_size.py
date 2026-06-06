"""base_min_cluster_size sweep — re-cluster the DEPLOYED coords, no UMAP, no LLM.

The display geometry is held fixed (data/umap_coords.npz); we only vary the clusterer's
finest-layer min cluster size. Question: can a coarser clusterer fuse the 6 prize islands
into 1-2 regions WITHOUT (a) dissolving them into noise or (b) decimating the rest of the map?

Per value reports: n_layers, global regions/noise, prize_regions + prize_noise%, and the
prize cluster-size distribution (the tell: one big count = merged; lots of noise = dissolved).
"""

from __future__ import annotations

import sys
from collections import Counter

sys.path.insert(0, "pipeline")

import numpy as np
import pandas as pd
from config import TASK_EMB_NPZ, TOPONYMY_LABELS_PARQUET, UMAP_COORDS_NPZ
from toponymy import ToponymyClusterer

VALUES = [10, 12, 15, 20, 25, 30, 40, 50]  # base_min_cluster_size (10 = current)
SYNONYMS = [
    "Most [Superlative] Item Challenges",
    "Best Object Bring-In Tasks",
    "Best Thing Challenges",
    "Bring the Most Superlative Object",
    "Superlative Object Showcase Tasks",
    "Best/Nicest Object Superlative Challenges",
]


def main():
    crd = np.load(UMAP_COORDS_NPZ, allow_pickle=True)
    task_id = crd["task_id"]
    coords = crd["coords"].astype(np.float32)
    pos = {t: i for i, t in enumerate(task_id)}

    ed = np.load(TASK_EMB_NPZ, allow_pickle=True)
    row = {c: i for i, c in enumerate(ed["task_id"])}
    embeddings = ed["emb"][np.array([row[c] for c in task_id])].astype(np.float32)

    prod = pd.read_parquet(TOPONYMY_LABELS_PARQUET)
    prize_idx = np.array([pos[t] for t in prod.loc[prod.label_layer_0.isin(SYNONYMS), "task_id"]])
    print(f"{len(task_id)} tasks; {len(prize_idx)} prize-district tasks (deployed coords held fixed)\n")

    rows = []
    for bmcs in VALUES:
        try:
            cl = ToponymyClusterer(min_clusters=6, base_min_cluster_size=bmcs)
            cl.fit(clusterable_vectors=coords, embedding_vectors=embeddings, show_progress_bar=False)
        except Exception as e:  # noqa: BLE001
            print(f"base_min_cluster_size={bmcs}: FAILED ({type(e).__name__}: {e})")
            continue

        n_layers = len(cl.cluster_layers_)
        finest = np.asarray(cl.cluster_layers_[0].cluster_labels)
        n_clusters = len(set(finest[finest >= 0].tolist()))
        noise_pct = 100 * (finest < 0).mean()

        pf = finest[prize_idx]
        prize_regions = len(set(pf[pf >= 0].tolist()))
        prize_noise_pct = 100 * (pf < 0).mean()
        # prize cluster-size distribution: how the 134 prize tasks split across regions (+noise)
        sizes = sorted((c for lbl, c in Counter(pf[pf >= 0].tolist()).items()), reverse=True)
        spread = "/".join(str(s) for s in sizes) + (f" (+{(pf < 0).sum()}n)" if (pf < 0).any() else "")

        rows.append(
            {
                "base_min_cluster_size": bmcs,
                "n_layers": n_layers,
                "regions": n_clusters,
                "noise%": round(noise_pct, 1),
                "prize_regions": prize_regions,
                "prize_noise%": round(prize_noise_pct, 1),
                "prize_spread": spread,
            }
        )
        print(f"done: {rows[-1]}")

    print("\n" + "=" * 100)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
