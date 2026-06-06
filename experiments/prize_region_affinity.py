"""Where would prize tasks go if the format signal were removed? (no UMAP, no LLM)

Removes the linear task_format subspace from the embeddings (class-mean removal —
"LEACE-lite"), then assigns each of the 189 Prize tasks to its nearest *non-prize*
layer-0 region centroid (cosine) in that erased space. Tests the user's point: do
prize tasks have plausible content homes (Creative Use of Everyday Objects, Abstract
Open-Ended Creative, Pleasing the Taskmaster, Taskmaster Tributes, ...)?
"""

from __future__ import annotations

import sys

sys.path.insert(0, "pipeline")

import numpy as np
import pandas as pd
from config import TASK_EMB_NPZ, TASK_ROWS_PARQUET, TOPONYMY_LABELS_PARQUET

SYN = {
    "Most [Superlative] Item Challenges",
    "Best Object Bring-In Tasks",
    "Best Thing Challenges",
    "Bring the Most Superlative Object",
    "Superlative Object Showcase Tasks",
    "Best/Nicest Object Superlative Challenges",
    "Make or Build the Best",
}
HIGHLIGHT = {
    "Creative Use of Everyday Objects",
    "Abstract Open-Ended Creative Challenges",
    "Pleasing the Taskmaster",
    "Portraits of the Taskmaster",  # layer-0 pieces of "Taskmaster Tributes"
}


def main():
    ed = np.load(TASK_EMB_NPZ, allow_pickle=True)
    emb_by_id = {t: ed["emb"][i] for i, t in enumerate(ed["task_id"])}
    lab = pd.read_parquet(TOPONYMY_LABELS_PARQUET)
    fmt = pd.read_parquet(TASK_ROWS_PARQUET, columns=["task_id", "task_format", "embed_text"])
    df = lab.merge(fmt, on="task_id")
    df = df[df.task_id.isin(emb_by_id)].reset_index(drop=True)
    X = np.stack([emb_by_id[t] for t in df.task_id]).astype(np.float64)

    # ── erase the linear task_format subspace (class-mean removal) ──
    Xc = X - X.mean(0)
    means = np.stack([Xc[(df.task_format == c).values].mean(0) for c in df.task_format.unique()])
    _, S, Vt = np.linalg.svd(means, full_matrices=False)
    B = Vt[S > 1e-6]  # orthonormal basis of the format subspace (rank <= 3 after centering)
    Xe = Xc - (Xc @ B.T) @ B
    Xe /= np.linalg.norm(Xe, axis=1, keepdims=True) + 1e-9
    print(f"erased a {B.shape[0]}-dim task_format subspace from {X.shape[1]}-dim embeddings\n")

    # ── candidate regions = non-prize layer-0 names; assign each prize task to nearest centroid ──
    cand = [g for g in df.label_layer_0.unique() if g not in SYN and g != "Unlabelled"]
    cents = {g: Xe[(df.label_layer_0 == g).values].mean(0) for g in cand}
    cents = {g: v / (np.linalg.norm(v) + 1e-9) for g, v in cents.items()}
    cand_names = list(cents)
    C = np.stack([cents[g] for g in cand_names])

    prize = df[df.task_format == "Prize"].copy()
    Pe = Xe[prize.index.values]
    sims = Pe @ C.T
    best = sims.argmax(1)
    prize["nearest_region"] = [cand_names[i] for i in best]
    prize["sim"] = sims.max(1)

    print("=== where the 189 prize tasks land (nearest non-prize region, format erased) ===")
    print(prize.nearest_region.value_counts().to_string())

    print("\n=== the regions you named — sample prize tasks that map there ===")
    for g in HIGHLIGHT:
        sub = prize[prize.nearest_region == g].sort_values("sim", ascending=False)
        if not len(sub):
            print(f"\n[{0}] {g}: (none)")
            continue
        print(f"\n[{len(sub)}] {g}:")
        for _, r in sub.head(5).iterrows():
            print(f"    {r.sim:.2f}  {r.embed_text[:64]}")

    # spread: how concentrated vs dispersed is the landing?
    vc = prize.nearest_region.value_counts()
    print(
        f"\nspread: 189 prize tasks → {len(vc)} distinct regions; "
        f"top region holds {vc.iloc[0]} ({100 * vc.iloc[0] / len(prize):.0f}%); "
        f"median cosine to assigned region = {prize.sim.median():.2f}"
    )


if __name__ == "__main__":
    main()
