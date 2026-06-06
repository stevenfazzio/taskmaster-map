"""Proper LEACE erase of task_format -> re-UMAP -> re-cluster. No LLM.

Run with:  uv run --with concept-erasure --with plotly python experiments/leace_experiment.py

Tests whether erasing the format concept dissolves the prize island and mixes prize
tasks into content regions, and how surgical the erasure is on the other 3 formats.
Outputs data/experiments/leace_umap.html (baseline vs LEACE, coloured by task_format).
"""

from __future__ import annotations

import sys

sys.path.insert(0, "pipeline")

import numpy as np
import pandas as pd
import torch
import umap
from concept_erasure import LeaceEraser
from config import DATA_DIR, TASK_EMB_NPZ, TASK_ROWS_PARQUET, UMAP_COORDS_NPZ
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, f1_score, silhouette_score
from sklearn.model_selection import cross_val_predict
from toponymy import ToponymyClusterer

UMAP_KW = dict(n_components=2, n_neighbors=15, min_dist=0.05, metric="cosine", random_state=42)
COLORS = {"Filmed": "#3aa6a0", "Prize": "#e0a92a", "Live (studio)": "#8a6fc4", "Tiebreak": "#d1495b"}


def cluster(coords, emb):
    cl = ToponymyClusterer(min_clusters=6)
    cl.fit(
        clusterable_vectors=coords.astype(np.float32), embedding_vectors=emb.astype(np.float32), show_progress_bar=False
    )
    return len(cl.cluster_layers_), np.asarray(cl.cluster_layers_[0].cluster_labels)


def main():
    crd = np.load(UMAP_COORDS_NPZ, allow_pickle=True)
    tid = crd["task_id"]
    base_coords = crd["coords"].astype(np.float64)  # the deployed map
    ed = np.load(TASK_EMB_NPZ, allow_pickle=True)
    row = {c: i for i, c in enumerate(ed["task_id"])}
    X = ed["emb"][np.array([row[c] for c in tid])].astype(np.float64)
    fmt = (
        pd.read_parquet(TASK_ROWS_PARQUET, columns=["task_id", "task_format"])
        .set_index("task_id")
        .loc[tid, "task_format"]
        .values
    )
    codes, names = pd.factorize(fmt)
    is_prize = fmt == "Prize"
    print(f"{len(tid)} tasks; formats {dict(pd.Series(fmt).value_counts())}\n")

    # ── proper LEACE ──
    eraser = LeaceEraser.fit(torch.tensor(X), torch.tensor(codes))
    Xe = eraser(torch.tensor(X)).numpy()

    # 1) erasure check: can a linear model still predict format?
    def lin_acc(M):
        pred = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), M, codes, cv=5)
        return (pred == codes).mean(), f1_score(codes, pred, average="macro")

    a0, f0 = lin_acc(X)
    a1, f1 = lin_acc(Xe)
    maj = pd.Series(fmt).value_counts(normalize=True).iloc[0]
    print(
        f"linear format-predictability  raw: acc={a0:.2f} macroF1={f0:.2f}   "
        f"erased: acc={a1:.2f} macroF1={f1:.2f}   (majority baseline acc={maj:.2f})"
    )

    # 2) how surgical? per-task cosine(raw, erased) — prize should move more than the rest
    cos = (X * Xe).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Xe, axis=1) + 1e-9)
    print(
        f"embedding change (cos raw·erased)  prize median={np.median(cos[is_prize]):.3f}   "
        f"non-prize median={np.median(cos[~is_prize]):.3f}   (lower = moved more)\n"
    )

    # ── re-UMAP erased, re-cluster both ──
    leace_coords = umap.UMAP(**UMAP_KW).fit_transform(Xe)
    nl_b, lab_b = cluster(base_coords, X)
    nl_l, lab_l = cluster(leace_coords, Xe)

    def isolation(coords):
        pc = coords[is_prize].mean(0)
        intra = np.median(np.linalg.norm(coords[is_prize] - pc, axis=1))
        to_rest = np.median(np.linalg.norm(coords[~is_prize] - pc, axis=1))
        sil = silhouette_score(coords, is_prize.astype(int))
        return intra, to_rest, to_rest / intra, sil

    def prize_mix(lab):
        pf = lab[is_prize]
        prize_clusters = sorted(set(pf[pf >= 0].tolist()))
        mixed = absorbed = 0
        for cl in prize_clusters:
            members = lab == cl
            n_non = int((members & ~is_prize).sum())
            if n_non > 0:
                mixed += 1
            if n_non >= 0.5 * int(members.sum()):  # cluster majority non-prize -> prize tasks here are "absorbed"
                absorbed += int((members & is_prize).sum())
        return len(prize_clusters), mixed, absorbed

    for name, c, nl, lab in [("baseline", base_coords, nl_b, lab_b), ("LEACE", leace_coords, nl_l, lab_l)]:
        intra, rest, ratio, sil = isolation(c)
        nclust = len(set(lab[lab >= 0].tolist()))
        span, mixed, absorbed = prize_mix(lab)
        print(
            f"[{name}] layers={nl} clusters={nclust} noise={100 * (lab < 0).mean():.0f}%  | "
            f"prize island: intra={intra:.2f} to_rest={rest:.2f} ratio={ratio:.1f} silhouette={sil:.2f}  | "
            f"prize span {span} clusters, {mixed} shared w/ non-prize, "
            f"{absorbed}/{int(is_prize.sum())} prize tasks in majority-non-prize clusters"
        )

    ari_np = adjusted_rand_score(lab_b[~is_prize], lab_l[~is_prize])
    print(
        f"\nnon-prize partition ARI (baseline vs LEACE) = {ari_np:.2f}  "
        f"(caveat: reseeding UMAP alone gives ~0.25, so judge collateral against that floor, not 1.0)"
    )

    # ── viz: baseline vs LEACE, coloured by task_format ──
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Baseline (deployed)", "LEACE: format erased"))
        for col, coords in [(1, base_coords), (2, leace_coords)]:
            for f in names:
                msk = fmt == f
                fig.add_trace(
                    go.Scattergl(
                        x=coords[msk, 0],
                        y=coords[msk, 1],
                        mode="markers",
                        marker=dict(size=4, color=COLORS.get(f, "#888"), opacity=0.7),
                        name=f,
                        legendgroup=f,
                        showlegend=(col == 1),
                    ),
                    row=1,
                    col=col,
                )
        fig.update_layout(title="task_format on the map — before vs after LEACE erasure", height=620, width=1300)
        out = DATA_DIR / "experiments" / "leace_umap.html"
        fig.write_html(str(out))
        print(f"\nwrote {out}")
    except Exception as e:  # noqa: BLE001
        print(f"\n(viz skipped: {e})")


if __name__ == "__main__":
    main()
