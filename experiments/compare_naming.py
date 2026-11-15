"""Experiment: side-by-side region names for runs that share one layout + clustering.

Off-pipeline. Reads toponymy_labels.parquet from several run folders (made with
TASKMAP_DATA_DIR, see pipeline/config.py) that named the SAME clusters with different
naming models / rerolls, checks the clusters match (via the
cluster_layer_* columns stage 04 writes; matched by overlap, since clustering can wobble
by a few points between processes), and renders one HTML table: one row per
region, one column per run, plus a few member briefs so names can be judged against
their content. Each region is tagged:
  same       — every run gave the same name
  model      — stable within each model's rerolls, but the models differ
  reroll     — at least one model renamed it between its own rerolls (noise)

Run:  uv run python experiments/compare_naming.py data/experiments/compare B B2 C C2 \
          --groups "Opus 4.8:B,B2" "Opus 5.5:C,C2"
Writes <root>/naming_comparison.html.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


def norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--groups", nargs="+", required=True, help='"label:run1,run2" per model')
    ap.add_argument("--n-briefs", type=int, default=4)
    ap.add_argument("--base", help="run whose clusters define the rows (default: first run)")
    args = ap.parse_args()

    groups = {}
    for g in args.groups:
        label, members = g.split(":", 1)
        groups[label] = members.split(",")

    dfs = {r: pd.read_parquet(args.root / r / "toponymy_labels.parquet") for r in args.runs}
    base_run = args.base or args.runs[0]
    base = dfs[base_run]
    layers = sorted(int(c.rsplit("_", 1)[1]) for c in base.columns if c.startswith("cluster_layer_"))

    # Clustering should be identical across runs, but can wobble by a few points (and
    # renumber ids). Map each run's clusters onto the base run's by best overlap and
    # refuse if any region isn't essentially the same set of tasks (Jaccard < MIN_JACCARD).
    MIN_JACCARD = 0.8
    mapping = {}  # (run, layer, base_cid) -> mask of that run's matched cluster
    worst = {}
    for r, df in dfs.items():
        if not (df["task_id"].to_numpy() == base["task_id"].to_numpy()).all():
            raise SystemExit(f"{r}: task order differs")
        for li in layers:
            b = base[f"cluster_layer_{li}"].to_numpy()
            o = df[f"cluster_layer_{li}"].to_numpy()
            if set(b.tolist()) - {-1} and len(set(b.tolist()) - {-1}) != len(set(o.tolist()) - {-1}):
                raise SystemExit(f"{r}: layer {li} has a different number of clusters than {base_run}")
            for cid in set(b.tolist()) - {-1}:
                bm = b == cid
                cand, cnt = np.unique(o[bm & (o != -1)], return_counts=True)
                if len(cand) == 0:
                    raise SystemExit(f"{r}: layer {li} cluster {cid} has no counterpart")
                om = o == cand[cnt.argmax()]
                jac = (bm & om).sum() / (bm | om).sum()
                if jac < MIN_JACCARD:
                    raise SystemExit(f"{r}: layer {li} cluster {cid} best match Jaccard {jac:.2f} < {MIN_JACCARD}")
                mapping[(r, li, cid)] = om
                worst[r] = min(worst.get(r, 1.0), jac)
    print(
        f"base run {base_run}; worst region match per run (Jaccard): " + ", ".join(f"{r} {worst[r]:.3f}" for r in dfs)
    )

    briefs = pd.read_parquet(args.root / base_run / "task_rows.parquet").set_index("task_id")["embed_text"]

    rows, counts = [], {}
    for li in reversed(layers):  # coarsest first
        cl = base[f"cluster_layer_{li}"].to_numpy()
        for cid in sorted(set(cl.tolist()) - {-1}, key=lambda c: -int((cl == c).sum())):
            mask = cl == cid
            names = {r: str(dfs[r][f"label_layer_{li}"].to_numpy()[mapping[(r, li, cid)]][0]) for r in args.runs}
            stable = all(len({norm(names[r]) for r in m}) == 1 for m in groups.values())
            if len({norm(n) for n in names.values()}) == 1:
                tag = "same"
            elif stable:
                tag = "model"
            else:
                tag = "reroll"
            counts[(li, tag)] = counts.get((li, tag), 0) + 1
            ids = base["task_id"].to_numpy()[mask]
            rng = np.random.default_rng(int(cid) + 1000 * li)
            sample = rng.choice(ids, size=min(args.n_briefs, len(ids)), replace=False)
            rows.append((li, int(cid), int(mask.sum()), tag, names, [str(briefs.get(i, "")) for i in sample]))

    for li in reversed(layers):
        n = sum(v for (layer, _), v in counts.items() if layer == li)
        parts = ", ".join(f"{t} {counts.get((li, t), 0)}" for t in ("same", "model", "reroll"))
        print(f"layer {li}: {n} regions — {parts}")

    run_label = {r: f"{g} · {r}" for g, ms in groups.items() for r in ms}
    head = "".join(f"<th>{escape(run_label.get(r, r))}</th>" for r in args.runs)
    body = []
    for li, cid, size, tag, names, sample in rows:
        cells = "".join(f"<td>{escape(names[r])}</td>" for r in args.runs)
        ex = "".join(f"<li>{escape(b[:160])}</li>" for b in sample)
        body.append(
            f'<tr class="{tag}"><td>{li}</td><td>{size}</td><td><span class="tag">{tag}</span></td>{cells}'
            f"<td><ul>{ex}</ul></td></tr>"
        )
    summary = "<br>".join(
        f"Layer {li} ({'coarse' if li == layers[-1] else 'fine'}): "
        + ", ".join(f"{t} {counts.get((li, t), 0)}" for t in ("same", "model", "reroll"))
        for li in reversed(layers)
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Region names compared</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#e5e5ea; --same:#f2f2f7; --model:#e8f1ff; --reroll:#fff4e0; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1c1c1e; --fg:#f2f2f7; --muted:#a1a1a6; --line:#3a3a3c;
  --same:#2c2c2e; --model:#1e3250; --reroll:#4a3714; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.4 -apple-system, system-ui, sans-serif; margin:24px; }}
table {{ border-collapse:collapse; width:100%; }} th, td {{ border-bottom:1px solid var(--line); padding:6px 8px;
  vertical-align:top; text-align:left; }}
th {{ position:sticky; top:0; background:var(--bg); }} ul {{ margin:0; padding-left:16px; color:var(--muted);
  font-size:12px; }}
tr.same {{ background:var(--same); }} tr.model {{ background:var(--model); }} tr.reroll {{ background:var(--reroll); }}
.tag {{ font-size:11px; text-transform:uppercase; letter-spacing:.04em; }} p {{ color:var(--muted); }}
</style></head><body>
<h1>Region names: same clusters, different naming runs</h1>
<p>{summary}<br>same = identical in every run · model = stable across each model's rerolls but the models
differ · reroll = renamed between rerolls of the same model (noise). Coarsest layer first; regions by size.</p>
<table><tr><th>Layer</th><th>Tasks</th><th>Tag</th>{head}<th>Sample briefs</th></tr>
{"".join(body)}</table></body></html>"""
    out = args.root / "naming_comparison.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
