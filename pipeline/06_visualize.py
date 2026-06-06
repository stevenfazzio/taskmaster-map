"""Render the interactive DataMapPlot map of Taskmaster tasks.

Merges the layout (UMAP coords), region labels (Toponymy), and per-task fields
(scores + LLM taxonomy) on task_id, builds a Taskmaster-themed hover card, wires
up colormaps and a contestant search, and writes data/task_map.html + docs/index.html.

v1 scope (per the build plan): themed hover card + contestant filter (via the
built-in search, with contestant names in the search index) + colormaps. Deferred:
the heavy range-slider filter panel, a contestant dropdown, random-task button,
shareable per-task deep-links, and a methodology page.
"""

from __future__ import annotations

import json
from html import escape

import datamapplot
import glasbey
import numpy as np
import pandas as pd
from config import (
    DOCS_HTML,
    MAP_HTML,
    STRUCTURED_FIELDS_PARQUET,
    TASK_ROWS_PARQUET,
    TAXONOMY_JSON,
    TOPONYMY_LABELS_PARQUET,
    UMAP_COORDS_NPZ,
)

PROJECT_TITLE = "The Taskmaster Task Map"

_UNKNOWN_VALUES = {"", "Unknown", "Other", "Not Stated", "not_stated", "other", "—", "None"}

# Nicer display labels for slugs whose auto-prettified form reads awkwardly.
_DISPLAY_OVERRIDES = {}  # slug -> nicer label, for any slug whose auto-prettified form reads awkwardly

# Fixed palette for task_format — the card's headline pill and the "Task type"
# colormap share it, so the pill colour matches the map when coloured by type.
_TASK_TYPE_COLORS = {
    "Prize": "#e0a22b",  # amber — the prize-task "bring something" category
    "Filmed": "#2fa4a0",  # teal — pre-recorded lab / out-and-about tasks
    "Live (studio)": "#8a6bb1",  # violet — studio head-to-heads
    "Tiebreak": "#d9594c",  # coral — the dramatic decider
}


# ── small helpers (ported from the huggingface-dataset-map visualize stage) ──────


def _prettify(s: str) -> str:
    """Hyphens/underscores → spaces; title-case (preserving short acronyms)."""
    if not s:
        return s
    text = str(s).replace("-", " ").replace("_", " ")
    out = []
    for w in text.split():
        out.append(w if (w.isupper() and len(w) <= 4) else (w[:1].upper() + w[1:] if w else w))
    return " ".join(out)


def _label(slug: str) -> str:
    """Display label for an LLM slug: an override if defined, else prettified."""
    return _DISPLAY_OVERRIDES.get(slug, _prettify(slug))


def _top_n_plus_other(values: np.ndarray, n: int = 9, other_label: str = "Other") -> np.ndarray:
    s = pd.Series(values)
    top = s[s != other_label].value_counts().head(n).index.tolist()
    return s.where(s.isin(top), other_label).values


def _color_mapping(values: np.ndarray, other_label: str = "Other") -> dict:
    """Glasbey palette keyed by unique value; 'Other' pinned to neutral grey."""
    unique = sorted(set(values.tolist()))
    non_other = [v for v in unique if v != other_label]
    palette = glasbey.create_palette(palette_size=max(len(non_other), 1))
    mapping = dict(zip(non_other, palette))
    if other_label in unique:
        mapping[other_label] = "#bdbdbd"
    return mapping


def _maybe_json_list(v) -> list:
    if isinstance(v, list):
        return list(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, str) and v.startswith("["):
        try:
            out = json.loads(v)
            return out if isinstance(out, list) else []
        except Exception:
            return []
    return []


def _slug_or(raw, allowed: set[str], fallback: str = "other") -> str:
    return raw if (isinstance(raw, str) and raw in allowed) else fallback


def _load_allowed_slugs() -> dict[str, set[str]]:
    tax = json.loads(TAXONOMY_JSON.read_text())
    return {
        f: {c["name"] for c in spec.get("categories", [])}
        for f, spec in tax.items()
        if not f.startswith("_") and "categories" in spec
    }


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def _dim_cell(value: str) -> str:
    raw = (str(value) if value is not None else "").strip()
    v = escape(raw)
    if raw in _UNKNOWN_VALUES:
        return f'<span style="color:#a3a9b2;font-weight:normal;">{v or "—"}</span>'
    return v


def _pill(value: str, hex_color: str) -> str:
    raw = (value or "").strip()
    v = escape(raw)
    if raw in _UNKNOWN_VALUES:
        return f'<span style="background:#f1f3f5;color:#a3a9b2;padding:1px 8px;border-radius:10px;font-size:11px;">{v or "—"}</span>'  # noqa: E501
    bg = _hex_to_rgba(hex_color, 0.30)
    return (
        f'<span style="background:{bg};color:#1f2328;padding:1px 9px;border-radius:10px;'
        f'font-size:11px;font-weight:500;">{v}</span>'
    )


# ── hover card ───────────────────────────────────────────────────────────────

_LBL = "62px"
_BAR = (
    '<div style="background:#eaeef2;height:2px;width:54px;margin-top:3px;border-radius:1px;">'
    '<div style="background:#b0b6be;height:100%;width:{pct}%;border-radius:1px;"></div></div>'
)


def _meta_cell(label: str, placeholder: str) -> str:
    return (
        '<div style="overflow:hidden;white-space:nowrap;text-overflow:ellipsis;font-weight:500;">'
        f'<span style="display:inline-block;width:{_LBL};color:#8b949e;font-weight:normal;">{label}</span>'
        f"{{{placeholder}}}</div>"
    )


HOVER_TEMPLATE = (
    "<div style=\"font-family:'IBM Plex Sans',sans-serif;width:360px;padding:9px 11px;"
    'box-sizing:border-box;color:#1f2328;">'
    # context: series · episode
    '<div style="font-size:11.5px;color:#8b949e;margin-bottom:5px;overflow:hidden;'
    'text-overflow:ellipsis;white-space:nowrap;">{series_ep}</div>'
    # stats row: avg / divisiveness / top, each with a percentile bar
    '<div style="font-size:11px;color:#57606a;display:flex;gap:18px;margin-bottom:11px;">'
    "<div><div>Avg {avg}</div>" + _BAR.replace("{pct}", "{avg_pct}") + "</div>"
    "<div><div>Spread {spread}</div>" + _BAR.replace("{pct}", "{spread_pct}") + "</div>"
    "<div><div>Top {top}</div>" + _BAR.replace("{pct}", "{top_pct}") + "</div>"
    "</div>"
    # the brief (dominant)
    '<div style="font-size:14px;line-height:1.45;margin-bottom:10px;">{brief}</div>'
    # task-type pill (the card's headline category stamp)
    '<div style="margin-bottom:7px;">{type_pill}</div>'
    # 2-col metadata grid
    '<div style="font-size:11.5px;line-height:1.85;margin-bottom:6px;'
    'display:grid;grid-template-columns:1fr 1fr;gap:0 18px;">'
    + _meta_cell("Activity", "activity")
    + _meta_cell("Judged", "judged")
    + _meta_cell("Winner", "winner")
    + _meta_cell("Twist", "twist")
    + "</div>"
    # props + footer
    '<div style="font-size:11px;color:#57606a;margin-bottom:3px;overflow:hidden;'
    'text-overflow:ellipsis;white-space:nowrap;">🎒 {props}</div>'
    '<div style="font-size:11px;color:#8b949e;line-height:1.5;">📺 {air_date} &nbsp;·&nbsp; 👥 {players}</div>'
    "</div>"
)

CUSTOM_JS = "datamap.deckgl.setProps({controller: {scrollZoom: {speed: 0.05, smooth: true}}});"
CUSTOM_CSS = "#main-title { letter-spacing: -0.02em; line-height: 1.1 !important; color:#1f2328; }"

# Attribution footer (bottom-right) — the data is CC BY-SA 4.0, so the published map
# must credit its source on the page itself, not only in the README.
ATTRIBUTION_HTML = (
    '<div id="attribution" style="position:fixed;bottom:8px;right:12px;z-index:300;'
    "font-family:'IBM Plex Sans',system-ui,sans-serif;font-size:11px;color:#8b949e;"
    "background:rgba(255,255,255,0.78);-webkit-backdrop-filter:blur(4px);backdrop-filter:blur(4px);"
    'padding:4px 9px;border-radius:8px;border:1px solid rgba(0,0,0,0.06);max-width:46vw;line-height:1.4;">'
    'Data: <a href="https://taskmaster.fandom.com/" target="_blank" rel="noopener" '
    'style="color:#57606a;">Taskmaster Wiki</a> via '
    '<a href="https://github.com/silverdavi/taskmaster-uk-scores" target="_blank" rel="noopener" '
    'style="color:#57606a;">silverdavi/taskmaster-uk-scores</a> · '
    '<a href="https://creativecommons.org/licenses/by-sa/4.0/" target="_blank" rel="noopener" '
    'style="color:#57606a;">CC BY-SA 4.0</a> · '
    '<a href="https://github.com/stevenfazzio/taskmaster-map" target="_blank" rel="noopener" '
    'style="color:#57606a;">source</a>'
    "</div>"
)


def _inject_attribution(html_path) -> None:
    """Inject the CC BY-SA attribution footer into the rendered map HTML."""
    html = html_path.read_text()
    html_path.write_text(html.replace("</body>", ATTRIBUTION_HTML + "\n</body>", 1))


def main():
    crd = np.load(UMAP_COORDS_NPZ, allow_pickle=True)
    coords = crd["coords"].astype(np.float32)
    order = crd["task_id"]  # canonical point order

    tasks = pd.read_parquet(TASK_ROWS_PARQUET)
    labels = pd.read_parquet(TOPONYMY_LABELS_PARQUET)
    structured = pd.read_parquet(STRUCTURED_FIELDS_PARQUET)

    df = tasks.merge(labels, on="task_id", how="left").merge(structured, on="task_id", how="left")
    df = df.set_index("task_id").reindex(order).reset_index()  # row i ↔ coords[i]
    n = len(df)
    print(f"Assembled {n:,} tasks (coords {coords.shape})")

    allowed = _load_allowed_slugs()

    # ── colormap value arrays ───────────────────────────────────────────────
    fmt_vals = df["task_format"].fillna("Filmed").astype(str).values
    era_vals = df["channel_era"].fillna("Unknown").astype(str).values
    activity_vals = _top_n_plus_other(
        np.array([_label(_slug_or(v, allowed["activity_type"])) for v in df["activity_type"].values])
    )
    judged_vals = _top_n_plus_other(
        np.array([_prettify(_slug_or(v, allowed["judging_criterion"])) for v in df["judging_criterion"].values])
    )
    twist_vals = np.array(
        ["Has a twist" if len(_maybe_json_list(v)) else "Straightforward" for v in df["task_twist"].values]
    )

    avg = pd.to_numeric(df["avg_score"], errors="coerce")
    spread = pd.to_numeric(df["score_spread"], errors="coerce")
    top = pd.to_numeric(df["top_score"], errors="coerce")
    air = pd.to_datetime(df["air_date"], errors="coerce")
    air_num = (air - air.min()).dt.days.astype(float)
    air_num = air_num.fillna(air_num.median()).values

    # ── hover-card fields ───────────────────────────────────────────────────
    series_ep = [
        escape(f"{s} · Episode {int(e)}" + (f' · "{t}"' if isinstance(t, str) and t.strip() else ""))
        if pd.notna(e)
        else escape(str(s))
        for s, e, t in zip(df["series"], df["episode_num"], df["episode_title"])
    ]

    def _fmt_num(v, dec=False):
        if pd.isna(v):
            return "—"
        return f"{v:.1f}" if dec else f"{int(round(v))}"

    avg_disp = [_fmt_num(v, dec=True) for v in avg]
    spread_disp = [_fmt_num(v) for v in spread]
    top_disp = [_fmt_num(v) for v in top]
    avg_pct = avg.rank(pct=True).mul(100).round(1).fillna(0).tolist()
    spread_pct = spread.rank(pct=True).mul(100).round(1).fillna(0).tolist()
    top_pct = top.rank(pct=True).mul(100).round(1).fillna(0).tolist()

    brief = [escape(str(b)) for b in df["embed_text"].fillna("")]

    activity_disp = [_dim_cell(_label(_slug_or(v, allowed["activity_type"]))) for v in df["activity_type"].values]
    type_pill = [_pill(v, _TASK_TYPE_COLORS.get(v, "#bdbdbd")) for v in fmt_vals]

    judged_disp = [
        _dim_cell(_prettify(_slug_or(v, allowed["judging_criterion"], "not_stated")))
        for v in df["judging_criterion"].values
    ]
    twist_disp = [
        _dim_cell(", ".join(_prettify(t) for t in _maybe_json_list(v)) or "—") for v in df["task_twist"].values
    ]
    props_disp = [escape(", ".join(_maybe_json_list(v))) or "—" for v in df["key_props"].values]
    winner_disp = [_dim_cell(w) for w in df["winner"].fillna("").values]

    contestants_list = [_maybe_json_list(v) for v in df["contestants"].values]
    players_disp = [escape(", ".join(cs)) if cs else "—" for cs in contestants_list]
    air_disp = [a.strftime("%d %b %Y") if pd.notna(a) else "—" for a in air]
    wiki_url = [f"https://taskmaster.fandom.com/wiki/{escape(str(s).replace(' ', '_'))}" for s in df["series"]]

    # Search index drives the contestant "filter": typing a contestant name (or a
    # word from the brief/series) highlights matching tasks via DataMapPlot search.
    search_text = [
        f"{b} {s} {' '.join(cs)} {w}"
        for b, s, cs, w in zip(df["embed_text"].fillna(""), df["series"], contestants_list, df["winner"].fillna(""))
    ]

    # Marker size encodes divisiveness (bigger = more divisive); unscored → small.
    spread_filled = spread.fillna(0).values
    denom = (spread_filled.max() - spread_filled.min()) or 1.0
    marker_sizes = 5 + 9 * (spread_filled - spread_filled.min()) / denom

    extra_data = pd.DataFrame(
        {
            "series_ep": series_ep,
            "avg": avg_disp,
            "avg_pct": avg_pct,
            "spread": spread_disp,
            "spread_pct": spread_pct,
            "top": top_disp,
            "top_pct": top_pct,
            "brief": brief,
            "type_pill": type_pill,
            "activity": activity_disp,
            "judged": judged_disp,
            "winner": winner_disp,
            "twist": twist_disp,
            "props": props_disp,
            "air_date": air_disp,
            "players": players_disp,
            "wiki_url": wiki_url,
        }
    )

    # ── region labels (finest first) ────────────────────────────────────────
    label_cols = sorted(c for c in df.columns if c.startswith("label_layer_"))
    label_layers = [df[c].fillna("Unlabelled").astype(str).values for c in label_cols]

    # ── colormaps ───────────────────────────────────────────────────────────
    def _cat(field, desc, values):
        return {
            "field": field,
            "description": desc,
            "kind": "categorical",
            "color_mapping": _color_mapping(values),
            "show_legend": True,
        }

    rawdata = [
        fmt_vals,
        era_vals,
        activity_vals,
        judged_vals,
        twist_vals,
        avg.fillna(avg.min()).values,
        spread.fillna(0).values,
        top.fillna(top.min()).values,
        air_num,
    ]
    metadata = [
        {
            "field": "format",
            "description": "Task type",
            "kind": "categorical",
            "color_mapping": _TASK_TYPE_COLORS,
            "show_legend": True,
        },
        _cat("era", "Channel era", era_vals),
        _cat("activity", "Activity (LLM)", activity_vals),
        _cat("judged", "Judged on (LLM)", judged_vals),
        _cat("twist", "Has a twist (LLM)", twist_vals),
        {"field": "avg", "description": "Average score", "kind": "continuous", "cmap": "YlOrRd"},
        {"field": "spread", "description": "Divisiveness (score spread)", "kind": "continuous", "cmap": "magma"},
        {"field": "top", "description": "Top score", "kind": "continuous", "cmap": "viridis"},
        {"field": "air", "description": "Air date", "kind": "continuous", "cmap": "cividis"},
    ]

    plot = datamapplot.create_interactive_plot(
        coords,
        *label_layers,
        hover_text=search_text,
        hover_text_html_template=HOVER_TEMPLATE,
        extra_point_data=extra_data,
        on_click="window.open(`{wiki_url}`, `_blank`)",
        marker_size_array=marker_sizes,
        title=PROJECT_TITLE,
        sub_title=f"{n:,} tasks from Taskmaster (UK), placed by what each one asks you to do",
        enable_search=True,
        font_family="IBM Plex Sans",
        colormap_rawdata=rawdata,
        colormap_metadata=metadata,
        custom_js=CUSTOM_JS,
        custom_css=CUSTOM_CSS,
    )
    plot.save(str(MAP_HTML))
    _inject_attribution(MAP_HTML)
    print(f"Wrote {MAP_HTML} ({MAP_HTML.stat().st_size / 1e6:.1f} MB)")

    DOCS_HTML.write_text(MAP_HTML.read_text())
    print(f"Wrote {DOCS_HTML}")


if __name__ == "__main__":
    main()
