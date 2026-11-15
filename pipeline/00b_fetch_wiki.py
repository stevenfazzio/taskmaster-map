"""Scrape series the upstream dataset doesn't have yet from the Taskmaster Fandom wiki.

Upstream (stage 00, silverdavi/taskmaster-uk-scores) stopped at New Year Treat 6
(Jan 2026). This stage reads the same source it was compiled from — the Taskmaster
Fandom wiki (CC BY-SA 4.0) — for the series listed in WIKI_SERIES, and emits the
SAME four raw tables in upstream's conventions, so stage 01 just concatenates them:

  * `Prize:` / `Live:` prefixes become is_prize / is_live and are stripped;
    `Team:` (and `Team Live:`) stay in the brief, as upstream does.
  * task_num is the wiki's label (1..n, `S` series-long, `T` tiebreak, `2a`...).
  * every (task, contestant) gets a score row; `-` (didn't take part) and DQ are NaN,
    ✔/✘ are 1/0. A multi-row task keeps its first sub-row's brief (upstream's rule)
    but SUMS scores over its sub-rows (deliberately NOT upstream's first-row rule,
    which loses bonus points and zeroes branch tasks; see parse_episode).
  * task_id = series_code*10000 + episode*100 + row-in-episode: deterministic and
    disjoint from upstream's 1..~1045, so stage-05 caches stay keyed correctly.

Wiki pages are read through the MediaWiki API (action=parse, wikitext) and pinned
by revision id in pipeline/wiki_pins.json (like upstream's SOURCE_COMMIT): a pinned
page is fetched at that revision; an unpinned one at its latest revision, which is
then added to the pins. Raw API responses are cached in data/wiki_cache/.

  uv run python pipeline/00b_fetch_wiki.py            # scrape WIKI_SERIES -> data/wiki_*_raw.parquet
  uv run python pipeline/00b_fetch_wiki.py --validate ["Series 19"]  # re-scrape an upstream series, diff

Output: data/wiki_{tasks,scores,episodes,series}_raw.parquet
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from config import (
    SOURCE_CSVS,
    WIKI_API,
    WIKI_CACHE_DIR,
    WIKI_PINS_JSON,
    WIKI_RAW,
    WIKI_REQUEST_PAUSE,
    WIKI_SERIES,
    WIKI_USER_AGENT,
    WIKI_VALIDATION_SERIES,
)

# ── fetching ────────────────────────────────────────────────────────────────


def _load_pins() -> dict[str, int]:
    return json.loads(WIKI_PINS_JSON.read_text()) if WIKI_PINS_JSON.exists() else {}


def _save_pins(pins: dict[str, int]) -> None:
    tmp = WIKI_PINS_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(sorted(pins.items())), indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, WIKI_PINS_JSON)


def _cache_path(page: str, revid: int) -> Path:
    return WIKI_CACHE_DIR / f"{re.sub(r'[^A-Za-z0-9._-]', '_', page)}__{revid}.json"


def fetch_wikitext(page: str, pins: dict[str, int]) -> str:
    """Wikitext of `page` at its pinned revision (or latest, which then gets pinned)."""
    WIKI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pinned = pins.get(page)
    if pinned and _cache_path(page, pinned).exists():
        return json.loads(_cache_path(page, pinned).read_text())["parse"]["wikitext"]
    params = {"action": "parse", "prop": "wikitext|revid", "format": "json", "formatversion": "2"}
    params.update({"oldid": pinned} if pinned else {"page": page, "redirects": "1"})
    for attempt in range(4):
        try:
            r = requests.get(WIKI_API, params=params, headers={"User-Agent": WIKI_USER_AGENT}, timeout=60)
            r.raise_for_status()
            data = r.json()
            if "parse" not in data:
                raise RuntimeError(f"{page}: {str(data)[:200]}")
            break
        except Exception as e:  # noqa: BLE001 — transient network/API errors
            if attempt == 3:
                raise
            print(f"  fetch {page!r} failed ({e}); retrying")
            time.sleep(2 ** (attempt + 1))
    revid = int(data["parse"]["revid"])
    if pinned and revid != pinned:
        raise RuntimeError(f"{page}: asked for revision {pinned}, got {revid}")
    _cache_path(page, revid).write_text(json.dumps(data, ensure_ascii=False))
    pins[page] = revid
    time.sleep(WIKI_REQUEST_PAUSE)
    return data["parse"]["wikitext"]


# ── wikitext helpers ────────────────────────────────────────────────────────


def split_top_level(s: str, sep: str) -> list[str]:
    """Split on `sep` only outside [[...]] and {{...}}."""
    out, depth_l, depth_t, buf, i = [], 0, 0, [], 0
    while i < len(s):
        two = s[i : i + 2]
        if two == "[[":
            depth_l += 1
            buf.append(two)
            i += 2
            continue
        if two == "]]" and depth_l:
            depth_l -= 1
            buf.append(two)
            i += 2
            continue
        if two == "{{":
            depth_t += 1
            buf.append(two)
            i += 2
            continue
        if two == "}}" and depth_t:
            depth_t -= 1
            buf.append(two)
            i += 2
            continue
        if s.startswith(sep, i) and not depth_l and not depth_t:
            out.append("".join(buf))
            buf = []
            i += len(sep)
            continue
        buf.append(s[i])
        i += 1
    out.append("".join(buf))
    return out


_ATTR = re.compile(r'^\s*(?:[a-z-]+\s*=\s*("[^"]*"|\S+)\s*)+$', re.I)


def cell_attrs_and_text(cell: str) -> tuple[dict[str, str], str]:
    """`colspan="2" | 3` -> ({'colspan': '2'}, '3'); `3` -> ({}, '3')."""
    parts = split_top_level(cell, "|")
    if len(parts) >= 2 and _ATTR.match(parts[0]):
        attrs = dict((k.lower(), v.strip('"')) for k, v in re.findall(r'([a-z-]+)\s*=\s*("[^"]*"|\S+)', parts[0], re.I))
        return attrs, "|".join(parts[1:])
    return {}, cell


_UNKNOWN_TEMPLATES: set[str] = set()


def _template(m: re.Match) -> str:
    parts = split_top_level(m.group(1), "|")
    name = parts[0].strip().lower()
    args = [p for p in parts[1:] if "=" not in p.split("|")[0] or p.strip().startswith("http")]
    if name in ("wp", "w", "color", "colour", "nowrap", "small", "big", "sic"):
        return args[-1] if args else ""
    if name in ("tooltip", "abbr"):
        return args[0] if args else ""
    _UNKNOWN_TEMPLATES.add(name)
    return args[-1] if args else ""


def clean_markup(s: str) -> str:
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.S)
    s = re.sub(r"</?(onlyinclude|includeonly|noinclude)>", "", s)
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"</?(small|big|span|sup|sub|u|i|b|s)[^>]*>", "", s, flags=re.I)
    for _ in range(3):  # templates may nest
        s = re.sub(r"\{\{((?:[^{}]|\{(?!\{)|\}(?!\}))*)\}\}", _template, s)
    s = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", s)
    s = s.replace("'''", "").replace("''", "")
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


# ── table parsing ───────────────────────────────────────────────────────────


def table_rows(wikitext: str) -> list[tuple[str, list[str]]]:
    """[(row_class, [cell, ...]), ...] for the first `tmtable` in the page."""
    text = re.sub(r"</?(onlyinclude|includeonly|noinclude)>", "\n", wikitext)
    start = text.find('{| class="tmtable"')
    if start < 0:
        raise ValueError("no tmtable found")
    end = text.find("\n|}", start)
    body = text[start:end]
    rows: list[tuple[str, list[str]]] = []
    cls, cells = None, []
    for line in body.split("\n")[1:]:
        if line.startswith("|-"):
            if cls is not None:
                rows.append((cls, cells))
            m = re.search(r'class="([^"]+)"', line)
            cls, cells = (m.group(1) if m else ""), []
        elif line.startswith("!") or line.startswith("|"):
            sep = "!!" if line.startswith("!") else "||"
            cells.extend(split_top_level(line[1:], sep))
        elif cells:  # continuation of the previous cell
            cells[-1] += " " + line
    if cls is not None:
        rows.append((cls, cells))
    return rows


_ODD_SCORES: list[str] = []
_SCORE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(raw: str) -> float:
    t = clean_markup(raw)
    if t in ("", "-", "–", "—"):
        return np.nan
    if t.upper().startswith("DQ"):  # upstream records disqualification as no score
        return np.nan
    if t in ("✔", "✓"):  # pass/fail tiebreaks: upstream scores a tick 1, a cross 0
        return 1.0
    if t in ("✘", "✗", "×"):
        return 0.0
    m = _SCORE.search(t)
    if not m:  # e.g. a ✔ tick: no points. Recorded and reported, scored NaN.
        _ODD_SCORES.append(t)
        return np.nan
    return float(m.group(0))


def expand_grid(rows: list[list[str]], ncols: int) -> list[list[tuple[str, bool]]]:
    """Resolve rowspan/colspan into a full grid of (cell_text, carried_from_row_above)."""
    pending: dict[int, list] = {}  # col -> [rows_remaining, text]
    grid = []
    for cells in rows:
        it = iter(cells)
        out: list[tuple[str, bool]] = []
        col = 0
        while col < ncols:
            if col in pending:
                left, txt = pending[col]
                out.append((txt, True))
                if left <= 1:
                    del pending[col]
                else:
                    pending[col][0] -= 1
                col += 1
                continue
            try:
                attrs, txt = cell_attrs_and_text(next(it))
            except StopIteration:
                raise ValueError(f"row has too few cells: {cells!r}") from None
            span_r, span_c = int(attrs.get("rowspan", 1)), int(attrs.get("colspan", 1))
            for _ in range(span_c):
                out.append((txt, False))
                if span_r > 1:
                    pending[col] = [span_r - 1, txt]
                col += 1
        if next(it, None) is not None:
            raise ValueError(f"row has too many cells: {cells!r}")
        grid.append(out)
    return grid


def parse_episode(wikitext: str) -> tuple[list[str], list[dict]]:
    """-> (contestants in column order, [task dict with 'scores' list aligned to them]).

    Multi-part tasks are several table rows sharing one rowspan'd task-number cell
    (a prep row, branch rows, a bonus row). One task per task number; its brief is
    the FIRST sub-row's (upstream's convention), but each contestant's score is the
    SUM over sub-rows — upstream keeps only the first row's scores, which zeroes out
    branch tasks (S22E2 #4) and drops bonus points. A merged (rowspan) score cell
    counts once. Summing reproduces the wiki's own episode totals.
    """
    rows = table_rows(wikitext)
    header = next(c for k, c in rows if "tmtableheader" in k)
    contestants = [clean_markup(cell_attrs_and_text(c)[1]) for c in header[2:]]
    ncols = 2 + len(contestants)
    grid = expand_grid([cells for k, cells in rows if k == "tmtablerow"], ncols)
    groups: list[list] = []
    for row in grid:
        if row[0][1] and groups:  # task-number cell carried down: a sub-row of the previous task
            groups[-1].append(row)
        else:
            groups.append([row])
    tasks = []
    for g in groups:
        scores = []
        for col in range(2, ncols):
            vals = [parse_score(r[col][0]) for r in g if not r[col][1]]
            vals = [v for v in vals if not np.isnan(v)]
            scores.append(float(sum(vals)) if vals else np.nan)
        tasks.append({"task_num": clean_markup(g[0][0][0]), "desc_raw": g[0][1][0], "scores": scores, "n_rows": len(g)})
    return contestants, tasks


_PREFIX = re.compile(r"^\s*'''\s*([A-Za-z ]+?)\s*:\s*'''\s*|^\s*'''\s*([A-Za-z ]+?)\s*'''\s*:\s*")


def brief_and_flags(desc_raw: str, task_num: str) -> tuple[str, dict[str, int]]:
    """Upstream convention: strip `Prize:` / `Live:` into flags; keep `Team:` / `Team Live:` text.

    Upstream leaves `Team Live:` tasks with is_live=0 (all 27 of them), so we do too.
    """
    flags = {"is_prize": 0, "is_live": 0, "is_tiebreak": int(task_num.upper().startswith("T"))}
    m = _PREFIX.match(desc_raw)
    label = (m.group(1) or m.group(2)).strip().lower() if m else ""
    if label in ("prize", "live", "tiebreak", "tiebreaker"):
        desc_raw = desc_raw[m.end() :]
        flags["is_prize"] = int(label == "prize")
        flags["is_live"] = int(label == "live")
        flags["is_tiebreak"] |= int(label.startswith("tiebreak"))
    return clean_markup(desc_raw), flags


def parse_series_page(wikitext: str) -> tuple[dict, list[dict]]:
    """-> (infobox fields, [{'episode_num', 'episode_title', 'air_date'}...])."""
    box = {}
    m = re.search(r"\{\{Series infobox(.*?)\n\}\}", wikitext, re.S)
    if m:
        for line in m.group(1).split("\n"):
            if "=" in line:
                k, v = line.lstrip("| ").split("=", 1)
                box[k.strip()] = clean_markup(v)
    eps = []
    for m in re.finditer(
        r"Episode (\d+):.*?\(([^)]+)\)\s*\n\{\{:(.+?)\}\}",
        wikitext,
    ):
        eps.append({"episode_num": int(m.group(1)), "air_date": m.group(2).strip(), "episode_title": m.group(3)})
    return box, eps


# ── assembly ────────────────────────────────────────────────────────────────


def scrape_series(series: str, code: int, pins: dict[str, int]) -> dict[str, pd.DataFrame]:
    box, eps = parse_series_page(fetch_wikitext(series.replace(" ", "_"), pins))
    if not eps:
        raise ValueError(f"{series}: no episodes found on the series page")
    t_rows, s_rows, e_rows = [], [], []
    for ep in eps:
        contestants, tasks = parse_episode(fetch_wikitext(ep["episode_title"].replace(" ", "_"), pins))
        totals = [0.0] * len(contestants)
        for i, t in enumerate(tasks, start=1):
            brief, flags = brief_and_flags(t["desc_raw"], t["task_num"])
            task_id = code * 10000 + ep["episode_num"] * 100 + i
            base = {
                "task_id": task_id,
                "series": series,
                "episode_num": ep["episode_num"],
                "episode_title": ep["episode_title"],
                "task_num": t["task_num"],
                **flags,
            }
            t_rows.append({**base, "description": brief})
            for j, (c, sc) in enumerate(zip(contestants, t["scores"])):
                s_rows.append({**base, "contestant": c, "score": sc})
                if not flags["is_tiebreak"]:  # the wiki's episode totals exclude tiebreaks
                    totals[j] += 0 if np.isnan(sc) else sc
        e_rows.append(
            {
                "series": series,
                "episode_num": ep["episode_num"],
                "episode_title": ep["episode_title"],
                "air_date": ep["air_date"],
                "n_contestants": len(contestants),
                "contestants": "; ".join(contestants),
                **{f"total_{j + 1}": int(round(v)) for j, v in enumerate(totals)},
            }
        )
    kind = "series" if series.startswith("Series ") else "special"
    se = pd.DataFrame([{"series": series, "kind": kind, "air": box.get("year"), "episodes": len(eps)}])
    return {
        "tasks": pd.DataFrame(t_rows),
        "scores_long": pd.DataFrame(s_rows),
        "episodes": pd.DataFrame(e_rows),
        "series": se,
    }


def _write_parquet(df: pd.DataFrame, out: Path) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=out.parent, suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        assert len(pd.read_parquet(tmp_path)) == len(df)
        os.replace(tmp_path, out)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def validate(pins: dict[str, int], series: str, code: int) -> None:
    """Scrape a series upstream already has and diff field-by-field."""
    got = scrape_series(series, code, pins)
    up_t = pd.read_parquet(SOURCE_CSVS["tasks"]).query("series == @series")
    up_s = pd.read_parquet(SOURCE_CSVS["scores_long"]).query("series == @series")
    up_e = pd.read_parquet(SOURCE_CSVS["episodes"]).query("series == @series")
    gt = got["tasks"]
    print(f"\n{series}: wiki {len(gt)} tasks vs upstream {len(up_t)}")
    key = ["episode_num", "task_num"]
    m = gt.merge(up_t, on=key, how="outer", suffixes=("_wiki", "_up"), indicator=True)
    only = m[m["_merge"] != "both"]
    if len(only):
        print(f"  {len(only)} tasks on one side only:\n{only[key + ['_merge']].to_string()}")
    both = m[m["_merge"] == "both"]
    for col in ["is_prize", "is_live", "is_tiebreak", "description", "episode_title"]:
        diff = both[both[f"{col}_wiki"] != both[f"{col}_up"]]
        print(f"  {col}: {len(both) - len(diff)}/{len(both)} match")
        for _, r in diff.head(8).iterrows():
            print(
                f"     E{r.episode_num} #{r.task_num}\n"
                f"       wiki: {r[f'{col}_wiki']!r}\n       up:   {r[f'{col}_up']!r}"
            )
    # scores: align by (episode, task_num, contestant)
    gs = got["scores_long"].merge(gt[["task_id", *key]], on=["task_id", *key])
    us = up_s.merge(up_t[["task_id", *key]], on=["task_id", *key])
    sm = gs.merge(us, on=[*key, "contestant"], how="outer", suffixes=("_wiki", "_up"), indicator=True)
    same = (sm["score_wiki"] == sm["score_up"]) | (sm["score_wiki"].isna() & sm["score_up"].isna())
    print(f"  scores: {int(same.sum())}/{len(sm)} match ({int((sm['_merge'] != 'both').sum())} unaligned)")
    for _, r in sm[~same].head(8).iterrows():
        print(f"     E{r.episode_num} #{r.task_num} {r.contestant}: wiki {r.score_wiki} up {r.score_up}")
    ge = got["episodes"].merge(up_e, on="episode_num", suffixes=("_wiki", "_up"))
    for col in ["air_date", "contestants", "episode_title"]:
        print(f"  episodes.{col}: {int((ge[f'{col}_wiki'] == ge[f'{col}_up']).sum())}/{len(ge)} match")
    if _UNKNOWN_TEMPLATES:
        print(f"  unknown templates seen: {sorted(_UNKNOWN_TEMPLATES)}")
    if _ODD_SCORES:
        print(f"  non-numeric score cells (scored NaN): {_ODD_SCORES}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--validate",
        nargs="?",
        const=WIKI_VALIDATION_SERIES[0],
        metavar="SERIES",
        help=f'diff a scrape of an upstream series vs upstream (default "{WIKI_VALIDATION_SERIES[0]}")',
    )
    args = ap.parse_args()
    pins = _load_pins()
    try:
        if args.validate:
            m = re.match(r"^Series (\d+)$", args.validate)
            validate(pins, args.validate, int(m.group(1)) if m else 99)
            return
        parts: dict[str, list[pd.DataFrame]] = {k: [] for k in WIKI_RAW}
        for series, code in WIKI_SERIES:
            got = scrape_series(series, code, pins)
            n_t = len(got["tasks"])
            print(
                f"{series}: {len(got['episodes'])} episodes, {n_t} tasks, "
                f"{int(got['scores_long']['score'].notna().sum())} scores"
            )
            for k in parts:
                parts[k].append(got[k])
        for k, out in WIKI_RAW.items():
            df = pd.concat(parts[k], ignore_index=True)
            _write_parquet(df, out)
            print(f"  wrote {out} ({len(df):,} rows)")
        if _UNKNOWN_TEMPLATES:
            print(f"  unknown templates seen (check briefs): {sorted(_UNKNOWN_TEMPLATES)}")
        if _ODD_SCORES:
            print(f"  non-numeric score cells (scored NaN): {_ODD_SCORES}")
    finally:
        _save_pins(pins)


if __name__ == "__main__":
    main()
