"""Prepare tasks for embedding: clean the brief, build embed_text, derive fields.

One row per task (1 task = 1 node). embed_text is the task brief ONLY (pure task
semantics, so the layout groups by what the task asks rather than by series or
contestant). Everything else is layout-free metadata for the stage-06 colormaps,
contestant filter, and hover card — none of it touches embed_text, so changing a
derived field is cheap: re-run 01 then 06, no re-embedding.

Derived per-task fields:
  embed_text       whitespace-normalized description (the brief)
  task_format      Prize / Live (studio) / Tiebreak / Filmed  (from the flags)
  air_date         from the episodes table, joined on (series, episode_num)
  channel_era      Dave vs Channel 4, split at the Series 9 premiere (data-driven)
  series_num       int parsed from "Series N" (NaN for specials)
  is_special       True for Champion of Champions / New Year Treat
  avg_score        mean awarded score across contestants on the task
  top_score        max awarded score
  score_spread     top - min  (divisiveness: a 5-vs-1 task scores high here)
  n_contestants    number of contestants on the task
  winner           contestant(s) with the top score ("/"-joined on ties)
  contestants      list of all contestants on the task (drives the stage-06 filter)

Input:  data/{tasks,scores,episodes}_raw.parquet
Output: data/task_rows.parquet
"""

from __future__ import annotations

import os
import re
import tempfile

import numpy as np
import pandas as pd
from config import (
    MAX_TASKS,
    SOURCE_CSVS,
    SUBSET_SEED,
    TASK_ROWS_PARQUET,
)

_WS = re.compile(r"\s+")
_SERIES_NUM = re.compile(r"^Series (\d+)$")


def _norm(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.replace(_WS, " ", regex=True).str.strip()


def _task_format(row) -> str:
    """One categorical label per task, most-specific-wins."""
    if row["is_tiebreak"]:
        return "Tiebreak"
    if row["is_prize"]:
        return "Prize"
    if row["is_live"]:
        return "Live (studio)"
    return "Filmed"


def main():
    tasks = pd.read_parquet(SOURCE_CSVS["tasks"])
    scores = pd.read_parquet(SOURCE_CSVS["scores_long"])
    episodes = pd.read_parquet(SOURCE_CSVS["episodes"])
    print(f"tasks: {len(tasks):,}  scores: {len(scores):,}  episodes: {len(episodes):,}")

    df = tasks.copy()
    for c in ("is_prize", "is_live", "is_tiebreak"):
        df[c] = df[c].fillna(0).astype(int).astype(bool)

    # embed_text = the brief. Drop the tiny fraction with no description (nothing to embed).
    df["description"] = _norm(df["description"])
    blank = df["description"].eq("")
    if blank.any():
        print(f"  dropping {int(blank.sum())} tasks with a blank description")
        df = df.loc[~blank].reset_index(drop=True)
    df["embed_text"] = df["description"]

    # task_format (4-way categorical) + overlap diagnostics.
    overlap = df[["is_prize", "is_live", "is_tiebreak"]].astype(int)
    print(
        "  flag overlaps: "
        f"prize&live={int((overlap.is_prize & overlap.is_live).sum())} "
        f"prize&tiebreak={int((overlap.is_prize & overlap.is_tiebreak).sum())} "
        f"live&tiebreak={int((overlap.is_live & overlap.is_tiebreak).sum())}"
    )
    df["task_format"] = df.apply(_task_format, axis=1)

    # series_num + is_special.
    df["series_num"] = df["series"].map(lambda s: int(m.group(1)) if (m := _SERIES_NUM.match(str(s))) else np.nan)
    df["is_special"] = df["series_num"].isna()

    # air_date joined from episodes (UK day-first dates like "28 July 2015").
    episodes = episodes.copy()
    episodes["air_date"] = pd.to_datetime(episodes["air_date"], errors="coerce", dayfirst=True)
    ep_dates = episodes[["series", "episode_num", "air_date"]].drop_duplicates(["series", "episode_num"])
    df = df.merge(ep_dates, on=["series", "episode_num"], how="left")
    n_no_date = int(df["air_date"].isna().sum())
    if n_no_date:
        print(f"  WARNING: {n_no_date} tasks have no joined air_date")

    # channel_era: Taskmaster moved Dave -> Channel 4 at Series 9. Derive the split
    # date from the data (Series 9's first episode) so specials fall on the right side.
    s9 = episodes.loc[episodes["series"] == "Series 9", "air_date"]
    s9_start = s9.min()
    if pd.isna(s9_start):
        print("  WARNING: no Series 9 air_date; channel_era left Unknown")
        df["channel_era"] = "Unknown"
    else:
        df["channel_era"] = np.where(df["air_date"] < s9_start, "Dave", "Channel 4")
        df.loc[df["air_date"].isna(), "channel_era"] = "Unknown"
        print(f"  Series 9 premiere (Dave/C4 split): {s9_start.date()}")

    # Score-derived fields, per task_id, from the long scores table.
    scores = scores.copy()
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    scored = scores.dropna(subset=["score"])

    agg = scored.groupby("task_id")["score"].agg(avg_score="mean", top_score="max", min_score="min")
    agg["score_spread"] = agg["top_score"] - agg["min_score"]
    agg["avg_score"] = agg["avg_score"].round(2)

    contestants = (
        scores.groupby("task_id")["contestant"]
        .apply(lambda s: sorted(x for x in s.dropna().unique()))
        .rename("contestants")
    )
    # winner: contestant(s) achieving the per-task max score.
    top = scored.assign(_max=scored.groupby("task_id")["score"].transform("max"))
    winners = (
        top[top["score"] == top["_max"]]
        .groupby("task_id")["contestant"]
        .apply(lambda s: " / ".join(sorted(s.dropna().unique())))
        .rename("winner")
    )

    df = df.merge(agg[["avg_score", "top_score", "score_spread"]], on="task_id", how="left")
    df = df.merge(contestants, on="task_id", how="left")
    df = df.merge(winners, on="task_id", how="left")
    df["contestants"] = df["contestants"].apply(lambda v: v if isinstance(v, list) else [])
    df["n_contestants"] = df["contestants"].map(len)
    df["winner"] = df["winner"].fillna("")

    if MAX_TASKS is not None and len(df) > MAX_TASKS:
        df = df.sample(n=MAX_TASKS, random_state=SUBSET_SEED).reset_index(drop=True)
        print(f"  MAX_TASKS={MAX_TASKS}: subsampled to {len(df):,}")

    # Diagnostics.
    print(f"\nFinal: {len(df):,} tasks across {df['series'].nunique()} series")
    print(f"  task_format: {df['task_format'].value_counts().to_dict()}")
    print(f"  channel_era: {df['channel_era'].value_counts().to_dict()}")
    print(f"  specials: {int(df['is_special'].sum())} tasks; main-series: {int((~df['is_special']).sum())}")
    if df["air_date"].notna().any():
        print(f"  air_date range: {df['air_date'].min().date()} -> {df['air_date'].max().date()}")
    print(
        f"  scores: avg in [{df['avg_score'].min():.1f}, {df['avg_score'].max():.1f}]; "
        f"spread in [{df['score_spread'].min():.0f}, {df['score_spread'].max():.0f}]; "
        f"{int(df['avg_score'].isna().sum())} tasks unscored"
    )
    print(f"  contestants: {sorted({c for cs in df['contestants'] for c in cs}).__len__()} distinct")
    print(f"  brief length (words): median {int(df['embed_text'].str.split().map(len).median())}")

    out = TASK_ROWS_PARQUET
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(out), suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        verify = pd.read_parquet(tmp_path)
        assert len(verify) == len(df), f"row count mismatch: {len(verify)} vs {len(df)}"
        os.replace(tmp_path, out)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    print(f"\nWrote {out} ({out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
