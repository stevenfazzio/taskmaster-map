"""Download the Taskmaster UK dataset CSVs and write them as parquet.

Source: silverdavi/taskmaster-uk-scores (GitHub), pinned to SOURCE_COMMIT. The
repo is compiled from the Taskmaster Fandom Wiki (CC BY-SA 4.0) and provides
clean per-task briefs + per-contestant scores. Four CSVs:

  tasks        one row per task (task_id, series, episode, flags, description)
  scores_long  one row per (task, contestant) with the awarded score
  episodes     per-episode metadata (air_date, contestant lineup, totals)
  series       per-series metadata (kind, air range, episode count)

Output: data/{tasks,scores,episodes,series}_raw.parquet. Cleaning + the embed
input string are built in stage 01.
"""

import io
import os
import tempfile

import pandas as pd
import requests
from config import SOURCE_CSVS, SOURCE_RAW_BASE


def _fetch_csv(stem: str) -> pd.DataFrame:
    url = f"{SOURCE_RAW_BASE}/{stem}.csv"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def _write_parquet(df: pd.DataFrame, out) -> None:
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


def main():
    for stem, out in SOURCE_CSVS.items():
        print(f"Fetching {stem}.csv ...")
        df = _fetch_csv(stem)
        print(f"  {len(df):,} rows x {len(df.columns)} cols: {list(df.columns)}")
        _write_parquet(df, out)
        print(f"  wrote {out} ({out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
