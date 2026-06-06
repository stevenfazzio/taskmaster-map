"""Piece 2 of the LEACE map: render via stage 06, with input/output paths redirected
to data/experiments/ so production data and docs/index.html are NEVER touched.

Run:  uv run python experiments/leace_render.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, "pipeline")

import config

EXP = Path("data/experiments")
# Redirect the four paths stage 06 binds; leave TASK_ROWS / STRUCTURED_FIELDS = production.
config.UMAP_COORDS_NPZ = EXP / "leace_umap_coords.npz"
config.TOPONYMY_LABELS_PARQUET = EXP / "leace_toponymy_labels.parquet"
config.MAP_HTML = EXP / "leace_task_map.html"
config.DOCS_HTML = EXP / "leace_docs_throwaway.html"  # NOT docs/index.html

# Load stage 06 AFTER patching config, so its `from config import ...` picks up the patches.
spec = importlib.util.spec_from_file_location("viz06", "pipeline/06_visualize.py")
viz06 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viz06)
viz06.main()
print(f"\nrendered LEACE map -> {config.MAP_HTML}")
