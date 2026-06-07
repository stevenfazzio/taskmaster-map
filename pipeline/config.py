"""Central config for the taskmaster-map pipeline. Every stage does `from config
import ...` (the stage's own dir is on sys.path when run as `python pipeline/XX.py`,
so it's a bare import, not `from pipeline.config`). Edit constants here for smoke
tests rather than adding CLI args."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# API keys come from the environment (the user exports them in their shell). A
# local .env can supplement but does NOT override already-set env vars.
load_dotenv(PROJECT_ROOT / ".env")
CO_API_KEY = os.environ.get("CO_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

PROJECT_NAME = "Taskmaster Task Map"
PROJECT_TAGLINE = "Every Taskmaster task, laid out by what it asks you to do"

# --- Stage 00: fetch Taskmaster data ---
# silverdavi/taskmaster-uk-scores: wiki-derived CSVs (CC BY-SA 4.0), Series 1-20
# + Champion of Champions + New Year Treat. ~1,045 raw task rows with brief text +
# scores; upstream duplicates the single New Year Treat special as 6 identical series,
# which stage 01 collapses -> 990 tasks (see _dedupe_identical_series there).
# Pinned to a commit so re-fetches are reproducible. The dict key doubles as the
# upstream CSV stem (data/<stem>.csv); the value is our local parquet.
SOURCE_REPO = "silverdavi/taskmaster-uk-scores"
SOURCE_COMMIT = "2842413d327ec9c0e3638a9c124485718f5ab8e2"
SOURCE_RAW_BASE = f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_COMMIT}/data"
SOURCE_CSVS = {
    "tasks": DATA_DIR / "tasks_raw.parquet",
    "scores_long": DATA_DIR / "scores_raw.parquet",
    "episodes": DATA_DIR / "episodes_raw.parquet",
    "series": DATA_DIR / "series_raw.parquet",
}

# --- Stage 01: prepare ---
# One row per task. embed_text = the task brief (description) only — pure task
# semantics, so the layout groups by what the task asks, not by series/contestant.
TASK_ROWS_PARQUET = DATA_DIR / "task_rows.parquet"
# Smoke-test knob: cap to a random subset for a fast dry run. None = all tasks.
MAX_TASKS = None
SUBSET_SEED = 42

# --- Stage 02: embed tasks (Cohere embed-v4.0) ---
# input_type="clustering" because the only downstream use is grouping/visualization
# (UMAP + Toponymy's clusterer). See the sibling jeopardy-map for the full rationale
# (the two embedding spaces are bridged only by cluster membership, never a dot product).
COHERE_EMBED_MODEL = "embed-v4.0"
COHERE_INPUT_TYPE = "clustering"
COHERE_OUTPUT_DIM = 1024  # Matryoshka dim; 256/512/1024/1536 allowed.
EMBED_BATCH = 96  # Cohere embed max texts per call
EMBED_CHECKPOINT_EVERY = 50  # batches between progress checkpoints (mostly moot at ~1k)
TASK_EMB_NPZ = DATA_DIR / "task_embeddings.npz"  # float32 [N x dim] + aligned task_id

# --- Stage 03: UMAP layout ---
# n_neighbors=15 matches the sibling maps; at ~1k points this may over-smooth local
# structure, so it's a likely tuning knob in stage 03. random_state fixed (disables
# UMAP parallelism, fine for a one-shot run).
UMAP_COORDS_NPZ = DATA_DIR / "umap_coords.npz"
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.05
UMAP_RANDOM_STATE = 42

# --- Stage 04: Toponymy region labels (Opus naming) ---
ANTHROPIC_MODEL_NAMING = "claude-opus-4-8"
ANTHROPIC_MAX_CONCURRENCY = 12  # Toponymy fires naming calls concurrently via a Semaphore
TOPONYMY_LABELS_PARQUET = DATA_DIR / "toponymy_labels.parquet"

# --- Stage 05: per-task structured field extraction (Opus) ---
# Each task brief -> {value, quote} per taxonomy field. Constrained classification,
# so adaptive thinking + effort=low keeps cost down (bump effort if sample quality lags).
ANTHROPIC_MODEL_EXTRACT = "claude-opus-4-8"
EXTRACT_CONCURRENCY = 8
EXTRACT_EFFORT = "low"
EXTRACT_MAX_TOKENS = 4000  # ceiling only; covers adaptive-thinking tokens + the JSON
EXTRACT_MAX_RETRIES = 4
TAXONOMY_JSON = Path(__file__).resolve().parent / "taxonomy.json"
STRUCTURED_FIELDS_PARQUET = DATA_DIR / "structured_fields.parquet"
STRUCTURED_FIELDS_CACHE_DIR = DATA_DIR / "structured_fields_cache"  # per-task JSON, resumable

# Emoji summary: a separate, single-purpose pass (its own prompt + cache) so the
# tuned wit survives untouched and rerunning 05 regenerates only the emoji.
EMOJI_CACHE_DIR = DATA_DIR / "emoji_cache"  # per-task .txt, resumable
EMOJI_MAX_TOKENS = 64  # ceiling only; output is ≤4 emoji, headroom for any adaptive thinking

# Motif tags: a third single-purpose pass (its own prompt + cache) classifying which of the
# 16 filter motifs each task involves — decoupled from the witty gist, drives the filter buttons.
MOTIF_CACHE_DIR = DATA_DIR / "motif_cache"  # per-task .txt (space-separated motif keys), resumable
MOTIF_MAX_TOKENS = 128  # ceiling only; output is a short key list, headroom for adaptive thinking

# --- Stage 06: DataMapPlot visualization ---
MAP_HTML = DATA_DIR / "task_map.html"
DOCS_HTML = DOCS_DIR / "index.html"
