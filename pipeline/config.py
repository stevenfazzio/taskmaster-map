"""Central config for the taskmaster-map pipeline. Every stage does `from config
import ...` (the stage's own dir is on sys.path when run as `python pipeline/XX.py`,
so it's a bare import, not `from pipeline.config`). Edit constants here for smoke
tests rather than adding CLI args."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
# Experiment runs: TASKMAP_DATA_DIR redirects every stage's inputs/outputs to another
# folder (seed it with the upstream files the stage reads), and keeps stage 06 from
# touching docs/ — the deployed page is only ever written by a default run.
_RUN_DIR = os.environ.get("TASKMAP_DATA_DIR")
if _RUN_DIR:
    DATA_DIR = Path(_RUN_DIR).resolve()
    DOCS_DIR = DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

# API keys come from the environment (the user exports them in their shell). A
# local .env can supplement but does NOT override already-set env vars.
load_dotenv(PROJECT_ROOT / ".env")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CO_API_KEY = os.environ.get("CO_API_KEY")  # legacy: experiments/ only (see bottom of file)

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

# --- Stage 00b: series upstream doesn't have yet, scraped from the Fandom wiki ---
# Same CC BY-SA source upstream compiled from, emitted in upstream's table conventions
# (see 00b's docstring). (series name, numeric code): code feeds the deterministic
# task_id = code*10000 + episode*100 + row, disjoint from upstream's ids. Add new series
# here (e.g. ("New Year Treat 7", 9007)) — existing ids don't move.
WIKI_SERIES = [("Series 21", 21), ("Series 22", 22)]
WIKI_VALIDATION_SERIES = ("Series 20", 20)  # `00b --validate` re-scrapes this and diffs vs upstream
WIKI_API = "https://taskmaster.fandom.com/api.php"
WIKI_USER_AGENT = "taskmaster-map/0.1 (personal data-visualisation project)"
WIKI_REQUEST_PAUSE = 0.5  # seconds between uncached requests
WIKI_PINS_JSON = Path(__file__).resolve().parent / "wiki_pins.json"  # page -> revision id (checked in)
WIKI_CACHE_DIR = DATA_DIR / "wiki_cache"
WIKI_RAW = {
    "tasks": DATA_DIR / "wiki_tasks_raw.parquet",
    "scores_long": DATA_DIR / "wiki_scores_raw.parquet",
    "episodes": DATA_DIR / "wiki_episodes_raw.parquet",
    "series": DATA_DIR / "wiki_series_raw.parquet",
}

# --- Stage 01: prepare ---
# One row per task. embed_text = the task brief (description) only — pure task
# semantics, so the layout groups by what the task asks, not by series/contestant.
TASK_ROWS_PARQUET = DATA_DIR / "task_rows.parquet"
# Smoke-test knob: cap to a random subset for a fast dry run. None = all tasks.
MAX_TASKS = None
SUBSET_SEED = 42

# --- Stage 02: embed tasks (Qwen3-Embedding-8B via OpenRouter) ---
# Qwen3-Embedding is instruction-aware ("Instruct: ...\nQuery: <text>"); the instruction
# plays the role Cohere's input_type="clustering" did — steer the space toward what the
# downstream grouping should key on (the task's ask, not its wording). Changing the
# instruction, model, or dim changes the signature and forces a re-embed.
# Toponymy's keyphrase embedder (stage 04) uses the same model with NO instruction: it
# embeds candidate names in their own space and never dots them against task vectors.
EMBED_MODEL = "qwen/qwen3-embedding-8b"
EMBED_DIM = 1024  # Matryoshka truncation (native 4096); OpenRouter honours `dimensions`, returns unit-norm.
EMBED_INSTRUCTION = "Identify what the contestant is being asked to do in this Taskmaster task brief"
EMBED_BATCH = 64
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
# Opus 5.5: thinking is always on (can't be disabled) and thinking tokens count against
# max_tokens, so effort is the only cost/depth knob — and its API default is "medium",
# so set it explicitly. Toponymy passes max_tokens=2048 for naming; the namer raises it
# to at least NAMING_MIN_MAX_TOKENS so thinking can't starve the JSON answer.
# Env overrides (experiments): TASKMAP_NAMING_MODEL, and TASKMAP_NAMING_EFFORT="none" for
# the legacy Opus 4.8 call (no effort, no thinking, Toponymy's own max_tokens).
ANTHROPIC_MODEL_NAMING = os.environ.get("TASKMAP_NAMING_MODEL", "claude-opus-5-5")
NAMING_EFFORT = os.environ.get("TASKMAP_NAMING_EFFORT", "medium")
if NAMING_EFFORT.lower() == "none":
    NAMING_EFFORT = None
NAMING_MIN_MAX_TOKENS = 16000 if NAMING_EFFORT else 0
ANTHROPIC_MAX_CONCURRENCY = 12  # Toponymy fires naming calls concurrently via a Semaphore
TOPONYMY_LABELS_PARQUET = DATA_DIR / "toponymy_labels.parquet"

# --- Stage 05: per-task structured field extraction (Opus) ---
# Each task brief -> {value, quote} per taxonomy field. Constrained classification,
# so effort=low keeps thinking (always on for Opus 5.5) cheap; bump if quality lags.
ANTHROPIC_MODEL_EXTRACT = "claude-opus-5-5"
EXTRACT_CONCURRENCY = 8
EXTRACT_EFFORT = "low"
EXTRACT_MAX_TOKENS = 8000  # ceiling only; covers thinking tokens + the JSON
EXTRACT_MAX_RETRIES = 4
TAXONOMY_JSON = Path(__file__).resolve().parent / "taxonomy.json"
STRUCTURED_FIELDS_PARQUET = DATA_DIR / "structured_fields.parquet"
STRUCTURED_FIELDS_CACHE_DIR = DATA_DIR / "structured_fields_cache"  # per-task JSON, resumable

# Emoji summary: a separate, single-purpose pass (its own prompt + cache) so the
# tuned wit survives untouched and rerunning 05 regenerates only the emoji.
EMOJI_CACHE_DIR = DATA_DIR / "emoji_cache"  # per-task .txt, resumable
EMOJI_EFFORT = "low"
EMOJI_MAX_TOKENS = 4000  # ceiling only; output is ≤4 emoji, but thinking tokens count against it


# --- Stage 06: DataMapPlot visualization ---
MAP_HTML = DATA_DIR / "task_map.html"
DOCS_HTML = DOCS_DIR / "index.html"

# --- Legacy: experiments/ only ---
# The off-pipeline experiments (derive_taxonomy, diagnose_disambiguation) were run on
# Cohere embed-v4.0; kept so they still import and reproduce. The pipeline doesn't use these.
COHERE_EMBED_MODEL = "embed-v4.0"
COHERE_INPUT_TYPE = "clustering"
COHERE_OUTPUT_DIM = 1024
