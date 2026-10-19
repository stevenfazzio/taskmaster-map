# CLAUDE.md — Taskmaster Task Map

Agent notes for this repo. Read `README.md` first for the project overview, the
pipeline command list, and data/attribution; this file adds the conventions,
gotchas, and operational detail that aren't obvious from the code.

## What it is

An interactive 2D semantic map of every UK Taskmaster task — **one dot = one
task**, laid out by what each task asks you to do. Spine: fetch → embed (Cohere) →
UMAP → label (Toponymy + Opus) → LLM field-extract (Opus) → render (DataMapPlot) →
GitHub Pages. Sibling of `../jeopardy-map` (the lean spine) and
`../huggingface-dataset-map` (LLM extraction + rich hover card). This map
deliberately has **no custom filter UI** — it stays on DataMapPlot's public API
(see "Feature set" below for why).

## Commands

- `make install` (`uv sync --extra dev`), `make lint` (`ruff check . && ruff format --check .`), `make format`, `make test`.
- Run a stage: `uv run python pipeline/0X_*.py` (ordered 00→06; each reads the prior stage's parquet/npz — see README).
- **Re-render only** (the common iterate step after editing the viz/hover): `uv run python pipeline/06_visualize.py` → writes `data/task_map.html` + `docs/index.html`. Does NOT re-run the LLM stages.
- `05_extract_fields.py` runs **two** per-task LLM passes, each with its own cache so tuning one doesn't disturb the other: `data/structured_fields_cache/` (taxonomy JSON) and `data/emoji_cache/` (emoji gist `.txt`). (`data/motif_cache/` on disk is a leftover from the removed motif filter; nothing reads it.) Takes `--limit N` (random sample) and `--aggregate-only`. Only successes are cached, so reruns retry failures; delete a cache dir to force re-extraction (changing a prompt does NOT auto-invalidate — the prompt isn't in the cache key).

## Conventions

- `task_id` is the alignment key across **every** stage — always merge/reindex on it (`06` does `set_index("task_id").reindex(order)` so row i ↔ `coords[i]`).
- **Corpus is 990 tasks, not the ~1,045 raw rows.** Upstream duplicates the single New Year Treat special as six identical series (`New Year Treat 1`..`6`); `01`'s `_dedupe_identical_series` collapses any byte-identical series to one, keeping genuine same-brief recurrences like Series 2 & 3 "Buy a gift for the Taskmaster" (different casts). `05`'s `aggregate` is corpus-scoped so dropped tasks don't linger as orphaned cache rows.
- **Atomic writes everywhere**: tempfile in the same dir → verify → `os.replace`. Mirror this for any new output.
- API keys come from **environment variables**, not `.env` files (`config.py` only loads `PROJECT_ROOT/.env` if it happens to exist). Don't recreate `~/.config/data-apis/.env`; never read/echo secrets.
- Plain `.py` scripts, HTML visual output, `uv`/`ruff` (line-length 120, rules E/F/I) — see the global `~/.claude/CLAUDE.md`.

## External APIs

- **Cohere** `embed-v4.0`, `input_type="clustering"`, `output_dimension=1024`, `cohere.ClientV2`. Re-embeds when an embedding-signature in the npz changes.
- **Anthropic Opus 4.8** (`claude-opus-4-8`). It **rejects `temperature`/`top_p`/`top_k` and `budget_tokens` with a 400** — adaptive thinking only. Toponymy's `AnthropicNamer` passes `temperature`, so `04` subclasses it (`OpusAnthropicNamer`) to drop it. Cost: full pipeline ≈ $7–8; stage 05 taxonomy ≈ $5, emoji pass ≈ $2.

## DataMapPlot gotchas (most of these cost real debugging)

- **Point/hover data is encoded into the HTML, not greppable.** Verify rendered output at runtime via `datamap.metaData.<field>` in the browser, not by grepping `docs/index.html`. (CSS/JS strings like `scrollZoom` or the attribution footer ARE greppable — use them as deploy markers.)
- **Native search is whole-query, case-insensitive substring match with a "0 matches → show ALL" fallback** — NOT token-AND. The index is `06`'s `search_text` (brief, series, contestants, winner, episode title, twist labels); anything added there is matched as a raw substring, so short category words pollute ordinary queries.

## Toponymy (stage 04)

Place-naming, not document classification. `Unlabelled` = "unnamed region at this scale" — signal, keep those rows. `topic_names_[0]` is the FINEST layer, `[-1]` the coarsest. `experiments/derive_taxonomy.py` is an off-pipeline EVoC+Toponymy pass that discovered/validated `pipeline/taxonomy.json` from open-ended Opus characterisations; EVoC does its own dim-reduction, so pass full embeddings to both slots. Note the `CompatEVoCClusterer` shim for the evoc 0.3.1 API.

## Local preview & deploy

- **Preview:** serve `data/` over HTTP and open the `http://127.0.0.1` URL — never `file://` (the Chrome extension rewrites it to `https://file://` and fails). `python3 -m http.server <port> --bind 127.0.0.1 --directory data` then `http://127.0.0.1:<port>/task_map.html`.
- **Deploy:** GitHub Pages serves `/docs` on `main`; `06` writes `docs/index.html`, so a commit + `git push` redeploys (~15s). Live: <https://stevenfazzio.github.io/taskmaster-map/>. Verify with a greppable-marker poll (`curl … | grep <marker>`). Data is CC BY-SA 4.0 → keep the attribution footer (`ATTRIBUTION_HTML` in `06`). Commit/push only when asked.

## Feature set (in the code)

Hover card with an Opus-generated **emoji gist** (stage 05) above the brief; 4
colormaps (Task type, Activity, Judged on, Air date) whose legends select by
category; native text search over the brief, contestants, episode title and
**twist labels** (so "teamwork", "time limit", "body restriction" find those
tasks — twist is multi-valued, so it gets search rather than a colormap).

**Removed 2026-10-18, on purpose:** a 16-glyph motif filter strip (LLM `motif_tags`
pass + `motifs.py`), a live "N tasks" caption, and a blank-on-zero-matches patch.
They depended on DataMapPlot internals (`dataSelectionManager`, a monkey-patched
`highlightPoints`, `pointLayer.clone`) and were the map's main breakage risk.
Search can't fully replace the motifs (briefs are short — most motifs were only
20–40% findable by an obvious keyword), and that was accepted. Don't re-add custom
filter UI here without discussing it; the pattern lives in
`../huggingface-dataset-map/pipeline/filter_panel.html`.

**Deferred:** rewrite the title subtitle; the long-deferred extras (contestant
dropdown, random-task button, shareable deep-links, methodology page).
