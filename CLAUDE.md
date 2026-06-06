# CLAUDE.md — Taskmaster Task Map

Agent notes for this repo. Read `README.md` first for the project overview, the
pipeline command list, and data/attribution; this file adds the conventions,
gotchas, and operational detail that aren't obvious from the code.

## What it is

An interactive 2D semantic map of every UK Taskmaster task — **one dot = one
task**, laid out by what each task asks you to do. Spine: fetch → embed (Cohere) →
UMAP → label (Toponymy + Opus) → LLM field-extract (Opus) → render (DataMapPlot) →
GitHub Pages. Sibling of `../jeopardy-map` (the lean spine) and
`../huggingface-dataset-map` (LLM extraction + rich hover card + **the composable
filter pattern** — the reference for filtering on DataMapPlot).

## Commands

- `make install` (`uv sync --extra dev`), `make lint` (`ruff check . && ruff format --check .`), `make format`, `make test`.
- Run a stage: `uv run python pipeline/0X_*.py` (ordered 00→06; each reads the prior stage's parquet/npz — see README).
- **Re-render only** (the common iterate step after editing the viz/hover/strip): `uv run python pipeline/06_visualize.py` → writes `data/task_map.html` + `docs/index.html`. Does NOT re-run the LLM stages.
- `05_extract_fields.py` takes `--limit N` (random sample) and `--aggregate-only`. Per-task caches in `data/structured_fields_cache/` (taxonomy) and `data/emoji_cache/` (emoji gist) make it resumable; only successes are cached, so reruns retry failures. Delete a cache dir to force re-extraction.

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

- **The UI layer is `pointer-events: none`** (so the deck.gl canvas gets pan/zoom); only `.interactive-element` re-enables clicks. Custom controls must carry `container-box interactive-element` — which also gives the matching box chrome (shadow / 16px radius / `8px 16px` inset margin / 12px padding). A programmatic `.click()` bypasses this, so test *real* clicks with `read_page` → `computer.left_click` by `ref`.
- **Point/hover data is encoded into the HTML, not greppable.** Verify rendered output at runtime via `datamap.metaData.<field>` in the browser, not by grepping `docs/index.html`. (CSS/JS identifiers like `emoji-strip` ARE greppable — use them as deploy markers.)
- **Native search is whole-query substring match with a "0 matches → show ALL" fallback** — NOT token-AND. You can't compose filters by joining query strings.
- **Composable filters use `datamap.dataSelectionManager`**: `addSelection(indices, filterId)` / `removeSelection(filterId)`; the manager **intersects** all active filters. Native text search registers under `datamap.searchItemId` (`"text-search"`); the emoji strip registers `"emoji-filter"` — so they compose by intersection without touching each other's state. (Mirrors `../huggingface-dataset-map/pipeline/filter_panel.html`.)
- **Blank-on-empty:** because empty selection → show-all, a zero-match combo shows everything. Fix (from the HF filter panel): monkey-patch `highlightPoints` so that when `selectedIndicesByItem` is non-empty but `getSelectedIndices().size === 0`, do `dm.selected.fill(-1)` then re-clone the point layer + `deckgl.setProps({layers})` to force the blank. Our map has `pointRadiusMin/MaxPixels`, so keep the radius-preserving clone.
- **`window.datamap` / `dm.selected` aren't ready at strip-build time.** The strip builds as soon as `#search-container` exists, which can precede `dm.selected` being populated; reading it then throws a *silent* load-time exception (the console tracker misses it). Read `window.datamap` **lazily** inside update functions, with a retry — don't capture at build time.

## Toponymy (stage 04)

Place-naming, not document classification. `Unlabelled` = "unnamed region at this scale" — signal, keep those rows. `topic_names_[0]` is the FINEST layer, `[-1]` the coarsest. `experiments/derive_taxonomy.py` is an off-pipeline EVoC+Toponymy pass that discovered/validated `pipeline/taxonomy.json` from open-ended Opus characterisations; EVoC does its own dim-reduction, so pass full embeddings to both slots. Note the `CompatEVoCClusterer` shim for the evoc 0.3.1 API.

## Local preview & deploy

- **Preview:** serve `data/` over HTTP and open the `http://127.0.0.1` URL — never `file://` (the Chrome extension rewrites it to `https://file://` and fails). `python3 -m http.server <port> --bind 127.0.0.1 --directory data` then `http://127.0.0.1:<port>/task_map.html`.
- **Deploy:** GitHub Pages serves `/docs` on `main`; `06` writes `docs/index.html`, so a commit + `git push` redeploys (~15s). Live: <https://stevenfazzio.github.io/taskmaster-map/>. Verify with a greppable-marker poll (`curl … | grep emoji-strip`). Data is CC BY-SA 4.0 → keep the attribution footer (`ATTRIBUTION_HTML` in `06`). Commit/push only when asked.

## Feature set (in the code)

Hover card with an Opus-generated **emoji gist** (stage 05) above the brief; 4
colormaps (Task type, Activity, Judged on, Air date); a **16-emoji filter strip**
below the search box (single-select, composes with text search by intersection);
and a live **"N tasks"** caption that turns red **"No matches"** and blanks the map
when a filter combination matches zero.

**Deferred:** rewrite the title subtitle (the count caption now carries the task
total, freeing the subtitle to be purely descriptive); optionally hide region
labels on the blank zero-state; the long-deferred extras (contestant dropdown,
random-task button, shareable deep-links, methodology page).
