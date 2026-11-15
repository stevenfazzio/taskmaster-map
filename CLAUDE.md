# CLAUDE.md — Taskmaster Task Map

Agent notes for this repo. Read `README.md` first for the project overview, the
pipeline command list, and data/attribution; this file adds the conventions,
gotchas, and operational detail that aren't obvious from the code.

## What it is

An interactive 2D semantic map of every UK Taskmaster task — **one dot = one
task**, laid out by what each task asks you to do. Spine: fetch → embed (Qwen3 via OpenRouter) →
UMAP → label (Toponymy + Opus) → LLM field-extract (Opus) → render (DataMapPlot) →
GitHub Pages. Sibling of `../jeopardy-map` (the lean spine) and
`../huggingface-dataset-map` (LLM extraction + rich hover card). This map
deliberately has **no custom filter UI** — it stays on DataMapPlot's public API
(see "Feature set" below for why).

## Commands

- `make install` (`uv sync --extra dev`), `make lint` (`ruff check . && ruff format --check .`), `make format`, `make test`.
- Run a stage: `uv run python pipeline/0X_*.py` (ordered 00→00b→01→06; each reads the prior stage's parquet/npz — see README).
- **Experiment runs:** `TASKMAP_DATA_DIR=<folder>` redirects every stage's I/O there (seed it with the inputs the stage reads) and keeps 06 off `docs/`; `TASKMAP_NAMING_MODEL` / `TASKMAP_NAMING_EFFORT=none` override the stage-04 namer. `experiments/compare_naming.py` aligns region names across runs that share a layout.
- **Re-render only** (the common iterate step after editing the viz/hover): `uv run python pipeline/06_visualize.py` → writes `data/task_map.html` + `docs/index.html`. Does NOT re-run the LLM stages.
- `05_extract_fields.py` runs **two** per-task LLM passes, each with its own cache so tuning one doesn't disturb the other: `data/structured_fields_cache/` (taxonomy JSON) and `data/emoji_cache/` (emoji gist `.txt`). (`data/motif_cache/` on disk is a leftover from the removed motif filter; nothing reads it.) Takes `--limit N` (random sample) and `--aggregate-only`. Only successes are cached, so reruns retry failures; delete a cache dir to force re-extraction (changing a prompt or the model does NOT auto-invalidate — neither is in the cache key).

## Conventions

- `task_id` is the alignment key across **every** stage — always merge/reindex on it (`06` does `set_index("task_id").reindex(order)` so row i ↔ `coords[i]`).
- **New series come from the wiki (stage `00b`).** Upstream stopped at New Year Treat 6; `WIKI_SERIES` in config lists the series scraped from the Fandom MediaWiki API into `wiki_*_raw.parquet` (upstream's schema), which `01` concatenates. Pages are pinned by revision in `pipeline/wiki_pins.json` (checked in; unpinned pages get pinned on first fetch); raw responses cache in `data/wiki_cache/`. task_id = code·10000 + episode·100 + row (210103 = S21E1 row 3) — disjoint from upstream's 1..1045. Conventions match upstream (`Prize:`/`Live:` → flags and stripped; `Team:`/`Team Live:` kept in text with is_live=0; `-`/DQ → NaN; ✔/✘ → 1/0), validated with `00b --validate ["Series 19"|"Series 20"]` (S20: 52/52 tasks, 260/260 scores before the deliberate change below). One deliberate deviation: a multi-row task (rowspan'd task number — prep/branch/bonus rows) keeps the first sub-row's brief but **sums** scores over sub-rows; upstream keeps the first row only, which drops bonus points and zeroes branch tasks. Summing reproduces the wiki's episode totals (20/20). Upstream also silently lost a few S19 tasks (E9 #2, three `S` rows). Parser tests: `tests/test_wiki_parse.py`.
- **Corpus is 990 tasks, not the ~1,045 raw rows** (1,092 once the S21+S22 wiki rows are merged by `01`). Upstream duplicates the single New Year Treat special as six identical series (`New Year Treat 1`..`6`); `01`'s `_dedupe_identical_series` collapses any byte-identical series to one, keeping genuine same-brief recurrences like Series 2 & 3 "Buy a gift for the Taskmaster" (different casts). `05`'s `aggregate` is corpus-scoped so dropped tasks don't linger as orphaned cache rows.
- **Atomic writes everywhere**: tempfile in the same dir → verify → `os.replace`. Mirror this for any new output.
- API keys come from **environment variables**, not `.env` files (`config.py` only loads `PROJECT_ROOT/.env` if it happens to exist). Don't recreate `~/.config/data-apis/.env`; never read/echo secrets.
- Plain `.py` scripts, HTML visual output, `uv`/`ruff` (line-length 120, rules E/F/I) — see the global `~/.claude/CLAUDE.md`.

## External APIs

- **Embeddings: Qwen3-Embedding-8B via OpenRouter** (`qwen/qwen3-embedding-8b`, `OPENROUTER_API_KEY`), `dimensions=1024` (Matryoshka; returned unit-norm), through `pipeline/openrouter_embed.py` (plain `requests`; its `encode()` also satisfies Toponymy's embedder protocol, so `04` uses it for keyphrases). Task texts get an `Instruct: …\nQuery: …` prefix (`EMBED_INSTRUCTION`), the analogue of Cohere's old `input_type="clustering"`; keyphrases get none. Re-embeds when the npz signature (model/dim/instruction hash/corpus) changes. ~3 min for the corpus, fractions of a cent.
- **Anthropic Opus 5.5** (`claude-opus-5-5`). **Thinking is always on** (`disabled` and `budget_tokens` 400) and its tokens count against `max_tokens`, so `effort` (`output_config`) is the only knob — and its API default is `medium`, so every call sets it explicitly (`NAMING_EFFORT`, `EXTRACT_EFFORT`, `EMOJI_EFFORT`). `temperature`/`top_p`/`top_k` 400; Toponymy's `AnthropicNamer` passes `temperature`, so `04` subclasses it (`OpusAnthropicNamer`) to drop it and floor `max_tokens`. All calls treat a non-`end_turn` stop (thinking truncation, refusal) as a retryable failure. The emoji pass also uses structured output (`{"emoji": …}`) plus `emoji_problem()` validation (1–5 emoji, keycap runs = one): at low effort 5.5 leaked its deliberation into ~1.4% of plain-text answers. Cached emoji that fail validation are regenerated on the next run and blanked (never shipped) if regeneration fails. Server-side refusal `fallbacks` are NOT enabled: they need a far newer `anthropic` SDK than the 0.106 pin. Cost (measured on Opus 5.5, 990 tasks): stage 05 taxonomy ≈ $3.70 (low effort; thinking added ~nothing), emoji pass ≈ $0.80 (system prompt is cached); stage 04 naming ≈ $1–4. Stage-05 caches are per task and not keyed by model, so a rerun only pays for new tasks.
- **Cohere** is now only used by the off-pipeline `experiments/` scripts (legacy constants at the bottom of `config.py`).

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
