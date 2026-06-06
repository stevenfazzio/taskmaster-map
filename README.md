# Taskmaster Task Map

An interactive 2D semantic map of every [Taskmaster](https://taskmaster.fandom.com/)
(UK) task, laid out by what each task actually asks you to do. **One dot = one
task.** Each task's brief is embedded, projected to 2D with UMAP, and the regions
are named by Toponymy; an LLM also tags each task (activity, how it's judged, the
gimmick) for colouring, filtering, and the hover card.

Built as a sibling of `../jeopardy-map` (the lean fetch → embed → reduce → label →
visualize spine) and `../huggingface-dataset-map` (the LLM field-extraction stage
and the rich hover card).

## Pipeline

```bash
make install   # uv sync --extra dev

python pipeline/00_fetch_taskmaster.py   # silverdavi CSVs -> *_raw.parquet
python pipeline/01_prepare_tasks.py      # clean + derive fields -> task_rows.parquet
python pipeline/02_embed_tasks.py        # Cohere embed-v4.0 -> task_embeddings.npz
python pipeline/03_reduce_umap.py        # UMAP -> umap_coords.npz
python pipeline/04_label_topics.py       # Toponymy + Opus -> toponymy_labels.parquet
python pipeline/05_extract_fields.py     # Opus taxonomy -> structured_fields.parquet
python pipeline/06_visualize.py          # DataMapPlot -> docs/index.html
```

`task_id` is the alignment key across every stage. Constants (including the
`MAX_TASKS` smoke-test knob) live in `pipeline/config.py`.

## Data & attribution

Task briefs, scores, and episode metadata come from
[silverdavi/taskmaster-uk-scores](https://github.com/silverdavi/taskmaster-uk-scores)
(pinned commit), which is compiled from the
[Taskmaster Fandom Wiki](https://taskmaster.fandom.com/) and released under
**CC BY-SA 4.0**. This project inherits that licence for the derived data.
