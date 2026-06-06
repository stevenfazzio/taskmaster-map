"""Diagnostic: what do Toponymy's region names look like WITHOUT the disambiguation pass?

Mirrors pipeline/04_label_topics.py's fit exactly, but monkeypatches
ClusterLayer.disambiguate_topics to a no-op (recording the raw first-pass names).
The clustering is deterministic (fixed seed, fixed coords/embeddings), so region
membership matches the production run — only the *names* differ (raw vs disambiguated).

Writes to data/experiments/ ONLY; never touches the production parquet.

Outputs:
  data/experiments/toponymy_labels_nodisambig.parquet   (task_id + raw label_layer_*)
  data/experiments/raw_cluster_names.json               (per-layer raw per-cluster names)
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "pipeline")

import nest_asyncio
import numpy as np
import pandas as pd
from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MAX_CONCURRENCY,
    ANTHROPIC_MODEL_NAMING,
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    DATA_DIR,
    TASK_EMB_NPZ,
    TASK_ROWS_PARQUET,
    UMAP_COORDS_NPZ,
)
from toponymy.llm_wrappers import AsyncAnthropicNamer

nest_asyncio.apply()

MAX_DOC_CHARS = 2_000
OUT_PARQUET = DATA_DIR / "experiments" / "toponymy_labels_nodisambig.parquet"
OUT_RAW_JSON = DATA_DIR / "experiments" / "raw_cluster_names.json"


class OpusAnthropicNamer(AsyncAnthropicNamer):
    """Verbatim copy of pipeline/04_label_topics.py's namer: Opus 4.8 rejects
    `temperature`, so we don't forward it; join all text blocks for robust parse."""

    async def _call_single_llm(self, prompt, temperature, max_tokens):
        async with self.semaphore:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt + self.extra_prompting}],
            )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    async def _call_single_llm_with_system(self, system_prompt, user_prompt, temperature, max_tokens):
        async with self.semaphore:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt + self.extra_prompting}],
            )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


# ── monkeypatch: disable disambiguation, record the raw first-pass per-cluster names ──
RAW_CLUSTER_NAMES: dict[int, list[str]] = {}


def _noop_disambiguate(self, *args, **kwargs):
    # self.topic_names here are the raw first-pass names (disambiguation hasn't run).
    # Record only the first call per layer (the second pass would re-trigger otherwise).
    if self.layer_id not in RAW_CLUSTER_NAMES:
        RAW_CLUSTER_NAMES[self.layer_id] = list(self.topic_names)
    return  # skip the rename-to-distinct step entirely


def main():
    import toponymy.cluster_layer as cl
    from toponymy import Toponymy, ToponymyClusterer
    from toponymy.embedding_wrappers import CohereEmbedder

    cl.ClusterLayer.disambiguate_topics = _noop_disambiguate  # the patch

    crd = np.load(UMAP_COORDS_NPZ, allow_pickle=True)
    coords = crd["coords"].astype(np.float32)
    task_id = crd["task_id"]

    ed = np.load(TASK_EMB_NPZ, allow_pickle=True)
    row = {c: i for i, c in enumerate(ed["task_id"])}
    idx = np.array([row[c] for c in task_id], dtype=np.int64)
    embeddings = ed["emb"][idx].astype(np.float32)

    text_by_id = pd.read_parquet(TASK_ROWS_PARQUET, columns=["task_id", "embed_text"]).set_index("task_id")[
        "embed_text"
    ]
    documents = text_by_id.reindex(task_id).fillna("").str.slice(0, MAX_DOC_CHARS).tolist()
    print(f"Loaded {len(documents):,} tasks; embeddings {embeddings.shape}")

    llm = OpusAnthropicNamer(
        api_key=ANTHROPIC_API_KEY,
        model=ANTHROPIC_MODEL_NAMING,
        max_concurrent_requests=ANTHROPIC_MAX_CONCURRENCY,
    )
    embedder = CohereEmbedder(api_key=CO_API_KEY, model=COHERE_EMBED_MODEL)
    clusterer = ToponymyClusterer(min_clusters=6)

    topic_model = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description="Taskmaster tasks",
        corpus_description=(
            "Tasks from the comedy panel show Taskmaster, each given as the brief "
            "read or shown to the contestants (e.g. 'Paint the best picture of a "
            "horse whilst riding a horse.')"
        ),
        lowest_detail_level=0.5,
        highest_detail_level=1.0,
    )
    np.random.seed(42)
    topic_model.fit(objects=documents, embedding_vectors=embeddings, clusterable_vectors=coords)

    n_layers = len(topic_model.topic_name_vectors_)
    print(f"Toponymy produced {n_layers} cluster layer(s) (disambiguation DISABLED)")

    out = {"task_id": task_id}
    for i, names in enumerate(topic_model.topic_name_vectors_):
        names = np.asarray(names, dtype=object)
        out[f"label_layer_{i}"] = names
        n_named = int(np.sum(names != "Unlabelled"))
        uniq = sorted({n for n in names.tolist() if n != "Unlabelled"})
        print(f"  layer {i}: {len(uniq)} distinct names, {n_named}/{len(names)} tasks named")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(OUT_PARQUET) + ".tmp"
    pd.DataFrame(out).to_parquet(tmp, index=False)
    os.replace(tmp, OUT_PARQUET)
    print(f"Wrote {OUT_PARQUET}")

    OUT_RAW_JSON.write_text(json.dumps({str(k): v for k, v in RAW_CLUSTER_NAMES.items()}, indent=2))
    print(f"Wrote {OUT_RAW_JSON}")


if __name__ == "__main__":
    main()
