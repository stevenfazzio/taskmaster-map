"""Hierarchical region labels for the map via Toponymy + Claude (Opus 4.8).

Toponymy names *regions of the embedding space* (place-naming), not individual
tasks. The 2D UMAP coords are the substrate the named regions sit on
(clusterable_vectors); the task embeddings carry the semantic content used while
clustering (embedding_vectors). Toponymy's own keyphrase embedder (CohereEmbedder)
is INDEPENDENT of these vectors — it embeds candidate names in its own search_query
space and ranks them against each other, never against our doc vectors — so its
model/dim need not match ours. Output is a per-task region name at several zoom
levels, fed to DataMapPlot's label layers (finest first). Tasks in unnamed space
come back "Unlabelled"; at ~1k points that fraction is high by design (sparse
density), and is a gap on the map (signal), not a failure.

Opus 4.8 note: the stock AsyncAnthropicNamer passes `temperature`, which Opus 4.7/4.8
removed (400, fail-fast). OpusAnthropicNamer below drops it.

Inputs:  data/task_embeddings.npz, data/umap_coords.npz, data/task_rows.parquet
Output:  data/toponymy_labels.parquet  (task_id + label_layer_0..k, finest first)
"""

from __future__ import annotations

import os

import nest_asyncio
import numpy as np
import pandas as pd
from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MAX_CONCURRENCY,
    ANTHROPIC_MODEL_NAMING,
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    TASK_EMB_NPZ,
    TASK_ROWS_PARQUET,
    TOPONYMY_LABELS_PARQUET,
    UMAP_COORDS_NPZ,
)
from toponymy.llm_wrappers import AsyncAnthropicNamer

nest_asyncio.apply()

MAX_DOC_CHARS = 2_000


class OpusAnthropicNamer(AsyncAnthropicNamer):
    """AsyncAnthropicNamer with the Opus 4.8 call surface: `temperature` is removed
    on Opus 4.7/4.8 (the stock namer passes it and would 400 fail-fast). We keep the
    method signatures (Toponymy passes temperature/max_tokens) but don't forward
    temperature, and join all text blocks so the parse is robust to any preamble."""

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


def main():
    from toponymy import Toponymy, ToponymyClusterer
    from toponymy.embedding_wrappers import CohereEmbedder

    # coords define point order; align the embedding matrix to the same task_id order.
    crd = np.load(UMAP_COORDS_NPZ, allow_pickle=True)
    coords = crd["coords"].astype(np.float32)
    task_id = crd["task_id"]

    ed = np.load(TASK_EMB_NPZ, allow_pickle=True)
    row = {c: i for i, c in enumerate(ed["task_id"])}
    idx = np.array([row[c] for c in task_id], dtype=np.int64)
    embeddings = ed["emb"][idx].astype(np.float32)

    # Documents = the same task brief we embedded.
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
    if n_layers == 0:
        raise ValueError("Toponymy produced 0 cluster layers")
    print(f"Toponymy produced {n_layers} cluster layer(s)")

    out = {"task_id": task_id}
    for i, names in enumerate(topic_model.topic_name_vectors_):
        names = np.asarray(names, dtype=object)
        out[f"label_layer_{i}"] = names
        n_named = int(np.sum(names != "Unlabelled"))
        uniq = sorted({n for n in names.tolist() if n != "Unlabelled"})
        print(f"  layer {i}: {len(uniq)} regions, {n_named}/{len(names)} tasks named")
        if i == n_layers - 1:
            print(f"    coarsest regions: {uniq[:12]}")

    df = pd.DataFrame(out)
    tmp = str(TOPONYMY_LABELS_PARQUET) + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, TOPONYMY_LABELS_PARQUET)
    print(f"Wrote {TOPONYMY_LABELS_PARQUET} ({n_layers} layers)")


if __name__ == "__main__":
    main()
