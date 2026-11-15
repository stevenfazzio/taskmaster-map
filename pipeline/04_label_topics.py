"""Hierarchical region labels for the map via Toponymy + Claude (Opus 5.5).

Toponymy names *regions of the embedding space* (place-naming), not individual
tasks. The 2D UMAP coords are the substrate the named regions sit on
(clusterable_vectors); the task embeddings carry the semantic content used while
clustering (embedding_vectors). Toponymy's own keyphrase embedder (the same Qwen3 model
via OpenRouter, but with NO instruction) is INDEPENDENT of these vectors — it embeds
candidate names in their own space and ranks them against each other, never against
our doc vectors — so its instruction/dim need not match ours. Output is a per-task region name at several zoom
levels, fed to DataMapPlot's label layers (finest first). Tasks in unnamed space
come back "Unlabelled"; at ~1k points that fraction is high by design (sparse
density), and is a gap on the map (signal), not a failure.

Opus 5.5 note: the stock AsyncAnthropicNamer passes `temperature`, which current Opus
models reject (400, fail-fast). OpusAnthropicNamer below drops it, sets effort explicitly
(thinking is always on; the API default effort is "medium"), and floors max_tokens so
thinking can't starve the answer.

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
    EMBED_DIM,
    EMBED_MODEL,
    NAMING_EFFORT,
    NAMING_MIN_MAX_TOKENS,
    OPENROUTER_API_KEY,
    TASK_EMB_NPZ,
    TASK_ROWS_PARQUET,
    TOPONYMY_LABELS_PARQUET,
    UMAP_COORDS_NPZ,
)
from openrouter_embed import OpenRouterEmbedder
from toponymy.llm_wrappers import AsyncAnthropicNamer

nest_asyncio.apply()

MAX_DOC_CHARS = 2_000


class OpusAnthropicNamer(AsyncAnthropicNamer):
    """AsyncAnthropicNamer with the Opus 5.5 call surface. Keeps Toponymy's method
    signatures (it passes temperature/max_tokens) but: never forwards temperature
    (400 on current Opus); sets effort explicitly; floors max_tokens, since thinking is
    always on and its tokens count against the cap; raises on a non-`end_turn` stop
    (truncation/refusal) so Toponymy's retry + per-prompt error handling sees it instead
    of a half-empty parse; and joins all text blocks so the parse ignores thinking blocks."""

    effort = NAMING_EFFORT
    min_max_tokens = NAMING_MIN_MAX_TOKENS

    async def _create(self, max_tokens, messages, system=None):
        kwargs = {"system": system} if system else {}
        if self.effort:  # None = legacy Opus 4.8 call (no effort/thinking), as the original map ran
            kwargs["output_config"] = {"effort": self.effort}
        async with self.semaphore:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=max(max_tokens, self.min_max_tokens),
                messages=messages,
                **kwargs,
            )
        if resp.stop_reason != "end_turn":
            raise RuntimeError(f"naming call stopped with stop_reason={resp.stop_reason!r}")
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    async def _call_single_llm(self, prompt, temperature, max_tokens):
        return await self._create(max_tokens, [{"role": "user", "content": prompt + self.extra_prompting}])

    async def _call_single_llm_with_system(self, system_prompt, user_prompt, temperature, max_tokens):
        return await self._create(
            max_tokens, [{"role": "user", "content": user_prompt + self.extra_prompting}], system=system_prompt
        )


def main():
    from toponymy import Toponymy, ToponymyClusterer

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
    embedder = OpenRouterEmbedder(api_key=OPENROUTER_API_KEY, model=EMBED_MODEL, dimensions=EMBED_DIM)
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

    print(f"Naming model: {ANTHROPIC_MODEL_NAMING} (effort={NAMING_EFFORT})")
    out = {"task_id": task_id}
    for i, names in enumerate(topic_model.topic_name_vectors_):
        names = np.asarray(names, dtype=object)
        out[f"label_layer_{i}"] = names
        # Raw cluster ids (-1 = noise) alongside the names, so runs that name the same
        # layout can be checked for identical clusters and aligned region-by-region.
        out[f"cluster_layer_{i}"] = np.asarray(topic_model.cluster_layers_[i].cluster_labels, dtype=np.int32)
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
