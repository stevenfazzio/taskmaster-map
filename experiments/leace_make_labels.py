"""Piece 1 of the LEACE map: erase task_format -> re-UMAP -> Toponymy WITH naming.

Run:  uv run --with concept-erasure python experiments/leace_make_labels.py
Writes data/experiments/leace_umap_coords.npz and leace_toponymy_labels.parquet (only).
Mirrors stages 03+04 exactly, but on the LEACE-erased embeddings.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "pipeline")

import nest_asyncio
import numpy as np
import pandas as pd
import torch
import umap
from concept_erasure import LeaceEraser
from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MAX_CONCURRENCY,
    ANTHROPIC_MODEL_NAMING,
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    DATA_DIR,
    TASK_EMB_NPZ,
    TASK_ROWS_PARQUET,
    TOPONYMY_LABELS_PARQUET,
    UMAP_COORDS_NPZ,
)
from toponymy.llm_wrappers import AsyncAnthropicNamer

nest_asyncio.apply()
MAX_DOC_CHARS = 2_000
OUT_COORDS = DATA_DIR / "experiments" / "leace_umap_coords.npz"
OUT_LABELS = DATA_DIR / "experiments" / "leace_toponymy_labels.parquet"


class OpusAnthropicNamer(AsyncAnthropicNamer):
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

    crd = np.load(UMAP_COORDS_NPZ, allow_pickle=True)
    task_id = crd["task_id"]
    ed = np.load(TASK_EMB_NPZ, allow_pickle=True)
    row = {c: i for i, c in enumerate(ed["task_id"])}
    X = ed["emb"][np.array([row[c] for c in task_id])].astype(np.float64)
    fmt = (
        pd.read_parquet(TASK_ROWS_PARQUET, columns=["task_id", "task_format"])
        .set_index("task_id")
        .loc[task_id, "task_format"]
        .values
    )
    codes, _ = pd.factorize(fmt)

    # LEACE erase task_format, then re-UMAP at the production settings
    Xe = LeaceEraser.fit(torch.tensor(X), torch.tensor(codes))(torch.tensor(X)).numpy().astype(np.float32)
    coords = (
        umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.05, metric="cosine", random_state=42)
        .fit_transform(Xe)
        .astype(np.float32)
    )
    OUT_COORDS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(OUT_COORDS) + ".tmp.npz", coords=coords, task_id=task_id)
    os.replace(str(OUT_COORDS) + ".tmp.npz", OUT_COORDS)
    print(f"LEACE coords -> {OUT_COORDS}")

    text_by_id = pd.read_parquet(TASK_ROWS_PARQUET, columns=["task_id", "embed_text"]).set_index("task_id")[
        "embed_text"
    ]
    documents = text_by_id.reindex(task_id).fillna("").str.slice(0, MAX_DOC_CHARS).tolist()

    llm = OpusAnthropicNamer(
        api_key=ANTHROPIC_API_KEY, model=ANTHROPIC_MODEL_NAMING, max_concurrent_requests=ANTHROPIC_MAX_CONCURRENCY
    )
    embedder = CohereEmbedder(api_key=CO_API_KEY, model=COHERE_EMBED_MODEL)
    topic_model = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedder,
        clusterer=ToponymyClusterer(min_clusters=6),
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
    topic_model.fit(objects=documents, embedding_vectors=Xe, clusterable_vectors=coords)

    out = {"task_id": task_id}
    for i, names in enumerate(topic_model.topic_name_vectors_):
        out[f"label_layer_{i}"] = np.asarray(names, dtype=object)
    df = pd.DataFrame(out)
    df.to_parquet(str(OUT_LABELS) + ".tmp")
    os.replace(str(OUT_LABELS) + ".tmp", OUT_LABELS)
    print(f"LEACE labels -> {OUT_LABELS}  ({len(out) - 1} layers)")

    # how did the (former) prize tasks' labels change?
    prod = pd.read_parquet(TOPONYMY_LABELS_PARQUET).rename(columns={"label_layer_0": "prod0"})
    cmp = df.rename(columns={"label_layer_0": "leace0"}).merge(prod[["task_id", "prod0"]], on="task_id")
    cmp["is_prize"] = fmt == "Prize"
    pr = cmp[cmp.is_prize]
    print(f"\n=== former prize tasks ({len(pr)}): LEACE label_layer_0 distribution ===")
    print(pr.leace0.value_counts().head(20).to_string())
    print(f"\n  distinct LEACE labels over prize tasks: {pr.leace0.nunique()} (was 7 synonyms in production)")
    print(f"  prize tasks now Unlabelled: {(pr.leace0 == 'Unlabelled').sum()}/{len(pr)}")


if __name__ == "__main__":
    main()
