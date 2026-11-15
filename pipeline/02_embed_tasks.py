"""Embed each task brief with Qwen3-Embedding-8B (via OpenRouter).

Input is the embed_text column from stage 01 (the task brief), prefixed with
EMBED_INSTRUCTION in Qwen's "Instruct: ...\\nQuery: ..." format — the instruction
steers the space toward what the task asks, since the only downstream use is
grouping/visualization (UMAP + Toponymy's clusterer). One float32 vector per task.
Checkpointed + resumable, though at ~1k tasks this is ~16 batches and a few seconds.

Input:  data/task_rows.parquet
Output: data/task_embeddings.npz  (emb [N x dim] float32, task_id [N])
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd
from config import (
    EMBED_BATCH,
    EMBED_CHECKPOINT_EVERY,
    EMBED_DIM,
    EMBED_INSTRUCTION,
    EMBED_MODEL,
    OPENROUTER_API_KEY,
    TASK_EMB_NPZ,
    TASK_ROWS_PARQUET,
)
from openrouter_embed import OpenRouterEmbedder


def main():
    df = pd.read_parquet(TASK_ROWS_PARQUET, columns=["task_id", "embed_text"])
    task_ids = df["task_id"].to_numpy()
    texts = df["embed_text"].tolist()
    n = len(texts)
    print(f"Embedding {n:,} tasks with {EMBED_MODEL} (dim={EMBED_DIM}, instruction={EMBED_INSTRUCTION!r})")

    instr_hash = hashlib.sha1((EMBED_INSTRUCTION or "").encode()).hexdigest()[:8]
    sig = f"{task_ids[0]}_{task_ids[-1]}_{n}_{EMBED_MODEL}_{EMBED_DIM}_{instr_hash}"
    if TASK_EMB_NPZ.exists():
        cached = np.load(TASK_EMB_NPZ, allow_pickle=True)
        if str(cached["sig"]) == sig:
            print(f"  reusing cached embeddings ({n:,})")
            return

    embedder = OpenRouterEmbedder(
        api_key=OPENROUTER_API_KEY, model=EMBED_MODEL, dimensions=EMBED_DIM, instruction=EMBED_INSTRUCTION
    )

    emb = np.zeros((n, EMBED_DIM), dtype=np.float32)
    done = 0
    prog_path = str(TASK_EMB_NPZ) + ".progress.npz"
    if os.path.exists(prog_path):
        p = np.load(prog_path, allow_pickle=True)
        if str(p["sig"]) == sig:
            emb = p["emb"]
            done = int(p["done"])
            print(f"  resuming from {done:,}/{n:,}")

    batch_i = 0
    for start in range(done, n, EMBED_BATCH):
        chunk = texts[start : start + EMBED_BATCH]
        emb[start : start + len(chunk)] = embedder.encode(chunk)
        batch_i += 1
        if batch_i % EMBED_CHECKPOINT_EVERY == 0:
            np.savez(prog_path + ".tmp.npz", sig=sig, done=start + len(chunk), emb=emb)
            os.replace(prog_path + ".tmp.npz", prog_path)
            print(f"  embedded {start + len(chunk):,}/{n:,}")

    norms = np.linalg.norm(emb, axis=1)
    if not np.all(norms > 0.5):
        raise RuntimeError(f"{int((norms <= 0.5).sum())} rows look empty/unfilled (norm <= 0.5); not writing")

    tmp = str(TASK_EMB_NPZ) + ".tmp.npz"
    np.savez(tmp, sig=sig, emb=emb, task_id=task_ids)
    os.replace(tmp, TASK_EMB_NPZ)
    if os.path.exists(prog_path):
        os.unlink(prog_path)
    print(f"Wrote {TASK_EMB_NPZ} ({TASK_EMB_NPZ.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
