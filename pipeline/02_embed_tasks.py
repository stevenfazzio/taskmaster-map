"""Embed each task brief with Cohere embed-v4.0.

Input is the embed_text column from stage 01 (the task brief). input_type=
"clustering" because the only downstream use is grouping/visualization (UMAP +
Toponymy's clusterer). One float32 vector per task. Checkpointed + resumable,
though at ~1k tasks this is ~11 batches and a few seconds.

Input:  data/task_rows.parquet
Output: data/task_embeddings.npz  (emb [N x dim] float32, task_id [N])
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from config import (
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    COHERE_INPUT_TYPE,
    COHERE_OUTPUT_DIM,
    EMBED_BATCH,
    EMBED_CHECKPOINT_EVERY,
    TASK_EMB_NPZ,
    TASK_ROWS_PARQUET,
)


def embed_with_retry(client, chunk, max_retries=5):
    """Cohere embed of one batch, with exponential backoff on transient errors."""
    for attempt in range(max_retries):
        try:
            resp = client.embed(
                model=COHERE_EMBED_MODEL,
                input_type=COHERE_INPUT_TYPE,
                texts=chunk,
                output_dimension=COHERE_OUTPUT_DIM,
                embedding_types=["float"],
            )
            return np.asarray(resp.embeddings.float_, dtype=np.float32)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = min(2**attempt * 5, 60)
            print(f"  embed attempt {attempt + 1} failed ({type(e).__name__}: {e}); retry in {wait}s")
            time.sleep(wait)


def main():
    import cohere

    df = pd.read_parquet(TASK_ROWS_PARQUET, columns=["task_id", "embed_text"])
    task_ids = df["task_id"].to_numpy()
    texts = df["embed_text"].tolist()
    n = len(texts)
    print(f"Embedding {n:,} tasks with {COHERE_EMBED_MODEL} (dim={COHERE_OUTPUT_DIM}, input_type={COHERE_INPUT_TYPE})")

    sig = f"{task_ids[0]}_{task_ids[-1]}_{n}_{COHERE_EMBED_MODEL}_{COHERE_OUTPUT_DIM}"
    if TASK_EMB_NPZ.exists():
        cached = np.load(TASK_EMB_NPZ, allow_pickle=True)
        if str(cached["sig"]) == sig:
            print(f"  reusing cached embeddings ({n:,})")
            return

    if not CO_API_KEY:
        raise RuntimeError("CO_API_KEY not set; export it or add it to .env (see .env.example)")
    client = cohere.ClientV2(api_key=CO_API_KEY)

    emb = np.zeros((n, COHERE_OUTPUT_DIM), dtype=np.float32)
    done = 0
    prog_path = str(TASK_EMB_NPZ) + ".progress.npz"
    if os.path.exists(prog_path):
        p = np.load(prog_path, allow_pickle=True)
        if int(p["n"]) == n and int(p["dim"]) == COHERE_OUTPUT_DIM:
            emb = p["emb"]
            done = int(p["done"])
            print(f"  resuming from {done:,}/{n:,}")

    batch_i = 0
    for start in range(done, n, EMBED_BATCH):
        chunk = texts[start : start + EMBED_BATCH]
        emb[start : start + len(chunk)] = embed_with_retry(client, chunk)
        batch_i += 1
        if batch_i % EMBED_CHECKPOINT_EVERY == 0:
            np.savez(prog_path + ".tmp.npz", n=n, dim=COHERE_OUTPUT_DIM, done=start + len(chunk), emb=emb)
            os.replace(prog_path + ".tmp.npz", prog_path)
            print(f"  embedded {start + len(chunk):,}/{n:,}")

    tmp = str(TASK_EMB_NPZ) + ".tmp.npz"
    np.savez(tmp, sig=sig, emb=emb, task_id=task_ids)
    os.replace(tmp, TASK_EMB_NPZ)
    if os.path.exists(prog_path):
        os.unlink(prog_path)
    print(f"Wrote {TASK_EMB_NPZ} ({TASK_EMB_NPZ.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
