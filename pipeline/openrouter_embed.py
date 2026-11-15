"""Minimal OpenRouter embeddings client (OpenAI-compatible /embeddings endpoint).

Used by stage 02 (task embeddings) and stage 04 (Toponymy keyphrase embeddings).
`encode()` matches Toponymy's TextEmbedderProtocol, so one class serves both. Plain
`requests` rather than the openai SDK to avoid a dependency for one POST.
"""

from __future__ import annotations

import time

import numpy as np
import requests
from tqdm import tqdm

OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"


class OpenRouterEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int | None = None,
        instruction: str | None = None,
        batch_size: int = 64,
        max_retries: int = 5,
        timeout: float = 120.0,
    ):
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set; export it or add it to .env (see .env.example)")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        # Qwen3-Embedding is instruction-aware: "Instruct: <task>\nQuery: <text>". None = raw text.
        self.instruction = instruction
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout

    def _format(self, text: str) -> str:
        return f"Instruct: {self.instruction}\nQuery: {text}" if self.instruction else text

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        payload = {"model": self.model, "input": [self._format(t) for t in texts], "encoding_format": "float"}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.max_retries):
            try:
                r = requests.post(OPENROUTER_EMBED_URL, headers=headers, json=payload, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                if "data" not in data:  # OpenRouter can return 200 with an error body
                    raise RuntimeError(f"no data in response: {str(data)[:300]}")
                rows = sorted(data["data"], key=lambda d: d["index"])
                emb = np.asarray([d["embedding"] for d in rows], dtype=np.float32)
                if emb.shape[0] != len(texts):
                    raise RuntimeError(f"got {emb.shape[0]} embeddings for {len(texts)} texts")
                if self.dimensions and emb.shape[1] != self.dimensions:
                    raise RuntimeError(f"got dim {emb.shape[1]}, asked for {self.dimensions}")
                return emb
            except Exception as e:  # noqa: BLE001 — retry transient HTTP/provider errors
                if attempt == self.max_retries - 1:
                    raise
                wait = min(2**attempt * 5, 60)
                print(f"  embed attempt {attempt + 1} failed ({type(e).__name__}: {e}); retry in {wait}s")
                time.sleep(wait)

    def encode(
        self, texts: list[str], verbose: bool | None = None, show_progress_bar: bool | None = None
    ) -> np.ndarray:
        texts = list(texts)
        out = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="embedding", disable=not show_progress_bar):
            out.append(self._embed_batch(texts[i : i + self.batch_size]))
        return np.vstack(out) if out else np.zeros((0, self.dimensions or 0), dtype=np.float32)
