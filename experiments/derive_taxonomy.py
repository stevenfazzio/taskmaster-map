"""Experiment: derive a taxonomy bottom-up instead of hardcoding it.

Off-pipeline. For ONE axis (activity or judging): characterise each task open-ended
via Opus (no fixed slugs), embed the characterisations, cluster the FULL embeddings
with Toponymy's EVoC clusterer (EVoC reduces internally), and read the named regions
as a candidate taxonomy. Then cross-tab the discovered regions against the current
hand-rolled slug for that axis to see where they agree / diverge.

This DISCOVERS the taxonomy (the category SET); per-task assignment stays with
pipeline/05 (constrained). Toponymy names places, not documents — we use its region
NAMES as candidate slugs, not its per-doc labels (which carry an Unlabelled tail by
design). The AXIS (what to characterise) is our choice; the VALUES are discovered.

Outputs to data/experiments/ (gitignored). Resumable: per-task characterisations are
cached per axis. Run:  uv run python experiments/derive_taxonomy.py --axis judging
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import evoc
import nest_asyncio
import numpy as np
import pandas as pd
from anthropic import AsyncAnthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (  # noqa: E402
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL_EXTRACT,
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    COHERE_INPUT_TYPE,
    COHERE_OUTPUT_DIM,
    DATA_DIR,
    EXTRACT_CONCURRENCY,
    EXTRACT_MAX_RETRIES,
    STRUCTURED_FIELDS_PARQUET,
    TASK_ROWS_PARQUET,
)
from toponymy import Toponymy  # noqa: E402
from toponymy.clustering import Clusterer, EVoCClusterer  # noqa: E402
from toponymy.embedding_wrappers import CohereEmbedder  # noqa: E402
from toponymy.llm_wrappers import AsyncAnthropicNamer  # noqa: E402

nest_asyncio.apply()

OUT_DIR = DATA_DIR / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The examples in each prompt teach the abstraction LEVEL, not a list of allowed answers
# — deliberately varied so they don't seed the clusters.
_ACTIVITY_SYSTEM = (
    "You characterise tasks from the comedy show Taskmaster along ONE dimension: the core "
    "physical ACTIVITY — what the contestant actually DOES, abstracted away from the specific "
    "props and from how the task is scored.\n\n"
    "Given a task brief, reply with a SHORT phrase (3-7 words) naming that core activity at a "
    "level where two tasks involving the same kind of doing would get the same phrase. Reply "
    "with ONLY the phrase — no punctuation, no explanation.\n\n"
    "These examples show the abstraction LEVEL, not allowed answers — coin whatever fits:\n"
    "- 'In the lab is a watermelon. Eat as much as possible.' -> eating food quickly\n"
    "- 'Convince a stranger to give you their shoe.' -> persuading a member of the public\n"
    "- 'Build the tallest tower out of these boxes.' -> building a structure"
)
_JUDGING_SYSTEM = (
    "You characterise tasks from the comedy show Taskmaster along ONE dimension: how the WINNER "
    "is DECIDED — the criterion the Taskmaster measures or judges to rank the contestants, "
    "abstracted away from the specific activity and props.\n\n"
    "Given a task brief, reply with a SHORT phrase (3-7 words) naming that judging criterion at a "
    "level where two tasks judged the same way would get the same phrase. Reply with ONLY the "
    "phrase — no punctuation, no explanation.\n\n"
    "These examples show the abstraction LEVEL, not allowed answers — coin whatever fits:\n"
    "- 'In the lab is a watermelon. Eat as much as possible.' -> largest amount achieved\n"
    "- 'Paint the best picture of a horse.' -> subjective quality judgement\n"
    "- 'Get the potato into the hole fastest.' -> least time taken"
)
AXES = {
    "activity": {
        "system": _ACTIVITY_SYSTEM,
        "hand_field": "activity_type",
        "object_description": "task activities",
        "corpus_description": "short phrases describing the core activity of Taskmaster tasks",
    },
    "judging": {
        "system": _JUDGING_SYSTEM,
        "hand_field": "judging_criterion",
        "object_description": "task judging criteria",
        "corpus_description": "short phrases describing how Taskmaster tasks are judged to pick a winner",
    },
}


class OpusAnthropicNamer(AsyncAnthropicNamer):
    """AsyncAnthropicNamer minus `temperature` (removed on Opus 4.7/4.8); see pipeline/04."""

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


class CompatEVoCClusterer(EVoCClusterer):
    """toponymy 0.5.0's EVoCClusterer targets an older evoc API (min_num_clusters,
    next_cluster_size_quantile); evoc 0.3.1 renamed/removed those. Rebuild self.evoc with
    the 0.3.1 constructor and inherit the (compatible) fit()."""

    def __init__(self, base_min_cluster_size=20, noise_level=0.5, n_neighbors=15, min_samples=5, random_state=42):
        Clusterer.__init__(self)
        self.verbose = False
        self.evoc = evoc.EVoC(
            base_min_cluster_size=base_min_cluster_size,
            noise_level=noise_level,
            n_neighbors=n_neighbors,
            min_samples=min_samples,
            random_state=random_state,
        )


# ── stage 1: open-ended characterisation (Opus) ──────────────────────────────


async def _char_one(client, sem, system, task_id, brief) -> str:
    last = None
    for attempt in range(EXTRACT_MAX_RETRIES):
        try:
            async with sem:
                resp = await client.messages.create(
                    model=ANTHROPIC_MODEL_EXTRACT,
                    max_tokens=60,
                    system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": f"Task brief:\n{brief}"}],
                )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < EXTRACT_MAX_RETRIES - 1:
                await asyncio.sleep(min(2**attempt * 2, 30))
    raise RuntimeError(f"characterise failed for task {task_id}: {last}")


async def _characterise(df, system, cache_dir) -> None:
    from tqdm import tqdm

    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    sem = asyncio.Semaphore(EXTRACT_CONCURRENCY)
    todo = [(int(t), b) for t, b in zip(df.task_id, df.embed_text) if not (cache_dir / f"{int(t)}.txt").exists()]
    print(f"characterising {len(todo)} tasks ({len(df) - len(todo)} cached)")
    if not todo:
        return

    async def _do(tid, brief):
        (cache_dir / f"{tid}.txt").write_text(await _char_one(client, sem, system, tid, brief))

    tasks = [_do(t, b) for t, b in todo]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="characterising"):
        await coro


# ── stage 2: embed the characterisations (Cohere) ────────────────────────────


def _embed(texts) -> np.ndarray:
    import cohere

    client = cohere.ClientV2(api_key=CO_API_KEY)
    out = []
    for i in range(0, len(texts), 96):
        chunk = texts[i : i + 96]
        for attempt in range(5):
            try:
                resp = client.embed(
                    model=COHERE_EMBED_MODEL,
                    input_type=COHERE_INPUT_TYPE,
                    texts=chunk,
                    output_dimension=COHERE_OUTPUT_DIM,
                    embedding_types=["float"],
                )
                out.extend(resp.embeddings.float_)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2**attempt * 5, 60))
    return np.asarray(out, dtype=np.float32)


# ── stage 3: EVoC + Toponymy naming on the FULL embeddings ───────────────────


def _discover(characterisations, embeddings, object_description, corpus_description):
    llm = OpusAnthropicNamer(api_key=ANTHROPIC_API_KEY, model=ANTHROPIC_MODEL_EXTRACT, max_concurrent_requests=12)
    embedder = CohereEmbedder(api_key=CO_API_KEY, model=COHERE_EMBED_MODEL)
    clusterer = CompatEVoCClusterer(base_min_cluster_size=20)
    tm = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description=object_description,
        corpus_description=corpus_description,
        lowest_detail_level=0.5,
        highest_detail_level=1.0,
    )
    np.random.seed(42)
    # EVoC reduces internally, so the full embeddings go into both slots (the documented idiom).
    tm.fit(objects=characterisations, embedding_vectors=embeddings, clusterable_vectors=embeddings)
    return tm.topic_name_vectors_  # list of per-doc name arrays, finest first


# ── stage 4: analysis (candidate taxonomy + cross-tab vs the hand slugs) ──────


def _analyze(df_out, names_by_layer, hand_field, out_path) -> None:
    sf = pd.read_parquet(STRUCTURED_FIELDS_PARQUET, columns=["task_id", hand_field])
    df = df_out.merge(sf, on="task_id", how="left")
    for li, names in enumerate(names_by_layer):
        df[f"region_L{li}"] = np.asarray(names, dtype=object)
    n_layers = len(names_by_layer)

    print(f"\n=== EVoC produced {n_layers} layer(s) — candidate taxonomy per layer ===")
    for li in range(n_layers):
        vc = df[f"region_L{li}"].value_counts()
        named = vc[vc.index != "Unlabelled"]
        print(f"\nLayer {li}: {len(named)} regions, {int(vc.get('Unlabelled', 0))} unlabelled")
        for region, cnt in named.items():
            print(f"  [{cnt:>3}] {region}")

    # Cross-tab the layer whose region count is closest to ~12 against the hand slugs.
    best = min(range(n_layers), key=lambda li: abs(df[f"region_L{li}"].nunique() - 12))
    col = f"region_L{best}"
    print(f"\n=== cross-tab: discovered regions (layer {best}) vs hand {hand_field} ===")
    for region, sub in df[df[col] != "Unlabelled"].groupby(col):
        top = sub[hand_field].value_counts().head(3)
        top_str = ", ".join(f"{k}×{v}" for k, v in top.items())
        print(f"\n• {region}  (n={len(sub)})  hand: {top_str}")
        for e in sub["brief"].head(2):
            print(f"    e.g. {str(e)[:84]}")

    df.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=list(AXES), default="activity")
    args = parser.parse_args()
    cfg = AXES[args.axis]
    print(f"axis: {args.axis}  (cross-tab vs hand '{cfg['hand_field']}')")

    if not ANTHROPIC_API_KEY or not CO_API_KEY:
        raise SystemExit("need ANTHROPIC_API_KEY and CO_API_KEY")
    cache_dir = OUT_DIR / f"{args.axis}_char_cache"
    cache_dir.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{args.axis}_discovery.parquet"

    df = pd.read_parquet(TASK_ROWS_PARQUET, columns=["task_id", "embed_text"])
    df = df[df.embed_text.fillna("").str.len() > 0].reset_index(drop=True)
    print(f"{len(df)} tasks")

    asyncio.run(_characterise(df, cfg["system"], cache_dir))
    char = [(cache_dir / f"{int(t)}.txt").read_text().strip() for t in df.task_id]
    df_out = pd.DataFrame(
        {"task_id": df.task_id.astype(int).values, "brief": df.embed_text.values, "characterisation": char}
    )
    print("sample characterisations:")
    for b, c in list(zip(df_out.brief, df_out.characterisation))[:6]:
        print(f"  {c[:38]:<40} <- {b[:58]}")

    emb = _embed(char)
    print(f"embedded characterisations: {emb.shape}")

    names_by_layer = _discover(char, emb, cfg["object_description"], cfg["corpus_description"])
    _analyze(df_out, names_by_layer, cfg["hand_field"], out_path)


if __name__ == "__main__":
    main()
