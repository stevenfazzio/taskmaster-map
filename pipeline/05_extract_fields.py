"""Extract structured fields from each task brief via Claude Opus 4.8.

Loads the enum/rule schema from pipeline/taxonomy.json, calls Opus once per task,
validates responses against the schema, and emits a flat-column parquet for the
visualize stage. Resumable: per-task JSON files in data/structured_fields_cache/
are skipped on rerun.

Each field comes back as {value, quote} — the quote is a short verbatim span from
the brief justifying the choice (great hover evidence). Invalid slugs are coerced
to 'other' at render time (stage 06); the truth stays in the parquet.

A second, single-purpose pass generates a 2-4 emoji summary per task (its own
prompt + data/emoji_cache/, merged in as the `task_emoji` column). Keeping it
separate from the taxonomy call preserves the tuned wit and means a rerun after
the taxonomy is cached regenerates only the cheap emoji.

A third pass classifies which of the 16 filter motifs (motifs.py) each task
involves (own prompt + data/motif_cache/, merged in as `motif_tags`), decoupled
from the gist so tuning a motif boundary disturbs neither taxonomy nor emoji. A
final reconcile step keeps the gist and the filter buttons consistent: a concrete
glyph the classifier missed adds its motif; a metaphor-prone glyph the classifier
rejected is stripped from the gist (see motifs.STRIP_POLICY_KEYS).

Validate first on a sample (`--limit 50`), eyeball the tags, then run the full set
(the sample's cached results are reused). Opus 4.8 takes no `temperature`, so the
call is just model + max_tokens + system (cached) + messages.

Input:  data/task_rows.parquet  +  pipeline/taxonomy.json
Output: data/structured_fields.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import tempfile
from pathlib import Path

import pandas as pd
from anthropic import AsyncAnthropic
from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL_EXTRACT,
    EMOJI_CACHE_DIR,
    EMOJI_MAX_TOKENS,
    EXTRACT_CONCURRENCY,
    EXTRACT_MAX_RETRIES,
    EXTRACT_MAX_TOKENS,
    MOTIF_CACHE_DIR,
    MOTIF_MAX_TOKENS,
    STRUCTURED_FIELDS_CACHE_DIR,
    STRUCTURED_FIELDS_PARQUET,
    SUBSET_SEED,
    TASK_ROWS_PARQUET,
    TAXONOMY_JSON,
)
from motifs import GLYPH_TO_KEY, MOTIF_KEYS, MOTIFS, STRIP_POLICY_KEYS
from tqdm import tqdm


def _safe_filename(task_id) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "__", str(task_id))


def _field_block(name: str, spec: dict) -> str:
    t = spec["type"]
    if t == "single-select":
        lines = [f"FIELD: {name} (pick EXACTLY ONE slug)"]
        for cat in spec["categories"]:
            lines.append(f"  - {cat['name']}: {cat['description']}")
        return "\n".join(lines)
    if t == "multi-select":
        lines = [
            f"FIELD: {name} (return a JSON LIST of applicable slugs. "
            "If NO slug applies, return an empty list []. Do NOT invent 'not_stated' for this field.)"
        ]
        for cat in spec["categories"]:
            lines.append(f"  - {cat['name']}: {cat['description']}")
        if "rule" in spec:
            lines.append(f"  EXTRA RULE: {spec['rule']}")
        return "\n".join(lines)
    if t == "open-list":
        return f"FIELD: {name} (return a JSON LIST of raw strings)\n  RULE: {spec['rule']}"
    raise ValueError(f"Unknown field type: {t}")


def build_system_prompt(taxonomy: dict) -> tuple[str, list[str]]:
    fields = [k for k in taxonomy if not k.startswith("_")]
    field_blocks = "\n\n".join(_field_block(f, taxonomy[f]) for f in fields)

    shape = "{\n"
    for f in fields:
        shape += f'  "{f}": {{ "value": <per field rule>, "quote": "<≤20-word verbatim span from the brief, or \'not_stated\'>" }},\n'  # noqa: E501
    shape = shape.rstrip(",\n") + "\n}"

    system = (
        "You label tasks from the British comedy panel show Taskmaster against a fixed taxonomy. "
        "You are given one task's brief — the instruction read or shown to the contestants.\n\n"
        "RULES:\n"
        "- For slug-valued fields, return one of the provided slugs verbatim. No paraphrases, no combined values like 'a / b'.\n"  # noqa: E501
        '- For LIST-typed fields, `value` MUST be a JSON array even if only one item applies: ["item"]. Never a bare string.\n'  # noqa: E501
        "- Each field captures a DIFFERENT axis. Do NOT reuse a slug from one field as the value for another. "
        "`activity_type` = what the contestant physically DOES; `judging_criterion` = how the winner is DECIDED/measured; "  # noqa: E501
        "`task_twist` = special constraints or gimmicks (a LIST, usually empty or short); "
        "`key_props` = the notable physical objects (hover flavour only). Example: "
        "'Throw the most balls into the bucket in 60 seconds' -> activity_type='physical-feat', "
        "judging_criterion='most-quantity', task_twist=['time-limit'], key_props=['ball','bucket'].\n"
        "- For each field include `quote`: a short verbatim span from the brief that justified your choice. "
        "If the brief is silent on that axis, use the sentinel 'not_stated'.\n"
        "- Judge from the brief itself. Prize-task briefs are often terse (e.g. 'Most unusual item.') — lean on the format hint.\n"  # noqa: E501
        "- Output strictly valid JSON. No prose, no markdown fences, no commentary outside the JSON object.\n\n"
        "FIELD DEFINITIONS:\n\n"
        f"{field_blocks}\n\n"
        "OUTPUT SHAPE (fill in values; do not change the structure):\n"
        f"{shape}"
    )
    return system, fields


def _build_user_message(brief: str, task_format: str) -> str:
    return f"Taskmaster task (format: {task_format}):\n---\n{brief}\n---"


async def _extract_one(client, sem, system, task_id, brief, task_format) -> dict:
    last_err = None
    for attempt in range(EXTRACT_MAX_RETRIES):
        try:
            async with sem:
                resp = await client.messages.create(
                    model=ANTHROPIC_MODEL_EXTRACT,
                    max_tokens=EXTRACT_MAX_TOKENS,
                    system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": _build_user_message(brief, task_format)}],
                )
            raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return {
                "task_id": int(task_id),
                "raw_text": raw,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001 — one bad task shouldn't crash the batch
            last_err = e
            if attempt < EXTRACT_MAX_RETRIES - 1:
                await asyncio.sleep(min(2**attempt * 2, 30))
    return {"task_id": int(task_id), "raw_text": None, "error": f"{type(last_err).__name__}: {last_err}"}


def _save_result(result: dict) -> None:
    out_path = STRUCTURED_FIELDS_CACHE_DIR / f"{_safe_filename(result['task_id'])}.json"
    tmp_fd, tmp_path = tempfile.mkstemp(dir=STRUCTURED_FIELDS_CACHE_DIR, suffix=".json.tmp")
    os.close(tmp_fd)
    try:
        Path(tmp_path).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


async def _run_extractions(rows: pd.DataFrame, system: str) -> None:
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    sem = asyncio.Semaphore(EXTRACT_CONCURRENCY)

    todo = []
    for _, row in rows.iterrows():
        cache_path = STRUCTURED_FIELDS_CACHE_DIR / f"{_safe_filename(row['task_id'])}.json"
        if cache_path.exists():
            continue
        todo.append((row["task_id"], row["embed_text"], row["task_format"]))
    print(f"{len(todo)} to extract ({len(rows) - len(todo)} already cached)")
    if not todo:
        return

    async def _do(task_id, brief, task_format):
        res = await _extract_one(client, sem, system, task_id, brief, task_format)
        _save_result(res)

    tasks = [_do(tid, b, fmt) for tid, b, fmt in todo]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="extracting"):
        await coro


# ── emoji summary pass ──────────────────────────────────────────────────────
# A separate, single-purpose call: the taxonomy prompt above classifies; this one
# performs. Bundling the two dilutes the wit, so the emoji gets its own prompt and
# its own cache — a rerun regenerates only the emoji once the taxonomy is cached.
# The literal-use guardrail below is built from motifs.py so it can't drift from the classifier.
_MOTIF_GUARDRAIL = " · ".join(f"{m['glyph']} {m['sense']}" for m in MOTIFS)
EMOJI_SYSTEM = (
    "You produce a short emoji summary of a task from the comedy panel show Taskmaster — "
    "a glanceable, witty visual gist. These tasks are deliberately silly; the emoji should be "
    "playful, and visual puns are welcome (e.g. 🕳️🥊 for a 'hole punch').\n\n"
    "Use 2-4 emoji, most evocative first, capturing in priority order: (1) the funniest or most "
    "defining element, (2) the core action + key object, (3) a memorable twist, constraint, or count. "
    "Add a 3rd/4th emoji ONLY when it adds a DISTINCT meaningful element — never a duplicate, a "
    "near-synonym, or decoration.\n\n"
    "Rules:\n"
    "- For abstract 'bring the most/best X' prize tasks, depict the QUALITY being judged "
    "(🤯 extraordinary, 😬 awkward, 🤑 expensive, 🪞 narcissistic) rather than a generic 🎁.\n"
    "- Avoid filler (✨ 🤔 ❓) and generic 🎁 unless genuinely the best fit.\n"
    "- Prefer specific, recognisable emoji over vague ones.\n"
    "- These 16 glyphs are filter-button icons, so use each ONLY in the sense given here, never as "
    "a metaphor (so a button never contradicts a gist it appears in):\n"
    f"  {_MOTIF_GUARDRAIL}.\n"
    "Every OTHER emoji is free to be punny or metaphorical — 🎩 for 'posh', 🪖 for 'a soldier' — "
    "only these 16 are reserved for these senses.\n"
    "- Reply with ONLY the emoji — no words, no spaces, no explanation.\n\n"
    "Examples (style/length, not content):\n"
    "- 'Eat as much watermelon as possible.' -> 🍉😋\n"
    "- 'Do something that will look impressive in reverse.' -> ⏪🤸\n"
    "- 'Most extraordinary souvenir.' -> 🌍🤯😮"
)


async def _emoji_one(client, sem, task_id, brief) -> str:
    last_err = None
    for attempt in range(EXTRACT_MAX_RETRIES):
        try:
            async with sem:
                resp = await client.messages.create(
                    model=ANTHROPIC_MODEL_EXTRACT,
                    max_tokens=EMOJI_MAX_TOKENS,
                    system=[{"type": "text", "text": EMOJI_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": f"Task brief:\n{brief}"}],
                )
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        except Exception as e:  # noqa: BLE001 — one bad task shouldn't crash the batch
            last_err = e
            if attempt < EXTRACT_MAX_RETRIES - 1:
                await asyncio.sleep(min(2**attempt * 2, 30))
    print(f"  emoji failed for {task_id}: {type(last_err).__name__}: {last_err}")
    return ""  # not cached (see _run_emoji) so a later rerun retries it


def _load_emoji(task_id) -> str:
    p = EMOJI_CACHE_DIR / f"{_safe_filename(task_id)}.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def _save_emoji(task_id, emoji: str) -> None:
    out_path = EMOJI_CACHE_DIR / f"{_safe_filename(task_id)}.txt"
    tmp_fd, tmp_path = tempfile.mkstemp(dir=EMOJI_CACHE_DIR, suffix=".txt.tmp")
    os.close(tmp_fd)
    try:
        Path(tmp_path).write_text(emoji, encoding="utf-8")
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


async def _run_emoji(rows: pd.DataFrame) -> None:
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    sem = asyncio.Semaphore(EXTRACT_CONCURRENCY)

    todo = []
    for _, row in rows.iterrows():
        if (EMOJI_CACHE_DIR / f"{_safe_filename(row['task_id'])}.txt").exists():
            continue
        todo.append((row["task_id"], row["embed_text"]))
    print(f"{len(todo)} emoji to generate ({len(rows) - len(todo)} already cached)")
    if not todo:
        return

    async def _do(task_id, brief):
        emoji = await _emoji_one(client, sem, task_id, brief)
        if emoji:  # only cache successes — empties stay uncached so reruns retry them
            _save_emoji(task_id, emoji)

    tasks = [_do(tid, b) for tid, b in todo]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="emoji"):
        await coro


# ── motif-tag pass ──────────────────────────────────────────────────────────
# A third single-purpose call, fully decoupled from the witty gist: it classifies which of the
# 16 filter motifs (motifs.py) a task CENTRALLY involves, from the brief alone. Own prompt + cache
# so tuning a motif boundary doesn't disturb the cached taxonomy or emoji. Drives the 06 buttons.
MOTIF_SYSTEM = (
    "You tag a task from the British comedy panel show Taskmaster with which of 16 recurring "
    "motifs it CENTRALLY involves — these power the site's filter buttons. A motif applies only "
    "when it is a core element of what the task asks, never an incidental mention or a forbidden "
    "side-effect ('do not break the vase' is NOT the smash motif). A task may match several "
    "motifs, or none. For abstract 'best/most X' prize tasks, tag a motif only if the judged "
    "quality IS that motif (heaviest -> weigh; sturdiest -> none). Do NOT tag music, film, or art "
    "merely because a task is a 'performance' or 'present X' — tag music only for ACTUAL music or "
    "singing, film only for actual filming/video or acting a scripted scene, art only for actual "
    "drawing/painting/sculpture.\n\n"
    "MOTIFS — tag the key when its rule fits:\n"
    + "\n".join(f"  {m['key']}: {m['definition']}" for m in MOTIFS)
    + "\n\nReply with ONLY the applicable keys, space-separated, lowercase, drawn from this exact "
    "list:\n  " + " ".join(MOTIF_KEYS) + "\nIf none apply, reply with the single word: none."
)


async def _motif_one(client, sem, task_id, brief) -> str | None:
    last_err = None
    valid = set(MOTIF_KEYS)
    for attempt in range(EXTRACT_MAX_RETRIES):
        try:
            async with sem:
                resp = await client.messages.create(
                    model=ANTHROPIC_MODEL_EXTRACT,
                    max_tokens=MOTIF_MAX_TOKENS,
                    system=[{"type": "text", "text": MOTIF_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": f"Task brief:\n{brief}"}],
                )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").lower()
            tags = [w for w in re.split(r"[^a-z_]+", text) if w in valid]
            return " ".join(dict.fromkeys(tags))  # dedupe, keep order; "" when none apply
        except Exception as e:  # noqa: BLE001 — one bad task shouldn't crash the batch
            last_err = e
            if attempt < EXTRACT_MAX_RETRIES - 1:
                await asyncio.sleep(min(2**attempt * 2, 30))
    print(f"  motif failed for {task_id}: {type(last_err).__name__}: {last_err}")
    return None  # not cached (see _run_motif) so a later rerun retries it


def _load_motif(task_id) -> str:
    p = MOTIF_CACHE_DIR / f"{_safe_filename(task_id)}.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def _save_motif(task_id, tags: str) -> None:
    out_path = MOTIF_CACHE_DIR / f"{_safe_filename(task_id)}.txt"
    tmp_fd, tmp_path = tempfile.mkstemp(dir=MOTIF_CACHE_DIR, suffix=".txt.tmp")
    os.close(tmp_fd)
    try:
        Path(tmp_path).write_text(tags, encoding="utf-8")
        os.replace(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


async def _run_motif(rows: pd.DataFrame) -> None:
    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    sem = asyncio.Semaphore(EXTRACT_CONCURRENCY)

    todo = []
    for _, row in rows.iterrows():
        if (MOTIF_CACHE_DIR / f"{_safe_filename(row['task_id'])}.txt").exists():
            continue
        todo.append((row["task_id"], row["embed_text"]))
    print(f"{len(todo)} motif tags to generate ({len(rows) - len(todo)} already cached)")
    if not todo:
        return

    async def _do(task_id, brief):
        tags = await _motif_one(client, sem, task_id, brief)
        if tags is not None:  # cache successes incl. a genuine empty 'none'; failures retry
            _save_motif(task_id, tags)

    tasks = [_do(tid, b) for tid, b in todo]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="motif"):
        await coro


def _reconcile_motifs(emoji: str, tags: str) -> tuple[str, str]:
    # Reconcile the witty gist's glyphs with the classifier's tags so a filter button and the gist
    # it appears in never contradict each other. For a glyph present in the gist but NOT tagged:
    #   - concrete glyph (egg, clothing, ...): ADD the tag — the classifier missed a literal use.
    #   - metaphor-prone glyph (STRIP_POLICY_KEYS): STRIP it from the gist — the appearance is almost
    #     always metaphorical (target for "match", burst for "wow"), so trust the classifier.
    # Variation selectors are dropped for the membership test so the scales glyph matches a bare one.
    out_tags = tags.split()
    out_emoji = emoji or ""
    for glyph, key in GLYPH_TO_KEY.items():
        if glyph.replace("\ufe0f", "") in out_emoji.replace("\ufe0f", "") and key not in out_tags:
            if key in STRIP_POLICY_KEYS:
                out_emoji = out_emoji.replace(glyph + "\ufe0f", "").replace(glyph, "")
            else:
                out_tags.append(key)
    return out_emoji, " ".join(out_tags)


def _parse_json(raw_text):
    if raw_text is None:
        return None, "no_raw_text"
    m = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not m:
        return None, "no_json_object"
    try:
        return json.loads(m.group(0)), None
    except json.JSONDecodeError as e:
        return None, f"json_decode: {e}"


def _validate(parsed, taxonomy, fields):
    issues = []
    for f in fields:
        spec = taxonomy[f]
        t = spec["type"]
        entry = parsed.get(f)
        if not isinstance(entry, dict) or "value" not in entry:
            issues.append(f"{f}:missing")
            continue
        val = entry["value"]
        allowed = {c["name"] for c in spec.get("categories", [])}
        if t == "single-select":
            if not isinstance(val, str) or val not in allowed:
                issues.append(f"{f}:invalid_slug:{val!r}")
        elif t == "multi-select":
            if not isinstance(val, list):
                issues.append(f"{f}:not_list")
            elif any(not isinstance(v, str) or v not in allowed for v in val):
                bad = [v for v in val if not isinstance(v, str) or v not in allowed]
                issues.append(f"{f}:invalid_slugs:{bad}")
        elif t == "open-list":
            if not isinstance(val, list):
                issues.append(f"{f}:not_list")
    return issues


def aggregate(taxonomy, fields, corpus_task_ids) -> pd.DataFrame:
    # Restrict to the current corpus so dropped tasks (e.g. deduped series) don't
    # linger as orphaned cache entries in the output. Cache files for tasks no longer
    # in task_rows.parquet are simply skipped (harmless to leave on disk).
    corpus = {int(t) for t in corpus_task_ids}
    rows = []
    for p in sorted(STRUCTURED_FIELDS_CACHE_DIR.glob("*.json")):
        data = json.loads(p.read_text())
        if int(data["task_id"]) not in corpus:
            continue
        parsed, parse_err = _parse_json(data.get("raw_text"))
        issues = _validate(parsed, taxonomy, fields) if parsed else []
        row = {
            "task_id": data["task_id"],
            "error": data.get("error"),
            "parse_error": parse_err,
            "validation_issues": "; ".join(issues) if issues else None,
        }
        for f in fields:
            entry = (parsed or {}).get(f) or {}
            val = entry.get("value")
            row[f] = json.dumps(val, ensure_ascii=False) if isinstance(val, list) else val
            row[f"{f}_quote"] = entry.get("quote")
        rows.append(row)

    df = pd.DataFrame(rows)
    raw_emoji = [_load_emoji(t) for t in df["task_id"]]
    raw_motifs = [_load_motif(t) for t in df["task_id"]]
    reconciled = [_reconcile_motifs(e, m) for e, m in zip(raw_emoji, raw_motifs)]
    df["task_emoji"] = [e for e, _ in reconciled]
    df["motif_tags"] = [t for _, t in reconciled]
    n_added = sum(len(t.split()) > len(rm.split()) for (_, t), rm in zip(reconciled, raw_motifs))
    n_stripped = sum(e != re_ for (e, _), re_ in zip(reconciled, raw_emoji))
    tmp_fd, tmp_path = tempfile.mkstemp(dir=STRUCTURED_FIELDS_PARQUET.parent, suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path, index=False)
        verify = pd.read_parquet(tmp_path)
        assert len(verify) == len(df), f"{len(verify)} != {len(df)}"
        os.replace(tmp_path, STRUCTURED_FIELDS_PARQUET)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    print(f"\nWrote {len(df)} rows to {STRUCTURED_FIELDS_PARQUET}")
    print(f"  API errors:        {df['error'].notna().sum()}")
    print(f"  Parse errors:      {df['parse_error'].notna().sum()}")
    print(f"  Validation issues: {df['validation_issues'].notna().sum()}")
    print(f"  emoji present:     {(df['task_emoji'].str.len() > 0).sum()} / {len(df)}")
    print(f"  motif tags present:{(df['motif_tags'].str.len() > 0).sum()} / {len(df)}")
    print(f"  reconcile:         +{n_added} tags added (concrete glyphs), {n_stripped} orphan glyphs stripped")
    for f in fields:
        if taxonomy[f]["type"] == "single-select":
            print(f"  {f}: {df[f].value_counts().to_dict()}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="extract a random N-task sample (for taxonomy validation)"
    )
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    STRUCTURED_FIELDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EMOJI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MOTIF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not ANTHROPIC_API_KEY and not args.aggregate_only:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    taxonomy = json.loads(TAXONOMY_JSON.read_text())
    system, fields = build_system_prompt(taxonomy)
    print(f"Fields: {fields}")

    df = pd.read_parquet(TASK_ROWS_PARQUET, columns=["task_id", "embed_text", "task_format"])
    df = df[df.embed_text.notna() & (df.embed_text.str.len() > 0)].reset_index(drop=True)
    corpus_task_ids = df["task_id"].tolist()  # full corpus; aggregate spans this, not the --limit sample
    if args.limit is not None:
        df = df.sample(n=min(args.limit, len(df)), random_state=SUBSET_SEED).reset_index(drop=True)
    print(f"Corpus: {len(df)} tasks")

    if not args.aggregate_only:
        asyncio.run(_run_extractions(df, system))
        asyncio.run(_run_emoji(df))
        asyncio.run(_run_motif(df))

    aggregate(taxonomy, fields, corpus_task_ids)


if __name__ == "__main__":
    main()
