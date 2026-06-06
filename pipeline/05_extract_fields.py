"""Extract structured fields from each task brief via Claude Opus 4.8.

Loads the enum/rule schema from pipeline/taxonomy.json, calls Opus once per task,
validates responses against the schema, and emits a flat-column parquet for the
visualize stage. Resumable: per-task JSON files in data/structured_fields_cache/
are skipped on rerun.

Each field comes back as {value, quote} — the quote is a short verbatim span from
the brief justifying the choice (great hover evidence). Invalid slugs are coerced
to 'other' at render time (stage 06); the truth stays in the parquet.

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
    EXTRACT_CONCURRENCY,
    EXTRACT_MAX_RETRIES,
    EXTRACT_MAX_TOKENS,
    STRUCTURED_FIELDS_CACHE_DIR,
    STRUCTURED_FIELDS_PARQUET,
    SUBSET_SEED,
    TASK_ROWS_PARQUET,
    TAXONOMY_JSON,
)
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


def aggregate(taxonomy, fields) -> pd.DataFrame:
    rows = []
    for p in sorted(STRUCTURED_FIELDS_CACHE_DIR.glob("*.json")):
        data = json.loads(p.read_text())
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
    if not ANTHROPIC_API_KEY and not args.aggregate_only:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    taxonomy = json.loads(TAXONOMY_JSON.read_text())
    system, fields = build_system_prompt(taxonomy)
    print(f"Fields: {fields}")

    df = pd.read_parquet(TASK_ROWS_PARQUET, columns=["task_id", "embed_text", "task_format"])
    df = df[df.embed_text.notna() & (df.embed_text.str.len() > 0)].reset_index(drop=True)
    if args.limit is not None:
        df = df.sample(n=min(args.limit, len(df)), random_state=SUBSET_SEED).reset_index(drop=True)
    print(f"Corpus: {len(df)} tasks")

    if not args.aggregate_only:
        asyncio.run(_run_extractions(df, system))

    aggregate(taxonomy, fields)


if __name__ == "__main__":
    main()
