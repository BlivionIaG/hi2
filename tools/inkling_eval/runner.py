# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Run a plan against one endpoint, appending every item to a resumable JSONL.

One line per item: the rendered prompt, its hash, the completion, the score and
the transport facts.  Rerunning the same command skips complete items whose
prompt hash still matches and retries missing, failed or changed ones.  A
different label, URL, plan, seed or template refuses to resume into the same
file: use a new ``--out``.

Greedy decoding is fixed by the client (temperature 0, seed).  Completion and
scoring happen before the line is appended, so a crash never leaves a record
that claims a score it does not have.
"""

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tools.inkling_eval import scoring
from tools.inkling_eval.client import TransportError

SCHEMA = 1
TRUNCATION_GATE = 0.05
META_KEYS = (
    "schema",
    "label",
    "url",
    "plan_sha256",
    "set",
    "seed",
    "reasoning_effort",
    "template",
    "haystack_sha256",
    "decoding",
)


class RunMismatch(RuntimeError):
    """The existing output file belongs to a different run configuration."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def meta_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".meta.json")


def load_records(out: Path) -> dict[str, dict]:
    """Latest record per item id; a torn last line is ignored."""
    records: dict[str, dict] = {}
    if not out.exists():
        return records
    with out.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("item_id"):
                records[record["item_id"]] = record
    return records


def read_meta(out: Path) -> dict | None:
    path = meta_path(out)
    return json.loads(path.read_text()) if path.exists() else None


def truncation_summary(records: dict[str, dict], plan_counts: dict | None = None) -> dict:
    """Per-category length truncation and the 5% reliability gate.

    ``length`` means the server hit ``max_tokens`` before the model stopped, so
    the completion is not a wrong answer: it is an unfinished one.  Such items
    are excluded from accuracy and make a run unreliable when more than 5% of a
    category is truncated.  The denominator prefers the planned count from the
    run meta so an interrupted run cannot hide truncation.
    """
    plan_counts = plan_counts or {}
    categories = set(plan_counts) | {record.get("category") for record in records.values()}
    categories.discard(None)
    by_category = {}
    failed = []
    for category in sorted(categories):
        planned = plan_counts.get(category)
        planned = int(planned.get("items") or 0) if isinstance(planned, dict) else 0
        seen = 0
        truncated = 0
        for record in records.values():
            if record.get("category") != category:
                continue
            seen += 1
            if record.get("status") == "truncated" or record.get("finish_reason") == "length":
                truncated += 1
        items = max(planned, seen)
        rate = truncated / items if items else 0.0
        gate_failed = rate > TRUNCATION_GATE
        by_category[category] = {
            "items": items,
            "truncated": truncated,
            "rate": round(rate, 4),
            "gate_failed": gate_failed,
        }
        if gate_failed:
            failed.append(category)
    return {
        "gate": TRUNCATION_GATE,
        "by_category": by_category,
        "failed": failed,
        "reliable": not failed,
    }


def _check_meta(existing: dict, wanted: dict, *, force: bool) -> None:
    differences = {}
    for key in META_KEYS:
        if existing.get(key) != wanted.get(key):
            differences[key] = {"existing": existing.get(key), "wanted": wanted.get(key)}
    if differences and not force:
        raise RunMismatch(
            "the output file was written with a different run configuration: "
            + json.dumps(differences, sort_keys=True)
            + " (use a new --out, or --force to resume anyway)"
        )


def run_plan(
    plan: dict,
    client,
    out: Path,
    *,
    label: str,
    workers: int = 4,
    resume: bool = True,
    force: bool = False,
    log=None,
    progress_every: int = 10,
) -> dict:
    """Execute pending items; return counts and the number of requests this call made."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = log or (lambda message: print(message, file=sys.stderr, flush=True))
    meta = {
        "schema": SCHEMA,
        "label": label,
        "url": client.url,
        "model": client.model_name(),
        "plan_sha256": plan["meta"]["plan_sha256"],
        "set": plan["meta"]["set"],
        "seed": plan["meta"]["seed"],
        "reasoning_effort": plan["meta"]["reasoning_effort"],
        "template": plan["meta"]["template"],
        "haystack_sha256": plan["meta"]["haystack_sha256"],
        "contexts": plan["meta"]["contexts"],
        "skipped_contexts": plan["meta"]["skipped_contexts"],
        "plan_counts": plan["meta"]["counts"],
        "workers": workers,
        "decoding": {
            "temperature": 0,
            "top_p": 1,
            "n": 1,
            "seed": plan["meta"]["seed"],
            "skip_special_tokens": False,
        },
        "transport": "/v1/completions",
        "started_at": _now(),
        "pid": os.getpid(),
    }
    existing_meta = read_meta(out) if resume else None
    if existing_meta:
        _check_meta(existing_meta, meta, force=force)
        meta["started_at"] = existing_meta.get("started_at", meta["started_at"])
    meta_path(out).write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    records = load_records(out) if resume else {}
    pending = []
    for item in plan["items"]:
        record = records.get(item["id"])
        if (
            record
            and record.get("status") == "complete"
            and record.get("finish_reason") != "length"
            and record.get("prompt_sha256") == item["prompt_sha256"]
            and isinstance(record.get("score"), dict)
        ):
            continue
        pending.append(item)
    counters = {"complete": 0, "ok": 0, "wrong": 0, "truncated": 0, "error": 0}
    lock = threading.Lock()

    def one(item: dict) -> dict:
        started = time.monotonic()
        record = {
            "schema": SCHEMA,
            "item_id": item["id"],
            "category": item["category"],
            "subcategory": item["subcategory"],
            "label": label,
            "prompt_sha256": item["prompt_sha256"],
            "prompt": item["prompt"],
            "max_tokens": item["max_tokens"],
            "approx_prompt_tokens": item["approx_prompt_tokens"],
            "timestamp": _now(),
        }
        try:
            response = client.complete(item["prompt"], item["max_tokens"])
            truncated = response.get("finish_reason") == "length"
            record.update(
                {
                    "status": "truncated" if truncated else "complete",
                    "completion": response["text"],
                    "finish_reason": response.get("finish_reason"),
                    "usage": response.get("usage") or {},
                }
            )
            # Keep the score next to a truncated completion as diagnostic
            # evidence; it is not counted as right or wrong.
            record["score"] = scoring.score(item, response["text"])
        except TransportError as error:
            record.update({"status": "error", "error": f"transport: {error}"})
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            record.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
        record["latency_s"] = round(time.monotonic() - started, 3)
        return record

    if out.exists() and out.stat().st_size:
        with out.open("rb") as tail:
            tail.seek(-1, os.SEEK_END)
            if tail.read(1) != b"\n":
                with out.open("a", encoding="utf-8") as repair:
                    repair.write("\n")
    with out.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(one, item): item for item in pending}
            finished = 0
            try:
                for future in as_completed(futures):
                    record = future.result()
                    with lock:
                        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        handle.flush()
                    status = record["status"]
                    if status == "complete":
                        counters["complete"] += 1
                        if record["score"]["ok"]:
                            counters["ok"] += 1
                        else:
                            counters["wrong"] += 1
                    elif status == "truncated":
                        counters["truncated"] += 1
                    else:
                        counters["error"] += 1
                    finished += 1
                    if finished % progress_every == 0 or finished == len(pending):
                        log(
                            f"[{label}] {finished}/{len(pending)} requested; "
                            f"{counters['ok']} ok, {counters['wrong']} wrong, "
                            f"{counters['truncated']} truncated, {counters['error']} errors"
                        )
            except KeyboardInterrupt:
                for future in futures:
                    future.cancel()
                raise
    records = load_records(out)
    def _truncated(record: dict) -> bool:
        return record.get("status") == "truncated" or record.get("finish_reason") == "length"

    complete = sum(
        1 for record in records.values() if record.get("status") == "complete" and not _truncated(record)
    )
    ok = sum(
        1
        for record in records.values()
        if record.get("status") == "complete"
        and not _truncated(record)
        and (record.get("score") or {}).get("ok")
    )
    errors = sum(1 for record in records.values() if record.get("status") == "error")
    truncated = sum(1 for record in records.values() if _truncated(record))
    truncation = truncation_summary(records, plan["meta"].get("counts"))
    summary = {
        "label": label,
        "planned": len(plan["items"]),
        "skipped_complete": len(plan["items"]) - len(pending),
        "requested": len(pending),
        "complete": complete,
        "ok": ok,
        "wrong": complete - ok,
        "truncated": truncated,
        "errors": errors,
        "missing": len(plan["items"]) - complete - truncated - errors,
        "truncation": truncation,
        "reliable": not truncation["failed"]
        and len(plan["items"]) - complete - truncated - errors == 0
        and errors == 0,
        "finished_at": _now(),
    }
    log(json.dumps(summary, sort_keys=True))
    return summary
