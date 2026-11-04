# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Run evaluation tasks against an OpenAI-compatible server and store every item.

Greedy decoding, each task's fixed ``max_tokens``. Items of all tasks share one pool of
``--workers`` concurrent requests. The output keeps, per item, the completion, ``ok`` and
the score details, plus the configuration (task versions, seed, fingerprints) and the
timing, and is rewritten as items finish: rerunning the same command resumes, redoing
only missing or failed items (a changed task version or seed redoes that task).

    python bench/run.py --url http://127.0.0.1:18731 --model NAME \
        --tasks gsm8k,code,ifeval,toolcall,needle,mc,decision --workers 16 --out results/TAG.json
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import tasks as registry  # noqa: E402
from bench.client import Client, strip_think  # noqa: E402


def task_config(t, n: int, limit, seed: int) -> dict:
    cfg = {
        "version": t.VERSION,
        "seed": seed,
        "limit": limit,
        "n": n,
        "max_tokens": t.MAX_TOKENS,
        "effective_limit": limit if limit is not None else t.DEFAULT_LIMIT,
        "datasets": [list(spec) for spec in getattr(t, "DATASETS", ())],
    }
    for key in ("AGREE", "METRICS"):
        if hasattr(t, key):
            cfg[key.lower()] = [
                list(x) if isinstance(x, tuple) else x for x in getattr(t, key)
            ]
    if hasattr(t, "fingerprint"):
        cfg["fingerprint"] = t.fingerprint()
    return cfg


def save(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, path)


def completed(row: dict) -> bool:
    """Wrong answers are completed work; request/scoring failures are not."""
    return (
        "ok" in row
        and "error" not in row
        and "score_error" not in row.get("details", {})
        and row.get("status", "complete") == "complete"
    )


def update_completion(state: dict, *, finished: bool) -> None:
    completion = state["completion"]
    for name, task in completion["tasks"].items():
        rows = state["results"].get(name, {}) if "error" not in task else {}
        task["completed"] = sum(completed(row) for row in rows.values())
        task["request_errors"] = sum("error" in row for row in rows.values())
        task["score_errors"] = sum(
            "score_error" in row.get("details", {}) for row in rows.values()
        )
        expected = task["expected"]
        task["missing"] = max(0, expected - len(rows)) if expected is not None else None
        if "error" not in task:
            task["status"] = (
                "complete"
                if expected and task["completed"] == expected
                else "incomplete"
                if finished
                else "running"
            )
    state["status"] = (
        "complete"
        if completion["tasks"]
        and not completion["errors"]
        and all(t["status"] == "complete" for t in completion["tasks"].values())
        else "incomplete"
        if finished
        else "running"
    )


def work(client: Client, t, item: dict) -> dict:
    row = {"id": item["id"], "group": item.get("group")}
    t0 = time.time()
    try:
        if hasattr(t, "request"):
            completion, usage = t.request(client, item)
        else:
            completion, usage = client.generate(t.prompt(item), t.MAX_TOKENS)
    except (
        Exception,
        SystemExit,
    ) as e:  # record failures for resume, including task exits
        return row | {
            "status": "request_error",
            "error": f"{type(e).__name__}: {e}",
            "ok": False,
            "latency": time.time() - t0,
        }
    t1 = time.time()
    try:
        res = dict(t.score(item, strip_think(completion)))
        ok = bool(res.pop("ok"))
        status = "complete"
    except (Exception, SystemExit) as e:
        ok = False
        status = "score_error"
        res = {"score_error": f"{type(e).__name__}: {e}"}
    return row | {
        "status": status,
        "completion": completion,
        "ok": ok,
        "details": res,
        "usage": usage,
        "latency": t1 - t0,
        "score_s": time.time() - t1,
    }


def run_suite(
    client: Client,
    tasks: dict,
    limit,
    seed: int,
    workers: int,
    out: Path,
    save_every: float = 20.0,
    log=print,
    command: list[str] | None = None,
) -> dict:
    state = {"config": {}, "results": {}, "timing": {}, "attempts": []}
    if out.exists():
        state = json.loads(out.read_text())
        for key, value in (
            ("model", client.model),
            ("url", client.url),
            ("chat_extra", client.chat_extra),
        ):
            if key in state["config"] and state["config"][key] != value:
                raise ValueError(
                    f"cannot resume with a different {key}; use a new --out"
                )
    state["schema_version"] = 1
    state["status"] = "running"
    state["completion"] = {
        "requested_tasks": list(tasks),
        "tasks": {name: {"status": "pending", "expected": None} for name in tasks},
        "errors": [],
    }
    state.setdefault("attempts", []).append(
        {"started_at": time.time(), "command": command, "workers": workers}
    )
    state["config"] |= {
        "model": client.model,
        "url": client.url,
        "seed": seed,
        "chat_extra": client.chat_extra,
        "requested_tasks": list(tasks),
        "limit": limit,
        "workers": workers,
        "timeout": client.timeout,
        "retries": client.retries,
    }
    state["config"].setdefault("tasks", {})
    save(out, state)
    todo = []
    for name, t in tasks.items():
        task = {"status": "running", "expected": None}
        state["completion"]["tasks"][name] = task
        try:
            if isinstance(t, BaseException):
                raise t
            items = t.load(limit if limit is not None else t.DEFAULT_LIMIT, seed)
            cfg = task_config(t, len(items), limit, seed)
            cfg["items_sha256"] = hashlib.sha256(
                json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            ids = {it["id"] for it in items}
            if len(ids) != len(items) or any(not isinstance(i, str) for i in ids):
                raise ValueError("task item IDs must be unique strings")
            if not items:
                raise ValueError("task loaded no items")
        except (Exception, SystemExit) as e:
            task |= {"status": "load_error", "error": f"{type(e).__name__}: {e}"}
            log(f"{name}: incomplete, {task['error']}")
            continue
        except KeyboardInterrupt:
            task |= {"status": "incomplete", "error": "KeyboardInterrupt"}
            state["completion"]["errors"].append(
                "KeyboardInterrupt while loading tasks"
            )
            update_completion(state, finished=True)
            state["attempts"][-1] |= {
                "finished_at": time.time(),
                "status": state["status"],
            }
            save(out, state)
            raise
        task["expected"] = len(items)
        old = state["config"]["tasks"].get(name, {})
        keep = state["results"].get(name, {})
        if any(
            old.get(k) != cfg.get(k)
            for k in ("version", "seed", "fingerprint", "items_sha256")
        ):
            keep = {}
        keep = {i: r for i, r in keep.items() if i in ids}
        state["config"]["tasks"][name] = cfg
        state["results"][name] = keep
        pending = [
            (name, t, it)
            for it in items
            if it["id"] not in keep or not completed(keep[it["id"]])
        ]
        todo.extend(pending)
        log(
            f"{name}: {len(items)} items, {len(items) - len(pending)} done, v{t.VERSION}"
        )
    if not tasks:
        state["completion"]["errors"].append("no tasks requested")
    update_completion(state, finished=False)
    save(out, state)
    last = time.time()
    started = {}
    t_start = time.time()
    try:
        with ThreadPoolExecutor(max(1, workers)) as pool:
            futs = {}
            for name, t, it in todo:
                started.setdefault(name, time.time())
                futs[pool.submit(work, client, t, it)] = name
            for k, f in enumerate(as_completed(futs), 1):
                name = futs[f]
                row = f.result()
                state["results"][name][row["id"]] = row
                tm = state["timing"].setdefault(name, {"requests": 0, "seconds": 0.0})
                tm["requests"] += 1
                tm["latency_sum"] = tm.get("latency_sum", 0.0) + row["latency"]
                tm["end"] = time.time()
                if time.time() - last > save_every:
                    update_completion(state, finished=False)
                    save(out, state)
                    last = time.time()
                if k % 100 == 0:
                    log(f"{k}/{len(todo)} requests")
    except (Exception, SystemExit) as e:
        state["completion"]["errors"].append(f"{type(e).__name__}: {e}")
        log(f"harness incomplete: {type(e).__name__}: {e}")
    except KeyboardInterrupt:
        state["completion"]["errors"].append("KeyboardInterrupt")
        raise
    finally:
        for name, tm in state["timing"].items():
            if name in started and "end" in tm:
                tm["seconds"] = round(tm["seconds"] + tm.pop("end") - started[name], 2)
        state["timing"]["_wall"] = round(
            state["timing"].get("_wall", 0.0) + time.time() - t_start, 2
        )
        update_completion(state, finished=True)
        state["attempts"][-1] |= {"finished_at": time.time(), "status": state["status"]}
        save(out, state)
    for name in tasks:
        rows = state["results"].get(name)
        if rows is None:
            continue
        done = [r for r in rows.values() if completed(r)]
        acc = sum(r["ok"] for r in done) / len(done) if done else float("nan")
        status = state["completion"]["tasks"][name]["status"]
        log(
            f"{name:<10} n={len(done):<5} acc={acc:.4f} "
            f"errors={len(rows) - len(done)} status={status}"
        )
    return state


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument(
        "--tasks",
        default="all",
        help=f"comma list of {registry.NAMES}, or code (humaneval,mbpp), or all",
    )
    ap.add_argument("--limit", type=int, default=None, help="max items per task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument(
        "--thinking",
        action="store_true",
        help="leave the chat template's thinking mode on",
    )
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.limit is not None and a.limit <= 0:
        ap.error("--limit must be positive")
    if a.workers <= 0:
        ap.error("--workers must be positive")
    extra = {} if a.thinking else {"chat_template_kwargs": {"enable_thinking": False}}
    client = Client(a.url, a.model, timeout=a.timeout, chat_extra=extra)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    selected = {}
    names = dict.fromkeys(
        name
        for token in a.tasks.split(",")
        for name in registry.ALIASES.get(token.strip(), (token.strip(),))
        if name
    )
    for name in names:
        try:
            selected.update(registry.resolve(name))
        except (Exception, SystemExit) as e:
            selected[name] = e
    try:
        state = run_suite(
            client,
            selected,
            a.limit,
            a.seed,
            a.workers,
            a.out,
            command=["bench/run.py", *(sys.argv[1:] if argv is None else argv)],
        )
    except (OSError, ValueError) as e:
        print(f"benchmark failed: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
