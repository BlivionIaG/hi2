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
import json
import os
import sys
import threading
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


def work(client: Client, t, item: dict) -> dict:
    row = {"id": item["id"], "group": item.get("group")}
    t0 = time.time()
    try:
        if hasattr(t, "request"):
            completion, usage = t.request(client, item)
        else:
            completion, usage = client.generate(t.prompt(item), t.MAX_TOKENS)
    except Exception as e:  # a failed request is recorded and retried on resume
        return row | {
            "error": f"{type(e).__name__}: {e}",
            "ok": False,
            "latency": time.time() - t0,
        }
    t1 = time.time()
    try:
        res = dict(t.score(item, strip_think(completion)))
    except Exception as e:
        res = {"ok": False, "score_error": f"{type(e).__name__}: {e}"}
    ok = bool(res.pop("ok"))
    return row | {
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
) -> dict:
    state = {"config": {}, "results": {}, "timing": {}}
    if out.exists():
        state = json.loads(out.read_text())
        if state["config"].get("model") not in (None, client.model):
            log(
                f"warning: resuming a run of {state['config']['model']} with {client.model}"
            )
    state["config"] |= {
        "model": client.model,
        "url": client.url,
        "seed": seed,
        "chat_extra": client.chat_extra,
    }
    state["config"].setdefault("tasks", {})
    todo = []
    for name, t in tasks.items():
        try:
            items = t.load(limit if limit is not None else t.DEFAULT_LIMIT, seed)
        except Exception as e:
            log(f"{name}: skipped, {e}")
            continue
        cfg = task_config(t, len(items), limit, seed)
        old = state["config"]["tasks"].get(name, {})
        keep = state["results"].get(name, {})
        if any(old.get(k) != cfg.get(k) for k in ("version", "seed", "fingerprint")):
            keep = {}
        ids = {it["id"] for it in items}
        keep = {i: r for i, r in keep.items() if i in ids}
        state["config"]["tasks"][name] = cfg
        state["results"][name] = keep
        todo += [
            (name, t, it)
            for it in items
            if it["id"] not in keep or "error" in keep[it["id"]]
        ]
        log(
            f"{name}: {len(items)} items, {len(items) - sum(1 for x in todo if x[0] == name)}"
            f" done, v{t.VERSION}"
        )
    lock, last = threading.Lock(), [time.time()]
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
                with lock:
                    state["results"][name][row["id"]] = row
                    tm = state["timing"].setdefault(
                        name, {"requests": 0, "seconds": 0.0}
                    )
                    tm["requests"] += 1
                    tm["latency_sum"] = tm.get("latency_sum", 0.0) + row["latency"]
                    tm["end"] = time.time()
                    if time.time() - last[0] > save_every:
                        save(out, state)
                        last[0] = time.time()
                if k % 100 == 0:
                    log(f"{k}/{len(todo)} requests")
    finally:
        for name, tm in state["timing"].items():
            if name in started and "end" in tm:
                tm["seconds"] = round(tm["seconds"] + tm.pop("end") - started[name], 2)
        state["timing"]["_wall"] = round(
            state["timing"].get("_wall", 0.0) + time.time() - t_start, 2
        )
        save(out, state)
    for name in tasks:
        rows = state["results"].get(name)
        if rows is None:
            continue
        done = [r for r in rows.values() if "error" not in r]
        acc = sum(r["ok"] for r in done) / len(done) if done else float("nan")
        log(f"{name:<10} n={len(done):<5} acc={acc:.4f} errors={len(rows) - len(done)}")
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
    extra = {} if a.thinking else {"chat_template_kwargs": {"enable_thinking": False}}
    client = Client(a.url, a.model, timeout=a.timeout, chat_extra=extra)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    run_suite(client, registry.resolve(a.tasks), a.limit, a.seed, a.workers, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
