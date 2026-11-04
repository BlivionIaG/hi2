# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Teacher-forced log-probability gaps on a reference model's completions.

For every item, the server scores the reference completion after the task prompt.
The comparison reports the token-weighted mean of ``log p_ref(t) - log p_cand(t)``.
Reference benchmark completions are greedy, not samples from the reference
distribution: this signed score gap is not KL divergence and may be negative.
Only matching items and lengths are paired; callers must ensure identical
tokenizers and chat-template settings. Token identity is not verified here.

    # under the reference server, then under each candidate server:
    python bench/logprob_gap.py score --url URL --model NAME --ref-results ref_bench.json \
        --tasks gsm8k,toolcall --limit 200 --out lp_TAG.json
    python bench/logprob_gap.py compare lp_ref.json lp_cand.json [...]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import tasks as registry  # noqa: E402
from bench.client import Client  # noqa: E402
from bench.run import completed, save  # noqa: E402

CHAT_TASKS = ("gsm8k", "humaneval", "mbpp", "ifeval", "toolcall", "decision")


def token_logprobs(client: Client, messages, completion: str) -> list[float]:
    """Log-probabilities of the tokens of ``completion`` after ``messages``."""
    p = client.post(
        "/tokenize",
        {
            "model": client.model,
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": client.chat_extra.get("chat_template_kwargs", {}),
        },
    )["tokens"]
    c = client.tokenize(completion)
    lp = client.prompt_logprobs(p + c)
    completion_lp = lp[len(p) :]
    if not c or len(completion_lp) != len(c) or any(x is None for x in completion_lp):
        raise ValueError(
            "server did not return every completion token's log-probability"
        )
    return [float(x) for x in completion_lp]


def score(
    client, ref_results: dict, names, limit, workers, out_path: Path | None = None
) -> dict:
    names = list(names)
    out = {
        "schema_version": 1,
        "status": "running",
        "config": {"model": client.model, "url": client.url, "limit": limit},
        "results": {},
        "completion": {
            "requested_tasks": names,
            "tasks": {name: {"status": "pending", "expected": None} for name in names},
            "errors": [],
        },
    }
    if ref_results.get("status") != "complete":
        out["completion"]["errors"].append(
            "reference benchmark is not explicitly complete"
        )
    if not names:
        out["completion"]["errors"].append("no eligible chat tasks requested")
    if out_path is not None:
        save(out_path, out)
    last = time.time()
    for name in names:
        rows = ref_results["results"].get(name, {})
        ids = list(rows)[:limit]
        task = {
            "status": "incomplete",
            "expected": len(ids),
            "completed": 0,
            "errors": 0,
        }
        out["completion"]["tasks"][name] = task
        out["results"][name] = {}
        try:
            t = registry.get(name)
            if name not in CHAT_TASKS:
                raise ValueError(
                    "teacher-forced scoring requires a supported chat task"
                )
            if not ids:
                raise ValueError("reference task has no items")
            items = {it["id"]: it for it in t.load(None, ref_results["config"]["seed"])}

            def one(i, t=t, items=items, rows=rows):
                try:
                    if not completed(rows[i]) or not rows[i].get("completion"):
                        raise ValueError(
                            "reference item has no completed response to score"
                        )
                    return i, token_logprobs(
                        client, t.prompt(items[i]), rows[i]["completion"]
                    )
                except (Exception, SystemExit) as e:
                    return i, {"error": f"{type(e).__name__}: {e}"}

            with ThreadPoolExecutor(workers) as pool:
                for i, row in pool.map(one, ids):
                    out["results"][name][i] = row
                    task["completed"] += isinstance(row, list)
                    task["errors"] += isinstance(row, dict)
                    if out_path is not None and time.time() - last > 20:
                        save(out_path, out)
                        last = time.time()
            task["completed"] = sum(
                isinstance(row, list) for row in out["results"][name].values()
            )
            task["errors"] = len(ids) - task["completed"]
            if task["completed"] == len(ids):
                task["status"] = "complete"
        except (Exception, SystemExit) as e:
            task |= {"status": "load_error", "error": f"{type(e).__name__}: {e}"}
        except KeyboardInterrupt:
            task["error"] = "KeyboardInterrupt"
            out["status"] = "incomplete"
            raise
        finally:
            if out_path is not None:
                save(out_path, out)
    out["status"] = (
        "complete"
        if names
        and not out["completion"]["errors"]
        and all(t["status"] == "complete" for t in out["completion"]["tasks"].values())
        else "incomplete"
    )
    if out_path is not None:
        save(out_path, out)
    return out


def mean_logprob_gap(ref: dict, cand: dict) -> dict:
    """Per task: mean over tokens of ``lp_ref - lp_cand`` (nats/token) and the item count.
    Items are weighted by their token count, so long completions count more."""
    res = {}
    for name in dict.fromkeys([*ref, *cand]):
        s = n = items = 0
        for i, a in ref.get(name, {}).items():
            b = cand.get(name, {}).get(i)
            if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and a:
                s += sum(x - y for x, y in zip(a, b))
                n += len(a)
                items += 1
        expected = len(ref.get(name, {}).keys() | cand.get(name, {}).keys())
        res[name] = {
            "mean_logprob_gap": s / n if n else None,
            "tokens": n,
            "items": items,
            "skipped": expected - items,
            "status": "complete" if items and items == expected else "incomplete",
        }
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--url", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--ref-results", type=Path, required=True)
    s.add_argument(
        "--tasks", help="chat tasks; default: chat tasks requested by the reference"
    )
    s.add_argument("--limit", type=int, default=200)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("ref", type=Path)
    c.add_argument("cands", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "score":
        if a.limit <= 0 or a.workers <= 0:
            ap.error("--limit and --workers must be positive")
        extra = {"chat_template_kwargs": {"enable_thinking": False}}
        client = Client(a.url, a.model, chat_extra=extra)
        ref = json.loads(a.ref_results.read_text())
        names = (
            list(dict.fromkeys(n.strip() for n in a.tasks.split(",") if n.strip()))
            if a.tasks is not None
            else [
                n
                for n in ref.get("config", {}).get("requested_tasks", ref["results"])
                if n in CHAT_TASKS
            ]
        )
        a.out.parent.mkdir(parents=True, exist_ok=True)
        try:
            report = score(client, ref, names, a.limit, a.workers, a.out)
        except KeyboardInterrupt:
            return 130
        report["config"]["reference"] = str(a.ref_results.resolve())
        report["config"]["command"] = [
            "bench/logprob_gap.py",
            *(sys.argv[1:] if argv is None else argv),
        ]
        save(a.out, report)
        return 0 if report["status"] == "complete" else 1
    ref = json.loads(a.ref.read_text())
    status = 0
    for p in a.cands:
        cand = json.loads(p.read_text())
        r = mean_logprob_gap(ref["results"], cand["results"])
        print(p.name, json.dumps(r, allow_nan=False))
        if (
            not r
            or ref["status"] != "complete"
            or cand["status"] != "complete"
            or any(t["status"] != "complete" for t in r.values())
        ):
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
