# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Command line for the task-level evaluation.

    python -m tools.inkling_eval.cli plan
    python -m tools.inkling_eval.cli dry-run
    python -m tools.inkling_eval.cli run --url http://127.0.0.1:18731 --label stock --out runs/stock.jsonl
    python -m tools.inkling_eval.cli compare --ref runs/stock.jsonl --cand runs/hi2.jsonl

Exit codes: 0 only when the requested operation completed (a wrong answer is a
completed measurement), 1 for an incomplete or mismatched run/comparison, 2 for
invalid arguments.
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from tools.inkling_eval import compare as compare_mod
from tools.inkling_eval import mock_server, runner, sandbox, tasks
from tools.inkling_eval.client import Client, TransportError

def _categories(value: str):
    names = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in tasks.CATEGORIES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown categories: {unknown}")
    return names


def _contexts(value: str):
    try:
        lengths = [int(part) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"context lengths must be integers: {error}")
    if any(length < 512 for length in lengths):
        raise argparse.ArgumentTypeError("context lengths must be >= 512")
    return lengths


def _plan(args, max_model_len):
    return tasks.build_plan(
        seed=args.seed,
        contexts=args.contexts,
        max_model_len=max_model_len,
        categories=args.categories,
        limit=args.limit,
        reasoning_effort=args.reasoning_effort,
        set_name=args.set,
    )


def cmd_plan(args) -> int:
    max_model_len = args.max_model_len
    if max_model_len is None and args.url:
        client = Client(args.url, args.model, timeout=60, retries=1)
        max_model_len = client.max_model_len()
    plan = _plan(args, max_model_len)
    payload = {"meta": plan["meta"], "items": len(plan["items"])}
    payload["meta"]["approx_prompt_tokens"] = {
        "max": max(item["approx_prompt_tokens"] for item in plan["items"]),
        "total": sum(item["approx_prompt_tokens"] for item in plan["items"]),
    }
    print(json.dumps(payload, indent=1, sort_keys=True))
    return 0


def cmd_run(args) -> int:
    client = Client(
        args.url,
        args.model,
        timeout=args.timeout,
        retries=args.retries,
        seed=args.seed,
    )
    max_model_len = args.max_model_len
    if max_model_len is None:
        max_model_len = client.max_model_len()
    if max_model_len is None:
        max_model_len = 32768
        print(
            "warning: /v1/models did not report max_model_len; assuming 32768",
            file=sys.stderr,
        )
        if max(args.contexts) > 32768:
            print("contexts above the assumed limit are skipped", file=sys.stderr)
    plan = _plan(args, max_model_len)
    if plan["meta"]["skipped_contexts"]:
        print(
            f"skipped contexts: {plan['meta']['skipped_contexts']}",
            file=sys.stderr,
        )
    summary = runner.run_plan(
        plan,
        client,
        Path(args.out),
        label=args.label,
        workers=args.workers,
        resume=not args.no_resume,
        force=args.force,
        progress_every=args.progress_every,
    )
    return 0 if summary["reliable"] else 1


def _dry_run_assertions(plan: dict, summary: dict, out: Path) -> list[str]:
    problems = []
    records = runner.load_records(out)
    for item in plan["items"]:
        record = records.get(item["id"])
        if not record or record.get("status") != "complete":
            problems.append(f"{item['id']}: not complete")
        elif not (record.get("score") or {}).get("ok"):
            problems.append(f"{item['id']}: gold completion did not score ok")
    if summary["requested"] != len(plan["items"]):
        problems.append(f"first pass requested {summary['requested']} of {len(plan['items'])}")
    return problems


def cmd_dry_run(args) -> int:
    started = time.monotonic()
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path(
            tempfile.mkdtemp(prefix="inkling-eval-dry-", dir=str(sandbox.scratch_root()))
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "dry-run.jsonl"
    for path in (out, runner.meta_path(out), out_dir / "answers.jsonl"):
        if path.exists():
            path.unlink()
    plan = _plan(args, 32768)
    if not plan["items"]:
        print("dry-run: the plan is empty", file=sys.stderr)
        return 1
    answers_path = out_dir / "answers.jsonl"
    count = mock_server.write_answers(plan["items"], answers_path)
    answers = mock_server.load_answers(answers_path)
    if count != len(plan["items"]) or len(answers) != count:
        print("dry-run: duplicate or missing canned answers", file=sys.stderr)
        return 1
    server, thread, url = mock_server.serve(answers, max_model_len=32768)
    try:
        client = Client(url, "canned", timeout=60, retries=0, seed=0)
        summary = runner.run_plan(
            plan,
            client,
            out,
            label="canned",
            workers=args.workers,
            progress_every=max(1, len(plan["items"]) // 5),
        )
        problems = _dry_run_assertions(plan, summary, out)
        resumed = runner.run_plan(
            plan,
            client,
            out,
            label="canned",
            workers=args.workers,
            progress_every=1000,
        )
        if resumed["requested"] != 0 or resumed["skipped_complete"] != len(plan["items"]):
            problems.append(
                f"resume requested {resumed['requested']} and skipped "
                f"{resumed['skipped_complete']} of {len(plan['items'])}"
            )
        report = compare_mod.compare(out, out)
        if report["status"] != "complete":
            problems.append(f"self-comparison status {report['status']}")
        for name, entry in report["categories"].items():
            if entry["delta"] not in (0.0, None):
                problems.append(f"self-comparison delta {entry['delta']} in {name}")
        if report["pairing"]["paired"] != len(plan["items"]):
            problems.append(
                f"self-comparison paired {report['pairing']['paired']} of {len(plan['items'])}"
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    counts = {name: entry["items"] for name, entry in plan["meta"]["counts"].items()}
    for name in plan["meta"]["categories"]:
        if name not in counts:
            problems.append(f"category {name} is empty")
    if problems:
        print("dry-run FAILED:", file=sys.stderr)
        for problem in problems[:20]:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    seconds = round(time.monotonic() - started, 1)
    print(
        "dry-run OK: " + json.dumps(counts, sort_keys=True) + f"; {len(plan['items'])} items, "
        f"0 requests on resume, self-comparison delta 0; {seconds}s; nothing measured"
    )
    return 0


def cmd_compare(args) -> int:
    report = compare_mod.compare(
        Path(args.ref),
        Path(args.cand),
        floor_path=Path(args.floor) if args.floor else None,
        reps=args.reps,
    )
    compare_mod.write_report(report, Path(args.out), Path(args.md) if args.md else None)
    print(compare_mod.markdown(report), end="")
    return 0 if report["status"] == "complete" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, *, needs_plan=True):
        if needs_plan:
            p.add_argument("--set", choices=("main", "agentic-probe", "vision"), default="main")
            p.add_argument("--categories", type=_categories, default=list(tasks.CATEGORIES))
            p.add_argument("--contexts", type=_contexts, default=[8192, 16384, 32768])
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--limit", type=int, default=None, help="items per category")
            p.add_argument("--reasoning-effort", type=float, default=0.0)

    p_plan = sub.add_parser("plan", help="print the plan summary")
    common(p_plan)
    p_plan.add_argument("--url")
    p_plan.add_argument("--model")
    p_plan.add_argument("--max-model-len", type=int)
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser("run", help="run a plan against one endpoint")
    common(p_run)
    p_run.add_argument("--url", required=True)
    p_run.add_argument("--model")
    p_run.add_argument("--label", required=True)
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--workers", type=int, default=4)
    p_run.add_argument("--timeout", type=float, default=1800)
    p_run.add_argument("--retries", type=int, default=2)
    p_run.add_argument("--max-model-len", type=int)
    p_run.add_argument("--no-resume", action="store_true")
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--progress-every", type=int, default=10)
    p_run.set_defaults(func=cmd_run)

    p_dry = sub.add_parser("dry-run", help="validate every task against a canned server")
    common(p_dry)
    p_dry.add_argument("--out-dir")
    p_dry.add_argument("--workers", type=int, default=2)
    p_dry.set_defaults(func=cmd_dry_run)

    p_cmp = sub.add_parser("compare", help="paired comparison of two runs")
    p_cmp.add_argument("--ref", required=True)
    p_cmp.add_argument("--cand", required=True)
    p_cmp.add_argument("--floor", help="a second reference (stock) run for the noise floor")
    p_cmp.add_argument("--out", required=True)
    p_cmp.add_argument("--md")
    p_cmp.add_argument("--reps", type=int, default=10000)
    p_cmp.set_defaults(func=cmd_compare)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except runner.RunMismatch as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except TransportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
