# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Domain KLD without generation: the reference model's own greedy completions, scored
teacher-forced under a served model.

For every item of a task, the prompt (chat template applied by the server) plus the
reference completion is sent as one prompt; the server returns the log-probability of every
completion token. The log-probabilities under the reference and under a candidate differ by
``log p_ref(t) - log p_cand(t)`` per token, whose mean over tokens drawn from the reference
is an estimate of KL(reference || candidate) on this task's own text (a prefill-only pass,
a few minutes per task).

    # under the reference server, then under each candidate server:
    python bench/tfkld.py score --url URL --model NAME --ref-results ref_bench.json \
        --tasks gsm8k,toolcall --limit 200 --out lp_TAG.json
    python bench/tfkld.py compare lp_ref.json lp_cand.json [...]
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import tasks as registry  # noqa: E402
from bench.client import Client  # noqa: E402

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
    return [float(x) for x in lp[len(p) :]]


def score(client, ref_results: dict, names, limit, workers) -> dict:
    out = {}
    for name in names:
        t = registry.get(name)
        rows = ref_results["results"].get(name, {})
        ids = [i for i, r in rows.items() if r.get("completion")][:limit]
        items = {
            it["id"]: it for it in t.load(None, ref_results["config"][name]["seed"])
        }
        todo = [(i, items[i], rows[i]["completion"]) for i in ids if i in items]

        def one(args, t=t):
            i, item, comp = args
            try:
                return i, token_logprobs(client, t.prompt(item), comp)
            except Exception as e:
                return i, {"error": f"{type(e).__name__}: {e}"}

        with ThreadPoolExecutor(workers) as pool:
            out[name] = dict(pool.map(one, todo))
    return out


def tf_kld(ref: dict, cand: dict) -> dict:
    """Per task: mean over tokens of ``lp_ref - lp_cand`` (nats/token) and the item count.
    Items are weighted by their token count, so long completions count more."""
    res = {}
    for name in ref:
        s = n = items = 0
        for i, a in ref[name].items():
            b = cand.get(name, {}).get(i)
            if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and a:
                s += sum(x - y for x, y in zip(a, b))
                n += len(a)
                items += 1
        res[name] = {"kld": s / n if n else float("nan"), "tokens": n, "items": items}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--url", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--ref-results", type=Path, required=True)
    s.add_argument("--tasks", default=",".join(CHAT_TASKS))
    s.add_argument("--limit", type=int, default=200)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("ref", type=Path)
    c.add_argument("cands", type=Path, nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "score":
        extra = {"chat_template_kwargs": {"enable_thinking": False}}
        client = Client(a.url, a.model, chat_extra=extra)
        ref = json.loads(a.ref_results.read_text())
        names = [n for n in a.tasks.split(",") if n in ref["results"]]
        a.out.write_text(json.dumps(score(client, ref, names, a.limit, a.workers)))
        return 0
    ref = json.loads(a.ref.read_text())
    for p in a.cands:
        r = tf_kld(ref, json.loads(p.read_text()))
        print(p.name, json.dumps({k: round(v["kld"], 5) for k, v in r.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
