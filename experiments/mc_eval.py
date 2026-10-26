# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Multiple-choice accuracy through a vLLM server (prompt log-probabilities).

Datasets are the llama.cpp validation sets converted to JSON (ARC-Easy, ARC-Challenge,
HellaSwag, MMLU): each item has a context and answers with a 0/1 label. An answer is scored
by the log-likelihood of its tokens given the context: ``acc`` takes the highest total, and
``acc_norm`` the highest total per character. Contexts and answers are tokenised separately
at the boundary to avoid merge artefacts.

    python mc_eval.py --port 18301 --model-dir <dir with tokenizer> --served-name NAME \
        --data <dir with the .json files> --out result.json [--limit 2000]
"""

import argparse
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TASKS = {
    "arc_easy": "arc-easy-validation.json",
    "arc_challenge": "arc-challenge-validation.json",
    "hellaswag": "hellaswag-validation.json",
    "mmlu": "mmlu-test.json",
}


def logprobs(port: int, name: str, ids: list[int]) -> list[float]:
    body = json.dumps(
        {
            "model": name,
            "prompt": ids,
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        body,
        {"Content-Type": "application/json"},
    )
    out = json.loads(urllib.request.urlopen(req, timeout=900).read())
    plp = out["choices"][0]["prompt_logprobs"]
    return [None if e is None else e[str(t)]["logprob"] for t, e in zip(ids, plp)]


def score_item(port, name, tok, item):
    sc = item["single_correct"]
    ctx = item["question"]
    ctx_ids = tok(ctx, add_special_tokens=False).input_ids
    res = []
    for ans in sc["answers"]:
        ans_ids = tok(" " + ans, add_special_tokens=False).input_ids
        ids = ctx_ids + ans_ids
        lp = logprobs(port, name, ids)[len(ctx_ids) :]
        res.append((sum(lp), len(" " + ans)))
    pick = max(range(len(res)), key=lambda i: res[i][0])
    pick_n = max(range(len(res)), key=lambda i: res[i][0] / res[i][1])
    gold = sc["labels"].index(1)
    return pick == gold, pick_n == gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--tasks", default=",".join(TASKS))
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model_dir)
    result = {}
    for task in a.tasks.split(","):
        items = [
            x
            for x in json.loads((a.data / TASKS[task]).read_text())
            if x["single_correct"]["answers"]
        ]
        if len(items) > a.limit:
            items = random.Random(0).sample(items, a.limit)
        with ThreadPoolExecutor(a.workers) as pool:
            res = list(
                pool.map(lambda it: score_item(a.port, a.served_name, tok, it), items)
            )
        n = len(res)
        result[task] = {
            "n": n,
            "acc": sum(r[0] for r in res) / n,
            "acc_norm": sum(r[1] for r in res) / n,
        }
        print(task, result[task], flush=True)
    a.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
