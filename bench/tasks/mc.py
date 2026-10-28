# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Multiple choice by prompt log-likelihood (the logic of ``experiments/mc_eval.py``):
ARC-Easy, ARC-Challenge, HellaSwag and MMLU from the llama.cpp validation JSON files in
``$HI2_BENCH_MC_DATA``. An answer's score is the sum of its token log-probabilities given
the context, tokenised separately at the boundary through the server's ``/tokenize``;
``ok`` = the best answer is right, ``acc_norm`` = the best per character is right.
``limit`` is split evenly over the four sets (default 500 each)."""

import json
import os
import random
from pathlib import Path

from bench.data import DatasetMissing

NAME = "mc"
VERSION = "1"
MAX_TOKENS = 1
DEFAULT_LIMIT = 2000
METRICS = ("acc_norm",)
ENV = "HI2_BENCH_MC_DATA"
SETS = {
    "arc_easy": "arc-easy-validation.json",
    "arc_challenge": "arc-challenge-validation.json",
    "hellaswag": "hellaswag-validation.json",
    "mmlu": "mmlu-test.json",
}


def load(limit, seed):
    root = os.environ.get(ENV, "")
    if not root:
        raise DatasetMissing(
            f"set ${ENV} to the directory with the llama.cpp validation JSON files "
            f"(ikawrakow/validation-datasets-for-llama.cpp, converted with its "
            f"`convert` tool)"
        )
    d = Path(root)
    per = None if limit is None else max(1, limit // len(SETS))
    items = []
    for name, fn in SETS.items():
        f = d / fn
        if not f.exists():
            raise DatasetMissing(
                f"{f} not found: set ${ENV} to the directory with the llama.cpp "
                f"validation JSON files (ikawrakow/validation-datasets-for-llama.cpp, "
                f"converted with its `convert` tool)"
            )
        rows = [
            (i, x)
            for i, x in enumerate(json.loads(f.read_text()))
            if x["single_correct"]["answers"]
        ]
        if per is not None and len(rows) > per:
            rows = sorted(random.Random(seed).sample(rows, per), key=lambda r: r[0])
        for i, x in rows:
            sc = x["single_correct"]
            items.append(
                {
                    "id": f"{name}/{i}",
                    "group": name,
                    "question": x["question"],
                    "answers": sc["answers"],
                    "gold": sc["labels"].index(1),
                }
            )
    return items


def prompt(item):
    return item["question"]


def request(client, item):
    ctx = client.tokenize(item["question"])
    out = []
    for ans in item["answers"]:
        ids = ctx + client.tokenize(" " + ans)
        out.append(sum(client.prompt_logprobs(ids)[len(ctx) :]))
    return json.dumps(out), {"prompt_tokens": len(ctx), "requests": len(out)}


def score(item, completion):
    lps = json.loads(completion)
    lens = [len(" " + a) for a in item["answers"]]
    pick = max(range(len(lps)), key=lambda i: lps[i])
    pick_n = max(range(len(lps)), key=lambda i: lps[i] / lens[i])
    return {
        "ok": pick == item["gold"],
        "acc_norm": pick_n == item["gold"],
        "pick": pick,
    }
