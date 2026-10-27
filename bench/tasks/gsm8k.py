# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""GSM8K test (1319 grade-school word problems): zero-shot chain of thought with a fixed
prompt; correct when the final number equals the reference answer."""

import re

from bench.data import hf_rows, sample

NAME = "gsm8k"
VERSION = "1"
MAX_TOKENS = 768
DEFAULT_LIMIT = None
DATASETS = [("openai/gsm8k", "main", "test")]

PROMPT = (
    "Solve the following math problem step by step. At the end, write the final answer "
    'on its own line as "The answer is N." where N is a number only.\n\nProblem: {q}'
)
NUM = r"-?\$?\s*-?[\d,]*\.?\d+"


def normalize(s: str) -> str | None:
    s = s.replace(",", "").replace("$", "").replace(" ", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else repr(round(v, 6))


def extract(text: str) -> str | None:
    """The final answer: the last "answer is N" or \\boxed{N}, else the last number."""
    found = re.findall(r"answer is[:\s*]*(" + NUM + ")", text, re.I)
    found = found or re.findall(r"\\boxed\{\s*(" + NUM + r")\s*\}", text)
    found = found or re.findall(NUM, text)
    return normalize(found[-1]) if found else None


def load(limit, seed):
    rows = hf_rows(*DATASETS[0])
    items = [
        {
            "id": f"gsm8k/{i}",
            "question": r["question"],
            "answer": normalize(r["answer"].split("####")[-1].strip()),
        }
        for i, r in enumerate(rows)
    ]
    return sample(items, limit, seed)


def prompt(item):
    return [{"role": "user", "content": PROMPT.format(q=item["question"])}]


def score(item, completion):
    pred = extract(completion)
    return {
        "ok": pred is not None and pred == item["answer"],
        "pred": pred,
        "gold": item["answer"],
    }
