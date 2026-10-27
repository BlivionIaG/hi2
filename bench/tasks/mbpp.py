# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""MBPP sanitized test split (257 problems): pass@1 of the greedy completion against the
three asserts, run in the sandbox. The prompt shows the asserts (the original MBPP
format), which fixes the function name and signature."""

from bench.data import hf_rows, sample
from bench.sandbox import extract_code, run_python

NAME = "mbpp"
VERSION = "1"
MAX_TOKENS = 768
DEFAULT_LIMIT = None
DATASETS = [("google-research-datasets/mbpp", "sanitized", "test")]
TIMEOUT = 10.0

PROMPT = (
    "You are an expert Python programmer. {text}\nYour code should pass these tests:\n\n"
    "{tests}\n\nReply with the code only, in a single ```python code block."
)


def load(limit, seed):
    rows = hf_rows(*DATASETS[0])
    items = [
        {
            "id": f"mbpp/{r['task_id']}",
            "text": r["prompt"],
            "tests": list(r["test_list"]),
            "imports": list(r["test_imports"]),
        }
        for r in rows
    ]
    return sample(items, limit, seed)


def prompt(item):
    text = PROMPT.format(text=item["text"], tests="\n".join(item["tests"]))
    return [{"role": "user", "content": text}]


def program(item, completion: str) -> str:
    code = extract_code(completion)
    return "\n".join([*item["imports"], code, "", *item["tests"], ""])


def score(item, completion):
    r = run_python(program(item, completion), timeout=TIMEOUT)
    return {"ok": r["passed"], "status": r["status"], "stderr": r["stderr"]}
