# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""HumanEval (164 functions): pass@1 of the greedy completion, by running the dataset's
tests on it in the sandbox."""

from bench.data import hf_rows, sample
from bench.sandbox import extract_code, run_python

NAME = "humaneval"
VERSION = "1"
MAX_TOKENS = 768
DEFAULT_LIMIT = None
DATASETS = [("openai/openai_humaneval", None, "test")]
TIMEOUT = 10.0

PROMPT = (
    "Complete the following Python function. Reply with the complete function, including "
    "its signature and any imports it needs, in a single ```python code block.\n\n"
    "```python\n{prompt}```"
)


def load(limit, seed):
    rows = hf_rows(*DATASETS[0])
    items = [
        {
            "id": r["task_id"],
            "prompt": r["prompt"],
            "test": r["test"],
            "entry_point": r["entry_point"],
        }
        for r in rows
    ]
    return sample(items, limit, seed)


def prompt(item):
    return [{"role": "user", "content": PROMPT.format(prompt=item["prompt"])}]


def program(item, completion: str) -> str:
    """The dataset prompt (imports, helpers, the stub) then the reply: a full definition
    replaces the stub and a bare body completes it. Then the tests."""
    code = extract_code(completion, item["entry_point"])
    return (
        f"{item['prompt']}\n{code}\n\n{item['test']}\n\ncheck({item['entry_point']})\n"
    )


def score(item, completion):
    r = run_python(program(item, completion), timeout=TIMEOUT)
    return {"ok": r["passed"], "status": r["status"], "stderr": r["stderr"]}
