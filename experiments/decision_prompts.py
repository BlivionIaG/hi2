# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Build the decision-fidelity prompt set (spec 12.3): coding-agent conversations over this
repository's own files, one tool call to decide per prompt.

Each task is shown at five stages: fresh, after a directory listing, after reading a file,
after a failing test run, and after a passing run. Six vague requests test whether the model
asks a question. The set is written to ``data/decision_prompts.json`` and committed, so every
model sees the same prompts; rebuild it only to change the set.

    python experiments/decision_prompts.py        # from the repository root
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SYSTEM = """You are a coding agent working in a Python repository. You act by calling exactly \
ONE tool per message, written as XML:

<tool_name>
<param>value</param>
</tool_name>

Tools:
- read_file: <path>
- write_to_file: <path>, <content>
- replace_in_file: <path>, <diff>
- execute_command: <command>
- search_files: <path>, <regex>
- list_files: <path>
- ask_followup_question: <question>
- attempt_completion: <result>

After each tool call you receive its result. Think briefly, then call one tool. When the task \
is done, call attempt_completion. If the request is too vague to act on, call \
ask_followup_question."""

# (task, file the task is about, test file for the command stages)
TASKS = [
    (
        "Find where the persistent cache key for the statistics is computed and explain it.",
        "hi2/cache.py",
        "tests/test_cache.py",
    ),
    (
        "Add a --seed option to the `hi2 eval` command.",
        "hi2/__main__.py",
        "tests/test_pipeline.py",
    ),
    (
        "`hi2 size` reports a wrong bpw when a tensor has no rows. Fix it.",
        "hi2/size.py",
        "tests/test_size.py",
    ),
    (
        "Rename the function default_ratio to default_error_ratio everywhere.",
        "hi2/pipeline.py",
        "tests/test_allocate_fast.py",
    ),
    (
        "Run the tests of the allocation module and fix any failure.",
        "hi2/allocate.py",
        "tests/test_allocate_fast.py",
    ),
    (
        "Explain how allocate_by_quality chooses the bits of each row.",
        "hi2/allocate.py",
        "tests/test_allocate_fast.py",
    ),
    (
        "Make the calibration loader fail with a clear message when the text file is empty.",
        "hi2/pipeline.py",
        "tests/test_pipeline.py",
    ),
    (
        "Add type hints to hi2/evalq.py.",
        "hi2/evalq.py",
        "tests/test_pipeline.py",
    ),
    (
        "Write a new test that restacks a checkpoint to the stacked layout and back.",
        "hi2/stack.py",
        "tests/test_stack.py",
    ),
    (
        "Document the --max-error-ratio default in docs/usage.md.",
        "docs/usage.md",
        "tests/test_allocate_fast.py",
    ),
    (
        "List every module in hi2/ that imports torch.",
        "hi2/format.py",
        "tests/test_format.py",
    ),
    (
        "Why does QAD keep the PTQ encoding for some layers?",
        "hi2/qad.py",
        "tests/test_qad.py",
    ),
]

VAGUE = [
    "Make it faster.",
    "Fix the bug.",
    "Improve the quantizer.",
    "Clean this up.",
    "Do the thing we discussed.",
    "Why is it broken?",
]


def call(tool: str, **params: str) -> str:
    body = "\n".join(f"<{k}>{v}</{k}>" for k, v in params.items())
    return f"<{tool}>\n{body}\n</{tool}>"


def result(tool: str, text: str) -> str:
    return f"[{tool} result]\n{text}"


def excerpt(path: str, limit: int = 2200) -> str:
    text = (ROOT / path).read_text()
    return text[:limit].rsplit("\n", 1)[0]


def listing() -> str:
    return "\n".join(sorted(p.name for p in (ROOT / "hi2").glob("*.py")))


def failure(test: str) -> str:
    return (
        f"FAILED {test}::test_case - AssertionError: assert 3 == 4\n"
        f"E   assert 3 == 4\n{test}:41: AssertionError\n"
        "1 failed, 14 passed in 2.31s"
    )


def build() -> list[dict]:
    out = []

    def add(pid, stage, task, history):
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": task},
        ]
        for role, content in history:
            msgs.append({"role": role, "content": content})
        out.append({"id": pid, "stage": stage, "messages": msgs})

    for i, (task, path, test) in enumerate(TASKS):
        cmd = f"python -m pytest -q {test}"
        stages = {
            "fresh": [],
            "listing": [
                ("assistant", call("list_files", path="hi2")),
                ("user", result("list_files", listing())),
            ],
            "read": [
                ("assistant", call("read_file", path=path)),
                ("user", result("read_file", excerpt(path))),
            ],
            "test_fail": [
                ("assistant", call("execute_command", command=cmd)),
                ("user", result("execute_command", failure(test))),
            ],
            "test_pass": [
                ("assistant", call("execute_command", command=cmd)),
                ("user", result("execute_command", "15 passed in 2.40s")),
            ],
        }
        for stage, history in stages.items():
            add(f"t{i:02d}-{stage}", stage, task, history)
    for i, task in enumerate(VAGUE):
        add(f"v{i:02d}-vague", "vague", task, [])
    return out


if __name__ == "__main__":
    prompts = build()
    dest = ROOT / "experiments" / "data" / "decision_prompts.json"
    dest.write_text(json.dumps(prompts, indent=1) + "\n")
    print(f"{len(prompts)} prompts -> {dest}")
