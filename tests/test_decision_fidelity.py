# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Decision-fidelity helpers (CPU): tool-call parsing, garbage checks, agreement."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"))

import decision_fidelity as df  # noqa: E402
import decision_prompts as dp  # noqa: E402


def test_parse_well_formed_and_malformed_calls():
    ok = df.parse("Let me look.\n<read_file>\n<path>hi2/cache.py</path>\n</read_file>")
    assert ok == {"tool": "read_file", "valid": True, "arg": "hi2/cache.py"}
    assert (
        df.parse("<execute_command>\n<command>pytest -q</command>\n</execute_command>")[
            "arg"
        ]
        == "pytest -q"
    )
    cut = df.parse("<read_file>\n<path>hi2/cache.py</path>")  # no closing tag
    assert cut["tool"] == "read_file" and not cut["valid"]
    missing = df.parse("<write_to_file>\n<path>a.py</path>\n</write_to_file>")
    assert missing["tool"] == "write_to_file" and not missing["valid"]
    assert df.parse("It computes a digest.")["tool"] == "none"


def test_garbage_checks():
    assert df.garbage("") == ["empty"]
    assert "repeat" in df.garbage("abc" * 40)
    assert "bang-flood" in df.garbage("ok !!!!!!!!!! no")
    assert df.garbage("<read_file>\n<path>a</path>\n</read_file>") == []


def test_agreement_counts_tool_and_argument(tmp_path):
    def row(i, tool, arg):
        return {
            "id": i,
            "stage": "fresh",
            "tool": tool,
            "arg": arg,
            "valid": True,
            "garbage": [],
            "text": "",
        }

    a = {"1": row("1", "read_file", "a"), "2": row("2", "list_files", "hi2")}
    b = {"1": row("1", "read_file", "b"), "2": row("2", "search_files", "x")}
    assert df.agreement(a, b) == (0.5, 0.0)
    assert df.agreement(a, a) == (1.0, 1.0)
    lo, hi = df.wilson(30, 60)
    assert lo < 0.5 < hi


def test_prompt_set_is_committed_and_reproducible():
    built = dp.build()
    stored = json.loads(
        dp.ROOT.joinpath("experiments/data/decision_prompts.json").read_text()
    )
    assert len(built) == len(stored) == len(dp.TASKS) * 5 + len(dp.VAGUE)
    assert [p["id"] for p in built] == [p["id"] for p in stored]
    assert len({p["id"] for p in built}) == len(built)
