# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Mutants of the scorers and the sandbox: the probes must kill each one.

Each mutation is applied to a temporary copy of the module, the copy is imported,
and a probe that holds on the original is run against it.  A surviving mutant is
a hole in the tests.  CPU only, no server:

    python -m tools.inkling_eval.mutants

Exit code 0 only when every mutant is killed and every probe holds on the
original.
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _mutated(name: str, source_file: str, old: str, new: str, tmp: Path):
    source = HERE / source_file
    text = source.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{name}: mutation text occurs {count} times in {source_file}")
    copy = tmp / f"{name}_{source.name}"
    copy.write_text(text.replace(old, new), encoding="utf-8")
    return _load(f"mut_{name}", copy)


# --------------------------------------------------------------------------- probes


def _probe_needle(module) -> None:
    item = {"expected": {"value": "AB123-4567", "distractors": []}}
    assert module.score_needle(item, "no code in this reply")["ok"] is False
    assert module.score_needle(item, "the code is AB123-4567")["ok"] is True


def _probe_agentic(module) -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    }
    item = {"expected": {"name": "get_weather", "args": {"location": "Berlin"}, "tools": [tool]}}
    wrong = (
        '<|message_model|>get_weather<|content_invoke_tool_json|>'
        '{"name":"get_weather","args":{"location":"Lisbon"}}<|end_message|>'
    )
    right = (
        '<|message_model|>get_weather<|content_invoke_tool_json|>'
        '{"name":"get_weather","args":{"location":"Berlin"}}<|end_message|>'
    )
    assert module.score_agentic(item, wrong)["ok"] is False
    assert module.score_agentic(item, right)["ok"] is True


def _probe_math(module) -> None:
    item = {"expected": {"answer": "18"}}
    assert module.score_math(item, "The answer is 19.")["ok"] is False
    assert module.score_math(item, "The answer is 18.")["ok"] is True


def _probe_chrf(module) -> None:
    item = {
        "expected": {
            "reference": "Der Zug fährt um sieben Uhr morgens ab.",
            "lang": "de",
            "min_chrf": 0.2,
        }
    }
    assert module.score_chrf(item, "The train leaves at seven in the morning.")["ok"] is False
    assert module.score_chrf(item, "Der Zug fährt um sieben Uhr morgens ab.")["ok"] is True


def _probe_required(module) -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}, "unit": {"type": "string"}},
                "required": ["location", "unit"],
            },
        },
    }
    errors = module.validate_args({"location": "Berlin"}, tool)
    assert any("unit" in error for error in errors), errors


def _probe_channel(module) -> None:
    text = (
        "<|content_thinking|>preamble saying 99<|end_message|>"
        "<|content_text|>72<|end_message|>"
    )
    assert module.answer_text(text) == "72", module.answer_text(text)


def _probe_last_channel(module) -> None:
    text = (
        "<|content_text|>reasoning with 99<|end_message|><|message_model|>"
        "<|content_text|>Paris<|end_message|>"
    )
    assert module.answer_text(text) == "Paris", module.answer_text(text)


def _probe_final_segment(module) -> None:
    assert module.final_segment('Final should be "Paris".Paris') == "Paris"


def _probe_fence(module) -> None:
    text = (
        "Need exactly one ```python code block and nothing else."
        "```python\ndef add(a, b):\n    return a + b\n```"
    )
    code = module.extract_code(text, "add", "python")
    assert "return a + b" in code, code


def _probe_truncation_gate(module) -> None:
    records = {
        f"agentic-{index}": {
            "category": "agentic",
            "status": "truncated" if index < 2 else "complete",
        }
        for index in range(20)
    }
    summary = module.truncation_summary(records, {"agentic": {"items": 20}})
    assert summary["failed"] == ["agentic"], summary


def _probe_truncation_by_finish_reason(module) -> None:
    records = {
        "agentic-0": {"category": "agentic", "status": "complete", "finish_reason": "length"}
    }
    summary = module.truncation_summary(records, {"agentic": {"items": 1}})
    assert summary["by_category"]["agentic"]["truncated"] == 1, summary


def _probe_python(module) -> None:
    good = module.run_python("def add(a, b):\n    return a + b\n", "assert add(1, 2) == 3\nprint('OK')")
    bad = module.run_python("def add(a, b):\n    return a - b\n", "assert add(1, 2) == 3\nprint('OK')")
    assert good["status"] == "pass", good
    assert bad["status"] == "fail", bad


MUTANTS = [
    (
        "needle_always_true",
        "scoring.py",
        '    return {\n        "ok": found,',
        '    return {\n        "ok": True,',
        _probe_needle,
    ),
    (
        "agentic_args_always_match",
        "scoring.py",
        '    args_ok = graded["args"] is not None and canon(graded["args"]) == canon(expected["args"])',
        "    args_ok = True",
        _probe_agentic,
    ),
    (
        "math_accepts_any_number",
        "scoring.py",
        "    return {\"ok\": got is not None and got == expected, \"detail\": {\"answer\": got, \"expected\": expected}}",
        "    return {\"ok\": got is not None, \"detail\": {\"answer\": got, \"expected\": expected}}",
        _probe_math,
    ),
    (
        "chrf_ignores_floor",
        "scoring.py",
        '        "ok": bool(body) and score >= expected["min_chrf"] and signal["state"] != "mismatch",',
        '        "ok": bool(body) and score >= 0.0 and signal["state"] != "mismatch",',
        _probe_chrf,
    ),
    (
        "required_arguments_ignored",
        "scoring.py",
        '        for key in schema.get("required") or []:',
        "        for key in []:",
        _probe_required,
    ),
    (
        "thinking_preamble_kept",
        "scoring.py",
        "    if blocks:\n        return _clean(blocks[-1])",
        "    if blocks:\n        return _clean(text)",
        _probe_channel,
    ),
    (
        "first_text_channel",
        "scoring.py",
        "    if blocks:\n        return _clean(blocks[-1])",
        "    if blocks:\n        return _clean(blocks[0])",
        _probe_last_channel,
    ),
    (
        "final_segment_whole_line",
        "scoring.py",
        "    segment = parts[-1] if parts else line",
        "    segment = line",
        _probe_final_segment,
    ),
    (
        "fence_accepts_inline_mention",
        "scoring.py",
        'r"```([A-Za-z0-9_+.-]*)[ \\t]*\\n(.*?)(```|\\Z)"',
        'r"```([A-Za-z0-9_+.-]*)[ \\t]*\\n?(.*?)(```|\\Z)"',
        _probe_fence,
    ),
    (
        "truncation_gate_disabled",
        "runner.py",
        "        gate_failed = rate > TRUNCATION_GATE",
        "        gate_failed = False",
        _probe_truncation_gate,
    ),
    (
        "truncation_counts_only_status",
        "runner.py",
        '            if record.get("status") == "truncated" or record.get("finish_reason") == "length":',
        '            if record.get("status") == "truncated":',
        _probe_truncation_by_finish_reason,
    ),
    (
        "python_failure_passes",
        "sandbox.py",
        '    result["language"] = "python"\n'
        '    result["network_isolation"] = network_isolation()\n'
        '    if result["status"] == "pass" and "OK" not in result.get("stdout", ""):\n'
        '        result["status"] = "fail"\n'
        "    return result",
        '    result["language"] = "python"\n'
        '    result["network_isolation"] = network_isolation()\n'
        '    result["status"] = "pass"\n'
        "    return result",
        _probe_python,
    ),
]


def run_mutants(verbose: bool = True) -> list[dict]:
    originals = {
        "scoring.py": _load("mutant_original_scoring", HERE / "scoring.py"),
        "sandbox.py": _load("mutant_original_sandbox", HERE / "sandbox.py"),
        "runner.py": _load("mutant_original_runner", HERE / "runner.py"),
    }
    results = []
    scratch = Path.home() / ".cache" / "inkling-eval"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mutants-", dir=str(scratch)) as tmp:
        root = Path(tmp)
        for name, source_file, old, new, probe in MUTANTS:
            try:
                probe(originals[source_file])
            except Exception as error:  # noqa: BLE001
                raise SystemExit(f"{name}: the probe does not hold on the original: {error}")
            module = _mutated(name, source_file, old, new, root)
            try:
                probe(module)
            except Exception as error:  # noqa: BLE001 - a killed mutant is any probe failure
                results.append({"name": name, "killed": True, "by": type(error).__name__})
                if verbose:
                    print(f"killed   {name} ({type(error).__name__})")
            else:
                results.append({"name": name, "killed": False, "by": None})
                if verbose:
                    print(f"SURVIVED {name}")
    return results


def main(argv=None) -> int:
    results = run_mutants()
    survived = [result["name"] for result in results if not result["killed"]]
    print(f"{len(results) - len(survived)}/{len(results)} mutants killed")
    return 0 if not survived else 1


if __name__ == "__main__":
    raise SystemExit(main())
