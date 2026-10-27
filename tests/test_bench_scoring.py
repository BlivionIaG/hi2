# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""bench task scoring (CPU, no server): answer extraction, code execution, tool calls,
multiple choice, decision fidelity, and dataset loading from a jsonl export."""

import json

import pytest

from bench import data, tasks
from bench.client import strip_think
from bench.tasks import decision, gsm8k, humaneval, mbpp, mc, toolcall


def test_registry_resolves_aliases():
    assert list(tasks.resolve("code,gsm8k,mbpp")) == ["humaneval", "mbpp", "gsm8k"]
    assert list(tasks.resolve("all")) == list(tasks.NAMES)
    with pytest.raises(KeyError):
        tasks.resolve("nope")


def test_gsm8k_extraction():
    assert gsm8k.extract("so 9 * 2 = 18.\nThe answer is 18.") == "18"
    assert gsm8k.extract("The answer is $1,234.") == "1234"
    assert gsm8k.extract("we get \\boxed{42}") == "42"
    assert gsm8k.extract("first 3, then 7.5 cookies") == "7.5"
    assert gsm8k.extract("The answer is: -4.0") == "-4"
    assert gsm8k.extract("no numbers here") is None
    item = {"answer": gsm8k.normalize("18")}
    assert gsm8k.score(item, "The answer is 18.")["ok"]
    assert not gsm8k.score(item, "The answer is 17.")["ok"]


def test_strip_think():
    assert (
        strip_think("<think>maybe 3</think>\n\nThe answer is 4.") == "The answer is 4."
    )
    assert strip_think("plain") == "plain"


HE = {
    "id": "HumanEval/x",
    "entry_point": "add",
    "prompt": 'def add(a: int, b: int) -> int:\n    """Sum of a and b."""\n',
    "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n",
}


def test_humaneval_full_function_body_only_and_wrong():
    full = "Sure:\n```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```\n"
    assert humaneval.score(HE, full)["ok"]
    assert humaneval.score(HE, "    return a + b\n")["ok"]  # bare body, no fence
    bad = humaneval.score(HE, "```python\ndef add(a, b):\n    return a - b\n```")
    assert (
        not bad["ok"]
        and bad["status"] == "failed"
        and "AssertionError" in bad["stderr"]
    )


def test_mbpp_program():
    item = {
        "id": "mbpp/1",
        "text": "Square a number.",
        "imports": ["import math"],
        "tests": ["assert sq(3) == 9", "assert sq(0) == 0"],
    }
    assert "assert sq(3) == 9" in mbpp.prompt(item)[0]["content"]
    assert mbpp.score(item, "```python\ndef sq(x):\n    return x * x\n```")["ok"]
    assert not mbpp.score(item, "```python\ndef sq(x):\n    return x + x\n```")["ok"]


def test_toolcall_formats_and_f1():
    gold = toolcall.parse_python_calls('[Get Weather(city="Paris", days=3), getTime()]')
    item = {"gold": gold}
    assert toolcall.score(item, '[getTime(), Get Weather(days=3, city="paris")]')["ok"]
    js = '[{"name": "Get Weather", "arguments": {"city": "Paris", "days": 3.0}}, {"name": "getTime", "arguments": {}}]'
    assert toolcall.score(item, js)["ok"]
    tc = (
        '<tool_call>\n{"name": "Get Weather", "arguments": {"city": "Paris", "days": 3}}'
        '\n</tool_call>\n<tool_call>\n{"name": "getTime", "arguments": {}}\n</tool_call>'
    )
    assert toolcall.score(item, tc)["ok"]
    half = toolcall.score(item, '[Get Weather(city="Rome", days=3), getTime()]')
    assert not half["ok"] and half["name_ok"]
    # gold triples: (gw), (gw,city), (gw,days), (gt) = 4; pred shares 3 of 4
    assert half["arg_f1"] == pytest.approx(0.75)
    none = toolcall.score(item, "I cannot help with that.")
    assert none == {"ok": False, "parsed": False, "name_ok": False, "arg_f1": 0.0}
    nested = toolcall.parse_python_calls('[f(x=[{"a": 1}], flag=true, s="a, b)")]')
    assert nested == [
        {"name": "f", "arguments": {"x": [{"a": 1}], "flag": True, "s": "a, b)"}}
    ]


def test_toolcall_load_from_export(tmp_path, monkeypatch):
    rows = [
        {
            "system": "S",
            "conversations": [
                {"from": "user", "value": "weather?"},
                {"from": "assistant", "value": '[W(city="X")]'},
            ],
        },
        {
            "system": "S",
            "conversations": [
                {"from": "user", "value": "hi"},
                {"from": "assistant", "value": "Hello!"},
            ],
        },
    ]
    f = tmp_path / data.export_name(*toolcall.DATASETS[0])
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setenv(data.ENV_DATA, str(tmp_path))
    items = toolcall.load(None, 0)
    assert [i["id"] for i in items] == ["toolace/0"]
    assert toolcall.prompt(items[0])[0] == {"role": "system", "content": "S"}


def test_missing_dataset_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv(data.ENV_DATA, str(tmp_path))
    monkeypatch.setitem(__import__("sys").modules, "datasets", None)  # not importable
    with pytest.raises(data.DatasetMissing, match="HI2_BENCH_DATA"):
        gsm8k.load(None, 0)
    monkeypatch.setenv(mc.ENV, str(tmp_path))
    with pytest.raises(data.DatasetMissing, match="HI2_BENCH_MC_DATA"):
        mc.load(10, 0)


def test_sample_is_seeded_and_ordered():
    items = list(range(100))
    a = data.sample(items, 10, 3)
    assert a == data.sample(items, 10, 3) and a == sorted(a) and len(a) == 10
    assert data.sample(items, 200, 3) == items


def test_mc_load_and_score(tmp_path, monkeypatch):
    rows = [
        {
            "question": f"Q{i}",
            "single_correct": {"answers": ["a", "bbbb"], "labels": [0, 1]},
        }
        for i in range(6)
    ]
    for fn in mc.SETS.values():
        (tmp_path / fn).write_text(json.dumps(rows))
    monkeypatch.setenv(mc.ENV, str(tmp_path))
    items = mc.load(8, 0)
    assert len(items) == 8 and {i["group"] for i in items} == set(mc.SETS)
    it = items[0]
    # gold is "bbbb"; per-character lengths are 2 (" a") and 5 (" bbbb")
    r = mc.score(it, json.dumps([-3.0, -2.0]))
    assert r["ok"] and r["acc_norm"]
    r = mc.score(it, json.dumps([-0.5, -2.0]))
    assert not r["ok"] and not r["acc_norm"]
    r = mc.score(
        it, json.dumps([-1.0, -2.0])
    )  # raw picks a; per char 0.5 vs 0.4 picks b
    assert not r["ok"] and r["acc_norm"]


def test_mc_request_uses_tokenize_and_logprobs():
    class Fake:
        def tokenize(self, text):
            return [ord(c) for c in text]

        def prompt_logprobs(self, ids):
            return [None] + [-0.1] * (len(ids) - 1)

    item = {"question": "Q:", "answers": ["x", "yy"], "gold": 0}
    completion, usage = mc.request(Fake(), item)
    assert json.loads(completion) == pytest.approx([-0.2, -0.3])
    assert usage["requests"] == 2


def test_decision_task_wraps_the_experiment():
    items = decision.load(None, 0)
    assert len(items) == 66 and items[0]["group"] == "fresh"
    r = decision.score(items[0], "<read_file>\n<path>hi2/cache.py</path>\n</read_file>")
    assert r["ok"] and r["tool"] == "read_file" and r["arg"] == "hi2/cache.py"
    assert not decision.score(items[0], "")["ok"]
