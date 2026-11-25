# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Scorers: extraction rules, chrF, language guard, tool-call schema and gold checks."""

import json

import pytest

from tools.inkling_eval import multilingual_data, scoring, tasks


def test_last_number_prefers_the_committed_answer():
    assert scoring.last_number("steps 1, 2, 3. The answer is 1,250.") == "1250"
    assert scoring.last_number("so \\boxed{18}") == "18"
    assert scoring.last_number("nothing numeric") is None
    assert scoring.last_number("first 3 then 9") == "9"
    assert scoring.canonical_number("18.0") == "18"
    assert scoring.canonical_number("+7") == "7"
    assert scoring.canonical_number("1,200") == "1200"
    assert scoring.canonical_number(None) is None


def test_chrf_edges_and_partial_credit():
    assert scoring.chrf("identical text", "identical text") == 1.0
    assert scoring.chrf("", "") == 1.0
    assert scoring.chrf("abc", "") == 0.0
    assert scoring.chrf("ab", "ac") == pytest.approx(1 / 12, abs=1e-9)
    assert 0 < scoring.chrf("The train leaves at seven.", "The train leaves at seven in the morning.") < 1


def test_language_guard_rejects_the_wrong_language():
    english = "The train leaves at seven in the morning."
    assert scoring.language_signal("The train leaves at seven in the morning.", "de")["state"] == "undetermined"
    assert scoring.language_signal("Le train part à sept heures du matin.", "de")["state"] == "mismatch"
    assert scoring.language_signal("火车在早上七点出发。", "zh")["state"] == "match"
    assert scoring.language_signal("火車は朝七時に出発します。", "zh")["state"] == "mismatch"
    assert scoring.language_signal("Поезд отправляется в семь часов утра.", "ru")["state"] == "match"
    assert scoring.language_signal(english, "ru")["state"] == "mismatch"


def test_authored_translation_references_pass_the_language_guard():
    for spec in multilingual_data.TRANSLATIONS:
        for lang, _name in multilingual_data.LANGUAGES:
            state = scoring.language_signal(spec["refs"][lang], lang)["state"]
            assert state != "mismatch", (spec["id"], lang, state)


def test_qa_numeric_and_word():
    numeric = {"expected": {"answer": "7"}}
    assert scoring.score_qa_numeric(numeric, "7")["ok"] is True
    assert scoring.score_qa_numeric(numeric, "7.0")["ok"] is True
    assert scoring.score_qa_numeric(numeric, "8")["ok"] is False
    word = {"expected": {"answers": ["blau"]}}
    assert scoring.score_qa_word(word, "Blau.")["ok"] is True
    assert scoring.score_qa_word(word, "blue")["ok"] is False
    word_ar = {"expected": {"answers": ["أزرق"]}}
    assert scoring.score_qa_word(word_ar, "أزرق.")["ok"] is True


def test_needle_and_distractors():
    item = {"expected": {"value": "AB123-4567", "distractors": ["ZZ999-0000"]}}
    result = scoring.score_needle(item, "AB123-4567")
    assert result["ok"] is True and result["detail"]["distractors_present"] == 0
    result = scoring.score_needle(item, "AB123-4567 or ZZ999-0000")
    assert result["ok"] is True and result["detail"]["distractors_present"] == 1
    assert scoring.score_needle(item, "ZZ999-0000")["ok"] is False


def test_summarize_requires_the_answer_and_the_length_bound():
    item = {"expected": {"answer": "Team Cedar", "summary_max_words": 60}}
    good = "The window is 03:00-05:00 UTC on 2026-12-05; Team Cedar owns rollback. Answer: Team Cedar"
    assert scoring.score_summarize(item, good)["ok"] is True
    assert scoring.score_summarize(item, "A summary without an answer")["ok"] is False
    long_summary = " ".join(["word"] * 100) + " Answer: Team Cedar"
    assert scoring.score_summarize(item, long_summary)["ok"] is False


def _tool(name, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def test_tool_call_formats_and_schema():
    tool = _tool(
        "get_weather",
        {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            "days": {"type": "integer"},
        },
        ["location", "unit", "days"],
    )
    item = {
        "expected": {
            "name": "get_weather",
            "args": {"location": "Lisbon", "unit": "celsius", "days": 2},
            "tools": [tool],
        }
    }
    inkling = (
        '<|message_model|>get_weather<|content_invoke_tool_json|>'
        '{"name":"get_weather","args":{"location":"Lisbon","unit":"celsius","days":2}}'
        "<|end_message|>"
    )
    assert scoring.score_agentic(item, inkling)["ok"] is True
    hermes = (
        "<tool_call>"
        + json.dumps(
            {
                "name": "get_weather",
                "arguments": {"location": "Lisbon", "unit": "celsius", "days": 2},
            }
        )
        + "</tool_call>"
    )
    assert scoring.score_agentic(item, hermes)["ok"] is True
    fenced = (
        "```json\n"
        + json.dumps(
            {
                "name": "get_weather",
                "arguments": {"location": "Lisbon", "unit": "celsius", "days": 2},
            }
        )
        + "\n```"
    )
    assert scoring.score_agentic(item, fenced)["ok"] is True
    invalid_json = '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":'
    result = scoring.score_agentic(item, invalid_json)
    assert result["ok"] is False
    missing = (
        '<|message_model|>get_weather<|content_invoke_tool_json|>'
        '{"name":"get_weather","args":{"location":"Lisbon","unit":"celsius"}}<|end_message|>'
    )
    result = scoring.score_agentic(item, missing)
    assert result["ok"] is False and result["detail"]["schema_ok"] is False
    extra = (
        '<|message_model|>get_weather<|content_invoke_tool_json|>'
        '{"name":"get_weather","args":{"location":"Lisbon","unit":"celsius","days":2,"rain":true}}'
    )
    assert scoring.score_agentic(item, extra)["ok"] is False
    wrong_enum = (
        '<|message_model|>get_weather<|content_invoke_tool_json|>'
        '{"name":"get_weather","args":{"location":"Lisbon","unit":"kelvin","days":2}}'
    )
    result = scoring.score_agentic(item, wrong_enum)
    assert result["ok"] is False and result["detail"]["name_ok"] is True
    wrong_name = (
        '<|message_model|>lookup_customer<|content_invoke_tool_json|>'
        '{"name":"lookup_customer","args":{"location":"Lisbon","unit":"celsius","days":2}}'
    )
    result = scoring.score_agentic(item, wrong_name)
    assert result["ok"] is False and result["detail"]["name_ok"] is False
    partial = (
        '<|message_model|>get_weather<|content_invoke_tool_json|>'
        '{"name":"get_weather","args":{"location":"Porto","unit":"celsius","days":2}}'
    )
    result = scoring.score_agentic(item, partial)
    assert result["ok"] is False and 0 < result["detail"]["arg_f1"] < 1


def test_tool_call_arguments_as_json_string_and_bare_object():
    tool = _tool("lookup_customer", {"email": {"type": "string"}}, ["email"])
    item = {"expected": {"name": "lookup_customer", "args": {"email": "ada@example.com"}, "tools": [tool]}}
    hermes = (
        "<tool_call>"
        + json.dumps({"name": "lookup_customer", "arguments": '{"email": "ada@example.com"}'})
        + "</tool_call>"
    )
    assert scoring.score_agentic(item, hermes)["ok"] is True
    bare = 'Sure: {"name": "lookup_customer", "arguments": {"email": "ada@example.com"}}'
    assert scoring.score_agentic(item, bare)["ok"] is True


def test_extract_code_prefers_the_entry_point_block():
    text = (
        "Here is a helper:\n```python\ndef helper():\n    return 1\n```\n"
        "and the answer:\n```python\ndef target():\n    return 2\n```\n"
    )
    assert "def target" in scoring.extract_code(text, "target", "python")
    no_fence = scoring.extract_code("def target():\n    return 2\n", "target", "python")
    assert no_fence.startswith("def target")


def test_gold_completions_score_ok_for_every_scorer():
    plan = tasks.build_plan(limit=2, max_model_len=32768)
    extra = tasks.build_plan(categories=["long_context", "multilingual"], max_model_len=32768)
    seen = set()
    for item in plan["items"] + extra["items"]:
        result = scoring.score(item, item["gold"])
        assert result["ok"], (item["id"], result["detail"])
        seen.add(item["scorer"])
    assert {"agentic", "code", "chrf", "qa_numeric", "qa_word", "needle", "summarize", "math"} <= seen


def test_answer_text_uses_the_last_text_channel_not_the_thinking_preamble():
    text = (
        "<|message_model|><|content_thinking|>The user asks ... nine times eight is 72."
        " Final should be 72.<|end_message|><|content_text|>72<|end_message|>"
        "<|content_model_end_sampling|>"
    )
    assert scoring.answer_text(text) == "72"
    assert scoring.last_number(text) == "72"
    item = {"expected": {"answer": "72", "lang": "en"}}
    assert scoring.score_qa_numeric(item, text)["ok"]


def test_answer_text_uses_the_last_of_multiple_text_channel_blocks():
    text = (
        "<|content_text|>reasoning with 99<|end_message|><|message_model|>"
        "<|content_text|>Paris<|end_message|>"
    )
    assert scoring.answer_text(text) == "Paris"
    assert scoring.final_segment(text) == "Paris"


def test_final_segment_takes_the_committed_answer_after_a_plain_preamble():
    assert scoring.final_segment('Good. Final should be "Paris".Paris') == "Paris"
    assert scoring.final_segment("reasoning\n\nThe answer is 18") == "The answer is 18"


def test_qa_word_uses_the_final_segment_not_the_preamble():
    item = {"expected": {"answers": ["blue"], "lang": "en"}}
    completion = "The user asks what colour the sky is. Sky is blue.\n\nBlue."
    assert scoring.final_segment(completion) == "Blue."
    assert scoring.score_qa_word(item, completion)["ok"] is True


def test_chrf_scores_the_answer_channel_and_ignores_a_thinking_translation():
    item = {
        "expected": {
            "reference": "Der Zug fährt um sieben Uhr morgens ab.",
            "lang": "de",
            "min_chrf": 0.2,
        }
    }
    good = (
        "<|content_thinking|>Need translate.<|end_message|>"
        "<|content_text|>Der Zug fährt um sieben Uhr morgens ab.<|end_message|>"
    )
    wrong = (
        "<|content_thinking|>Der Zug fährt um sieben Uhr morgens ab.<|end_message|>"
        "<|content_text|>I do not know.<|end_message|>"
    )
    assert scoring.score_chrf(item, good)["ok"] is True
    assert scoring.score_chrf(item, wrong)["ok"] is False


def test_fenced_code_ignores_inline_prose_mentions_and_truncated_blocks():
    text = (
        "Need exactly one ```python code block and nothing else."
        "```python\ndef add(a, b):\n    return a + b\n```\n"
        "A cut-off second attempt:\n```python\ndef add(a, b):\n    \"\"\"unterminated"
    )
    assert "return a + b" in scoring.extract_code(text, "add", "python")
