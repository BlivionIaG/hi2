# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""The Inkling chat-template transcription: golden strings and parity with Jinja."""

import json
import os
import re
from pathlib import Path

import pytest

from tools.inkling_eval import chat_template as ct


def test_user_prompt_golden():
    assert ct.user_prompt("hi") == (
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
        "<|message_user|><|content_text|>hi<|end_message|>"
    )
    assert ct.user_prompt("hi", reasoning_effort="none") == ct.user_prompt("hi", 0.0)


def test_effort_numbers_and_bounds():
    for value, expected in ((0.0, "0"), (0.7, "0.7"), ("high", "0.9"), (0.99, "0.99")):
        assert ct.effort_message(value) == (
            "<|message_system|><|content_text|>Thinking effort level: "
            f"{expected}<|end_message|>"
        )
    with pytest.raises(ValueError):
        ct.effort_message(1.0)
    with pytest.raises(ValueError):
        ct.effort_message("extreme")


def test_tools_declaration_and_sorted_arguments():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]
    messages = [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Weather in Lisbon?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"unit": "celsius", "location": "Lisbon"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "12 C"},
        {"role": "user", "content": "And in Berlin?"},
    ]
    rendered = ct.render(messages, tools=tools, reasoning_effort=0.0)
    specs = json.dumps(
        [
            {
                "description": "Current weather for a location.",
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
                "type": "function",
            }
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    assert rendered.startswith(
        "<|message_system|>tool_declare<|content_xml|>" + specs + "<|end_message|>"
    )
    assert "<|message_model|>get_weather<|content_invoke_tool_json|>" in rendered
    assert (
        '{"name":"get_weather","args":{"location":"Lisbon","unit":"celsius"}}'
        in rendered
    )
    # The tool result takes the function name from the call id it answers.
    assert "<|message_tool|>get_weather<|content_text|>12 C<|end_message|>" in rendered
    assert rendered.endswith("<|message_user|><|content_text|>And in Berlin?<|end_message|>")
    assert rendered.count("<|content_model_end_sampling|>") == 1


def test_tool_call_arguments_must_be_parsed():
    with pytest.raises(ValueError):
        ct.render(
            [
                {"role": "user", "content": "x"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "f",
                                "arguments": '{"a": 1}',
                            }
                        }
                    ],
                },
            ]
        )
    with pytest.raises(ValueError):
        ct.render([{"role": "narrator", "content": "x"}])


def test_content_parts_and_effort_at_end():
    rendered = ct.render(
        [{"role": "user", "content": [{"type": "text", "text": "one"}, {"type": "image"}]}]
    )
    assert "<|content_text|>one<|end_message|>" in rendered
    assert "<|content_image|><|unused_200054|>" in rendered
    system_only = ct.render([{"role": "system", "content": "s"}], reasoning_effort=0.2)
    assert system_only == (
        "<|message_system|><|content_text|>s<|end_message|>"
        "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>"
    )


def _reference_template() -> Path | None:
    candidates = []
    if os.environ.get("INKLING_CHAT_TEMPLATE"):
        candidates.append(Path(os.environ["INKLING_CHAT_TEMPLATE"]))
    local = Path.home() / ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots"
    candidates.extend(sorted(local.glob("*/chat_template.jinja")))
    candidates.extend(
        [
            Path("/home/chenco_adm/w4a8_runs/inkling-hi2/stage4/runs/bpw3.5/chat_template.jinja"),
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def test_matches_reference_jinja_template():
    pytest.importorskip("jinja2")
    import jinja2.sandbox
    from jinja2.utils import htmlsafe_json_dumps

    template_path = _reference_template()
    if template_path is None:
        pytest.skip("no Inkling chat template on this machine")
    text = template_path.read_text(encoding="utf-8")
    # `{% generation %}` is a transformers-only masking extension; it emits nothing.
    text = re.sub(r"\{%-?\s*(?:end)?generation\s*-?%\}", "", text)

    def raise_exception(message):
        raise ValueError(message)

    environment = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    )
    environment.filters["tojson"] = htmlsafe_json_dumps
    environment.globals["raise_exception"] = raise_exception
    template = environment.from_string(text)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_customer",
                "description": "Look up a customer by email.",
                "parameters": {
                    "type": "object",
                    "properties": {"email": {"type": "string"}},
                    "required": ["email"],
                },
            },
        }
    ]
    conversations = [
        [{"role": "user", "content": "hello"}],
        [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Look up ada@example.com."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup_customer",
                            "arguments": {"email": "ada@example.com"},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "C-1040"},
            {"role": "user", "content": "Now email her."},
        ],
        [
            {"role": "user", "content": "part one"},
            {"role": "assistant", "content": "answer"},
        ],
    ]
    for effort in (0.0, 0.7, "none"):
        for messages in conversations:
            mine = ct.render(messages, tools=tools, reasoning_effort=effort)
            theirs = template.render(messages=messages, tools=tools, reasoning_effort=effort)
            assert mine == theirs
