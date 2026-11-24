# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Render Inkling-2 conversations with the model's own chat template.

This is a Python transcription of the template shipped with Inkling-2-Mini
(``chat_template.jinja``; the box copy is
``~/w4a8_runs/inkling-hi2/stage4/runs/bpw3.5/chat_template.jinja``).  Rendering
locally, and sending the result to ``/v1/completions``, is what makes the task
prompts independent of which server template version is running: stock and
quantized endpoints receive byte-identical prompts.

The transcription is checked against the original Jinja template by
``tests/test_inkling_eval_template.py`` when a template file is available.

Message roles: ``system``, ``user``, ``assistant`` and ``tool``.  A tool result
carries ``name`` (preferred) or the ``tool_call_id`` of the assistant call it
answers.  Assistant tool calls carry parsed-object ``arguments``; the template
rejects JSON strings, so callers must parse first.
"""

import json

ROLE_TOKEN = {
    "user": "<|message_user|>",
    "assistant": "<|message_model|>",
    "system": "<|message_system|>",
    "tool": "<|message_tool|>",
}
EFFORT_MAP = {
    "none": 0.0,
    "minimal": 0.1,
    "low": 0.2,
    "medium": 0.7,
    "high": 0.9,
    "xhigh": 0.99,
    "max": 0.99,
}
END = "<|end_message|>"
TEXT = "<|content_text|>"
XML = "<|content_xml|>"
THINKING = "<|content_thinking|>"
INVOKE = "<|content_invoke_tool_json|>"
MODEL_END = "<|content_model_end_sampling|>"


def _dumps(obj, **overrides) -> str:
    """Jinja's ``tojson``: html-safe JSON, sorted keys, compact separators."""
    kwargs = {"sort_keys": True, "separators": (",", ":"), "ensure_ascii": True}
    kwargs.update(overrides)
    text = json.dumps(obj, **kwargs)
    return (
        text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("'", "\\u0027")
    )


def _effort(value) -> float:
    if isinstance(value, str):
        key = value.strip()
        if key not in EFFORT_MAP:
            raise ValueError(f"unknown reasoning_effort: {value!r}")
        return EFFORT_MAP[key]
    num = float(value)
    if num < 0.0 or num > 0.99:
        raise ValueError("reasoning_effort must be in [0.0, 0.99]")
    return num


def _effort_number(num: float) -> str:
    return "0" if num == 0.0 else str(num)


def effort_message(value) -> str:
    """The template's implicit system turn that selects reasoning effort."""
    num = _effort(value)
    return (
        f"{ROLE_TOKEN['system']}{TEXT}Thinking effort level: "
        f"{_effort_number(num)}{END}"
    )


def _spec(tool: dict) -> dict:
    """A tool declaration in the template's key order (it sorts keys anyway)."""
    fn = tool.get("function", tool)
    return {
        "description": fn.get("description") or "",
        "name": fn["name"],
        "parameters": fn.get("parameters") or {},
        "type": tool.get("type") or "function",
    }


def tools_block(tools) -> str:
    specs = [_spec(tool) for tool in tools]
    return f"{ROLE_TOKEN['system']}tool_declare{XML}{_dumps(specs)}{END}"


def _tool_header(messages_prefix: list, message: dict) -> str:
    if message.get("name"):
        return ROLE_TOKEN["tool"] + message["name"]
    call_id = message.get("tool_call_id")
    if call_id:
        for prev in messages_prefix:
            if prev.get("role") != "assistant":
                continue
            for call in prev.get("tool_calls") or []:
                if call.get("id") == call_id and call.get("function", {}).get("name"):
                    return ROLE_TOKEN["tool"] + call["function"]["name"]
    return ROLE_TOKEN["tool"]


def _call_block(call: dict) -> str:
    fn = call["function"]
    name = fn["name"]
    if not isinstance(name, str):
        raise ValueError("tool call function name must be a string")
    args = fn.get("arguments") or {}
    if isinstance(args, str):
        raise ValueError(
            "tool call arguments must be a parsed object, not a JSON string"
        )
    if not isinstance(args, dict):
        raise ValueError("tool call arguments must be an object")
    body = ",".join(f"{_dumps(key)}:{_dumps(args[key])}" for key in sorted(args))
    return (
        f"{ROLE_TOKEN['assistant']}{name}{INVOKE}"
        f'{{"name":{_dumps(name)},"args":{{{body}}}}}{END}'
    )


def render(messages, tools=None, reasoning_effort=0.0) -> str:
    """The prompt string the served model sees, matching ``chat_template.jinja``."""
    out = []
    if tools:
        out.append(tools_block(tools))
    effort_done = False
    seen = []
    for message in messages:
        role = message.get("role")
        if role not in ROLE_TOKEN:
            raise ValueError(f"unknown message role: {role!r}")
        if not effort_done and role != "system":
            out.append(effort_message(reasoning_effort))
            effort_done = True
        if role == "tool":
            header = _tool_header(seen, message)
        elif role == "assistant" and message.get("name"):
            header = ROLE_TOKEN[role] + message["name"]
        else:
            header = ROLE_TOKEN[role]
        if role == "assistant" and message.get("reasoning_content"):
            out.append(f"{header}{THINKING}{message['reasoning_content']}{END}")
        content = message.get("content")
        has_calls = bool(message.get("tool_calls"))
        if isinstance(content, str) and (
            content or not (role == "assistant" and (has_calls or message.get("reasoning_content")))
        ):
            out.append(f"{header}{TEXT}{content}{END}")
        elif content:
            for part in content:
                if isinstance(part, str):
                    out.append(f"{header}{TEXT}{part}{END}")
                elif part.get("type") in (None, "text", "input_text"):
                    out.append(f"{header}{TEXT}{part.get('text') or ''}{END}")
                elif part.get("type") in ("image", "input_image", "image_url"):
                    if role not in ("user", "tool"):
                        raise ValueError(
                            "image content is only supported for user and tool messages"
                        )
                    out.append(f"{header}<|content_image|><|unused_200054|>{END}")
                elif part.get("type") in ("audio", "input_audio", "audio_url"):
                    raise ValueError("Inkling 2 does not accept audio input")
                else:
                    raise ValueError(f"unsupported content part type: {part.get('type')}")
        if role == "assistant" and has_calls:
            for call in message["tool_calls"]:
                out.append(_call_block(call))
        if role == "assistant":
            out.append(MODEL_END)
        seen.append(message)
    if not effort_done:
        out.append(effort_message(reasoning_effort))
    return "".join(out)


def user_prompt(text: str, tools=None, reasoning_effort=0.0) -> str:
    """A single user turn, the shape every non-agentic task uses."""
    return render(
        [{"role": "user", "content": text}],
        tools=tools,
        reasoning_effort=reasoning_effort,
    )
