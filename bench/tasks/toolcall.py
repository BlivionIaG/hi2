# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Tool calling on ToolACE: the dataset's system prompt (function list and call format)
and the first user turn whose reference reply is a call list. ``ok`` = the same calls
(names and arguments, order-free); details: the names match, and the argument-wise F1
over (function, argument, value) triples.

Replies are read in the ToolACE format ``[f(a=1, b="x"), g()]``, as JSON
(``{"name", "arguments"}`` objects or a list of them) or as ``<tool_call>`` JSON blocks.
"""

import ast
import json
import re
from collections import Counter

from bench.data import hf_rows, sample

NAME = "toolcall"
VERSION = "1"
MAX_TOKENS = 512
DEFAULT_LIMIT = 500
DATASETS = [("Team-ACE/ToolACE", None, "train")]
METRICS = ("arg_f1",)
CONSTS = {"true": True, "false": False, "null": None, "none": None}


def _split_top(s: str, sep: str = ",") -> list[str]:
    """Split at ``sep`` outside brackets and string literals."""
    parts, depth, quote, cur, i = [], 0, None, [], 0
    while i < len(s):
        ch = s[i]
        if quote:
            if ch == "\\":
                cur.append(s[i : i + 2])
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _literal(node):
    if isinstance(node, ast.Name) and node.id.lower() in CONSTS:
        return CONSTS[node.id.lower()]
    if isinstance(node, ast.List | ast.Tuple):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def parse_python_calls(text: str) -> list[dict] | None:
    """``[f(a=1), g name(b="x")]`` -> calls; None when the text is not such a list."""
    s = text.strip()
    start, end = s.find("["), s.rfind("]")
    if start < 0 or end < start:
        return None
    calls = []
    for part in _split_top(s[start + 1 : end]):
        m = re.fullmatch(r"([^()\[\]{}=\"']+?)\s*\((.*)\)", part, re.S)
        if not m:
            return None
        try:
            tree = ast.parse(f"f({m.group(2)})", mode="eval").body
            args = {kw.arg: _literal(kw.value) for kw in tree.keywords}
        except (SyntaxError, ValueError, TypeError):
            return None
        if tree.args:
            return None
        calls.append({"name": m.group(1).strip(), "arguments": args})
    return calls


def _json_calls(obj) -> list[dict] | None:
    objs = obj if isinstance(obj, list) else [obj]
    calls = []
    for o in objs:
        if not isinstance(o, dict) or "name" not in o:
            return None
        o = o.get("function", o) if isinstance(o.get("function"), dict) else o
        args = o.get("arguments", o.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return None
        if not isinstance(args, dict):
            return None
        calls.append({"name": str(o["name"]).strip(), "arguments": args})
    return calls


def parse_calls(text: str) -> list[dict] | None:
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    if blocks:
        try:
            return _json_calls([json.loads(b) for b in blocks])
        except ValueError:
            return None
    body = re.sub(r"^\s*```(?:json|python)?\s*|\s*```\s*$", "", text.strip())
    try:
        return _json_calls(json.loads(body))
    except ValueError:
        pass
    return parse_python_calls(body)


def norm(v):
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, int | float):
        return round(float(v), 6)
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, list | tuple):
        return tuple(norm(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((str(k), norm(x)) for k, x in v.items()))
    return str(v)


def triples(calls: list[dict]) -> Counter:
    out = Counter()
    for c in calls:
        name = c["name"].lower()
        out[(name, None, None)] += 1
        for k, v in c["arguments"].items():
            out[(name, k, norm(v))] += 1
    return out


def load(limit, seed):
    items = []
    for i, r in enumerate(hf_rows(*DATASETS[0])):
        conv = r["conversations"]
        if len(conv) < 2 or conv[0]["from"] != "user" or conv[1]["from"] != "assistant":
            continue
        gold = parse_python_calls(conv[1]["value"])
        if not gold or not conv[1]["value"].strip().startswith("["):
            continue
        items.append(
            {
                "id": f"toolace/{i}",
                "system": r["system"],
                "user": conv[0]["value"],
                "gold": gold,
                "group": "single" if len(gold) == 1 else "multi",
            }
        )
    return sample(items, limit, seed)


def prompt(item):
    return [
        {"role": "system", "content": item["system"]},
        {"role": "user", "content": item["user"]},
    ]


def score(item, completion):
    pred = parse_calls(completion)
    if pred is None:
        return {"ok": False, "parsed": False, "name_ok": False, "arg_f1": 0.0}
    g, p = triples(item["gold"]), triples(pred)
    tp = sum((g & p).values())
    f1 = 2 * tp / (sum(g.values()) + sum(p.values()))
    names = sorted(c["name"].lower() for c in pred) == sorted(
        c["name"].lower() for c in item["gold"]
    )
    return {"ok": g == p, "parsed": True, "name_ok": names, "arg_f1": f1}
