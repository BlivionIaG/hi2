# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Decision fidelity (spec 12.3): what a quantized model decides in an agent loop.

``run`` sends the prompt set (``decision_prompts.py``) to an OpenAI-compatible server with
greedy decoding and records, for each prompt, the first tool the model calls, whether the
call is well formed, and the garbage checks. ``compare`` scores runs against a reference
run: the share of prompts with the same first tool, the same first tool and the same
argument, the rate of well-formed calls, and the garbage count. A second run of the
unquantized model gives the spread that greedy decoding has on this stack; only
differences beyond that spread count.

    python decision_fidelity.py run --url http://127.0.0.1:18731 --model NAME --out run.json
    python decision_fidelity.py compare ref.json ref2.json other.json [...]
"""

import argparse
import json
import re
import statistics
import sys
import urllib.request
from pathlib import Path

PROMPTS = Path(__file__).parent / "data" / "decision_prompts.json"
REQUIRED = {
    "read_file": ("path",),
    "write_to_file": ("path", "content"),
    "replace_in_file": ("path", "diff"),
    "execute_command": ("command",),
    "search_files": ("path", "regex"),
    "list_files": ("path",),
    "ask_followup_question": ("question",),
    "attempt_completion": ("result",),
}
KEY_PARAM = {  # the argument that says what the call acts on
    "read_file": "path",
    "write_to_file": "path",
    "replace_in_file": "path",
    "execute_command": "command",
    "search_files": "regex",
    "list_files": "path",
}
TOOL = re.compile(r"<(" + "|".join(REQUIRED) + r")>")


def parse(text: str) -> dict:
    """First tool call of a reply: ``tool`` (or ``none``), ``valid`` and its key argument."""
    m = TOOL.search(text)
    if not m:
        return {"tool": "none", "valid": False, "arg": ""}
    tool = m.group(1)
    end = text.find(f"</{tool}>", m.end())
    body = text[m.end() : end] if end >= 0 else ""
    params = {
        k: v.strip() for k, v in re.findall(r"<(\w+)>(.*?)</\1>", body, flags=re.S)
    }
    valid = end >= 0 and all(params.get(p) for p in REQUIRED[tool])
    arg = params.get(KEY_PARAM.get(tool, ""), "")
    return {"tool": tool, "valid": valid, "arg": re.sub(r"\s+", " ", arg).strip("/ ")}


def garbage(text: str) -> list[str]:
    flags = []
    if not text.strip():
        flags.append("empty")
    if re.search(r"(.{3,40}?)\1{9,}", text, flags=re.S):
        flags.append("repeat")
    if text and sum(c < " " and c not in "\n\t\r" for c in text) / len(text) > 0.02:
        flags.append("non-printable")
    if "!!!!!!!!" in text:
        flags.append("bang-flood")
    return flags


def ask(url: str, model: str, messages: list[dict], max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def run(a) -> int:
    prompts = json.loads(PROMPTS.read_text())
    rows = []
    for p in prompts:
        try:
            r = ask(a.url, a.model, p["messages"], a.max_tokens)
            text = r["choices"][0]["message"]["content"] or ""
            tokens = r["usage"]["completion_tokens"]
        except Exception as e:  # keep going: a failed request is a result too
            text, tokens = f"<<request failed: {e}>>", 0
        row = {"id": p["id"], "stage": p["stage"], "text": text, "tokens": tokens}
        row |= parse(text) | {"garbage": garbage(text)}
        rows.append(row)
    out = {"model": a.model, "n": len(rows), "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1))
    ok = sum(r["valid"] for r in rows)
    print(f"{a.model}: {len(rows)} prompts, {ok} well-formed calls")
    return 0


def _rows(path: str) -> dict[str, dict]:
    return {r["id"]: r for r in json.loads(Path(path).read_text())["rows"]}


def agreement(a: dict, b: dict) -> tuple[float, float]:
    """(same first tool, same first tool and argument) over the shared prompts."""
    ids = a.keys() & b.keys()
    same = sum(a[i]["tool"] == b[i]["tool"] for i in ids)
    arg = sum(a[i]["tool"] == b[i]["tool"] and a[i]["arg"] == b[i]["arg"] for i in ids)
    return same / len(ids), arg / len(ids)


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return c - h, c + h


def compare(a) -> int:
    ref = _rows(a.runs[0])
    n = len(ref)
    print(
        f"reference {a.runs[0]} ({n} prompts); well-formed = parseable call with its parameters"
    )
    head = f"{'run':<28}{'tool agree':>12}{'95% CI':>15}{'+argument':>11}{'well-formed':>13}{'garbage':>9}"
    print(head)
    for path in a.runs:
        rows = _rows(path)
        tool, arg = (1.0, 1.0) if path == a.runs[0] else agreement(ref, rows)
        lo, hi = wilson(tool * n, n)
        valid = sum(r["valid"] for r in rows.values())
        bad = sum(bool(r["garbage"]) for r in rows.values())
        print(
            f"{Path(path).stem:<28}{tool:>12.1%}{f'{lo:.0%}-{hi:.0%}':>15}"
            f"{arg:>11.1%}{valid / len(rows):>13.1%}{bad:>9}"
        )
    stages = sorted({r["stage"] for r in ref.values()})
    print("\ntool agreement with the reference, by stage:")
    print(f"{'run':<28}" + "".join(f"{s:>11}" for s in stages))
    for path in a.runs[1:]:
        rows = _rows(path)
        cells = []
        for s in stages:
            ids = [i for i, r in ref.items() if r["stage"] == s and i in rows]
            cells.append(
                statistics.mean(ref[i]["tool"] == rows[i]["tool"] for i in ids)
            )
        print(f"{Path(path).stem:<28}" + "".join(f"{c:>11.0%}" for c in cells))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-tokens", type=int, default=384)
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+", help="reference first, then the others")
    a = ap.parse_args()
    return run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    sys.exit(main())
