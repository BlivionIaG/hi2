# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Sandbox execution: Python, C, TypeScript and the type-stripping fallback."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.inkling_eval import sandbox

DATA = Path(__file__).resolve().parents[1] / "tools" / "inkling_eval" / "data"


def test_python_status_pass_fail_timeout():
    good = sandbox.run_python("def add(a, b):\n    return a + b\n", "assert add(2, 3) == 5\nprint('OK')")
    assert good["status"] == "pass", good
    bad = sandbox.run_python("def add(a, b):\n    return a - b\n", "assert add(2, 3) == 5\nprint('OK')")
    assert bad["status"] == "fail", bad
    slow = sandbox.run_python("while True:\n    pass\n", "print('OK')", timeout=1)
    assert slow["status"] == "timeout", slow
    # The answer must print OK: exit 0 alone is not enough.
    quiet = sandbox.run_python("print('done')\n", "")
    assert quiet["status"] == "fail", quiet


@pytest.mark.skipif(not any(shutil.which(name) for name in ("gcc", "cc", "clang")), reason="no C compiler")
def test_c_pass_fail_and_missing_entry():
    row = next(json.loads(line) for line in (DATA / "c_code.jsonl").open() if json.loads(line)["id"] == "gcd")
    tests = row["prototype"] + "\n" + row["tests"]
    assert sandbox.run_c(row["solution"], tests)["status"] == "pass"
    assert sandbox.run_c(row["solution"].replace("a % b", "a + b"), tests)["status"] == "fail"
    assert sandbox.run_c("int not_gcd(int a, int b) { return 0; }", tests)["status"] != "pass"


def test_engine_is_reported_when_present():
    engine = sandbox.js_engine()
    if engine["name"] is None:
        pytest.skip("no JavaScript engine on this machine")
    assert engine["name"] in ("node", "bun", "deno", "qjs", "quickjs")


@pytest.mark.skipif(sandbox.js_engine()["name"] is None, reason="no JavaScript engine")
def test_typescript_pass_fail():
    row = next(json.loads(line) for line in (DATA / "ts_code.jsonl").open() if json.loads(line)["id"] == "sum_even")
    passed = sandbox.run_typescript(row["solution"], row["tests"])
    assert passed["status"] == "pass", passed
    failed = sandbox.run_typescript(row["solution"].replace("total += x", "total -= x"), row["tests"])
    assert failed["status"] == "fail", failed


def test_strip_typescript_documented_subset():
    cases = {
        "function f(a: number, b: string): number { return a; }": "function f(a, b)",
        "let y: number[] = [1, 2];": "let y=",
        "const z = value as number;": "const z = value;",
        "function h<T>(x: T): T { return x; }": "function h(x)",
        "const o = { a: 1, b: 2 };": "const o = { a: 1, b: 2 };",
        'const s = "interface P { a: number }";': '"interface P { a: number }"',
        "function f(a?: number) { return a; }": "function f(a)",
        "function g(cb: (x: number) => void): void { cb(1); }": "function g(cb)",
    }
    for source, expected in cases.items():
        assert expected in sandbox.strip_typescript(source), (source, sandbox.strip_typescript(source))
    stripped = sandbox.strip_typescript(
        "function map2<T, U>(xs: T[], fn: (x: T) => U): U[] { const out: U[] = []; return out; }"
    )
    assert "function map2(xs, fn)" in stripped
    assert "<T" not in stripped and ": U[]" not in stripped


@pytest.mark.skipif(not shutil.which("node"), reason="no node to run stripped JavaScript")
def test_stripped_typescript_is_runnable(tmp_path):
    source = (
        "interface Point { x: number; y: number }\n"
        "function norm(p: Point): number { const s: number = p.x * p.x + p.y * p.y; return s; }\n"
        "const v = norm({ x: 3, y: 4 } as Point);\n"
        "console.log(v);\n"
    )
    path = tmp_path / "candidate.mjs"
    path.write_text(sandbox.strip_typescript(source))
    result = subprocess.run([shutil.which("node"), str(path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "25"


def test_scratch_root_honours_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(sandbox.SCRATCH_ENV, str(tmp_path))
    assert sandbox.scratch_root() == tmp_path
    result = sandbox.run_python("print('OK')\n", "")
    assert result["status"] == "pass"
    assert not any(tmp_path.iterdir()), "the sandbox directory is removed"
