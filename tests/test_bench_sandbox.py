# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""bench sandbox: exit status, timeout kill, disabled network and file removal."""

import time

from bench.sandbox import extract_code, run_python


def test_pass_and_fail():
    assert run_python("assert 1 + 1 == 2\n")["status"] == "passed"
    r = run_python("raise ValueError('boom')\n")
    assert r["status"] == "failed" and "boom" in r["stderr"]


def test_timeout_kills_the_child():
    t0 = time.time()
    r = run_python("while True:\n    pass\n", timeout=1.0)
    assert r["status"] == "timeout" and not r["passed"]
    assert time.time() - t0 < 5


def test_sleeping_child_is_killed():
    r = run_python("import time\ntime.sleep(30)\n", timeout=0.5)
    assert r["status"] == "timeout"


def test_network_and_removal_disabled(tmp_path):
    r = run_python("import socket\nsocket.create_connection(('1.1.1.1', 80))\n")
    assert r["status"] == "failed" and "disabled" in r["stderr"]
    victim = tmp_path / "keep.txt"
    victim.write_text("x")
    r = run_python(f"import os\nos.remove({str(victim)!r})\n")
    assert r["status"] == "failed" and victim.exists()


def test_runs_in_an_empty_temp_dir():
    r = run_python(
        "import os\nassert os.listdir('.') == ['prog.py'], os.listdir('.')\n"
    )
    assert r["passed"], r


def test_extract_code():
    text = "x\n```python\nimport os\n```\nthen\n```python\ndef f(a):\n    return a\n```"
    assert extract_code(text, "f").startswith("def f")
    assert extract_code(text).startswith("import os")
    assert extract_code("def f(): pass") == "def f(): pass"
    assert "def g" in extract_code(
        "```\ndef g():\n    return 1\n"
    )  # unterminated fence
