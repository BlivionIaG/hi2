# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""bench run.py against a fake in-process OpenAI-compatible server (http.server)."""

import json
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bench import run
from bench.client import Client


class Handler(BaseHTTPRequestHandler):
    fail = set()  # prompts that get an HTTP 500
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/chat/completions":
            q = body["messages"][-1]["content"]
            Handler.seen.append(q)
            out = {
                "choices": [{"message": {"content": f"<think>x</think>{q.upper()}"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        elif self.path == "/v1/completions":
            q = body["prompt"]
            Handler.seen.append(q)
            out = {"choices": [{"text": q[::-1]}], "usage": {}}
        else:
            self.send_error(404)
            return
        if q in Handler.fail:
            self.send_error(500)
            return
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    Handler.fail, Handler.seen = set(), []
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def fake_task(version="1"):
    words = ["alpha", "beta", "gamma", "delta", "eps"]
    return types.SimpleNamespace(
        NAME="fake",
        VERSION=version,
        MAX_TOKENS=8,
        DEFAULT_LIMIT=None,
        METRICS=("len",),
        load=lambda limit, seed: [
            {"id": f"f/{w}", "group": w[0], "w": w} for w in words
        ][:limit],
        prompt=lambda it: [{"role": "user", "content": it["w"]}],
        score=lambda it, c: {
            "ok": c == it["w"].upper() and it["w"] != "eps",
            "len": len(c),
        },
    )


def text_task():
    return types.SimpleNamespace(
        NAME="txt",
        VERSION="1",
        MAX_TOKENS=4,
        DEFAULT_LIMIT=None,
        load=lambda limit, seed: [{"id": "t/0", "s": "abc"}],
        prompt=lambda it: it["s"],
        score=lambda it, c: {"ok": c == "cba"},
    )


def test_run_store_and_resume(server, tmp_path):
    out = tmp_path / "r.json"
    client = Client(server, "m", retries=0)
    Handler.fail = {"gamma"}
    st = run.run_suite(
        client,
        {"fake": fake_task(), "txt": text_task()},
        None,
        0,
        4,
        out,
        log=lambda *a: None,
    )
    rows = st["results"]["fake"]
    assert len(rows) == 5 and "error" in rows["f/gamma"]
    assert rows["f/alpha"]["ok"] and not rows["f/eps"]["ok"]
    assert rows["f/alpha"]["completion"].startswith("<think>")  # raw reply kept
    assert rows["f/alpha"]["details"] == {"len": 5}  # scored without the think block
    assert st["results"]["txt"]["t/0"]["ok"]  # text completion endpoint
    saved = json.loads(out.read_text())
    assert saved["config"]["tasks"]["fake"]["version"] == "1"
    assert saved["config"]["tasks"]["fake"]["metrics"] == ["len"]
    assert saved["timing"]["fake"]["requests"] == 5
    # resume: only the failed item is requested again
    Handler.fail, Handler.seen = set(), []
    st = run.run_suite(
        client,
        {"fake": fake_task(), "txt": text_task()},
        None,
        0,
        4,
        out,
        log=lambda *a: None,
    )
    assert Handler.seen == ["gamma"]
    assert st["results"]["fake"]["f/gamma"]["ok"]
    assert st["timing"]["fake"]["requests"] == 6
    # a new task version redoes the task
    Handler.seen = []
    run.run_suite(
        client, {"fake": fake_task("2")}, None, 0, 2, out, log=lambda *a: None
    )
    assert sorted(Handler.seen) == sorted(["alpha", "beta", "gamma", "delta", "eps"])


def test_limit_and_main(server, tmp_path, monkeypatch):
    monkeypatch.setattr(run.registry, "resolve", lambda spec: {"fake": fake_task()})
    out = tmp_path / "sub" / "r.json"
    assert (
        run.main(
            [
                "--url",
                server,
                "--model",
                "m",
                "--tasks",
                "fake",
                "--limit",
                "2",
                "--workers",
                "2",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    st = json.loads(out.read_text())
    assert sorted(st["results"]["fake"]) == ["f/alpha", "f/beta"]
    assert st["config"]["chat_extra"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_score_exception_is_recorded(server, tmp_path):
    t = fake_task()
    t.score = lambda it, c: 1 / 0
    st = run.run_suite(
        Client(server, "m"),
        {"fake": t},
        1,
        0,
        1,
        tmp_path / "r.json",
        log=lambda *a: None,
    )
    row = st["results"]["fake"]["f/alpha"]
    assert not row["ok"] and "ZeroDivisionError" in row["details"]["score_error"]
