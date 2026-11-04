# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""bench run.py against a fake in-process OpenAI-compatible server (http.server)."""

import json
import os
import subprocess
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from bench import run, tfkld
from bench.client import Client
from bench.data import DatasetMissing


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
    assert saved["status"] == "incomplete"
    assert saved["completion"]["tasks"]["fake"]["request_errors"] == 1
    assert saved["completion"]["tasks"]["fake"]["completed"] == 4
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
    assert st["status"] == "complete"
    assert st["completion"]["tasks"]["fake"]["completed"] == 5
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
    assert st["status"] == "incomplete"
    assert st["completion"]["tasks"]["fake"]["score_errors"] == 1


def main_args(server, out, tasks="fake"):
    return [
        "--url",
        server,
        "--model",
        "m",
        "--tasks",
        tasks,
        "--workers",
        "1",
        "--out",
        str(out),
    ]


@pytest.mark.parametrize("failure", ["dataset", "import", "empty", "request", "score"])
def test_main_incomplete_preserves_other_tasks(server, tmp_path, monkeypatch, failure):
    bad = fake_task()

    def fail(*args):
        if failure == "dataset":
            raise DatasetMissing("fixture dataset unavailable")
        raise RuntimeError("fixture scorer failed")

    if failure == "dataset":
        bad.load = fail
    elif failure == "empty":
        bad.load = lambda limit, seed: []
    elif failure == "request":
        Handler.fail = {"gamma"}
    elif failure == "score":
        bad.score = fail

    def resolve(name):
        if name == "broken":
            if failure == "import":
                raise ImportError("fixture task import unavailable")
            return {"broken": bad}
        return {"txt": text_task()}

    monkeypatch.setattr(run.registry, "resolve", resolve)
    out = tmp_path / "partial.json"
    assert run.main(main_args(server, out, "broken,txt")) == 1
    report = json.loads(out.read_text())
    assert report["schema_version"] == 1 and report["status"] == "incomplete"
    assert report["completion"]["requested_tasks"] == ["broken", "txt"]
    assert report["completion"]["tasks"]["broken"]["status"] in (
        "load_error",
        "incomplete",
    )
    assert report["completion"]["tasks"]["txt"]["status"] == "complete"
    assert report["results"]["txt"]["t/0"]["ok"]
    assert report["attempts"][-1]["command"][0] == "bench/run.py"


def test_low_accuracy_is_completed_execution(server, tmp_path, monkeypatch):
    task = fake_task()
    task.score = lambda item, text: {"ok": False}
    monkeypatch.setattr(run.registry, "resolve", lambda spec: {"fake": task})
    out = tmp_path / "wrong.json"
    assert run.main(main_args(server, out)) == 0
    report = json.loads(out.read_text())
    assert report["status"] == "complete"
    assert report["completion"]["tasks"]["fake"]["completed"] == 5
    assert all(not row["ok"] for row in report["results"]["fake"].values())


def test_harness_failure_saves_incomplete_report(server, tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("unexpected harness failure")

    monkeypatch.setattr(run, "work", fail)
    monkeypatch.setattr(run.registry, "resolve", lambda spec: {"fake": fake_task()})
    out = tmp_path / "harness.json"
    assert run.main(main_args(server, out)) == 1
    report = json.loads(out.read_text())
    assert report["status"] == "incomplete"
    assert "unexpected harness failure" in report["completion"]["errors"][0]
    assert report["completion"]["tasks"]["fake"]["missing"] == 5


@pytest.mark.parametrize("stage", ["load", "score", "work"])
def test_task_system_exit_zero_is_still_incomplete(tmp_path, monkeypatch, stage):
    task = fake_task()

    def exit_zero(*args):
        raise SystemExit(0)

    if stage == "work":
        monkeypatch.setattr(run, "work", exit_zero)
    elif stage == "load":
        task.load = exit_zero
    else:
        task.request = lambda client, item: ("answer", {})
        task.score = exit_zero
    monkeypatch.setattr(run.registry, "resolve", lambda spec: {"fake": task})
    out = tmp_path / "exit-zero.json"
    assert run.main(main_args("http://unused", out)) == 1
    assert json.loads(out.read_text())["status"] == "incomplete"


def test_loading_interruption_preserves_explicit_incomplete_report(
    tmp_path, monkeypatch
):
    task = fake_task()

    def interrupt(*args):
        raise KeyboardInterrupt

    task.load = interrupt
    monkeypatch.setattr(run.registry, "resolve", lambda spec: {"fake": task})
    out = tmp_path / "interrupted.json"
    assert run.main(main_args("http://unused", out)) == 130
    report = json.loads(out.read_text())
    assert report["status"] == "incomplete"
    assert report["completion"]["tasks"]["fake"]["error"] == "KeyboardInterrupt"
    assert report["attempts"][-1]["status"] == "incomplete"


def test_score_failures_are_retried_on_resume(server, tmp_path):
    task = fake_task()
    good_score = task.score
    task.score = lambda item, text: {}  # malformed scorer response is a harness error
    out = tmp_path / "scorer.json"
    client = Client(server, "m")
    first = run.run_suite(client, {"fake": task}, 1, 0, 1, out)
    assert first["status"] == "incomplete"
    assert "KeyError" in first["results"]["fake"]["f/alpha"]["details"]["score_error"]
    task.score = good_score
    Handler.seen = []
    second = run.run_suite(client, {"fake": task}, 1, 0, 1, out)
    assert second["status"] == "complete"
    assert Handler.seen == ["alpha"]


def test_changed_model_does_not_mix_resume_results(server, tmp_path, monkeypatch):
    monkeypatch.setattr(run.registry, "resolve", lambda spec: {"fake": fake_task()})
    out = tmp_path / "identity.json"
    args = main_args(server, out)
    assert run.main(args) == 0
    before = out.read_bytes()
    args[args.index("--model") + 1] = "different-model"
    assert run.main(args) == 1
    assert out.read_bytes() == before


def test_dataset_content_change_invalidates_resume(server, tmp_path):
    task = fake_task()
    client = Client(server, "m")
    out = tmp_path / "dataset.json"
    first = run.run_suite(client, {"fake": task}, 1, 0, 1, out)
    task.load = lambda limit, seed: [{"id": "f/alpha", "w": "changed"}]
    Handler.seen = []
    second = run.run_suite(client, {"fake": task}, 1, 0, 1, out)
    assert Handler.seen == ["changed"]
    assert (
        first["config"]["tasks"]["fake"]["items_sha256"]
        != second["config"]["tasks"]["fake"]["items_sha256"]
    )


@pytest.mark.parametrize("failure", ["dataset", "request", "missing_task"])
def test_teacher_forced_failures_return_nonzero(tmp_path, monkeypatch, failure):
    task = fake_task()
    ref = {
        "status": "complete",
        "config": {"seed": 0, "requested_tasks": ["gsm8k"]},
        "results": {
            "gsm8k": {
                "f/alpha": {"ok": True, "completion": "alpha"},
                "f/beta": {"ok": False, "completion": "beta"},
            }
        },
    }
    if failure == "dataset":

        def missing(limit, seed):
            raise DatasetMissing("fixture dataset missing")

        task.load = missing

    def logprobs(client, messages, completion):
        if failure == "request" and completion == "beta":
            raise ConnectionError("fixture server unavailable")
        return [-1.0, -2.0]

    monkeypatch.setattr(tfkld.registry, "get", lambda name: task)
    monkeypatch.setattr(tfkld, "token_logprobs", logprobs)
    source, out = tmp_path / "ref.json", tmp_path / "lp.json"
    source.write_text(json.dumps(ref))
    tasks = "gsm8k,toolcall" if failure == "missing_task" else "gsm8k"
    assert (
        tfkld.main(
            [
                "score",
                "--url",
                "http://unused",
                "--model",
                "m",
                "--tasks",
                tasks,
                "--ref-results",
                str(source),
                "--out",
                str(out),
            ]
        )
        == 1
    )
    report = json.loads(out.read_text())
    assert report["status"] == "incomplete"
    if failure == "request":
        assert report["results"]["gsm8k"]["f/alpha"] == [-1.0, -2.0]
        assert "error" in report["results"]["gsm8k"]["f/beta"]
    elif failure == "missing_task":
        assert report["completion"]["tasks"]["toolcall"]["status"] == "load_error"
    else:
        assert (
            "fixture dataset missing" in report["completion"]["tasks"]["gsm8k"]["error"]
        )


def test_incomplete_prompt_logprobs_are_not_scored(monkeypatch):
    client = Client("http://unused", "m")
    monkeypatch.setattr(
        client,
        "post",
        lambda path, body: {
            "choices": [{"prompt_logprobs": [None]}],
        },
    )
    with pytest.raises(ValueError, match="incomplete prompt_logprobs"):
        client.prompt_logprobs([1, 2])


@pytest.fixture
def wrapper_env(tmp_path):
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "stage = 'bench' if Path(sys.argv[1]).name == 'run.py' else 'lp'\n"
        "code = int(os.environ.get(stage.upper() + '_RC', '0'))\n"
        "out = Path(os.environ['OUT'])\n"
        "(out / (stage + '.json')).write_text(json.dumps({\n"
        "    'status': 'incomplete' if code else 'complete', 'partial': True}))\n"
        "raise SystemExit(code)\n"
    )
    fake_python.chmod(0o755)
    return os.environ | {
        "PY": str(fake_python),
        "OUT": str(tmp_path),
        "PORT": "1",
        "SERVED": "fixture",
        "BENCH_ATTEMPT": "fixture-attempt",
        "TF_KLD": "1",
        "BENCH_RC": "0",
        "LP_RC": "0",
    }


@pytest.mark.parametrize(("bench_rc", "lp_rc"), [(7, 0), (0, 9), (0, 0)])
def test_hook_propagates_stage_exit_codes(tmp_path, wrapper_env, bench_rc, lp_rc):
    wrapper_env |= {"BENCH_RC": str(bench_rc), "LP_RC": str(lp_rc)}
    hook = Path(run.__file__).with_name("hook.sh")
    proc = subprocess.run(
        ["bash", str(hook)], env=wrapper_env, capture_output=True, text=True
    )
    expected = bench_rc or lp_rc
    assert proc.returncode == expected
    assert proc.stdout.splitlines()[-1] == ("FAIL" if expected else "PASS")
    assert (tmp_path / "bench.exit").read_text().split()[:2] == [
        "fixture-attempt",
        str(expected),
    ]
    assert (tmp_path / "lp.json").exists() == (bench_rc == 0)


@pytest.mark.parametrize(
    ("bench_rc", "lp_rc", "serve_rc", "run_hook", "expected"),
    [
        (7, 0, 0, True, 7),
        (0, 9, 0, True, 9),
        (0, 0, 11, True, 11),
        (0, 0, 0, False, 1),
        (0, 0, 0, True, 0),
    ],
)
def test_serve_wrapper_preserves_failure_even_if_external_driver_masks_it(
    tmp_path,
    wrapper_env,
    bench_rc,
    lp_rc,
    serve_rc,
    run_hook,
    expected,
):
    serve = tmp_path / "serve.sh"
    serve.write_text(
        "#!/usr/bin/env bash\n"
        'mkdir -p "$OUT"\n'
        "export SERVED=fixture\n"
        'if [[ $RUN_HOOK == 1 ]]; then bash "$LOAD_HOOK" || true; fi\n'
        'exit "$SERVE_RC"\n'
    )
    root = tmp_path / "results"
    wrapper_env |= {
        "SERVE_VALIDATE": str(serve),
        "GPU": "0",
        "ROOT": str(root),
        "BENCH_RC": str(bench_rc),
        "LP_RC": str(lp_rc),
        "SERVE_RC": str(serve_rc),
        "RUN_HOOK": "1" if run_hook else "0",
    }
    # A prior successful marker must not certify a new invocation that never ran.
    stale = root / "serve-first"
    stale.mkdir(parents=True)
    (stale / "bench.exit").write_text("old-attempt 0\n")
    wrapper = Path(run.__file__).with_name("serve_run.sh")
    proc = subprocess.run(
        ["bash", str(wrapper), "first=fixture", "second=fixture"],
        env=wrapper_env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == expected, proc.stderr
    if run_hook:
        partial = json.loads((root / "first.json").read_text())
        assert partial["status"] == ("incomplete" if bench_rc else "complete")
        assert partial["partial"]
        assert (
            root / "second.json"
        ).exists()  # a failed first arm does not stop later arms
        if lp_rc:
            assert (
                json.loads((root / "first.lp.json").read_text())["status"]
                == "incomplete"
            )
    else:
        assert not (root / "first.json").exists()


def test_teacher_forcing_does_not_claim_success_for_unknown_reference_completeness(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(tfkld.registry, "get", lambda name: fake_task())
    monkeypatch.setattr(tfkld, "token_logprobs", lambda *args: [-1.0])
    ref = {
        "config": {"seed": 0},
        "results": {"gsm8k": {"f/alpha": {"ok": True, "completion": "alpha"}}},
    }
    report = tfkld.score(
        Client("http://unused", "m"), ref, ["gsm8k"], 1, 1, tmp_path / "lp.json"
    )
    assert report["status"] == "incomplete"
    assert report["results"]["gsm8k"]["f/alpha"] == [-1.0]
    assert "not explicitly complete" in report["completion"]["errors"][0]
