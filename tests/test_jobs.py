# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Real subprocess/SQLite jobs, local CPU weights, and kernel-owned FIFO barriers.

No injected dispatch functions: blocked jobs execute the real checkpoint validator
against a FIFO config.json, or pause at a real run/receipt publication boundary.
This makes cancellation, resource exclusion and crashes deterministic without
sleeps standing in for successful numerical work.
"""

import errno
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

from hi2.jobs import (
    HEARTBEAT_STALE,
    MAX_EVENT_BYTES,
    MAX_EVENTS,
    TERMINAL,
    IdempotencyConflict,
    JobStore,
    _owned,
)

REPO = Path(__file__).resolve().parents[1]
SUBMIT = """
import json, sys
from pathlib import Path
from hi2.jobs import JobStore
root, operation, request, key = json.loads(sys.argv[1])
print(json.dumps(JobStore(Path(root)).submit(operation, request, key)), flush=True)
"""

# Launch the production supervisor and worker with timing faults only; numerical
# execution, validation, atomic publication, signals and ownership stay real.
FAULT_SUPERVISOR = """
import sys
from hi2 import job_worker
worker_script, boundary, fifo = sys.argv[1:4]
del sys.argv[1:4]
command = job_worker._command
def faulted_command(*args):
    return [
        sys.executable, "-c", worker_script, boundary, fifo, *command(*args)[3:]
    ]
job_worker._command = faulted_command
raise SystemExit(job_worker.main())
"""
FAULT_WORKER = """
import sys
from pathlib import Path
from hi2 import job_worker, runtime
boundary, fifo = sys.argv[1:3]
del sys.argv[1:3]

if boundary == "log_rollover":
    from hi2 import jobs
    jobs.MAX_EVENTS = 4

def pause():
    with Path(fifo).open("rb", buffering=0) as stream:
        stream.read(1)

read_run = runtime.read_run
def read_after_dispatch(workdir):
    manifest = read_run(workdir)
    if (
        boundary == "dispatch"
        and sys._getframe(1).f_code is job_worker._dispatch.__code__
    ):
        pause()
    return manifest
runtime.read_run = read_after_dispatch

atomic_json = job_worker.atomic_json
def write_receipt(path, value):
    receipt = path.name == "receipt.json" and value.get("state") == "succeeded"
    if receipt and boundary in {"before_receipt", "log_rollover"}:
        pause()
    atomic_json(path, value)
    if (
        receipt and boundary == "receipt"
        or path.name == "receipt.json"
        and boundary == f"{value.get('state')}_receipt"
    ):
        pause()
job_worker.atomic_json = write_receipt

phase = runtime.RunSession.phase
def publish_phase(self, name, status, **details):
    phase(self, name, status, **details)
    if (
        boundary in {"precommit", "interrupted_receipt"}
        and (name, status) == ("validation", "running")
        or boundary == "before_binding" and name == "run"
        and status in {"running", "reused"}
    ):
        pause()
runtime.RunSession.phase = publish_phase
raise SystemExit(job_worker.main())
"""


def _row(store, job_id):
    with sqlite3.connect(store.database) as db:
        db.row_factory = sqlite3.Row
        return dict(
            db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        )


def _sql(store, statement, args=()):
    with sqlite3.connect(store.database) as db:
        return db.execute(statement, args).fetchall()


def _wait(store, job_id, predicate, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = store.status(job_id)
        if predicate(status):
            return status
        if status["state"] in TERMINAL:
            pytest.fail(f"job reached unexpected terminal state: {status}")
        time.sleep(0.05)
    pytest.fail(f"job did not reach expected state: {store.status(job_id)}")


def _terminal(store, job_id, timeout=60):
    return _wait(store, job_id, lambda s: s["state"] in TERMINAL, timeout)


def _queued_after_lease_attempt(store, job_id):
    heartbeat = _row(store, job_id)["supervisor_heartbeat"]
    # Observe two real supervisor loop iterations, not just the initial submit
    # timestamp. At least one lease acquisition has actually been attempted.
    for _ in range(2):
        state = _wait(
            store,
            job_id,
            lambda s, after=heartbeat: s["ownership"]["supervisor_heartbeat"] > after,
        )
        heartbeat = state["ownership"]["supervisor_heartbeat"]
    assert state["state"] == "queued"
    assert _row(store, job_id)["worker_started_at"] is None


def _fifo(root):
    root.mkdir(parents=True)
    path = root / "config.json"
    os.mkfifo(path)
    return path


def _reader(store, job_id, fifo):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            return os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno != errno.ENXIO:
                raise
        status = store.status(job_id)
        assert status["state"] not in TERMINAL, status
        time.sleep(0.05)
    pytest.fail(f"real validator never opened its FIFO: {store.status(job_id)}")


def _submit_process(store, operation, request, key):
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            SUBMIT,
            json.dumps([str(store.root), operation, request, key]),
        ],
        cwd=REPO,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.fixture(autouse=True)
def offline_cpu(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")


@pytest.fixture
def store(tmp_path):
    registry = JobStore(tmp_path / "jobs")
    yield registry
    active = _sql(
        registry,
        "SELECT job_id FROM jobs WHERE state NOT IN ('succeeded', 'failed', 'cancelled', 'interrupted')",
    )
    for (job_id,) in active:
        registry.cancel(job_id)
    for (job_id,) in active:
        _terminal(registry, job_id, timeout=20)


@pytest.fixture
def barriers(store, tmp_path):
    writers = []

    def start(name, *, operation="validate", request=None):
        path = _fifo(tmp_path / name)
        if request is None:
            request = {"path": str(path.parent)}
        else:
            request = {**request, "dst": str(path.parent)}
        job = store.submit(operation, request, name)
        fd = _reader(store, job["job_id"], path)
        writers.append(fd)
        return job["job_id"], path

    yield start
    for fd in writers:
        os.close(fd)


@pytest.fixture
def faulted_job(store, tmp_path, monkeypatch):
    import hi2.jobs as jobs

    writers = []
    started = []

    def start(boundary, operation, request, key="commit-boundary"):
        fifo = _fifo(tmp_path / f"{key}-barrier")
        command = jobs._command

        def launch(*args):
            return [
                sys.executable,
                "-c",
                FAULT_SUPERVISOR,
                FAULT_WORKER,
                boundary,
                str(fifo),
                *command(*args)[3:],
            ]

        with monkeypatch.context() as patch:
            patch.setattr(jobs, "_command", launch)
            job_id = store.submit(operation, request, key)["job_id"]
        started.append(job_id)
        writers.append(_reader(store, job_id, fifo))
        return job_id

    yield start
    for job_id in started:
        if _owned(store.root / job_id / "worker.lock"):
            with suppress(ProcessLookupError):
                os.kill(_row(store, job_id)["worker_pid"], signal.SIGCONT)
    for fd in writers:
        os.close(fd)


@pytest.fixture
def source(tmp_path):
    import torch
    from safetensors.torch import save_file

    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "hidden_size": 64,
                "num_hidden_layers": 1,
            }
        )
    )
    save_file(
        {
            "model.layers.0.mlp.up_proj.weight": torch.randn(
                64, 64, generator=torch.Generator().manual_seed(17)
            )
        },
        root / "model.safetensors",
    )
    return root


def _recipe(source, root, **overrides):
    config = {
        "src": str(source),
        "dst": str(root / "output"),
        "workdir": str(root / "work"),
        "devices": ["cpu"],
        "cfg": {
            "effort": "fast",
            "rotation": "off",
            "rounding": "nearest",
            "candidates": [[2, 32], [4, 32]],
        },
        "moe_layout": "per-expert",
    }
    config.update(overrides)
    return {"schema_version": 1, "config": config}


def _assert_committed_quantization(store, job_id, recipe, final):
    from hi2.checkpoint import validate_checkpoint

    work = Path(recipe["config"]["workdir"])
    output = Path(recipe["config"]["dst"])
    manifest = json.loads((work / "run.json").read_text())
    published = json.loads((output / "hi2_run.json").read_text())
    assert final["state"] == "succeeded", final
    assert final["error"] is None
    assert manifest["state"] == published["state"] == "succeeded"
    assert manifest["run_id"] == published["run_id"]
    assert manifest["attempt_id"] == published["attempt_id"]
    assert final["result"] == manifest["result"] == published["result"]
    assert final["result"]["feasible"] is True
    assert final["result"]["avg_bpw"] > 0
    validation = validate_checkpoint(output, log=lambda _: None)
    assert validation["checked"] == validation["expected"] == 1
    assert validation["coverage"] == 1
    assert not final["ownership"]["worker"]
    assert _sql(store, "SELECT * FROM leases WHERE job_id = ?", (job_id,)) == []


@pytest.mark.parametrize("reused", [False, True])
@pytest.mark.parametrize("interruption", ["sigint", "sigkill"])
def test_committed_quantization_wins_dispatch_interruption(
    store, source, tmp_path, faulted_job, reused, interruption
):
    recipe = _recipe(source, tmp_path / "committed")
    if reused:
        previous = store.submit("quantize", recipe, "previous-success")["job_id"]
        assert _terminal(store, previous)["state"] == "succeeded"
        previous_manifest = json.loads(
            (Path(recipe["config"]["workdir"]) / "run.json").read_text()
        )
    job_id = faulted_job("dispatch", "quantize", recipe)
    directory = store.root / job_id
    work = Path(recipe["config"]["workdir"])
    manifest = json.loads((work / "run.json").read_text())
    assert manifest["state"] == "succeeded"
    if reused:
        assert manifest["run_id"] == previous_manifest["run_id"]
        assert manifest["attempt_id"] != previous_manifest["attempt_id"]
    assert not (directory / "receipt.json").exists()
    row = _row(store, job_id)

    # A committed artifact does not free the live worker's leases. Leave the
    # cancellation pending on a stopped process so this observation cannot race.
    os.kill(row["worker_pid"], signal.SIGSTOP)
    offline = source.with_name("committed-source-offline")
    source.rename(offline)
    try:
        ack = store.cancel(job_id)
        assert ack["state"] == "cancelling" and ack["ownership"]["worker"]
        assert _sql(store, "SELECT resource FROM leases WHERE job_id = ?", (job_id,))
        _wait(
            store,
            job_id,
            lambda status: any(
                event.get("phase") == "cancellation" and event["status"] == "sigint"
                for event in status["events"]
            ),
        )
        os.kill(
            row["worker_pid"],
            signal.SIGKILL if interruption == "sigkill" else signal.SIGCONT,
        )
        final = _terminal(store, job_id)
        # Recovery must not need source weights/assets to recompute an identity.
        assert final["state"] == "succeeded", final
    finally:
        offline.rename(source)
    if interruption == "sigint":
        # Interpreter teardown may outlast the cancellation grace period; the
        # durable receipt, not a particular process exit code, proves this path.
        receipt = json.loads((directory / "receipt.json").read_text())
        assert receipt["state"] == "interrupted"
        assert receipt["error"]["type"] == "KeyboardInterrupt"
    else:
        assert final["returncode"] == -signal.SIGKILL
        assert not (directory / "receipt.json").exists()
    _assert_committed_quantization(store, job_id, recipe, final)
    binding = json.loads((directory / "run-binding.json").read_text())
    assert binding == {
        "schema_version": 1,
        "job_id": job_id,
        "token": row["supervisor_token"],
        "workdir": str(work.resolve()),
        "run_id": manifest["run_id"],
        "attempt_id": manifest["attempt_id"],
    }
    event = next(
        event
        for event in final["events"]
        if event.get("phase") == "run"
        and event.get("status") == ("reused" if reused else "running")
    )
    assert event["run_id"] == binding["run_id"]
    assert event["attempt_id"] == binding["attempt_id"]
    assert event["workdir"] == binding["workdir"]


def test_success_receipt_survives_teardown_termination(
    store, source, tmp_path, faulted_job
):
    recipe = _recipe(source, tmp_path / "teardown")
    job_id = faulted_job("receipt", "quantize", recipe)
    directory = store.root / job_id
    receipt = json.loads((directory / "receipt.json").read_text())
    assert receipt["state"] == "succeeded"
    # Isolate receipt recovery: no run-binding fallback can rescue this result.
    (directory / "run-binding.json").unlink(missing_ok=True)
    ack = store.cancel(job_id)
    assert ack["state"] == "cancelling" and ack["ownership"]["worker"]
    assert _sql(store, "SELECT resource FROM leases WHERE job_id = ?", (job_id,))
    os.kill(_row(store, job_id)["worker_pid"], signal.SIGTERM)
    final = _terminal(store, job_id)
    assert final["returncode"] == -signal.SIGTERM
    assert final["result"] == receipt["result"]
    _assert_committed_quantization(store, job_id, recipe, final)


@pytest.mark.parametrize("boundary", ["dispatch", "receipt"])
@pytest.mark.parametrize("via_leases", [False, True])
def test_orphaned_commit_reconciles_only_after_worker_exit(
    store, source, tmp_path, faulted_job, boundary, via_leases
):
    recipe = _recipe(source, tmp_path / "orphaned-commit")
    job_id = faulted_job(boundary, "quantize", recipe)
    if boundary == "receipt":
        (store.root / job_id / "run-binding.json").unlink(missing_ok=True)
    row = _row(store, job_id)
    os.kill(row["worker_pid"], signal.SIGSTOP)
    os.kill(row["supervisor_pid"], signal.SIGKILL)
    status = _wait(store, job_id, lambda value: not value["ownership"]["supervisor"])
    assert status["state"] == "running"
    assert status["ownership"]["orphaned_worker"]
    assert _sql(store, "SELECT resource FROM leases WHERE job_id = ?", (job_id,))
    reconnected = JobStore(store.root)
    assert reconnected.status(job_id)["state"] == "running"
    if via_leases:
        successor = reconnected.submit(
            "validate", {"path": recipe["config"]["dst"]}, "after-orphaned-commit"
        )["job_id"]
        _queued_after_lease_attempt(reconnected, successor)
    os.kill(row["worker_pid"], signal.SIGKILL)
    if via_leases:
        # No status call on the orphan: the queued supervisor must reconcile its
        # dead lease holder using exactly the same commit evidence as reconnect.
        validation = _terminal(reconnected, successor)
        assert validation["state"] == "succeeded"
        assert validation["result"]["checked"] == 1
        assert _row(store, job_id)["state"] == "succeeded"
    final = _terminal(reconnected, job_id)
    # The missing parent cannot supply a waitpid returncode; do not invent zero.
    assert final["returncode"] is None
    _assert_committed_quantization(reconnected, job_id, recipe, final)


def test_precommit_cancellation_preserves_resumable_quantization(
    store, source, tmp_path, faulted_job
):
    recipe = _recipe(source, tmp_path / "precommit")
    job_id = faulted_job("precommit", "quantize", recipe)
    work = Path(recipe["config"]["workdir"])
    output = Path(recipe["config"]["dst"])
    running = json.loads((work / "run.json").read_text())
    assert running["state"] == "running"
    assert {"phase": "encode", "status": "completed"} in running["phases"]
    assert (work / "shards_done").exists()
    assert (store.root / job_id / "run-binding.json").is_file()
    assert not (output / "hi2_run.json").exists()
    store.cancel(job_id)
    final = _terminal(store, job_id)
    assert final["state"] == "cancelled" and final["result"] is None
    assert final["returncode"] == 1
    assert not final["ownership"]["worker"]
    assert _sql(store, "SELECT * FROM leases WHERE job_id = ?", (job_id,)) == []
    assert not (output / "hi2_run.json").exists()
    interrupted = json.loads((work / "run.json").read_text())
    assert interrupted["state"] == "interrupted"
    assert interrupted["run_id"] == running["run_id"]
    assert interrupted["attempt_id"] == running["attempt_id"]
    assert (work / "shards_done").exists()
    resumed = store.submit("quantize", recipe, "resume-precommit")["job_id"]
    assert resumed != job_id
    _assert_committed_quantization(store, resumed, recipe, _terminal(store, resumed))
    assert json.loads((work / "run.json").read_text())["run_id"] == running["run_id"]
    assert (
        json.loads((work / "run.json").read_text())["attempt_id"]
        != running["attempt_id"]
    )


@pytest.mark.parametrize("via_leases", [False, True])
def test_precommit_orphan_cannot_adopt_later_same_recipe_foreground_success(
    store, source, tmp_path, faulted_job, via_leases
):
    from hi2.checkpoint import validate_checkpoint
    from hi2.service import Progress, config_from_dict, execute_run

    recipe = _recipe(source, tmp_path / "later-foreground")
    job_id = faulted_job("precommit", "quantize", recipe)
    directory = store.root / job_id
    work = Path(recipe["config"]["workdir"])
    output = Path(recipe["config"]["dst"])
    original = json.loads((work / "run.json").read_text())
    assert original["state"] == "running"
    assert not (output / "hi2_run.json").exists()
    row = _row(store, job_id)
    os.kill(row["worker_pid"], signal.SIGSTOP)
    os.kill(row["supervisor_pid"], signal.SIGKILL)
    os.kill(row["worker_pid"], signal.SIGKILL)
    # Do not call status: recovery must first see the later foreground commit,
    # not preemptively classify the orphan before mistaken adoption is possible.
    deadline = time.monotonic() + 10
    while any(_owned(directory / f"{role}.lock") for role in ("supervisor", "worker")):
        assert time.monotonic() < deadline, "killed processes retained ownership"
        time.sleep(0.05)
    assert _row(store, job_id)["state"] == "running"
    assert not (directory / "receipt.json").exists()

    result = execute_run(config_from_dict(recipe), Progress(quiet=True))
    later = json.loads((work / "run.json").read_text())
    published = json.loads((output / "hi2_run.json").read_text())
    assert later["state"] == published["state"] == "succeeded"
    assert later["run_id"] == original["run_id"]
    assert result["feasible"] and result["avg_bpw"] > 0
    assert validate_checkpoint(output, log=lambda _: None)["coverage"] == 1
    reconnected = JobStore(store.root)
    if via_leases:
        successor = reconnected.submit(
            "validate", {"path": str(output)}, "after-foreground"
        )["job_id"]
        validation = _terminal(reconnected, successor)
        assert validation["state"] == "succeeded"
        assert validation["result"]["checked"] == 1
        assert _row(store, job_id)["state"] == "interrupted"
    final = _terminal(reconnected, job_id)
    assert final["state"] == "interrupted" and final["result"] is None
    assert final["error"]["type"] == "OwnershipLost"
    assert final["returncode"] is None
    assert _sql(store, "SELECT * FROM leases WHERE job_id = ?", (job_id,)) == []
    binding = json.loads((directory / "run-binding.json").read_text())
    assert binding["attempt_id"] == original["attempt_id"]
    assert later["attempt_id"] == published["attempt_id"] != original["attempt_id"]


@pytest.mark.parametrize("recovery", ["matched", "cancelled", "stale"])
def test_orphaned_interrupted_receipt_preserves_worker_error(
    store, source, tmp_path, faulted_job, recovery
):
    from hi2.runtime import atomic_json

    recipe = _recipe(source, tmp_path / "interrupted-receipt")
    job_id = faulted_job("interrupted_receipt", "quantize", recipe)
    row = _row(store, job_id)
    receipt_path = store.root / job_id / "receipt.json"
    os.kill(row["worker_pid"], signal.SIGINT)
    _wait(store, job_id, lambda _: receipt_path.exists())
    receipt = json.loads(receipt_path.read_text())
    assert receipt["state"] == "interrupted"
    assert receipt["error"]["type"] == "KeyboardInterrupt"
    manifest = json.loads((Path(recipe["config"]["workdir"]) / "run.json").read_text())
    assert manifest["state"] == "interrupted"
    assert not (Path(recipe["config"]["dst"]) / "hi2_run.json").exists()
    if recovery == "stale":
        atomic_json(receipt_path, {**receipt, "token": "previous-supervisor"})
    os.kill(row["worker_pid"], signal.SIGSTOP)
    os.kill(row["supervisor_pid"], signal.SIGKILL)
    owned = _wait(store, job_id, lambda s: not s["ownership"]["supervisor"])
    assert owned["state"] == "running" and owned["ownership"]["orphaned_worker"]
    if recovery == "cancelled":
        assert store.cancel(job_id)["state"] == "cancelling"
    os.kill(row["worker_pid"], signal.SIGKILL)
    final = _terminal(JobStore(store.root), job_id)
    assert final["state"] == ("cancelled" if recovery == "cancelled" else "interrupted")
    assert final["returncode"] is None and final["result"] is None
    if recovery == "cancelled":
        assert final["error"]["type"] == "Cancelled"
        assert final["error"]["worker_error"] == receipt["error"]
    elif recovery == "stale":
        assert final["error"]["type"] == "OwnershipLost"
    else:
        assert final["error"] == receipt["error"]
    assert _sql(store, "SELECT * FROM leases WHERE job_id = ?", (job_id,)) == []


@pytest.mark.parametrize("stale_binding", [False, True])
def test_previous_success_cannot_complete_cancelled_reuse(
    store, source, tmp_path, faulted_job, stale_binding
):
    recipe = _recipe(source, tmp_path / "reuse-cancelled")
    previous = store.submit("quantize", recipe, "previous-success")["job_id"]
    assert _terminal(store, previous)["state"] == "succeeded"
    work = Path(recipe["config"]["workdir"])
    marker = Path(recipe["config"]["dst"]) / "hi2_run.json"
    published = marker.read_bytes()
    job_id = faulted_job("before_binding", "quantize", recipe)
    # RunSession must invalidate the old run manifest before a new binding can
    # be persisted, even though the previous publication still exists.
    assert json.loads((work / "run.json").read_text())["state"] == "running"
    binding = store.root / job_id / "run-binding.json"
    assert not binding.exists()
    if stale_binding:
        binding.write_bytes((store.root / previous / "run-binding.json").read_bytes())
    store.cancel(job_id)
    final = _terminal(store, job_id)
    assert final["state"] == "cancelled" and final["result"] is None
    assert json.loads((work / "run.json").read_text())["state"] == "interrupted"
    assert marker.read_bytes() == published
    assert not final["ownership"]["worker"]
    assert _sql(store, "SELECT * FROM leases WHERE job_id = ?", (job_id,)) == []


@pytest.mark.parametrize(
    "mismatch",
    [
        "missing",
        "schema_version",
        "job_id",
        "token",
        "workdir",
        "run_id",
        "attempt_id",
        "missing_attempt_id",
    ],
)
def test_commit_without_matching_generation_binding_is_not_success(
    store, source, tmp_path, faulted_job, mismatch
):
    from hi2.runtime import atomic_json

    recipe = _recipe(source, tmp_path / "unbound")
    unrelated = None
    if mismatch == "workdir":
        other_recipe = _recipe(source, tmp_path / "unrelated")
        other = store.submit("quantize", other_recipe, "unrelated-success")["job_id"]
        assert _terminal(store, other)["state"] == "succeeded"
        unrelated = json.loads((store.root / other / "run-binding.json").read_text())
    job_id = faulted_job("dispatch", "quantize", recipe)
    path = store.root / job_id / "run-binding.json"
    binding = json.loads(path.read_text())
    if mismatch == "missing":
        path.unlink()
    else:
        if mismatch == "workdir":
            binding["workdir"] = unrelated["workdir"]
            binding["run_id"] = unrelated["run_id"]
            binding["attempt_id"] = unrelated["attempt_id"]
        elif mismatch == "missing_attempt_id":
            binding.pop("attempt_id")
        else:
            binding[mismatch] = {
                "schema_version": 999,
                "job_id": "job_" + "0" * 32,
                "token": "stale-supervisor-generation",
                "run_id": "0" * 64,
                "attempt_id": "0" * 32,
            }[mismatch]
        atomic_json(path, binding)
    os.kill(_row(store, job_id)["worker_pid"], signal.SIGKILL)
    final = _terminal(store, job_id)
    assert final["state"] == "interrupted" and final["result"] is None
    assert final["returncode"] == -signal.SIGKILL
    assert not (store.root / job_id / "receipt.json").exists()
    assert not final["ownership"]["worker"]
    assert _sql(store, "SELECT * FROM leases WHERE job_id = ?", (job_id,)) == []
    # There really is a current committed artifact; its existence alone is not
    # permission to attribute its quantization result to this job generation.
    work = Path(recipe["config"]["workdir"])
    output = Path(recipe["config"]["dst"])
    assert json.loads((work / "run.json").read_text())["state"] == "succeeded"
    assert (output / "hi2_run.json").is_file()


@pytest.mark.parametrize("operation", ["validate", "portable", "quality"])
def test_other_operations_cannot_inherit_quantization_commit(
    store, quantized_model, tmp_path, faulted_job, operation
):
    from hi2.runtime import atomic_json

    source, output = quantized_model
    manifest = json.loads((output / "hi2_run.json").read_text())
    text = tmp_path / "held-out.txt"
    text.write_text("a b c d e f g h i j k l m n o " * 4)
    request = {
        "validate": {"path": str(output)},
        "portable": {
            "src": str(output),
            "out": str(tmp_path / "portable"),
            "assets_from": str(source),
        },
        "quality": {
            "src": str(source),
            "dst": str(output),
            "text": str(text),
            "device": "cpu",
            "dtype": "float32",
            "samples": 1,
            "seq_len": 8,
        },
    }[operation]
    job_id = faulted_job("before_receipt", operation, request)
    directory = store.root / job_id
    assert not (directory / "receipt.json").exists()
    # Even a binding to a genuine success with matching job/token must not turn
    # another operation's interrupted handoff into a quantization result.
    atomic_json(
        directory / "run-binding.json",
        {
            "schema_version": 1,
            "job_id": job_id,
            "token": _row(store, job_id)["supervisor_token"],
            "workdir": manifest["config"]["workdir"],
            "run_id": manifest["run_id"],
            "attempt_id": manifest["attempt_id"],
        },
    )
    os.kill(_row(store, job_id)["worker_pid"], signal.SIGKILL)
    final = _terminal(store, job_id)
    assert final["state"] == "interrupted" and final["result"] is None
    assert final["returncode"] == -signal.SIGKILL
    assert not final["ownership"]["worker"]
    assert _sql(store, "SELECT * FROM leases WHERE job_id = ?", (job_id,)) == []


def test_registry_discovery_has_no_compute_imports(tmp_path):
    script = """
import sys
from pathlib import Path
from hi2.jobs import JobStore
JobStore(Path(sys.argv[1]))
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "registry")],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr


def test_quantization_survives_submitter_exit_and_closed_stdio(
    store, source, tmp_path, barriers
):
    text = tmp_path / "eval.txt"
    text.write_text("local held-out text")
    blocker, _ = barriers(
        "cpu-owner",
        operation="quality",
        request={
            "src": str(source),
            "text": str(text),
            "device": "cpu",
        },
    )
    recipe = _recipe(source, tmp_path / "quant")
    proc = _submit_process(store, "quantize", recipe, "detached")
    stdout, stderr = proc.communicate(timeout=60)
    assert proc.returncode == 0, stderr
    ack = json.loads(stdout)  # Worker stdout cannot leak into the submitter's pipe.
    job_id = ack["job_id"]
    assert ack["schema_version"] == 1 and ack["state"] == "queued"
    persisted = _row(store, job_id)
    assert persisted["worker_started_at"] is None
    assert persisted["idempotency_key"] == "detached"
    assert json.loads(persisted["request"])["config"]["cfg"]["rounding"] == "nearest"
    store.cancel(blocker)
    assert _terminal(store, blocker)["state"] == "cancelled"
    # Do not call status until the supervisor has completed autonomously: status
    # can recover an unstarted queue entry and would otherwise mask lost detachment.
    deadline = time.monotonic() + 60
    while _row(store, job_id)["state"] not in TERMINAL and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _row(store, job_id)["state"] == "succeeded"
    reconnected = JobStore(store.root)
    result = reconnected.status(job_id)
    assert result["state"] == "succeeded", result
    assert result["returncode"] == 0 and result["error"] is None
    assert result["result"]["feasible"] is True
    assert result["result"]["avg_bpw"] > 0
    work = Path(recipe["config"]["workdir"])
    output = Path(recipe["config"]["dst"])
    manifest = json.loads((work / "run.json").read_text())
    assert manifest["state"] == "succeeded"
    assert (
        json.loads((output / "hi2_run.json").read_text())["run_id"]
        == manifest["run_id"]
    )
    from hi2.checkpoint import validate_checkpoint

    validation = validate_checkpoint(output, log=lambda _: None)
    assert validation["checked"] == validation["expected"] == 1
    assert validation["coverage"] == 1
    assert not result["ownership"]["worker"]
    assert _sql(store, "SELECT * FROM leases WHERE job_id = ?", (job_id,)) == []
    assert store.submit("quantize", recipe, "detached")["job_id"] == job_id
    offline = source.with_name("source-offline")
    source.rename(offline)
    try:
        assert (
            JobStore(store.root).submit("quantize", recipe, "detached")["job_id"]
            == job_id
        )
    finally:
        offline.rename(source)


def test_concurrent_identical_submissions_start_exactly_one_worker(store, tmp_path):
    fifo = _fifo(tmp_path / "shared")
    request = {"path": str(fifo.parent)}
    processes = [
        _submit_process(store, "validate", request, "one-key") for _ in range(6)
    ]
    try:
        responses = []
        for proc in processes:
            out, err = proc.communicate(timeout=60)
            assert proc.returncode == 0, err
            responses.append(json.loads(out))
        assert len({item["job_id"] for item in responses}) == 1
        job_id = responses[0]["job_id"]
        writer = _reader(store, job_id, fifo)
        try:
            assert _sql(store, "SELECT count(*) FROM jobs") == [(1,)]
            events = store.status(job_id)["events"]
            assert (
                sum(
                    e.get("phase") == "worker" and e.get("status") == "started"
                    for e in events
                )
                == 1
            )
            assert (
                sum(
                    e.get("phase") == "job" and e.get("status") == "running"
                    for e in events
                )
                == 1
            )
            other_spelling = {"path": str(fifo.parent / ".")}
            assert (
                store.submit("validate", other_spelling, "one-key")["job_id"] == job_id
            )
            with pytest.raises(
                IdempotencyConflict, match="different operation/request"
            ):
                store.submit(
                    "validate", {"path": str(tmp_path / "different")}, "one-key"
                )
            assert _sql(store, "SELECT count(*) FROM jobs") == [(1,)]
        finally:
            store.cancel(job_id)
            _terminal(store, job_id)
            os.close(writer)
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)


def test_conflicting_key_race_has_one_durable_winner(store, tmp_path):
    fifos = [_fifo(tmp_path / name) for name in ("left", "right")]
    processes = [
        _submit_process(store, "validate", {"path": str(f.parent)}, "raced-key")
        for f in fifos
    ]
    try:
        outputs = [p.communicate(timeout=60) for p in processes]
        assert sorted(p.returncode for p in processes) == [0, 1]
        winner = next(i for i, p in enumerate(processes) if p.returncode == 0)
        loser = 1 - winner
        assert "IdempotencyConflict" in outputs[loser][1]
        job_id = json.loads(outputs[winner][0])["job_id"]
        writer = _reader(store, job_id, fifos[winner])
        try:
            assert _sql(store, "SELECT count(*) FROM jobs") == [(1,)]
            store.cancel(job_id)
            assert _terminal(store, job_id)["state"] == "cancelled"
        finally:
            os.close(writer)
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)


@pytest.mark.parametrize("resource", ["output", "workdir", "device"])
def test_conflicting_resources_queue_until_actual_worker_exit(
    store, source, tmp_path, barriers, resource
):
    overrides = {}
    if resource == "device":
        text = tmp_path / "eval.txt"
        text.write_text("held out")
        blocker, _ = barriers(
            "resource-owner",
            operation="quality",
            request={
                "src": str(source),
                "text": str(text),
                "device": "cpu",
            },
        )
    else:
        blocker, fifo = barriers("resource-owner")
        overrides["dst" if resource == "output" else "workdir"] = str(
            fifo.parent / "nested"
        )
    recipe = _recipe(source, tmp_path / "waiting", **overrides)
    waiting = store.submit("quantize", recipe, "waiting")["job_id"]
    assert store.status(waiting)["state"] == "queued"
    assert _row(store, waiting)["worker_started_at"] is None
    assert store.status(blocker)["ownership"]["worker"]
    # The queued supervisor is alive and has tried to acquire leases.
    _wait(store, waiting, lambda s: s["ownership"]["supervisor_heartbeat"] is not None)
    assert _row(store, waiting)["worker_started_at"] is None
    _queued_after_lease_attempt(store, waiting)
    store.cancel(blocker)
    assert _terminal(store, blocker)["state"] == "cancelled"
    done = _terminal(store, waiting)
    assert done["state"] == "succeeded", done
    assert (
        _row(store, waiting)["worker_started_at"] >= _row(store, blocker)["updated_at"]
    )


def test_independent_resources_execute_concurrently(store, barriers):
    first, _ = barriers("independent-a")
    second, _ = barriers("independent-b")
    assert first != second
    for job_id in (first, second):
        status = store.status(job_id)
        assert status["state"] == "running" and status["ownership"]["worker"]
        assert _sql(
            store, "SELECT count(*) FROM leases WHERE job_id = ?", (job_id,)
        ) == [(1,)]
    assert _row(store, first)["worker_pid"] != _row(store, second)["worker_pid"]
    for job_id in (first, second):
        store.cancel(job_id)
        assert _terminal(store, job_id)["state"] == "cancelled"


def test_queued_cancellation_never_starts_a_worker(store, barriers):
    blocker, fifo = barriers("blocked-queue")
    queued = store.submit(
        "validate", {"path": str(fifo.parent / "child")}, "queued-cancel"
    )["job_id"]
    assert store.status(queued)["state"] == "queued"
    ack = store.cancel(queued)
    assert ack["cancellation"] in {"requested", "cancelled"}
    final = _terminal(store, queued)
    assert final["state"] == "cancelled"
    assert _row(store, queued)["worker_started_at"] is None
    assert not (store.root / queued / "receipt.json").exists()
    assert store.status(blocker)["state"] == "running"


def test_cancel_ack_waits_for_noncooperative_process_and_escalates(store, barriers):
    job_id, _ = barriers("uncooperative")
    worker_pid = _row(store, job_id)["worker_pid"]
    os.kill(worker_pid, signal.SIGSTOP)
    try:
        ack = store.cancel(job_id)
        assert ack["state"] == "cancelling" and ack["cancellation"] == "requested"
        assert ack["ownership"]["worker"]
        assert _sql(
            store, "SELECT count(*) FROM leases WHERE job_id = ?", (job_id,)
        ) == [(1,)]
        final = _terminal(store, job_id, timeout=20)
        assert final["state"] == "cancelled"
        # Darwin may terminate a stopped process at SIGTERM; Linux can leave it
        # pending until SIGKILL. Both must retain ownership until actual exit.
        assert final["returncode"] in {-signal.SIGTERM, -signal.SIGKILL}
        assert final["result"] is None and not final["ownership"]["worker"]
        signals = [
            e["status"]
            for e in final["events"]
            if e.get("phase") in {"cancellation", "termination"}
        ]
        expected = ["sigint", "sigterm"]
        if final["returncode"] == -signal.SIGKILL:
            expected.append("sigkill")
        assert signals == expected
        assert _sql(store, "SELECT * FROM leases WHERE job_id = ?", (job_id,)) == []
        assert store.cancel(job_id)["cancellation"] == "cancelled"
    finally:
        if _owned(store.root / job_id / "worker.lock"):
            os.kill(worker_pid, signal.SIGCONT)


def test_registry_pids_and_stale_heartbeat_never_override_real_ownership(
    store, barriers
):
    job_id, _ = barriers("reused-pid")
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _sql(
            store,
            "UPDATE jobs SET supervisor_pid = ?, worker_pid = ?, supervisor_heartbeat = ?, worker_heartbeat = ? WHERE job_id = ?",
            (
                unrelated.pid,
                unrelated.pid,
                time.time() - HEARTBEAT_STALE * 2,
                0,
                job_id,
            ),
        )
        assert store.status(job_id)["state"] == "running"
        store.cancel(job_id)
        final = _terminal(store, job_id)
        assert final["state"] == "cancelled"
        assert not final["ownership"]["worker"]
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=10)


def test_worker_death_releases_only_after_witnessed_exit(store, barriers):
    job_id, fifo = barriers("dead-worker")
    os.kill(_row(store, job_id)["worker_pid"], signal.SIGKILL)
    final = _terminal(store, job_id)
    assert final["state"] == "interrupted" and final["result"] is None
    assert final["returncode"] == -signal.SIGKILL
    assert final["error"]["type"] == "WorkerExited"
    assert _sql(store, "SELECT * FROM leases WHERE job_id = ?", (job_id,)) == []
    successor = store.submit("validate", {"path": str(fifo.parent)}, "after-death")[
        "job_id"
    ]
    writer = _reader(store, successor, fifo)
    try:
        assert store.status(successor)["state"] == "running"
        store.cancel(successor)
        assert _terminal(store, successor)["state"] == "cancelled"
    finally:
        os.close(writer)


def test_dead_supervisor_does_not_free_a_stopped_matching_worker(store, barriers):
    job_id, fifo = barriers("orphaned-worker")
    owned = _row(store, job_id)
    os.kill(owned["worker_pid"], signal.SIGSTOP)
    try:
        os.kill(owned["supervisor_pid"], signal.SIGKILL)
        _wait(store, job_id, lambda s: not s["ownership"]["supervisor"])
        _sql(
            store,
            "UPDATE jobs SET supervisor_heartbeat = 0, worker_heartbeat = 0 WHERE job_id = ?",
            (job_id,),
        )
        status = store.status(job_id)
        assert status["state"] == "running"
        assert status["ownership"]["orphaned_worker"]
        assert status["ownership"]["heartbeat_stale"]
        assert _sql(
            store, "SELECT count(*) FROM leases WHERE job_id = ?", (job_id,)
        ) == [(1,)]
        next_id = store.submit(
            "validate", {"path": str(fifo.parent)}, "after-supervisor"
        )["job_id"]
        assert store.status(next_id)["state"] == "queued"
        assert _row(store, next_id)["worker_started_at"] is None
        _queued_after_lease_attempt(store, next_id)
        os.kill(owned["worker_pid"], signal.SIGCONT)
        lost = _terminal(store, job_id, timeout=20)
        assert lost["state"] == "interrupted"
        assert lost["result"] is None
        writer = _reader(store, next_id, fifo)
        try:
            assert store.status(next_id)["state"] == "running"
            store.cancel(next_id)
            assert _terminal(store, next_id)["state"] == "cancelled"
        finally:
            os.close(writer)
    finally:
        if _owned(store.root / job_id / "worker.lock"):
            os.kill(owned["worker_pid"], signal.SIGCONT)


def test_reconnection_restarts_only_an_unstarted_queue_entry(store, barriers):
    blocker, fifo = barriers("queue-reconnect")
    queued = store.submit("validate", {"path": str(fifo.parent)}, "reconnect")["job_id"]
    before = _row(store, queued)
    assert before["worker_started_at"] is None
    os.kill(before["supervisor_pid"], signal.SIGKILL)
    deadline = time.monotonic() + 10
    while (
        _owned(store.root / queued / "supervisor.lock") and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    assert not _owned(store.root / queued / "supervisor.lock")
    reconnected = JobStore(store.root)
    assert reconnected.status(queued)["state"] == "queued"
    assert _row(store, queued)["supervisor_pid"] != before["supervisor_pid"]
    assert _row(store, queued)["worker_started_at"] is None
    store.cancel(blocker)
    assert _terminal(store, blocker)["state"] == "cancelled"
    writer = _reader(reconnected, queued, fifo)
    try:
        reconnected.cancel(queued)
        assert _terminal(reconnected, queued)["state"] == "cancelled"
    finally:
        os.close(writer)


def test_event_retention_is_bounded_and_cursors_are_monotonic(store, barriers):
    job_id, _ = barriers("events")
    with store._transaction() as db:
        for index in range(MAX_EVENTS + 7):
            store._event(
                db, job_id, {"phase": "fixture", "status": "measured", "index": index}
            )
        store._event(db, job_id, {"message": "\x00" * (MAX_EVENT_BYTES * 2)})
    assert _sql(store, "SELECT count(*) FROM events WHERE job_id = ?", (job_id,)) == [
        (MAX_EVENTS,)
    ]
    largest = _sql(
        store,
        "SELECT max(length(CAST(payload AS BLOB))) FROM events WHERE job_id = ?",
        (job_id,),
    )[0][0]
    assert largest <= MAX_EVENT_BYTES
    page = store.status(job_id, limit=7)
    assert page["events_truncated"] and len(page["events"]) == 7
    first = page["oldest_cursor"]
    cursors = [e["cursor"] for e in page["events"]]
    while page["next_cursor"] < page["last_cursor"]:
        page = store.status(job_id, cursor=page["next_cursor"], limit=100)
        assert len(page["events"]) <= 100
        cursors.extend(e["cursor"] for e in page["events"])
    assert cursors == list(range(first, first + MAX_EVENTS))
    assert page["events"][-1]["truncated"] is True
    empty = store.status(job_id, cursor=page["next_cursor"])
    assert empty["events"] == [] and empty["next_cursor"] == page["next_cursor"]
    with pytest.raises(ValueError, match="limit"):
        store.status(job_id, limit=101)
    with pytest.raises(ValueError, match="cursor"):
        store.status(job_id, cursor=-1)


def test_real_worker_progress_survives_event_rollover(
    store, source, tmp_path, faulted_job
):
    recipe = _recipe(source, tmp_path / "logged-quantization")
    # A small retention window rolls over on real numerical progress; no fixture
    # events or post-completion appends stand in for the worker's diagnostics.
    job_id = faulted_job("log_rollover", "quantize", recipe)
    page = store.status(job_id)
    assert page["state"] == "running" and page["ownership"]["worker"]
    assert not (store.root / job_id / "receipt.json").exists()
    assert page["events_truncated"] and page["oldest_cursor"] > 1
    assert page["last_cursor"] > len(page["events"])
    progress = []
    for line in Path(page["log_path"]).read_text().splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue  # Libraries may also write plain stdout/stderr.
        if isinstance(item, dict) and {"phase", "status", "elapsed_s"} <= item.keys():
            progress.append(item)
    diagnostics = [item for item in progress if item["phase"] == "log"]
    assert diagnostics and all(item["message"] for item in diagnostics)
    retained = [
        {key: value for key, value in item.items() if key not in {"cursor", "at"}}
        for item in page["events"]
    ]
    assert all(item in progress for item in retained)
    assert diagnostics[0] not in retained
    assert any(
        item["phase"] == "run" and item["status"] == "running" for item in progress
    )
    assert any(
        item["phase"] == "encode" and item["status"] == "completed" for item in progress
    )
    os.kill(_row(store, job_id)["worker_pid"], signal.SIGTERM)
    final = _terminal(store, job_id)
    assert final["returncode"] == -signal.SIGTERM
    _assert_committed_quantization(store, job_id, recipe, final)


@pytest.mark.parametrize(
    "job_id",
    [
        "../outside",
        "/tmp/absolute",
        ".",
        "job_" + "a" * 31,
        "job_" + "a" * 32 + "/../other",
    ],
)
def test_job_ids_cannot_traverse_the_registry(store, job_id):
    with pytest.raises(ValueError, match="job ID"):
        store.status(job_id)
    with pytest.raises(ValueError, match="job ID"):
        store.cancel(job_id)
    assert _sql(store, "SELECT count(*) FROM jobs") == [(0,)]


def test_fixed_dispatch_rejects_executable_and_extra_fields_before_persistence(
    store, tmp_path
):
    with pytest.raises(ValueError, match="unsupported job operation"):
        store.submit("shell", {"command": "touch forbidden"}, "shell")
    with pytest.raises(ValueError, match="request fields"):
        store.submit(
            "validate", {"path": str(tmp_path), "callable": "os.system"}, "callable"
        )
    with pytest.raises(ValueError, match="request fields"):
        store.submit(
            "quality",
            {
                "src": str(tmp_path),
                "dst": str(tmp_path),
                "text": str(tmp_path),
                "strict": False,
            },
            "weaken-gate",
        )
    assert _sql(store, "SELECT count(*) FROM jobs") == [(0,)]
    unknown = "job_" + "f" * 32
    with pytest.raises(KeyError, match="unknown job"):
        store.status(unknown)
    assert not (store.root / unknown).exists()


def test_validation_exception_is_durable_failure_not_success(store, tmp_path):
    artifact = tmp_path / "invalid"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    job_id = store.submit("validate", {"path": str(artifact)}, "invalid")["job_id"]
    final = _terminal(store, job_id)
    assert final["state"] == "failed" and final["returncode"] == 1
    assert final["result"] is None
    assert final["error"]["type"] == "ValueError"
    assert "unsupported HI2" in final["error"]["message"]
    assert JobStore(store.root).status(job_id)["error"] == final["error"]
    assert store.cancel(job_id)["cancellation"] == "already_finished"


@pytest.fixture(scope="module")
def quantized_model(tmp_path_factory):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    root = tmp_path_factory.mktemp("jobs-quality-cpu")
    source = root / "source"
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        LlamaForCausalLM(config).float().eval().save_pretrained(
            source, safe_serialization=True
        )
    vocab = {"[UNK]": 0, **{chr(ord("a") + i): i + 1 for i in range(15)}}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]"
    ).save_pretrained(source)
    recipe = _recipe(source, root / "quantized")
    registry = JobStore(root / "registry")
    job_id = registry.submit("quantize", recipe, "quantized-fixture")["job_id"]
    try:
        final = _terminal(registry, job_id)
        assert final["state"] == "succeeded", final
        yield source, Path(recipe["config"]["dst"])
    finally:
        if registry.status(job_id)["state"] not in TERMINAL:
            registry.cancel(job_id)
            _terminal(registry, job_id)


def test_real_quality_gate_rejection_keeps_measurements_and_failure(
    store, quantized_model, tmp_path
):
    source, output = quantized_model
    text = tmp_path / "held-out.txt"
    text.write_text("a b c d e f g h i j k l m n o " * 4)
    request = {
        "src": str(source),
        "dst": str(output),
        "text": str(text),
        "device": "cpu",
        "dtype": "float32",
        "samples": 1,
        "seq_len": 8,
        "max_ppl_ratio": 0.0,
    }
    job_id = store.submit("quality", request, "real-quality-rejection")["job_id"]
    final = _terminal(store, job_id)
    assert final["state"] == "failed" and final["returncode"] == 1
    assert final["result"] is None and final["error"]["type"] == "QualityRejected"
    report = final["error"]["report"]
    assert report["status"] == "target_not_met"
    assert "max_ppl_ratio" in report["failures"]
    assert report["metrics"]["tokens"] == 7 and report["metrics"]["ppl_ratio"] > 0
    assert json.loads((output / "hi2_quality.json").read_text()) == report
    assert JobStore(store.root).status(job_id)["error"]["report"] == report


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("via_leases", [False, True])
def test_orphaned_quality_rejection_preserves_measurements(
    store, quantized_model, tmp_path, faulted_job, cancelled, via_leases
):
    source, output = quantized_model
    text = tmp_path / "orphan-held-out.txt"
    text.write_text("a b c d e f g h i j k l m n o " * 4)
    request = {
        "src": str(source),
        "dst": str(output),
        "text": str(text),
        "device": "cpu",
        "dtype": "float32",
        "samples": 1,
        "seq_len": 8,
        "max_ppl_ratio": 0.0,
    }
    job_id = faulted_job("failed_receipt", "quality", request)
    receipt = json.loads((store.root / job_id / "receipt.json").read_text())
    assert receipt["state"] == "failed"
    assert receipt["error"]["type"] == "QualityRejected"
    report = receipt["error"]["report"]
    assert report["status"] == "target_not_met"
    assert "max_ppl_ratio" in report["failures"]
    assert report["metrics"]["tokens"] == 7 and report["metrics"]["ppl_ratio"] > 0
    assert json.loads((output / "hi2_quality.json").read_text()) == report
    row = _row(store, job_id)
    os.kill(row["worker_pid"], signal.SIGSTOP)
    os.kill(row["supervisor_pid"], signal.SIGKILL)
    owned = _wait(store, job_id, lambda s: not s["ownership"]["supervisor"])
    assert owned["state"] == "running" and owned["ownership"]["orphaned_worker"]
    reconnected = JobStore(store.root)
    if cancelled:
        assert reconnected.cancel(job_id)["state"] == "cancelling"
    if via_leases:
        successor = reconnected.submit(
            "validate", {"path": str(output)}, "after-quality-receipt"
        )["job_id"]
        _queued_after_lease_attempt(reconnected, successor)
    os.kill(row["worker_pid"], signal.SIGKILL)
    expected_state = "cancelled" if cancelled else "failed"
    if via_leases:
        validation = _terminal(reconnected, successor)
        assert validation["state"] == "succeeded"
        assert validation["result"]["coverage"] == 1
        assert _row(store, job_id)["state"] == expected_state
    final = _terminal(reconnected, job_id)
    assert final["state"] == expected_state
    assert final["returncode"] is None and final["result"] is None
    if cancelled:
        assert final["error"]["type"] == "Cancelled"
        error = final["error"]["worker_error"]
    else:
        error = final["error"]
    assert error == receipt["error"]
    assert error["report"] == report
    assert _sql(store, "SELECT * FROM leases WHERE job_id = ?", (job_id,)) == []


def test_validate_and_portable_return_actual_structural_results(
    store, quantized_model, tmp_path
):
    source, output = quantized_model
    validate = store.submit("validate", {"path": str(output)}, "valid-artifact")[
        "job_id"
    ]
    validation = _terminal(store, validate)
    assert validation["state"] == "succeeded", validation
    assert validation["result"]["mode"] == "full_structural"
    assert validation["result"]["coverage"] == 1
    assert validation["result"]["checked"] > 0
    destination = tmp_path / "portable"
    portable = store.submit(
        "portable",
        {"src": str(output), "out": str(destination), "assets_from": str(source)},
        "portable",
    )["job_id"]
    final = _terminal(store, portable)
    assert final["state"] == "succeeded", final
    result = final["result"]
    assert result["exported"] == str(destination)
    assert result["quantized_weights"] == "byte_identical"
    assert result["validation"]["coverage"] == 1
    assert (destination / "tokenizer.json").is_file()
    from hi2.runtime import file_digest

    for name in (p.name for p in output.glob("*.safetensors")):
        assert file_digest(output / name) == file_digest(destination / name)


def test_cancelled_quantization_cannot_publish_and_requires_explicit_resubmission(
    store, source, tmp_path, barriers
):
    text = tmp_path / "evaluation.txt"
    text.write_text("held out")
    blocker, _ = barriers(
        "quant-cancel-cpu",
        operation="quality",
        request={
            "src": str(source),
            "text": str(text),
            "device": "cpu",
        },
    )
    recipe = _recipe(source, tmp_path / "cancelled-quant")
    job_id = store.submit("quantize", recipe, "cancelled-quant")["job_id"]
    assert store.status(job_id)["state"] == "queued"
    # The queued job has passed real submission preflight. Block its worker at
    # the next real config read, without replacing the dispatcher or pipeline.
    config = source / "config.json"
    original = tmp_path / "original-config.json"
    config.rename(original)
    os.mkfifo(config)
    writer = None
    try:
        store.cancel(blocker)
        assert _terminal(store, blocker)["state"] == "cancelled"
        writer = _reader(store, job_id, config)
        ack = store.cancel(job_id)
        assert ack["cancellation"] in {"requested", "cancelled"}
        final = _terminal(store, job_id)
        assert final["state"] == "cancelled" and final["result"] is None
        output = Path(recipe["config"]["dst"])
        assert not (output / "hi2_run.json").exists()
        work = Path(recipe["config"]["workdir"])
        if (work / "run.json").exists():
            assert json.loads((work / "run.json").read_text())["state"] != "succeeded"
    finally:
        if store.status(job_id)["state"] not in TERMINAL:
            store.cancel(job_id)
            _terminal(store, job_id)
        if writer is not None:
            os.close(writer)
        config.unlink()
        original.rename(config)
    assert store.submit("quantize", recipe, "cancelled-quant")["job_id"] == job_id
    resumed = store.submit("quantize", recipe, "explicit-resubmission")["job_id"]
    assert resumed != job_id
    assert _terminal(store, resumed)["state"] == "succeeded"


def test_resolved_optional_defaults_and_dictionary_order_share_one_job(
    store, source, tmp_path, barriers
):
    text = tmp_path / "canonical-eval.txt"
    text.write_text("held out")
    job_id, fifo = barriers(
        "canonical-quality",
        operation="quality",
        request={
            "src": str(source),
            "text": str(text),
        },
    )
    equivalent = {
        "max_ppl_ratio": None,
        "excluded": None,
        "max_kld": None,
        "seq_len": 2048,
        "samples": 8,
        "dtype": "float32",
        "device": "cpu",
        "text": str(text),
        "dst": str(fifo.parent),
        "src": str(source),
    }
    assert store.submit("quality", equivalent, "canonical-quality")["job_id"] == job_id
    with pytest.raises(IdempotencyConflict):
        store.submit(
            "portable",
            {"src": str(source), "out": str(tmp_path / "portable")},
            "canonical-quality",
        )
    assert _sql(store, "SELECT count(*) FROM jobs") == [(1,)]
    store.cancel(job_id)
    assert _terminal(store, job_id)["state"] == "cancelled"


def test_private_entrypoint_cannot_replace_inherited_worker_ownership(store, barriers):
    job_id, _ = barriers("ownership-descriptor")
    row = _row(store, job_id)
    # A second open names the same inode, but does not own the inherited flock.
    fd = os.open(store.root / job_id / "worker.lock", os.O_RDWR)
    try:
        duplicate = subprocess.run(
            [
                sys.executable,
                "-m",
                "hi2.job_worker",
                "work",
                "--root",
                str(store.root),
                "--job-id",
                job_id,
                "--token",
                row["supervisor_token"],
                "--owner-fd",
                str(fd),
            ],
            pass_fds=(fd,),
            stdin=subprocess.DEVNULL,
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert duplicate.returncode != 0
        assert "BlockingIOError" in duplicate.stderr
        assert store.status(job_id)["state"] == "running"
        assert _row(store, job_id)["worker_pid"] == row["worker_pid"]
    finally:
        os.close(fd)
    store.cancel(job_id)
    assert _terminal(store, job_id)["state"] == "cancelled"


def test_supervisor_spawn_failure_is_durable_without_worker_or_leases(
    store, tmp_path, monkeypatch
):
    import hi2.jobs as jobs

    # Fail the real POSIX exec, rather than replacing Popen or the dispatcher.
    missing = tmp_path / "missing-supervisor"
    monkeypatch.setattr(jobs, "_command", lambda *args: [str(missing)])
    request = {"path": str(tmp_path / "artifact")}
    failed = store.submit("validate", request, "spawn-failure")
    job_id = failed["job_id"]
    assert failed["state"] == "failed"
    assert failed["error"]["type"] == "FileNotFoundError"
    assert failed["result"] is None and failed["returncode"] is None
    assert [event["status"] for event in failed["events"]] == ["queued", "failed"]
    row = _row(store, job_id)
    assert row["supervisor_pid"] is None
    assert row["worker_pid"] is None and row["worker_started_at"] is None
    assert not failed["ownership"]["supervisor"]
    assert not failed["ownership"]["worker"]
    assert _sql(store, "SELECT * FROM leases") == []

    restored = JobStore(store.root)
    replay = restored.submit("validate", request, "spawn-failure")
    assert replay["job_id"] == job_id and replay["state"] == "failed"
    assert replay["error"] == failed["error"]
    assert replay["events"] == failed["events"]
    cancelled = restored.cancel(job_id)
    assert cancelled["cancellation"] == "already_finished"
    assert cancelled["cancel_requested_at"] is None
    assert cancelled["events"] == failed["events"]
    assert _sql(store, "SELECT COUNT(*) FROM jobs") == [(1,)]


def test_stale_generation_cannot_publish_or_release_live_worker_leases(store, barriers):
    from hi2.job_worker import _publish_exit

    job_id, _ = barriers("generation-fencing")
    row = _row(store, job_id)
    token = row["supervisor_token"]
    marker = {"phase": "fixture", "status": "current-generation"}
    store._emit(job_id, token, marker)
    before = store.status(job_id)
    assert before["events"][-1]["status"] == marker["status"]
    leases = _sql(store, "SELECT * FROM leases WHERE job_id = ?", (job_id,))
    assert leases

    stale = "stale-" + token
    store._emit(job_id, stale, {"phase": "fixture", "status": "stale-generation"})
    assert store._acquire_leases(job_id, stale) is False
    _publish_exit(store, job_id, stale, 0)
    active = store.status(job_id)
    assert active["state"] == "running" and active["ownership"]["worker"]
    assert active["events"] == before["events"]
    assert active["last_cursor"] == before["last_cursor"]
    assert active["result"] is None and active["error"] is None
    assert _sql(store, "SELECT * FROM leases WHERE job_id = ?", (job_id,)) == leases
    assert _row(store, job_id)["worker_pid"] == row["worker_pid"]

    store.cancel(job_id)
    terminal = _terminal(store, job_id)
    assert terminal["state"] == "cancelled"
    # Even the current generation loses publication rights after terminal commit.
    store._emit(job_id, token, {"phase": "fixture", "status": "late-progress"})
    _publish_exit(store, job_id, token, 0)
    after = store.status(job_id)
    for field in (
        "state",
        "result",
        "error",
        "returncode",
        "updated_at",
        "cancel_requested_at",
        "last_cursor",
        "events",
    ):
        assert after[field] == terminal[field]
    assert _sql(store, "SELECT * FROM leases WHERE job_id = ?", (job_id,)) == []


def test_aborted_event_transaction_restores_payloads_and_cursor(store, barriers):
    job_id, _ = barriers("event-rollback")
    before = store.status(job_id)
    with pytest.raises(KeyboardInterrupt), store._transaction() as db:
        store._event(db, job_id, {"phase": "fixture", "status": "rolled-back"})
        raise KeyboardInterrupt
    after = store.status(job_id)
    assert after["events"] == before["events"]
    assert after["last_cursor"] == before["last_cursor"]
    assert after["updated_at"] == before["updated_at"]

    token = _row(store, job_id)["supervisor_token"]
    store._emit(job_id, token, {"phase": "fixture", "status": "committed"})
    committed = store.status(job_id)
    assert committed["last_cursor"] == before["last_cursor"] + 1
    assert committed["events"][:-1] == before["events"]
    assert committed["events"][-1]["status"] == "committed"
    store.cancel(job_id)
    assert _terminal(store, job_id)["state"] == "cancelled"


def test_surviving_submitter_reaps_supervisor_without_another_submission(
    store, tmp_path
):
    fifo = _fifo(tmp_path / "reap-supervisor")
    job_id = store.submit("validate", {"path": str(fifo.parent)}, "reap")["job_id"]
    writer = _reader(store, job_id, fifo)
    supervisor_pid = _row(store, job_id)["supervisor_pid"]
    try:
        assert store.status(job_id)["ownership"]["worker"]
        os.write(writer, b"{}")
    finally:
        os.close(writer)
    final = _terminal(store, job_id)
    assert final["state"] == "failed" and final["returncode"] == 1
    assert final["error"]["type"] == "ValueError"
    # No Popen/ps or waitpid here: either could reap an abandoned Popen child
    # itself and conceal the leak. signal 0 observes zombies without reaping them.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(supervisor_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("exited supervisor was not reaped by its surviving submitter")
