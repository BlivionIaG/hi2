# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Private POSIX supervisor/worker entrypoint used exclusively by JobStore.

The supervisor owns a real Popen child and waits for its exit before publishing a
terminal state or releasing leases. Explicit cancellation sends SIGINT, then after
2 seconds SIGTERM, then after another 2 seconds SIGKILL. A worker whose supervisor
dies applies the same escalation to itself; a stopped orphan keeps its leases.
Neither role accepts executable names, Python callables, or shell commands.
"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import threading
import time
import traceback
from pathlib import Path

from .jobs import TERMINAL, JobStore, _command, _environment, _owned, _take_lock
from .runtime import SCHEMA_VERSION, atomic_json

POLL_INTERVAL = 0.2
INTERRUPT_GRACE = 2.0
TERMINATE_GRACE = 2.0


def _check_owner(store, job_id, token, fd, role):
    expected = (store._directory(job_id) / f"{role}.lock").stat()
    actual = os.fstat(fd)
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise ValueError("ownership descriptor does not belong to this job")
    # An inode match alone is not ownership: a fresh open of the same lock file
    # must not be able to start a second worker beside the inherited descriptor.
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.set_inheritable(fd, False)
    with store._connect() as db:
        row = store._row(db, job_id)
        if row["supervisor_token"] != token or row["state"] in TERMINAL:
            raise ValueError("job ownership generation is no longer current")
        if role == "worker" and row["worker_started_at"] is None:
            raise ValueError(
                "worker was not launched by its resource-owning supervisor"
            )
        if role == "supervisor" and row["worker_started_at"] is not None:
            raise ValueError("an already started worker cannot be adopted or replayed")
    return row


def _dispatch(operation, request, progress):
    if operation == "quantize":
        from .runtime import read_run
        from .service import config_from_dict, execute_run

        config = config_from_dict(request)
        result = execute_run(config, progress)
        if read_run(config.workdir)["state"] != "succeeded":
            raise RuntimeError("quantization returned without a completed run manifest")
        return result
    if operation == "validate":
        from .checkpoint import validate_checkpoint

        return validate_checkpoint(Path(request["path"]), log=progress)
    if operation == "portable":
        from .checkpoint import export_portable

        return export_portable(
            Path(request["src"]),
            Path(request["out"]),
            assets_from=Path(request["assets_from"])
            if request.get("assets_from")
            else None,
        )
    if operation == "quality":
        from .quality import evaluate_quality

        options = {
            k: v
            for k, v in request.items()
            if k not in {"src", "dst", "text", "excluded"}
        }
        return evaluate_quality(
            Path(request["src"]),
            Path(request["dst"]),
            Path(request["text"]),
            excluded=[Path(p) for p in request.get("excluded") or []],
            strict=True,
            log=progress,
            **options,
        )
    raise ValueError(f"unsupported job operation: {operation}")


def _watch_supervisor(store, job_id, token, finished):
    lost_at = None
    sent_term = False
    while not finished.wait(POLL_INTERVAL):
        store._heartbeat(job_id, token, "worker")
        if lost_at is None and not _owned(store._directory(job_id) / "supervisor.lock"):
            lost_at = time.monotonic()
            os.kill(os.getpid(), signal.SIGINT)
        if lost_at is not None:
            elapsed = time.monotonic() - lost_at
            if elapsed >= INTERRUPT_GRACE + TERMINATE_GRACE:
                os.kill(os.getpid(), signal.SIGKILL)
            elif elapsed >= INTERRUPT_GRACE and not sent_term:
                sent_term = True
                os.kill(os.getpid(), signal.SIGTERM)


def _worker(store, job_id, token, owner_fd):
    row = _check_owner(store, job_id, token, owner_fd, "worker")
    finished = threading.Event()
    watcher = threading.Thread(
        target=_watch_supervisor, args=(store, job_id, token, finished), daemon=True
    )
    watcher.start()
    receipt = {"schema_version": SCHEMA_VERSION, "job_id": job_id, "token": token}
    request = json.loads(row["request"])
    code = 1
    try:
        from .service import Progress

        def emit(item):
            with store._connect() as db:
                current = store._row(db, job_id)
            if current["cancel_requested_at"] is not None:
                raise KeyboardInterrupt("job cancellation requested")
            if (
                row["operation"] == "quantize"
                and item.get("phase") == "run"
                and item.get("status") in {"running", "reused"}
            ):
                if item.get("workdir") != request["config"]["workdir"]:
                    raise ValueError("run progress does not match the job workdir")
                # RunSession has already replaced any previous success with a
                # running manifest. Persist this generation's identity before
                # allowing the pipeline to proceed towards its durable commit.
                atomic_json(
                    store._directory(job_id) / "run-binding.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "job_id": job_id,
                        "token": token,
                        "workdir": item["workdir"],
                        "run_id": item["run_id"],
                        "attempt_id": item["attempt_id"],
                    },
                )
            store._emit(job_id, token, item)

        # stderr is the append-only worker log; keep full diagnostics there even
        # after bounded registry events have rolled over or been truncated.
        progress = Progress(sink=emit, jsonl=True)
        progress.event("worker", "started", pid=os.getpid())
        result = _dispatch(row["operation"], request, progress)
        receipt.update(state="succeeded", result=result, error=None)
        code = 0
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        # Keep the real quality measurement/criteria, not just a generic failure.
        if hasattr(exc, "report"):
            error["report"] = exc.report
        receipt.update(
            state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            result=None,
            error=error,
        )
        traceback.print_exc()
    finally:
        # Ignore a late SIGINT while durably recording an already finished outcome.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        finished.set()
        watcher.join(timeout=POLL_INTERVAL * 2)
        receipt["finished_at"] = time.time()
        atomic_json(store._directory(job_id) / "receipt.json", receipt)
    return code


def _publish_exit(store, job_id, token, returncode):
    with store._transaction() as db:
        row = store._row(db, job_id)
        if row["supervisor_token"] != token:
            return
        cancelled = row["cancel_requested_at"] is not None
        receipt = store._read_receipt(row)
        result = store._committed_result(row, receipt)
        if result is not None:
            # A durable commit wins over interruption before the receipt, and a
            # valid receipt survives signals during interpreter teardown.
            store._finish(db, row, "succeeded", result=result, returncode=returncode)
            return
        if cancelled:
            store._finish(
                db,
                row,
                "cancelled",
                error={
                    "type": "Cancelled",
                    "message": "Cancellation requested and worker exit witnessed.",
                    "worker_error": receipt.get("error") if receipt else None,
                },
                returncode=returncode,
            )
        elif receipt is not None and receipt["state"] in {"failed", "interrupted"}:
            store._finish(
                db, row, receipt["state"], error=receipt["error"], returncode=returncode
            )
        else:
            store._finish(
                db,
                row,
                "interrupted",
                error={
                    "type": "WorkerExited",
                    "message": "Worker exited without a matching successful result.",
                    "returncode": returncode,
                },
                returncode=returncode,
            )


def _supervise(store, job_id, token, owner_fd):
    _check_owner(store, job_id, token, owner_fd, "supervisor")
    child = None
    stopping = False

    def interrupted(signum, frame):
        nonlocal stopping
        stopping = True

    # Supervisor termination is an interruption, not an explicit user cancellation.
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        while True:
            row = store._heartbeat(job_id, token, "supervisor")
            if row["state"] in TERMINAL or row["supervisor_token"] != token:
                return 0
            if row["cancel_requested_at"] is not None or stopping:
                with store._transaction() as db:
                    store._finish(
                        db,
                        store._row(db, job_id),
                        "cancelled"
                        if row["cancel_requested_at"] is not None
                        else "interrupted",
                        error={
                            "type": "Cancelled"
                            if row["cancel_requested_at"] is not None
                            else "SupervisorStopped",
                            "message": "Stopped before launching a worker.",
                        },
                    )
                return 0
            if store._acquire_leases(job_id, token):
                break
            time.sleep(POLL_INTERVAL)
        worker_fd = _take_lock(store._directory(job_id) / "worker.lock")
        if worker_fd is None:
            # Do not release leases even in an unexpected ownership collision.
            return 1
        try:
            with store._transaction() as db:
                row = store._row(db, job_id)
                if row["cancel_requested_at"] is not None or stopping:
                    store._finish(
                        db,
                        row,
                        "cancelled"
                        if row["cancel_requested_at"] is not None
                        else "interrupted",
                    )
                    return 0
                db.execute(
                    "UPDATE jobs SET state = 'running', worker_started_at = ?, updated_at = ? WHERE job_id = ?",
                    (time.time(), time.time(), job_id),
                )
                store._event(db, job_id, {"phase": "job", "status": "running"})
            with (store._directory(job_id) / "worker.log").open(
                "ab", buffering=0
            ) as log:
                child = subprocess.Popen(
                    _command("work", store.root, job_id, token, worker_fd),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    close_fds=True,
                    pass_fds=(worker_fd,),
                    start_new_session=True,
                    cwd=Path(__file__).resolve().parent.parent,
                    env=_environment(),
                )
            with store._transaction() as db:
                db.execute(
                    "UPDATE jobs SET worker_pid = ? WHERE job_id = ? AND supervisor_token = ?",
                    (child.pid, job_id, token),
                )
        finally:
            os.close(worker_fd)
        stop_at = None
        sent_term = sent_kill = False
        while child.poll() is None:
            row = store._heartbeat(job_id, token, "supervisor")
            if stop_at is None and (row["cancel_requested_at"] is not None or stopping):
                stop_at = time.monotonic()
                child.send_signal(signal.SIGINT)
                store._emit(
                    job_id,
                    token,
                    {
                        "phase": "cancellation" if not stopping else "interruption",
                        "status": "sigint",
                    },
                )
            if stop_at is not None:
                elapsed = time.monotonic() - stop_at
                if elapsed >= INTERRUPT_GRACE + TERMINATE_GRACE and not sent_kill:
                    child.kill()
                    sent_kill = True
                    store._emit(
                        job_id, token, {"phase": "termination", "status": "sigkill"}
                    )
                elif elapsed >= INTERRUPT_GRACE and not sent_term:
                    child.terminate()
                    sent_term = True
                    store._emit(
                        job_id, token, {"phase": "termination", "status": "sigterm"}
                    )
            time.sleep(POLL_INTERVAL)
        returncode = child.wait()
        _publish_exit(store, job_id, token, returncode)
        return 0
    except BaseException as exc:
        # Never publish or release a lease while our child could still be writing.
        if child is not None:
            if child.poll() is None:
                child.kill()
            child.wait()
        with store._transaction() as db:
            row = store._row(db, job_id)
            receipt = store._read_receipt(row)
            result = (
                store._committed_result(row, receipt) if child is not None else None
            )
            store._finish(
                db,
                row,
                "succeeded"
                if result is not None
                else "cancelled"
                if row["cancel_requested_at"] is not None
                else "interrupted",
                result=result,
                error=None
                if result is not None
                else {"type": type(exc).__name__, "message": str(exc)},
                returncode=child.returncode if child else None,
            )
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("supervise", "work"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--owner-fd", required=True, type=int)
    args = parser.parse_args()
    store = JobStore(args.root)
    # Keep the inherited descriptor through interpreter/atexit cleanup. Only the
    # kernel's process-exit close proves that an orphan can no longer write.
    if args.mode == "supervise":
        return _supervise(store, args.job_id, args.token, args.owner_fd)
    return _worker(store, args.job_id, args.token, args.owner_fd)


if __name__ == "__main__":
    raise SystemExit(main())
