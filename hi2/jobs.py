# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Durable, single-host jobs; use one shared registry for resource arbitration.

Submission commits before acknowledging and detaches from the client's session and
stdio. Conflicts queue, including CPU compute jobs; non-compute jobs with disjoint
paths can run together. Path leases cover ancestors and descendants, not just equal
spellings. Cancellation is explicit, never inferred from a disconnected client.

Ownership is an inherited flock, not a PID or a heartbeat timeout. A stopped or
orphaned worker retains its leases until its ownership descriptor closes. Status
can restart an unstarted queue entry, but never replays an interrupted operation.
Events retain the latest MAX_EVENTS entries, each bounded by MAX_EVENT_BYTES;
status paginates them without reading the potentially large worker log.
"""

import contextlib
import fcntl
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .runtime import SCHEMA_VERSION, object_digest

TERMINAL = frozenset({"succeeded", "failed", "cancelled", "interrupted"})
MAX_EVENTS = 1000
MAX_EVENT_BYTES = 8192
MAX_STATUS_EVENTS = 100
HEARTBEAT_STALE = 5.0
_JOB_ID = re.compile(r"job_[0-9a-f]{32}\Z")
_VISIBILITY = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


class IdempotencyConflict(ValueError):
    """An existing key belongs to a different immutable operation/request."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _job_id(value):
    if not isinstance(value, str) or not _JOB_ID.fullmatch(value):
        raise ValueError(
            "invalid job ID; expected job_ followed by 32 lowercase hex digits"
        )
    return value


def _take_lock(path):
    # Never unlink lock files; inherited descriptors must continue naming one inode.
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    except BaseException:
        os.close(fd)
        raise
    return fd


def _owned(path):
    fd = _take_lock(path)
    if fd is None:
        return True
    os.close(fd)
    return False


def _path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("job paths must be nonempty strings")
    return str(Path(value).expanduser().resolve())


def _submission_digest(operation, request, visibility):
    """Recognize exact retries even if their original inputs are no longer online."""
    if not isinstance(request, dict):
        raise ValueError("job request must be an object")
    submitted = json.loads(_json(request))
    paths = submitted.get("config", {}) if operation == "quantize" else submitted
    if isinstance(paths, dict):
        for name in (
            "src",
            "dst",
            "workdir",
            "cache_dir",
            "path",
            "out",
            "text",
            "assets_from",
        ):
            if paths.get(name) is not None:
                paths[name] = _path(paths[name])
        for name in ("calib", "excluded"):
            value = paths.get(name)
            if isinstance(value, str) and name == "calib":
                value = [p for p in value.split(",") if p]
            if isinstance(value, list):
                paths[name] = [_path(p) for p in value]
    return object_digest(
        {"operation": operation, "request": submitted, "visibility": visibility}
    )


def _devices(values):
    resources = set()
    for value in values:
        if re.fullmatch(r"cpu(?::\d+)?", value):
            resources.add("device:cpu")
            continue
        match = re.fullmatch(r"cuda(?::(\d+))?", value)
        if not match:
            raise ValueError(f"unsupported job device: {value}; use cpu or cuda:N")
        # Runtime UUIDs, not visible ordinals, identify the same CUDA/HIP device
        # across different visibility masks. This import is GPU-request-only.
        import torch

        index = int(match.group(1) or 0)
        try:
            identity = str(torch.cuda.get_device_properties(index).uuid)
        except (AttributeError, AssertionError, RuntimeError) as exc:
            raise ValueError(
                f"cannot establish stable ownership for job device {value}: {exc}"
            ) from exc
        if not identity:
            raise ValueError(f"job device {value} has no stable UUID for leasing")
        resources.add(f"device:cuda:{identity}")
    if not resources:
        raise ValueError("select at least one job device")
    return resources


def _prepare(operation, request):
    if operation not in {"quantize", "validate", "quality", "portable"}:
        raise ValueError(f"unsupported job operation: {operation}")
    if not isinstance(request, dict):
        raise ValueError("job request must be an object")
    # Detach nested caller data and reject non-JSON values, NaNs and infinities.
    request = json.loads(_json(request))
    resources = set()
    if operation == "quantize":
        from .service import config_dict, config_from_dict, output_paths, resolve_config

        rc = resolve_config(config_from_dict(request))
        request = {"schema_version": SCHEMA_VERSION, "config": config_dict(rc)}
        resources.update(_devices(rc.devices))
        resources.update(f"path:{p.resolve()}" for p in [rc.workdir, *output_paths(rc)])
    else:
        required, optional = {
            "validate": ({"path"}, set()),
            "portable": ({"src", "out"}, {"assets_from"}),
            "quality": (
                {"src", "dst", "text"},
                {
                    "max_kld",
                    "max_ppl_ratio",
                    "excluded",
                    "samples",
                    "seq_len",
                    "device",
                    "dtype",
                },
            ),
        }[operation]
        if required - request.keys() or request.keys() - required - optional:
            raise ValueError(
                f"invalid {operation} request fields; required={sorted(required)}, optional={sorted(optional)}"
            )
        for name in ("path", "src", "out", "dst", "text", "assets_from"):
            if name in request and request[name] is not None:
                request[name] = _path(request[name])
        for name in required:
            if request[name] is None:
                raise ValueError(f"{name} must be a path")
        if operation == "quality":
            request.setdefault("device", "cpu")
            request.setdefault("dtype", "float32")
            request.setdefault("samples", 8)
            request.setdefault("seq_len", 2048)
            request.setdefault("max_kld", None)
            request.setdefault("max_ppl_ratio", None)
            if request.get("excluded") is not None:
                if not isinstance(request["excluded"], list):
                    raise ValueError("excluded must be a list of paths")
                request["excluded"] = [_path(p) for p in request["excluded"]]
            else:
                request["excluded"] = []
            resources.update(_devices(request["device"].split(",")))
        elif operation == "portable":
            request.setdefault("assets_from", None)
        # Readers also exclude writers, including quality's hi2_quality.json write.
        resources.update(
            f"path:{request[k]}" for k in ("path", "src", "out", "dst") if k in request
        )
    return request, sorted(resources)


def _conflicts(left, right):
    if left == right:
        return True
    if left.startswith("path:") and right.startswith("path:"):
        a, b = Path(left[5:]), Path(right[5:])
        return a in b.parents or b in a.parents
    return False


def _command(mode, root, job_id, token, owner_fd):
    return [
        sys.executable,
        "-m",
        "hi2.job_worker",
        mode,
        "--root",
        str(root),
        "--job-id",
        job_id,
        "--token",
        token,
        "--owner-fd",
        str(owner_fd),
    ]


def _environment(visibility=None):
    env = os.environ.copy()
    if visibility is not None:
        for name in _VISIBILITY:
            env.pop(name, None)
        env.update(visibility)
    package_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [package_root, env.get("PYTHONPATH")])
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env


class JobStore:
    """SQLite registry and bounded status API; importing it does not import Torch."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.database = self.root / "jobs.sqlite3"
        with self._connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise ValueError(f"unsupported jobs registry schema_version: {version}")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE, request_digest TEXT NOT NULL,
                    submission_digest TEXT NOT NULL,
                    operation TEXT NOT NULL, request TEXT NOT NULL, resources TEXT NOT NULL,
                    visibility TEXT NOT NULL,
                    state TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    result TEXT, error TEXT, event_cursor INTEGER NOT NULL DEFAULT 0,
                    cancel_requested_at REAL, supervisor_token TEXT, supervisor_pid INTEGER,
                    supervisor_heartbeat REAL, worker_pid INTEGER, worker_heartbeat REAL,
                    worker_started_at REAL, returncode INTEGER
                );
                CREATE TABLE IF NOT EXISTS events (
                    job_id TEXT NOT NULL REFERENCES jobs(job_id), cursor INTEGER NOT NULL,
                    at REAL NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(job_id, cursor)
                );
                CREATE TABLE IF NOT EXISTS leases (
                    resource TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(job_id),
                    acquired_at REAL NOT NULL
                );
                PRAGMA user_version = 1;
            """)

    @contextlib.contextmanager
    def _connect(self):
        db = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA synchronous = FULL")
        try:
            yield db
        finally:
            db.close()

    @contextlib.contextmanager
    def _transaction(self):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.rollback()
                raise
            else:
                db.commit()

    def _directory(self, job_id):
        return self.root / _job_id(job_id)

    def _row(self, db, job_id):
        row = db.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (_job_id(job_id),)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown job: {job_id}")
        return row

    def _event(self, db, job_id, item):
        payload = _json(item)
        if len(payload.encode()) > MAX_EVENT_BYTES:
            item = {
                "phase": "log",
                "status": "message",
                "truncated": True,
                "message": payload.encode()[: MAX_EVENT_BYTES // 8].decode(
                    errors="replace"
                ),
            }
            payload = _json(item)
        now = time.time()
        db.execute(
            "UPDATE jobs SET event_cursor = event_cursor + 1, updated_at = ? WHERE job_id = ?",
            (now, job_id),
        )
        cursor = self._row(db, job_id)["event_cursor"]
        db.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?)", (job_id, cursor, now, payload)
        )
        db.execute(
            "DELETE FROM events WHERE job_id = ? AND cursor <= ?",
            (job_id, cursor - MAX_EVENTS),
        )

    def _emit(self, job_id, token, item):
        with self._transaction() as db:
            row = self._row(db, job_id)
            if row["supervisor_token"] == token and row["state"] not in TERMINAL:
                self._event(db, job_id, item)

    def _finish(self, db, row, state, *, result=None, error=None, returncode=None):
        if row["state"] in TERMINAL:
            return
        db.execute(
            "UPDATE jobs SET state = ?, result = ?, error = ?, returncode = ?, updated_at = ? WHERE job_id = ?",
            (
                state,
                _json(result) if result is not None else None,
                _json(error) if error is not None else None,
                returncode,
                time.time(),
                row["job_id"],
            ),
        )
        db.execute("DELETE FROM leases WHERE job_id = ?", (row["job_id"],))
        self._event(db, row["job_id"], {"phase": "job", "status": state})

    def _read_receipt(self, row):
        try:
            value = json.loads(
                (self._directory(row["job_id"]) / "receipt.json").read_text()
            )
        except (OSError, ValueError):
            return None
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != SCHEMA_VERSION
            or value.get("job_id") != row["job_id"]
            or not row["supervisor_token"]
            or value.get("token") != row["supervisor_token"]
            or value.get("state") not in {"succeeded", "failed", "interrupted"}
        ):
            return None
        return value

    def _committed_result(self, row, receipt):
        """Read commit evidence only after the worker has relinquished ownership."""
        if row["worker_started_at"] is None:
            return None
        if receipt is not None and receipt["state"] == "succeeded":
            result = receipt.get("result")
            if isinstance(result, dict):
                return result
        if row["operation"] != "quantize":
            return None
        try:
            binding = json.loads(
                (self._directory(row["job_id"]) / "run-binding.json").read_text()
            )
            workdir = json.loads(row["request"])["config"]["workdir"]
            if (
                not isinstance(binding, dict)
                or binding.get("schema_version") != SCHEMA_VERSION
                or binding.get("job_id") != row["job_id"]
                or not row["supervisor_token"]
                or binding.get("token") != row["supervisor_token"]
                or binding.get("workdir") != workdir
                or not isinstance(binding.get("run_id"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", binding["run_id"])
                or not isinstance(binding.get("attempt_id"), str)
                or not re.fullmatch(r"[0-9a-f]{32}", binding["attempt_id"])
            ):
                return None
            manifest = json.loads((Path(workdir) / "run.json").read_text())
        except (OSError, ValueError):
            return None
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("run_id") != binding["run_id"]
            or manifest.get("attempt_id") != binding["attempt_id"]
            or manifest.get("state") != "succeeded"
            or not isinstance(manifest.get("config"), dict)
            or manifest["config"].get("workdir") != workdir
            or not isinstance(manifest.get("result"), dict)
        ):
            return None
        # A fresh attempt binds this invocation, not merely its reusable recipe.
        # A later foreground run of the same recipe cannot complete this job.
        return manifest["result"]

    def _reconcile_unowned(self, db, row):
        """Called with both ownership locks held, so no matching process can run."""
        if row["state"] in TERMINAL:
            db.execute("DELETE FROM leases WHERE job_id = ?", (row["job_id"],))
            return
        receipt = self._read_receipt(row)
        result = self._committed_result(row, receipt)
        if result is not None:
            self._finish(
                db, row, "succeeded", result=result, returncode=row["returncode"]
            )
        elif row["cancel_requested_at"] is not None:
            self._finish(
                db,
                row,
                "cancelled",
                error={
                    "type": "Cancelled",
                    "message": "Cancellation requested; no worker retains ownership.",
                    "worker_error": receipt.get("error") if receipt else None,
                },
            )
        elif (
            row["worker_started_at"] is not None
            and receipt is not None
            and receipt["state"] in {"failed", "interrupted"}
        ):
            self._finish(
                db,
                row,
                receipt["state"],
                error=receipt.get("error"),
                returncode=row["returncode"],
            )
        elif row["worker_started_at"] is not None:
            self._finish(
                db,
                row,
                "interrupted",
                error={
                    "type": "OwnershipLost",
                    "message": "Supervisor and worker ownership were lost without a matching committed result.",
                },
            )
        else:
            db.execute("DELETE FROM leases WHERE job_id = ?", (row["job_id"],))

    def _start_or_reconcile(self, job_id):
        directory = self._directory(job_id)
        with self._connect() as db:
            row = self._row(db, job_id)
            if row["state"] in TERMINAL:
                return
        supervisor_fd = _take_lock(directory / "supervisor.lock")
        if supervisor_fd is None:
            return
        try:
            worker_fd = _take_lock(directory / "worker.lock")
            if worker_fd is None:
                # An orphan or stopped worker still owns every lease. Its watchdog
                # will stop it; status must not adopt it or trust a saved PID.
                return
            try:
                with self._transaction() as db:
                    row = self._row(db, job_id)
                    self._reconcile_unowned(db, row)
                    row = self._row(db, job_id)
                    if row["state"] in TERMINAL:
                        return
                    token = uuid.uuid4().hex
                    db.execute(
                        "UPDATE jobs SET supervisor_token = ?, supervisor_heartbeat = ?, updated_at = ? WHERE job_id = ?",
                        (token, time.time(), time.time(), job_id),
                    )
            finally:
                os.close(worker_fd)
            try:
                with (directory / "supervisor.log").open("ab", buffering=0) as log:
                    child = subprocess.Popen(
                        _command("supervise", self.root, job_id, token, supervisor_fd),
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        close_fds=True,
                        pass_fds=(supervisor_fd,),
                        start_new_session=True,
                        cwd=Path(__file__).resolve().parent.parent,
                        env=_environment(json.loads(row["visibility"])),
                    )
                # Retain and reap our child even if the submitter never calls us
                # again. A daemon waiter must not couple the job to client exit.
                threading.Thread(target=child.wait, daemon=True).start()
                with self._transaction() as db:
                    db.execute(
                        "UPDATE jobs SET supervisor_pid = ? WHERE job_id = ? AND supervisor_token = ?",
                        (child.pid, job_id, token),
                    )
            except OSError as exc:
                with self._transaction() as db:
                    self._finish(
                        db,
                        self._row(db, job_id),
                        "failed",
                        error={"type": type(exc).__name__, "message": str(exc)},
                    )
        finally:
            # Closing, not LOCK_UN: the detached child inherited this description.
            os.close(supervisor_fd)

    def submit(self, operation: str, request: dict, idempotency_key: str) -> dict:
        if (
            not isinstance(idempotency_key, str)
            or not idempotency_key
            or len(idempotency_key) > 512
        ):
            raise ValueError("idempotency_key must contain 1 to 512 characters")
        visibility = {
            name: os.environ[name] for name in _VISIBILITY if name in os.environ
        }
        submitted = _submission_digest(operation, request, visibility)
        with self._connect() as db:
            existing = db.execute(
                "SELECT job_id, submission_digest FROM jobs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        if existing is not None and existing["submission_digest"] == submitted:
            return self.status(existing["job_id"])
        request, resources = _prepare(operation, request)
        digest = object_digest(
            {"operation": operation, "request": request, "resources": resources}
        )
        with self._transaction() as db:
            existing = db.execute(
                "SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if existing is not None:
                if existing["request_digest"] != digest:
                    raise IdempotencyConflict(
                        "idempotency key already belongs to a different operation/request"
                    )
                job_id = existing["job_id"]
            else:
                job_id = "job_" + uuid.uuid4().hex
                self._directory(job_id).mkdir(mode=0o700)
                now = time.time()
                db.execute(
                    """INSERT INTO jobs (
                    job_id, schema_version, idempotency_key, request_digest,
                    submission_digest, operation, request, resources, visibility,
                    state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)""",
                    (
                        job_id,
                        SCHEMA_VERSION,
                        idempotency_key,
                        digest,
                        submitted,
                        operation,
                        _json(request),
                        _json(resources),
                        _json(visibility),
                        now,
                        now,
                    ),
                )
                self._event(db, job_id, {"phase": "job", "status": "queued"})
        return self.status(job_id)

    def _snapshot(self, job_id, cursor, limit):
        with self._connect() as db:
            db.execute("BEGIN")
            row = self._row(db, job_id)
            events = db.execute(
                "SELECT cursor, at, payload FROM events WHERE job_id = ? AND cursor > ? ORDER BY cursor LIMIT ?",
                (job_id, cursor, limit),
            ).fetchall()
            oldest = db.execute(
                "SELECT MIN(cursor) FROM events WHERE job_id = ?", (job_id,)
            ).fetchone()[0]
        directory = self._directory(job_id)
        supervisor, worker = (
            _owned(directory / "supervisor.lock"),
            _owned(directory / "worker.lock"),
        )
        heartbeat = row["supervisor_heartbeat"]
        result = {
            key: row[key]
            for key in (
                "schema_version",
                "job_id",
                "state",
                "operation",
                "created_at",
                "updated_at",
                "request_digest",
                "returncode",
                "cancel_requested_at",
            )
        }
        result.update(
            result=json.loads(row["result"]) if row["result"] is not None else None,
            error=json.loads(row["error"]) if row["error"] is not None else None,
            events=[
                {
                    **json.loads(event["payload"]),
                    "cursor": event["cursor"],
                    "at": event["at"],
                }
                for event in events
            ],
            next_cursor=events[-1]["cursor"] if events else cursor,
            oldest_cursor=oldest,
            last_cursor=row["event_cursor"],
            events_truncated=oldest is not None and cursor < oldest - 1,
            ownership={
                "supervisor": supervisor,
                "worker": worker,
                "supervisor_heartbeat": heartbeat,
                "worker_heartbeat": row["worker_heartbeat"],
                "heartbeat_stale": heartbeat is not None
                and time.time() - heartbeat > HEARTBEAT_STALE,
                "orphaned_worker": worker and not supervisor,
            },
            log_path=str(directory / "worker.log"),
        )
        return result

    def status(self, job_id: str, *, cursor=0, limit=100) -> dict:
        _job_id(job_id)
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("cursor must be a nonnegative integer")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_STATUS_EVENTS
        ):
            raise ValueError(f"limit must be between 1 and {MAX_STATUS_EVENTS}")
        self._start_or_reconcile(job_id)
        return self._snapshot(job_id, cursor, limit)

    def cancel(self, job_id: str) -> dict:
        with self._transaction() as db:
            row = self._row(db, job_id)
            if row["state"] not in TERMINAL and row["cancel_requested_at"] is None:
                db.execute(
                    "UPDATE jobs SET state = 'cancelling', cancel_requested_at = ?, updated_at = ? WHERE job_id = ?",
                    (time.time(), time.time(), job_id),
                )
                self._event(db, job_id, {"phase": "job", "status": "cancelling"})
        result = self.status(job_id)
        result["cancellation"] = (
            "cancelled"
            if result["state"] == "cancelled"
            else "requested"
            if result["state"] not in TERMINAL
            else "already_finished"
        )
        return result

    def _acquire_leases(self, job_id, token):
        with self._transaction() as db:
            row = self._row(db, job_id)
            if (
                row["supervisor_token"] != token
                or row["state"] in TERMINAL
                or row["cancel_requested_at"] is not None
            ):
                return False
            resources = json.loads(row["resources"])
            holders = db.execute(
                "SELECT DISTINCT j.* FROM jobs j JOIN leases l ON j.job_id = l.job_id WHERE j.job_id != ?",
                (job_id,),
            ).fetchall()
            for holder in holders:
                directory = self._directory(holder["job_id"])
                supervisor_fd = _take_lock(directory / "supervisor.lock")
                if supervisor_fd is None:
                    continue
                try:
                    worker_fd = _take_lock(directory / "worker.lock")
                    if worker_fd is not None:
                        try:
                            self._reconcile_unowned(db, holder)
                        finally:
                            os.close(worker_fd)
                finally:
                    os.close(supervisor_fd)
            leased = db.execute(
                "SELECT resource FROM leases WHERE job_id != ?", (job_id,)
            ).fetchall()
            if any(
                _conflicts(resource, lease["resource"])
                for resource in resources
                for lease in leased
            ):
                return False
            for resource in resources:
                db.execute(
                    "INSERT OR IGNORE INTO leases VALUES (?, ?, ?)",
                    (resource, job_id, time.time()),
                )
            return True

    def _heartbeat(self, job_id, token, role):
        if role not in {"supervisor", "worker"}:
            raise ValueError("invalid ownership role")
        with self._transaction() as db:
            db.execute(
                f"UPDATE jobs SET {role}_heartbeat = ? WHERE job_id = ? AND supervisor_token = ?",
                (time.time(), job_id, token),
            )
            return dict(self._row(db, job_id))
