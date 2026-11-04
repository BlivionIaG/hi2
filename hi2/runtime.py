# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Atomic local artifacts and an exclusive, identity-bound run lifecycle."""

import contextlib
import fcntl
import hashlib
import json
import os
import platform
import tempfile
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .errors import IncompatibleResume

SCHEMA_VERSION = 1


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@contextlib.contextmanager
def atomic_file(path: Path):
    """Commit one file only after its complete contents have been flushed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value) -> None:
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    with atomic_file(path) as stream:
        stream.write(content)
        stream.write(b"\n")


def atomic_torch_save(value, path: Path) -> None:
    import torch

    with atomic_file(Path(path)) as stream:
        torch.save(value, stream)


@contextlib.contextmanager
def exclusive_lock(path: Path):
    """Never unlink lock files: another process may still hold the inode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"resource is busy: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def environment() -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("hi2", "torch", "transformers", "safetensors"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    result["code_sha256"] = object_digest(
        {p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    )
    return result


def read_run(workdir: Path) -> dict:
    path = Path(workdir) / "run.json"
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read run manifest {path}: {exc}") from exc
    if record.get("schema_version") != SCHEMA_VERSION or not record.get("run_id"):
        raise ValueError(f"unsupported or incomplete run manifest: {path}")
    return record


def check_run_ownership(workdir: Path, outputs: list[Path], identity_factory):
    """Read-only ownership preflight; locks repeat these checks before mutation."""
    workdir = Path(workdir)
    identity = None
    run_id = None

    def expected_id():
        nonlocal identity, run_id
        if identity is None:
            identity = identity_factory()
            run_id = object_digest(identity)
        return run_id

    if workdir.exists():
        if not workdir.is_dir():
            raise ValueError(f"workdir is not a directory: {workdir}")
        if (workdir / "run.json").exists():
            if read_run(workdir)["run_id"] != expected_id():
                raise IncompatibleResume(
                    f"incompatible resume in {workdir}; use fresh work/output paths"
                )
        elif any(p.name != ".run.lock" for p in workdir.iterdir()):
            raise ValueError(
                f"unowned work directory {workdir}; choose a fresh workdir"
            )
    for output in outputs:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            marker = output / ".hi2-owner.json"
            if not marker.is_file():
                raise ValueError(f"unowned output already exists: {output}")
            if json.loads(marker.read_text()).get("run_id") != expected_id():
                raise IncompatibleResume(
                    f"output belongs to another run: {output}; choose fresh paths"
                )
    return identity


class RunSession:
    """Own a workdir and reject incompatible or unowned output reuse."""

    def __init__(self, workdir: Path, identity: dict, outputs: list[Path]):
        self.workdir = Path(workdir)
        self.outputs = [Path(p) for p in outputs]
        self.path = self.workdir / "run.json"
        self.run_id = object_digest(identity)
        # Recipe identity permits resume; invocation identity fences job recovery.
        self.attempt_id = uuid.uuid4().hex
        self.record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            **identity,
            "attempt_id": self.attempt_id,
            "outputs": [str(p.resolve()) for p in self.outputs],
            "state": "running",
            "phases": [],
        }
        self.previous = None
        self._lock = None
        self._outputs_lock = contextlib.ExitStack()
        self._publication_backups = {}

    def __enter__(self):
        self._lock = exclusive_lock(self.workdir / ".run.lock")
        self._lock.__enter__()
        try:
            if self.path.exists():
                old = read_run(self.workdir)
                if old["run_id"] != self.run_id:
                    raise IncompatibleResume(
                        f"incompatible resume in {self.workdir}; use a fresh work/output "
                        "directory for changed inputs, settings or implementation"
                    )
                self.previous = old
                self.record["phases"] = old.get("phases", [])
            else:
                leftovers = [p for p in self.workdir.iterdir() if p.name != ".run.lock"]
                if leftovers:
                    raise ValueError(
                        f"unowned work directory {self.workdir}; use a fresh workdir "
                        "rather than trusting legacy recovery artifacts"
                    )
            for output in sorted(self.outputs):
                lock_name = object_digest(str(output.resolve()))[:24]
                self._outputs_lock.enter_context(
                    exclusive_lock(output.parent / f".hi2-output-{lock_name}.lock")
                )
                owner = output / ".hi2-owner.json"
                if output.exists() and (not output.is_dir() or any(output.iterdir())):
                    if not owner.is_file():
                        raise ValueError(f"unowned output already exists: {output}")
                    stored = json.loads(owner.read_text())
                    if stored.get("run_id") != self.run_id:
                        raise ValueError(f"output belongs to a different run: {output}")
            # Check every destination before claiming any of them.
            for output in self.outputs:
                output.mkdir(parents=True, exist_ok=True)
                atomic_json(
                    output / ".hi2-owner.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": self.run_id,
                        "workdir": str(self.workdir.resolve()),
                    },
                )
            self.record["started_at"] = time.time()
            self.record["updated_at"] = time.time()
            atomic_json(self.path, self.record)
            return self
        except BaseException:
            self._outputs_lock.close()
            self._lock.__exit__(None, None, None)
            self._lock = None
            raise

    def phase(self, phase: str, status: str, **details) -> None:
        self.record["phases"].append({"phase": phase, "status": status, **details})
        self.record["phase_event_count"] = self.record.get("phase_event_count", 0) + 1
        self.record["phases"] = self.record["phases"][-256:]
        self.record["updated_at"] = time.time()
        atomic_json(self.path, self.record)

    def complete(self, result: dict) -> None:
        self.record.update(state="succeeded", result=result, updated_at=time.time())
        for output in self.outputs:
            marker = output / "hi2_run.json"
            self._publication_backups[marker] = (
                marker.read_bytes() if marker.exists() else None
            )
            atomic_json(marker, self.record)
        atomic_json(self.path, self.record)

    def __exit__(self, kind, exc, tb):
        try:
            if exc is not None:
                for marker, original in self._publication_backups.items():
                    if original is None:
                        marker.unlink(missing_ok=True)
                    else:
                        with atomic_file(marker) as stream:
                            stream.write(original)
                self.record.update(
                    state="interrupted"
                    if isinstance(exc, KeyboardInterrupt)
                    else "failed",
                    error={"type": type(exc).__name__, "message": str(exc)},
                    updated_at=time.time(),
                )
                atomic_json(self.path, self.record)
            elif self.record["state"] != "succeeded":
                self.record.update(state="interrupted", updated_at=time.time())
                atomic_json(self.path, self.record)
        finally:
            self._outputs_lock.close()
            if self._lock is not None:
                self._lock.__exit__(kind, exc, tb)
        return False
