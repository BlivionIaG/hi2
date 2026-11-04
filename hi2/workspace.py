# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Local operator capabilities for adapters; this module is not a transport.

Roots are operator configuration, never tool arguments. Registrations contain
canonical paths, not credentials. ``validate_recipe`` accepts and returns the
service's ``{schema_version: 1, config: {...}}`` document without executing a
model. Its work directories and shared cache are allocated beneath state_root;
only those exact paths can be replayed in a normalized recipe.

State must be owned by the current operator, mode 0700, and never receive user
uploads. Cache files can contain executable pickle: containment, ownership
markers and digests do NOT make a malicious pickle safe. The operator must also
prevent untrusted processes from changing roots/assets between validation and
use; this is not a filesystem sandbox or a multi-user authorization system.

Model assets may link to another explicitly trusted model root. Output and
artifact paths cannot expose private state or overwrite any configured input
root. All public rejections use WorkspacePolicyError with a stable ``code`` and
messages that do not echo submitted fields, values, paths or underlying errors.
"""

import contextlib
import dataclasses
import json
import math
import os
import stat
import types
import typing
import uuid
from pathlib import Path


class WorkspacePolicyError(ValueError):
    """Adapter-safe denial; ``code`` identifies the reason without user data."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _deny(code: str, message: str):
    raise WorkspacePolicyError(code, message) from None


def _absolute(value: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        _deny("invalid_path", "a nonempty absolute path or registered ID is required")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        _deny("invalid_path", "relative paths and parent traversal are not permitted")
    return path


def _canonical(path: Path, *, exists: bool = False) -> Path:
    try:
        return path.resolve(strict=exists)
    except (OSError, RuntimeError, ValueError):
        _deny("invalid_path", "path cannot be resolved safely")


def _beneath(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path.is_relative_to(root) for root in roots)


def _overlaps(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


_FORBIDDEN = {
    "token",
    "secret",
    "secrets",
    "password",
    "passwd",
    "credential",
    "credentials",
    "auth",
    "authorization",
    "apikey",
    "accesskey",
    "privatekey",
    "remotecode",
    "trustremotecode",
    "shell",
    "command",
    "commands",
    "function",
    "functions",
    "script",
    "code",
    "exec",
    "eval",
    "argv",
    "env",
    "environment",
}
_PICKLE_SUFFIXES = {".pt", ".pth", ".pkl", ".pickle", ".bin"}
_KINDS = {"model", "calibration", "output", "artifact"}


def _json_recipe(value) -> None:
    """Reject executable objects and capability keys before importing compute code.

    ``eval`` alone is executable, but a compound evaluation setting such as
    QadConfig.eval_every is not. Compound names still need to pass the service's
    dataclass-derived schema and declared types; this filter grants no fields.
    """
    if type(value) is dict:
        for key, child in value.items():
            if not isinstance(key, str):
                _deny("invalid_recipe", "recipe object keys must be strings")
            words = key.lower().replace("-", "_").split("_")
            if "".join(words) in _FORBIDDEN or (set(words) - {"eval"}) & _FORBIDDEN:
                _deny(
                    "forbidden_field",
                    "credentials and executable capabilities are operator-only",
                )
            _json_recipe(child)
    elif type(value) is list:
        for child in value:
            _json_recipe(child)
    elif type(value) is float:
        if not math.isfinite(value):
            _deny("invalid_recipe", "recipe numbers must be finite")
    elif value is not None and type(value) not in (str, int, bool):
        _deny("invalid_recipe", "recipe values must be JSON data")


def _matches_type(value, expected) -> bool:
    origin, args = typing.get_origin(expected), typing.get_args(expected)
    if origin in (types.UnionType, typing.Union):
        return any(_matches_type(value, item) for item in args)
    if origin in (list, tuple):
        if not isinstance(value, origin):
            return False
        if origin is list or (len(args) == 2 and args[1] is Ellipsis):
            return all(_matches_type(item, args[0]) for item in value)
        return len(value) == len(args) and all(
            _matches_type(item, kind) for item, kind in zip(value, args)
        )
    if expected is float:
        return type(value) in (float, int)
    if expected in (int, str, bool):
        return type(value) is expected
    return isinstance(value, expected)


def _typed_config(record) -> None:
    """Dataclass construction alone does not enforce the service's field types."""
    for field in dataclasses.fields(record):
        value = getattr(record, field.name)
        if field.name == "calib":  # RunConfig also accepts multiple input files.
            valid = value is None or (
                isinstance(value, list) and all(isinstance(p, Path) for p in value)
            )
        elif dataclasses.is_dataclass(value):
            _typed_config(value)
            continue
        else:
            valid = _matches_type(value, field.type)
        if not valid:
            _deny("invalid_recipe", "recipe settings do not match their declared types")


class WorkspacePolicy:
    """Operator-root policy, with persistent kind-scoped IDs.

    ``register`` accepts model/calibration/output/artifact, returning an opaque
    ``kind:UUID`` string. Model and calibration paths must exist; outputs may be
    new directories; artifacts must exist beneath output roots and cannot be
    pickle/cache inputs. Registering never grants access beyond configured roots.

    Error codes: invalid_path, unconfigured_root, path_escape, invalid_kind,
    unknown_id, id_kind, invalid_state, state_busy, invalid_recipe,
    forbidden_field, input_output_overlap, untrusted_cache.
    """

    def __init__(
        self,
        model_roots: list[Path],
        calibration_roots: list[Path],
        output_roots: list[Path],
        state_root: Path,
    ):
        def roots(values):
            result = []
            for value in values:
                path = _canonical(_absolute(str(value)))
                if path.exists() and not path.is_dir():
                    _deny("invalid_path", "configured roots must be directories")
                if path not in result:
                    result.append(path)
            return tuple(result)

        self.model_roots = roots(model_roots)
        self.calibration_roots = roots(calibration_roots)
        self.output_roots = roots(output_roots)
        original = _absolute(str(state_root))
        if original.is_symlink():
            _deny("invalid_state", "operator state must not be a symlink")
        self.state_root = _canonical(original)
        try:
            self.state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError:
            _deny("invalid_state", "operator state cannot be created")
        self._check_state()

    def _check_state(self) -> None:
        try:
            info = self.state_root.lstat()
            valid = (
                stat.S_ISDIR(info.st_mode)
                and info.st_uid == os.getuid()
                and stat.S_IMODE(info.st_mode) == 0o700
                and _canonical(self.state_root) == self.state_root
            )
        except OSError:
            valid = False
        if not valid:
            _deny(
                "invalid_state",
                "operator state must be a private, operator-owned 0700 directory",
            )

    def _state_entry(self, path: Path) -> None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return
        except OSError:
            _deny("invalid_state", "operator state cannot be inspected")
        if (
            info.st_uid != os.getuid()
            or info.st_mode & 0o022
            or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode))
            or (stat.S_ISREG(info.st_mode) and info.st_nlink != 1)
        ):
            _deny("invalid_state", "operator state contains an unsafe entry")

    @contextlib.contextmanager
    def _registry(self):
        from .runtime import exclusive_lock

        self._check_state()
        lock = self.state_root / ".workspace.lock"
        self._state_entry(lock)
        try:
            with exclusive_lock(lock):
                yield self._read_registry()
        except WorkspacePolicyError:
            raise
        except RuntimeError:
            _deny("state_busy", "workspace registrations are busy")
        except OSError:
            _deny("invalid_state", "workspace registrations cannot be accessed")

    def _read_registry(self) -> dict:
        path = self.state_root / "workspace.json"
        self._state_entry(path)
        if not path.exists():
            return {"schema_version": 1, "registrations": {}, "managed": []}
        try:
            record = json.loads(path.read_text())
            if (
                type(record) is not dict
                or set(record) != {"schema_version", "registrations", "managed"}
                or type(record["schema_version"]) is not int
                or record["schema_version"] != 1
                or type(record["registrations"]) is not dict
                or type(record["managed"]) is not list
                or any(not isinstance(p, str) for p in record["managed"])
            ):
                raise ValueError
            for identifier, item in record["registrations"].items():
                if (
                    type(item) is not dict
                    or set(item) != {"kind", "path"}
                    or item["kind"] not in _KINDS
                    or not identifier.startswith(item["kind"] + ":")
                    or not isinstance(item["path"], str)
                ):
                    raise ValueError
            return record
        except (OSError, UnicodeError, ValueError, TypeError):
            _deny("invalid_state", "workspace registration data is invalid")

    def _resolve(self, kind: str, value: str) -> Path:
        self._check_state()
        if not isinstance(value, str):
            _deny("invalid_path", "a path or registered ID must be a string")
        if not value.startswith("/") and ":" in value:
            with self._registry() as registry:
                item = registry["registrations"].get(value)
                if item is None:
                    _deny("unknown_id", "workspace ID is not registered")
                if item["kind"] != kind:
                    _deny("id_kind", "workspace ID has the wrong resource kind")
                value = item["path"]
        roots = {
            "model": self.model_roots,
            "calibration": self.calibration_roots,
            "output": self.output_roots,
            "artifact": self.output_roots,
        }[kind]
        if not roots:
            _deny("unconfigured_root", "this resource kind has no configured roots")
        path = _canonical(_absolute(value), exists=kind != "output")
        self._check_location(path, roots, kind)
        try:
            exists = path.exists()
            if kind in ("model", "output") and exists and not path.is_dir():
                _deny("invalid_path", "model and output paths must be directories")
            if kind == "calibration" and not path.is_file():
                _deny("invalid_path", "calibration paths must be regular files")
            if kind == "output" and path in roots:
                _deny(
                    "input_output_overlap",
                    "an output must be below its configured root",
                )
            if exists:
                self._check_tree(path, roots, kind)
        except OSError:
            _deny("invalid_path", "resource cannot be inspected safely")
        return path

    def _check_location(self, path: Path, roots: tuple[Path, ...], kind: str) -> None:
        if not _beneath(path, roots):
            _deny("path_escape", "path escapes the configured resource roots")
        if _overlaps(path, self.state_root):
            _deny("untrusted_cache", "private application state is not a tool resource")
        if kind in ("output", "artifact") and any(
            _overlaps(path, root) for root in self.model_roots + self.calibration_roots
        ):
            _deny(
                "input_output_overlap", "output resources must not overlap input roots"
            )

    def _check_tree(self, path: Path, roots: tuple[Path, ...], kind: str) -> None:
        pending, visited = [path], set()
        while pending:
            entry = pending.pop()
            current = _canonical(entry, exists=True)
            # Quantization links ancillary source assets (tokenizers, templates,
            # README files). These remain read-only model capabilities, never
            # output authority; weights, config and run evidence cannot escape.
            if (
                kind in ("output", "artifact")
                and entry != path
                and entry.is_symlink()
                and _beneath(current, self.model_roots)
                and not entry.name.startswith((".", "hi2_"))
                and entry.name not in {"config.json", "model.safetensors.index.json"}
                and entry.suffix.lower() not in _PICKLE_SUFFIXES | {".safetensors"}
            ):
                self._check_tree(current, self.model_roots, "model")
                continue
            self._check_location(current, roots, kind)
            try:
                info = current.stat()
                if stat.S_ISDIR(info.st_mode):
                    if current in visited:
                        continue
                    visited.add(current)
                    pending.extend(current.iterdir())
                elif not stat.S_ISREG(info.st_mode):
                    _deny(
                        "invalid_path",
                        "resource trees may contain only directories and regular files",
                    )
                elif kind == "output" and info.st_nlink != 1:
                    _deny(
                        "input_output_overlap",
                        "output files must not alias other files",
                    )
                elif kind != "output" and current.suffix.lower() in _PICKLE_SUFFIXES:
                    _deny(
                        "untrusted_cache",
                        "pickle and cache payloads are not tool resources",
                    )
            except OSError:
                _deny("invalid_path", "resource tree cannot be inspected safely")

    def register(self, kind: str, path: Path) -> str:
        """Persist a checked canonical path atomically; repeated registration is stable."""
        from .runtime import atomic_json

        if not isinstance(kind, str) or kind not in _KINDS:
            _deny("invalid_kind", "unsupported workspace resource kind")
        canonical = self._resolve(kind, str(path))
        item = {"kind": kind, "path": str(canonical)}
        with self._registry() as registry:
            for identifier, existing in registry["registrations"].items():
                if existing == item:
                    return identifier
            identifier = f"{kind}:{uuid.uuid4().hex}"
            registry["registrations"][identifier] = item
            atomic_json(self.state_root / "workspace.json", registry)
        return identifier

    def resolve_model(self, value: str) -> Path:
        return self._resolve("model", value)

    def resolve_calibration(self, value: str) -> Path:
        return self._resolve("calibration", value)

    def resolve_output(self, value: str) -> Path:
        return self._resolve("output", value)

    def resolve_artifact(self, value: str) -> Path:
        return self._resolve("artifact", value)

    def _managed(self, paths: list[Path]) -> None:
        from .runtime import atomic_json

        with self._registry() as registry:
            for path in paths:
                for part in reversed(path.relative_to(self.state_root).parents):
                    self._state_entry(self.state_root / part)
                self._state_entry(path)
                if path.exists() and not path.is_dir():
                    _deny(
                        "untrusted_cache",
                        "managed cache and work paths must be directories",
                    )
                if (
                    str(path) not in registry["managed"]
                    and path.exists()
                    and any(path.iterdir())
                ):
                    _deny(
                        "untrusted_cache",
                        "existing cache or work data was not created by this application",
                    )
                if path.exists():
                    for base, directories, files in os.walk(path, followlinks=False):
                        for name in directories + files:
                            self._state_entry(Path(base) / name)
            changed = False
            for path in paths:
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                path.mkdir(mode=0o700, exist_ok=True)
                if str(path) not in registry["managed"]:
                    registry["managed"].append(str(path))
                    changed = True
            if changed:
                atomic_json(self.state_root / "workspace.json", registry)

    def validate_recipe(self, document: dict) -> dict:
        """Validate/normalize JSON settings and allocate private cache/work paths.

        This enforces types and capabilities, not numerical feasibility or model
        compatibility; those remain the service's preflight responsibility. No
        models, tokenizers or pickle files are loaded here. No credentials are
        read from the environment, accepted as fields, or included in errors.
        """
        try:
            _json_recipe(document)
        except RecursionError:
            _deny("invalid_recipe", "recipe must be a finite JSON document")
        if (
            type(document) is not dict
            or set(document) != {"schema_version", "config"}
            or type(document["schema_version"]) is not int
            or document["schema_version"] != 1
            or type(document["config"]) is not dict
        ):
            _deny("invalid_recipe", "a version 1 service recipe is required")
        values = dict(document["config"])
        if not {"src", "dst"}.issubset(values):
            _deny("invalid_recipe", "recipe source and output are required")
        if values.get("calib_ids") is not None:
            _deny(
                "untrusted_cache",
                "calibration must use registered or rooted text files",
            )
        if "cfg" in values and type(values["cfg"]) is not dict:
            _deny("invalid_recipe", "quantization settings must be an object")
        if values.get("qad") is not None and type(values["qad"]) is not dict:
            _deny("invalid_recipe", "distillation settings must be an object")
        values["src"] = str(self.resolve_model(values["src"]))
        values["dst"] = str(self.resolve_output(values["dst"]))
        if values.get("calib") is not None:
            calibration = values["calib"]
            calibration = (
                calibration if isinstance(calibration, list) else [calibration]
            )
            if not calibration:
                _deny("invalid_recipe", "calibration file lists must not be empty")
            values["calib"] = [str(self.resolve_calibration(p)) for p in calibration]
        supplied = {key: values.pop(key, None) for key in ("workdir", "cache_dir")}
        values["workdir"] = str(self.state_root)

        from .runtime import object_digest
        from .service import config_dict, config_from_dict, output_paths

        try:
            rc = config_from_dict({"schema_version": 1, "config": values})
            _typed_config(rc)
            config = config_dict(rc)
            for output in output_paths(rc):
                if output != rc.dst:
                    self.resolve_output(str(output))
        except WorkspacePolicyError:
            raise
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            _deny("invalid_recipe", "recipe contains unknown or invalid settings")
        identity = {
            k: v for k, v in config.items() if k not in ("workdir", "cache_dir")
        }
        work = self.state_root / "work" / object_digest(identity)
        cache = self.state_root / "cache"
        for key, path in (("workdir", work), ("cache_dir", cache)):
            if supplied[key] is not None and supplied[key] != str(path):
                _deny(
                    "untrusted_cache",
                    "cache and work paths are assigned by the application",
                )
            config[key] = str(path)
        self._managed([work, cache])
        return {"schema_version": 1, "config": config}
