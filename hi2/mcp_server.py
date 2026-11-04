# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Optional local MCP transport; compute and ownership belong to JobStore.

Importing this module needs only the standard library. The executable reserves
stdout before importing the SDK or any numerical/model code.
"""

import argparse
import base64
import dataclasses
import functools
import itertools
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, get_type_hints

SDK_VERSION = "2.3.0"
MAX_RESOURCE_BYTES = 32768
MAX_PAGE_BYTES = 16384
MAX_EVENTS = 16
MAX_CURSOR = 2**53 - 1
_JOB_ID = re.compile(r"job_[0-9a-f]{32}\Z")
_ARTIFACT_ID = re.compile(r"artifact:[0-9a-f]{32}\Z")
_ARTIFACT_FILES = {
    "manifest": ("hi2_run.json", "hi2_portable_manifest.json"),
    "report": ("hi2_report.json",),
    "quality": ("hi2_quality.json",),
}
JobState = Literal[
    "queued", "running", "cancelling", "succeeded", "failed", "cancelled", "interrupted"
]
Operation = Literal["quantize", "quality", "validate", "portable"]


@dataclasses.dataclass
class ResourceReference:
    name: str
    uri: str


@dataclasses.dataclass
class JobHandle:
    schema_version: Literal[1]
    job_id: str
    state: JobState
    operation: Operation
    created_at: float
    updated_at: float
    request_digest: str
    cancellation: Literal["requested", "cancelled", "already_finished"] | None
    result: dict[str, Any] | None
    error: dict[str, Any] | None
    resources: list[ResourceReference]


@dataclasses.dataclass
class Inspection:
    schema_version: Literal[1]
    resource_id: str
    kind: Literal["model", "artifact"]
    summary: dict[str, Any]
    resources: list[ResourceReference]


@dataclasses.dataclass
class PlanSummary:
    schema_version: Literal[1]
    summary: dict[str, Any]
    output_ids: list[str]


def _compact(value) -> dict | None:
    """Bound previews, without embedding requests, tensor reports or full logs."""
    if value is None:
        return None
    remaining = 32

    def visit(item, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 3:
            return "[omitted]"
        if isinstance(item, str):
            return item[:256] + ("[truncated]" if len(item) > 256 else "")
        if isinstance(item, dict):
            return {
                str(k)[:64]: visit(v, depth + 1)
                for k, v in itertools.islice(item.items(), 12)
            }
        if isinstance(item, (list, tuple)):
            return [visit(v, depth + 1) for v in item[:12]]
        if isinstance(item, (int, float, bool)) or item is None:
            return item
        return "[omitted]"

    preview = visit(value)
    if not isinstance(preview, dict):
        preview = {"value": preview}
    if len(json.dumps(preview, ensure_ascii=True).encode()) > 4096:
        return {"omitted": True, "reason": "Read the bounded evidence resources."}
    return preview


def job_handle(record: dict) -> JobHandle:
    """Shared ordinary/Tasks projection of the same durable job snapshot."""
    job_id = record["job_id"]
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError("invalid job identifier")
    cancellation = record.get("cancellation")
    if cancellation is None and record.get("cancel_requested_at") is not None:
        cancellation = (
            "cancelled"
            if record["state"] == "cancelled"
            else "already_finished"
            if record["state"] in ("succeeded", "failed", "interrupted")
            else "requested"
        )
    return JobHandle(
        schema_version=1,
        job_id=job_id,
        state=record["state"],
        operation=record["operation"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        request_digest=record["request_digest"],
        cancellation=cancellation,
        result=_compact(record.get("result")),
        error=_compact(record.get("error")),
        resources=[
            ResourceReference(name, f"hi2://jobs/{job_id}/{name}")
            for name in ("status", "events", "log")
        ],
    )


def _json_resource(document) -> str:
    text = json.dumps(
        document, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    )
    if len(text.encode()) > MAX_RESOURCE_BYTES:
        raise ValueError("resource page exceeds the content limit")
    return text


def _page_bounds(cursor: int, limit: int, maximum: int):
    if type(cursor) is not int or not 0 <= cursor <= MAX_CURSOR:
        raise ValueError("cursor must be a nonnegative safe integer")
    if type(limit) is not int or not 1 <= limit <= maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")


def _file_page(directory: Path, name: str, cursor: int, limit: int) -> dict:
    """Open only an adapter-selected basename, refusing symlinks and devices."""
    _page_bounds(cursor, limit, MAX_PAGE_BYTES)
    parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        descriptor = os.open(
            name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
    finally:
        os.close(parent)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("resource must be a regular, unaliased file")
        data = b""
        if cursor < info.st_size:
            stream.seek(cursor)
            data = stream.read(limit)
    return {
        "schema_version": 1,
        "encoding": "base64",
        "data": base64.b64encode(data).decode("ascii"),
        "cursor": cursor,
        "next_cursor": cursor + len(data),
        "size_bytes": info.st_size,
        "eof": cursor + len(data) >= info.st_size,
    }


def _recipe_model(ConfigDict, Field, create_model):
    """Reuse service dataclasses, excluding application-owned paths and tensors."""
    from .calibrate import Config
    from .pipeline import RunConfig
    from .qad import QadConfig

    configuration = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
        validate_default=True,
    )
    path_type = Annotated[str, Field(min_length=1, max_length=4096)]
    positive = {
        "calib_samples",
        "seq_len",
        "sens_seqs",
        "sens_len",
        "sens_batch",
        "limit_layers",
        "samples",
        "batch",
        "val",
        "final_batch",
        "evals",
        "final_evals",
        "kl_tokens",
    }
    enums = {
        "rotation": Literal["on", "off", "auto"],
        "rounding": Literal["gptq", "nearest"],
        "effort": Literal["fast", "normal", "max"],
        "sensitivity": Literal["none", "probe"],
        "moe_layout": Literal["per-expert", "stacked"],
        "compute_dtype": Literal["float32", "float16", "bfloat16"],
        "sens_dtype": Literal["auto", "float32", "float16", "bfloat16"],
        "dtype": Literal["float32", "float16", "bfloat16"],
        "buffer_dtype": Literal["auto", "float32", "bfloat16"],
        "sampling": Literal["random", "balanced", "head"],
        "latent_units": Literal["abs", "step"],
        "init": Literal["weights", "ptq"],
    }

    def model(cls, overrides=None, omit=()):
        definitions = {}
        hints = get_type_hints(cls)
        for item in dataclasses.fields(cls):
            name = item.name
            if name in omit:
                continue
            annotation = (overrides or {}).get(name, enums.get(name, hints[name]))
            constraints = {}
            if annotation in (int, int | None) and name != "seed":
                constraints["ge"] = 1 if name in positive else 0
            elif annotation in (float, float | None):
                constraints["ge"] = 1 if name == "max_error_ratio" else 0
            elif annotation in (str, str | None):
                constraints["max_length"] = 4096
            if item.default_factory is not dataclasses.MISSING:
                default = item.default_factory()
                if dataclasses.is_dataclass(default):
                    default = json.loads(json.dumps(dataclasses.asdict(default)))
            elif item.default is not dataclasses.MISSING:
                default = (
                    list(item.default)
                    if isinstance(item.default, tuple)
                    else item.default
                )
            else:
                default = ...
            if name == "candidates":
                default = [list(pair) for pair in default]
            definitions[name] = (annotation, Field(default=default, **constraints))
        return create_model(
            cls.__name__ + "Input", __config__=configuration, **definitions
        )

    pair = Annotated[list[int], Field(min_length=2, max_length=2)]
    encoder = model(
        Config,
        {
            "candidates": Annotated[list[pair], Field(min_length=1, max_length=32)],
            "smooth_search": Annotated[list[float], Field(max_length=32)],
        },
    )
    qad = model(QadConfig)
    run = model(
        RunConfig,
        {
            "src": path_type,
            "dst": path_type,
            "cfg": encoder,
            "qad": qad | None,
            "calib": path_type
            | Annotated[list[path_type], Field(min_length=1, max_length=32)]
            | None,
            "devices": Annotated[
                list[
                    Annotated[
                        str, Field(pattern=r"^(cpu|cuda(:[0-9]+)?)$", max_length=32)
                    ]
                ],
                Field(min_length=1, max_length=32),
            ],
            "max_error_ratios": Annotated[
                list[Annotated[float, Field(ge=1)]], Field(min_length=1, max_length=16)
            ]
            | None,
        },
        omit={"workdir", "cache_dir", "calib_ids"},
    )
    return create_model(
        "Recipe",
        __config__=configuration,
        schema_version=(Annotated[int, Field(ge=1, le=1)], ...),
        config=(run, ...),
    )


def create_server(policy, jobs, *, enable_tasks=False):
    """Build an MCPServer over operator capabilities and one private job registry."""
    try:
        from importlib.metadata import version

        from mcp.server import MCPServer
        from mcp.server.mcpserver.exceptions import ResourceError, ToolError
        from mcp.server.mcpserver.tools import Tool
        from mcp.types import ToolAnnotations
        from pydantic import BaseModel, ConfigDict, Field, create_model
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional adapter with: python -m pip install 'hi2[mcp]' (mcp==2.3.0)."
        ) from exc
    if version("mcp") != SDK_VERSION:
        raise RuntimeError(
            "hi2-mcp requires mcp==2.3.0; reinstall with python -m pip install 'hi2[mcp]'."
        )

    from . import service
    from .workspace import WorkspacePolicyError

    job_root = policy.state_root / "jobs"
    info = job_root.lstat()
    if (
        job_root.is_symlink()
        or jobs.root != job_root
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("JobStore must use the private 0700 state_root/jobs directory")

    Recipe = _recipe_model(ConfigDict, Field, create_model)
    PathValue = Annotated[str, Field(min_length=1, max_length=4096)]
    JobId = Annotated[str, Field(pattern=r"^job_[0-9a-f]{32}$")]
    IdempotencyKey = Annotated[str, Field(min_length=1, max_length=512)]

    class Evaluation(BaseModel):
        model_config = ConfigDict(
            extra="forbid", strict=True, allow_inf_nan=False, hide_input_in_errors=True
        )
        src: PathValue
        dst: PathValue
        text: PathValue
        excluded: Annotated[list[PathValue], Field(max_length=32)] = Field(
            default_factory=list
        )
        max_kld: Annotated[float, Field(ge=0)] | None = None
        max_ppl_ratio: Annotated[float, Field(gt=0)] | None = None
        samples: Annotated[int, Field(ge=1, le=4096)] = 8
        seq_len: Annotated[int, Field(ge=2, le=131072)] = 2048
        device: Annotated[
            str, Field(pattern=r"^(cpu|cuda(:[0-9]+)?)$", max_length=32)
        ] = "cpu"
        dtype: Literal["float32", "float16", "bfloat16"] = "float32"

    tools = []

    def tool(*, readonly=False):
        def decorate(fn):
            @functools.wraps(fn)
            def guarded(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except WorkspacePolicyError as exc:
                    raise ToolError(f"{exc.code}: {exc}") from None
                except KeyError:
                    raise ToolError(
                        "unknown_job: no job is registered with that ID"
                    ) from None
                except (ValueError, OSError, SystemExit):
                    raise ToolError(
                        "invalid_request: check the recipe, resource roots, idempotency key and local input files"
                    ) from None

            registered = Tool.from_function(
                guarded,
                annotations=ToolAnnotations(
                    read_only_hint=readonly, open_world_hint=False
                ),
            )
            # SDK 2.3.0 defaults to ignoring unknown function arguments. Tighten
            # this registration, not global SDK state, and publish the same schema.
            arguments = registered.fn_metadata.arg_model
            arguments.model_config.update(
                extra="forbid", strict=True, hide_input_in_errors=True
            )
            arguments.model_rebuild(force=True)
            registered.parameters = arguments.model_json_schema()
            tools.append(registered)
            return guarded

        return decorate

    def artifact_references(identifier, path):
        names = [
            name
            for name, files in _ARTIFACT_FILES.items()
            if any((path / f).is_file() for f in files)
        ]
        return [
            ResourceReference(name, f"hi2://artifacts/{identifier}/{name}")
            for name in [*names, "support"]
        ]

    @tool(readonly=True)
    def hi2_inspect(
        path: PathValue, kind: Literal["model", "artifact"] = "model"
    ) -> Inspection:
        """Inspect safetensors metadata only; artifact validation is a separate durable job."""
        resolved = (
            policy.resolve_model(path)
            if kind == "model"
            else policy.resolve_artifact(path)
        )
        facts = service.inspect_model(resolved)
        identifier = policy.register(kind, resolved)
        facts.pop("path", None)
        return Inspection(
            1,
            identifier,
            kind,
            _compact(facts),
            artifact_references(identifier, resolved) if kind == "artifact" else [],
        )

    @tool(readonly=True)
    def hi2_plan(recipe: Recipe) -> PlanSummary:
        """Validate and plan without quantizing or creating an output checkpoint."""
        normalized = policy.validate_recipe(
            recipe.model_dump(mode="json", exclude_unset=True)
        )
        planned = service.plan_run(service.config_from_dict(normalized))
        outputs = [
            policy.register("output", policy.resolve_output(p))
            for p in planned["outputs"]
        ]
        selected = {
            key: planned[key]
            for key in (
                "phases",
                "rounding",
                "target_bpw",
                "max_error_ratio",
                "devices",
                "layers",
                "moe_layout",
                "native_runtime",
                "resource_estimates",
            )
            if key in planned
        }
        return PlanSummary(1, _compact(selected), outputs)

    @tool()
    def hi2_quantize(recipe: Recipe, idempotency_key: IdempotencyKey) -> JobHandle:
        """Submit quantization and return immediately with a reconnectable job ID."""
        normalized = policy.validate_recipe(
            recipe.model_dump(mode="json", exclude_unset=True)
        )
        return job_handle(jobs.submit("quantize", normalized, idempotency_key))

    @tool()
    def hi2_evaluate(request: Evaluation, idempotency_key: IdempotencyKey) -> JobHandle:
        """Submit held-out dense numerical quality evaluation, not a native runtime benchmark."""
        values = request.model_dump(mode="json")
        values["src"] = str(policy.resolve_model(values["src"]))
        values["dst"] = str(policy.resolve_artifact(values["dst"]))
        values["text"] = str(policy.resolve_calibration(values["text"]))
        values["excluded"] = [
            str(policy.resolve_calibration(p)) for p in values["excluded"]
        ]
        # A manifest is data, not authority to read calibration paths outside
        # the operator's roots when the worker checks held-out contamination.
        manifest_path = Path(values["dst"]) / "hi2_run.json"
        if manifest_path.exists():
            checked = policy.resolve_artifact(str(manifest_path))
            with checked.open("rb") as stream:
                manifest_bytes = stream.read(4 * 1024 * 1024 + 1)
            if len(manifest_bytes) > 4 * 1024 * 1024:
                raise ValueError(
                    "artifact run manifest exceeds the metadata size limit"
                )
            manifest = json.loads(manifest_bytes)
            for item in manifest.get("calibration", {}).get("paths", []):
                policy.resolve_calibration(item["path"])
        return job_handle(jobs.submit("quality", values, idempotency_key))

    @tool(readonly=True)
    def hi2_status(job_id: JobId) -> JobHandle:
        """Read compact durable status after reconnect; use resources to page evidence."""
        return job_handle(jobs.status(job_id, limit=1))

    @tool()
    def hi2_cancel(job_id: JobId) -> JobHandle:
        """Request cooperative cancellation; cancelling does not mean the worker exited."""
        return job_handle(jobs.cancel(job_id))

    @tool()
    def hi2_validate(path: PathValue, idempotency_key: IdempotencyKey) -> JobHandle:
        """Submit structural artifact validation; never validate synchronously in the transport."""
        resolved = policy.resolve_artifact(path)
        return job_handle(
            jobs.submit("validate", {"path": str(resolved)}, idempotency_key)
        )

    @tool()
    def hi2_portable(
        src: PathValue,
        out: PathValue,
        idempotency_key: IdempotencyKey,
        assets_from: PathValue | None = None,
    ) -> JobHandle:
        """Submit portable artifact export with rooted local tokenizer assets."""
        request = {
            "src": str(policy.resolve_artifact(src)),
            "out": str(policy.resolve_output(out)),
        }
        if assets_from is not None:
            request["assets_from"] = str(policy.resolve_model(assets_from))
        return job_handle(jobs.submit("portable", request, idempotency_key))

    extensions = []
    if enable_tasks:
        from .mcp_tasks import JobTasks

        extensions.append(JobTasks(jobs))
    server = MCPServer(
        "hi2",
        version="1",
        tools=tools,
        extensions=extensions,
        instructions="Local rooted checkpoints only. Plan first. Long operations return durable jobs. "
        "Cancellation is cooperative. Numerical quality is not native runtime support.",
    )

    def resource(uri):
        def decorate(fn):
            @functools.wraps(fn)
            def guarded(*args, **kwargs):
                try:
                    return fn(*args, **kwargs)
                except (WorkspacePolicyError, ValueError, OSError, KeyError):
                    raise ResourceError(
                        "resource_denied: unknown ID/type, unavailable evidence or invalid page bounds"
                    ) from None

            server.resource(uri, mime_type="application/json")(guarded)
            return guarded

        return decorate

    def snapshot(job_id, **kwargs):
        if not _JOB_ID.fullmatch(job_id):
            raise ValueError("invalid job ID")
        return jobs.status(job_id, **kwargs)

    @resource("hi2://jobs/{job_id}/status")
    def job_status(job_id: str) -> str:
        """Compact, reconnectable durable job status; contains no filesystem log paths."""
        return _json_resource(dataclasses.asdict(job_handle(snapshot(job_id, limit=1))))

    @resource("hi2://jobs/{job_id}/events{?cursor,limit}")
    def job_events(job_id: str, cursor: int = 0, limit: int = MAX_EVENTS) -> str:
        """Page retained events by monotonic event cursor (at most 16 per page)."""
        _page_bounds(cursor, limit, MAX_EVENTS)
        record = snapshot(job_id, cursor=cursor, limit=limit)
        page = {
            "schema_version": 1,
            "job_id": job_id,
            "cursor": cursor,
            "events": [],
            "next_cursor": cursor,
            "oldest_cursor": record.get("oldest_cursor"),
            "last_cursor": record.get("last_cursor"),
            "events_truncated": record.get("events_truncated", False),
        }
        for event in record["events"]:
            candidate = {
                **page,
                "events": [*page["events"], event],
                "next_cursor": event["cursor"],
            }
            if (
                len(json.dumps(candidate, ensure_ascii=True).encode())
                > MAX_RESOURCE_BYTES
            ):
                if not page["events"]:
                    page["events"] = [
                        {
                            "cursor": event["cursor"],
                            "omitted": True,
                            "reason": "Event exceeds resource cap; read the log resource.",
                        }
                    ]
                    page["next_cursor"] = event["cursor"]
                break
            page = candidate
        return _json_resource(page)

    @resource("hi2://jobs/{job_id}/log{?cursor,limit}")
    def job_log(job_id: str, cursor: int = 0, limit: int = MAX_PAGE_BYTES) -> str:
        """Page the private worker log as exact base64 bytes; cursor is a byte offset."""
        record = snapshot(job_id, limit=1)
        _page_bounds(cursor, limit, MAX_PAGE_BYTES)
        try:
            page = _file_page(job_root / job_id, "worker.log", cursor, limit)
        except FileNotFoundError:
            if record["state"] not in ("queued", "cancelling", "cancelled"):
                raise
            page = {
                "schema_version": 1,
                "encoding": "base64",
                "data": "",
                "cursor": cursor,
                "next_cursor": cursor,
                "size_bytes": 0,
                "eof": True,
                "available": False,
            }
        return _json_resource({"job_id": job_id, **page})

    @resource("hi2://artifacts/{artifact_id}/{kind}{?cursor,limit}")
    def artifact_evidence(
        artifact_id: str, kind: str, cursor: int = 0, limit: int = MAX_PAGE_BYTES
    ) -> str:
        """Read only registered manifest/report/quality/support evidence, never arbitrary paths."""
        if not _ARTIFACT_ID.fullmatch(artifact_id) or kind not in (
            *_ARTIFACT_FILES,
            "support",
        ):
            raise ValueError("invalid artifact resource")
        _page_bounds(cursor, limit, MAX_PAGE_BYTES)
        path = policy.resolve_artifact(artifact_id)
        if not path.is_dir():
            raise ValueError("artifact must be a checkpoint directory")
        if kind == "support":
            if cursor != 0:
                raise ValueError("support is a single compact document")
            return _json_resource(
                {
                    "schema_version": 1,
                    "artifact_id": artifact_id,
                    "evidence": [
                        r.name
                        for r in artifact_references(artifact_id, path)
                        if r.name != "support"
                    ],
                    "validation": "Use hi2_validate; metadata presence is not validation.",
                    "quality_context": "dense_numerical",
                    "native_runtime": "not_measured",
                    "sdk": SDK_VERSION,
                    "resource_page_bytes": MAX_PAGE_BYTES,
                }
            )
        for name in _ARTIFACT_FILES[kind]:
            if (path / name).is_file():
                checked = policy.resolve_artifact(str(path / name))
                if checked.parent != path:
                    raise ValueError(
                        "artifact evidence must remain inside its checkpoint"
                    )
                return _json_resource(
                    {
                        "artifact_id": artifact_id,
                        "kind": kind,
                        **_file_page(path, name, cursor, limit),
                    }
                )
        raise ValueError("artifact evidence is not available")

    return server


def _reserve_stdout():
    """Permanently move the protocol wire off fd 1, including at interpreter exit."""
    descriptor = os.dup(1)
    os.set_inheritable(descriptor, False)
    try:
        os.dup2(2, 1)
    except BaseException:
        os.close(descriptor)
        raise
    sys.stdout = sys.stderr
    sys.__stdout__ = sys.stderr
    return os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)


async def _serve_stdio(server, wire):
    import anyio
    from mcp.server.stdio import stdio_server

    # MCPServer.run_stdio_async has no explicit-stream option in pinned 2.3.0.
    # Use its same runner with an explicit wire so the SDK never restores fd 1.
    async with stdio_server(stdout=anyio.wrap_file(wire)) as (reader, writer):
        await server._lowlevel_server.run(
            reader, writer, server._lowlevel_server.create_initialization_options()
        )


def main(argv=None):
    wire = _reserve_stdout()
    try:
        parser = argparse.ArgumentParser(
            description="Root-scoped local hi2 MCP server (stdio only)."
        )
        parser.add_argument("--model-root", type=Path, action="append", default=[])
        parser.add_argument(
            "--calibration-root", type=Path, action="append", default=[]
        )
        parser.add_argument("--output-root", type=Path, action="append", default=[])
        parser.add_argument("--state-root", type=Path, required=True)
        parser.add_argument(
            "--tasks",
            action="store_true",
            help="Enable negotiated Tasks; ordinary jobs remain the default.",
        )
        args = parser.parse_args(argv)
        from .jobs import JobStore
        from .workspace import WorkspacePolicy

        policy = WorkspacePolicy(
            args.model_root, args.calibration_root, args.output_root, args.state_root
        )
        server = create_server(
            policy, JobStore(policy.state_root / "jobs"), enable_tasks=args.tasks
        )
        import anyio

        anyio.run(_serve_stdio, server, wire)
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        print(f"hi2-mcp: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    finally:
        wire.close()


if __name__ == "__main__":
    main()
