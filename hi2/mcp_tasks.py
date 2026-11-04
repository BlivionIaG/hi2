# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""The 2026-07-28 Tasks extension, projected over the existing JobStore.

Wire contract: https://github.com/modelcontextprotocol/ext-tasks/blob/main/
schema/2026-07-28/schema.ts . The SDK provides the Extension hooks and core wire
models; the extension-specific models below follow that immutable schema.

Install JobTasks only through create_server(..., enable_tasks=True). Both the
wire version and this request's client capability must permit Tasks. Ordinary
job tools remain usable without either opt-in. Task IDs are the durable job IDs;
there is no additional worker, task registry, session ownership, or task list.
The surrounding server is a local, single-operator workspace, not a multi-tenant
service. Possession of a job ID does not grant access to another registry.

Retention is unlimited (ttlMs: null): JobStore does not expire records or their
terminal result/error. Reconnecting requires the same operator-owned state root.
Keep that registry and its artifacts until their handles are no longer needed;
operator deletion removes the corresponding tasks. Never delete active state.
Polling is supported; input requests and task notifications are not emitted.

A failed/interrupted *job* is a completed tool call with isError=true, not a
failed Task: the spec reserves failed for JSON-RPC errors. The terminal tool
result uses the same JobHandle schema as the ordinary handler, including its
bounded result/error summary and references to the persisted full evidence.
Cancellation acknowledges intent only; cancelled comes solely from JobStore.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from typing import Any, Literal

from anyio import to_thread
from mcp import types
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.extension import Extension, MethodBinding
from mcp.server.mcpserver import require_client_extension
from mcp.shared.exceptions import MCPError
from pydantic import Field

from .jobs import JobStore

EXTENSION_ID = "io.modelcontextprotocol/tasks"
PROTOCOL_VERSION = "2026-07-28"
_TASK_TOOLS = frozenset({"hi2_quantize", "hi2_evaluate"})
TaskStatus = Literal["working", "input_required", "completed", "failed", "cancelled"]


class Task(types.Result):
    """Common Task fields, using the SDK's camelCase wire aliases."""

    task_id: str
    status: TaskStatus
    status_message: str | None = None
    created_at: str
    last_updated_at: str
    ttl_ms: int | None = Field(ge=0)
    poll_interval_ms: int | None = Field(default=None, ge=0)


class CreateTaskResult(Task):
    result_type: Literal["task"] = "task"


class WorkingTask(Task):
    result_type: Literal["complete"] = "complete"
    status: Literal["working"] = "working"


class InputRequiredTask(Task):
    result_type: Literal["complete"] = "complete"
    status: Literal["input_required"] = "input_required"
    input_requests: types.InputRequests


class CompletedTask(Task):
    result_type: Literal["complete"] = "complete"
    status: Literal["completed"] = "completed"
    result: dict[str, Any]


class FailedTask(Task):
    result_type: Literal["complete"] = "complete"
    status: Literal["failed"] = "failed"
    error: types.ErrorData


class CancelledTask(Task):
    result_type: Literal["complete"] = "complete"
    status: Literal["cancelled"] = "cancelled"


GetTaskResult = (
    WorkingTask | InputRequiredTask | CompletedTask | FailedTask | CancelledTask
)


class TaskParams(types.RequestParams):
    task_id: str


class UpdateTaskParams(TaskParams):
    input_responses: types.InputResponses


class GetTaskRequest(types.Request[TaskParams, Literal["tasks/get"]]):
    method: Literal["tasks/get"] = "tasks/get"
    params: TaskParams
    name_param = "taskId"


class CancelTaskRequest(types.Request[TaskParams, Literal["tasks/cancel"]]):
    method: Literal["tasks/cancel"] = "tasks/cancel"
    params: TaskParams
    name_param = "taskId"


class UpdateTaskRequest(types.Request[UpdateTaskParams, Literal["tasks/update"]]):
    method: Literal["tasks/update"] = "tasks/update"
    params: UpdateTaskParams
    name_param = "taskId"


def _wire(task: Task) -> dict[str, Any]:
    # The SDK dumps handler BaseModels with exclude_none=True. ttlMs is required
    # even when null, so hand it a dict with that field explicitly retained.
    return {
        **task.model_dump(by_alias=True, mode="json", exclude_none=True),
        "ttlMs": task.ttl_ms,
    }


def _task_fields(record: dict) -> dict:
    state = record["state"]
    if state in {"queued", "running", "cancelling"}:
        status = "working"
    elif state in {"succeeded", "failed", "interrupted"}:
        status = "completed"
    elif state == "cancelled":
        status = "cancelled"
    else:
        raise MCPError(code=types.INTERNAL_ERROR, message="Unknown durable job state")
    message = (
        "Cancellation requested; waiting for worker exit."
        if state == "cancelling"
        else f"Job {state}."
    )
    return {
        "task_id": record["job_id"],
        "status": status,
        "status_message": message,
        "created_at": datetime.fromtimestamp(
            record["created_at"], timezone.utc
        ).isoformat(),
        "last_updated_at": datetime.fromtimestamp(
            record["updated_at"], timezone.utc
        ).isoformat(),
        "ttl_ms": None,
        "poll_interval_ms": 1000,
    }


def _terminal_result(record: dict) -> dict:
    # Shared with the ordinary tool handlers; do not maintain a second schema or
    # retain a process-local copy of the submission result. The registry is truth.
    from .mcp_server import job_handle

    handle = asdict(job_handle(record))
    result = types.CallToolResult(
        content=[
            types.TextContent(type="text", text=json.dumps(handle, allow_nan=False))
        ],
        structured_content=handle,
        is_error=record["state"] in {"failed", "interrupted"},
    )
    return result.model_dump(by_alias=True, mode="json", exclude_none=True)


class JobTasks(Extension):
    """Opt-in, polling-only Tasks adapter; never submits or launches work itself."""

    identifier = EXTENSION_ID

    def __init__(self, jobs: JobStore):
        self.jobs = jobs

    def methods(self) -> Sequence[MethodBinding]:
        versions = frozenset({PROTOCOL_VERSION})
        return (
            MethodBinding(
                "tasks/get", TaskParams, self.get, protocol_versions=versions
            ),
            MethodBinding(
                "tasks/cancel", TaskParams, self.cancel, protocol_versions=versions
            ),
            MethodBinding(
                "tasks/update",
                UpdateTaskParams,
                self.update,
                protocol_versions=versions,
            ),
        )

    async def _record(self, task_id: str) -> dict:
        try:
            return await to_thread.run_sync(partial(self.jobs.status, task_id, limit=1))
        except (KeyError, ValueError):
            raise MCPError(
                code=types.INVALID_PARAMS, message="Unknown or invalid task ID"
            ) from None

    async def intercept_tool_call(
        self,
        params: types.CallToolRequestParams,
        ctx: ServerRequestContext[Any, Any],
        call_next: CallNext,
    ) -> HandlerResult:
        # Submit through the ordinary handler exactly once, including its policy
        # checks and idempotency. Nothing here can create a replacement job.
        result = await call_next(ctx)
        if params.name not in _TASK_TOOLS or ctx.protocol_version != PROTOCOL_VERSION:
            return result
        capabilities = ctx.session.client_capabilities
        if capabilities is None or self.identifier not in (
            capabilities.extensions or {}
        ):
            return result
        if not isinstance(result, types.CallToolResult) or result.is_error:
            return result
        handle = result.structured_content
        if not isinstance(handle, dict) or not isinstance(handle.get("job_id"), str):
            return result
        require_client_extension(ctx, self.identifier)
        record = await self._record(handle["job_id"])
        return _wire(CreateTaskResult(**_task_fields(record)))

    async def get(
        self, ctx: ServerRequestContext[Any, Any], params: TaskParams
    ) -> dict:
        require_client_extension(ctx, self.identifier)
        record = await self._record(params.task_id)
        fields = _task_fields(record)
        if fields["status"] == "completed":
            return _wire(CompletedTask(**fields, result=_terminal_result(record)))
        if fields["status"] == "cancelled":
            return _wire(CancelledTask(**fields))
        return _wire(WorkingTask(**fields))

    async def cancel(
        self, ctx: ServerRequestContext[Any, Any], params: TaskParams
    ) -> dict:
        require_client_extension(ctx, self.identifier)
        try:
            await to_thread.run_sync(self.jobs.cancel, params.task_id)
        except (KeyError, ValueError):
            raise MCPError(
                code=types.INVALID_PARAMS, message="Unknown or invalid task ID"
            ) from None
        return {"resultType": "complete"}

    async def update(
        self, ctx: ServerRequestContext[Any, Any], params: UpdateTaskParams
    ) -> dict:
        require_client_extension(ctx, self.identifier)
        await self._record(params.task_id)
        # No requests are outstanding. Per spec, ignore unknown/already-answered
        # keys, including on terminal tasks; never treat input as executable work.
        return {"resultType": "complete"}
