# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""2026 Tasks roundtrips over real durable CPU jobs, never an echo job store.

The raw peer uses the SDK's own request runner/dispatcher (the same path as its
in-process Client) so each request can independently advertise its capabilities.
JSON roundtripping keeps assertions about wire names/nulls, not Python models.
The FIFO fixture blocks the actual checkpoint validator, outside the production
workspace adapter; production workspace restrictions are not bypassed or patched.
"""

import errno
import json
import os
import signal
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime

import anyio
import pytest

pytest.importorskip("mcp.server.extension")

from mcp import Client, types
from mcp.client import advertise
from mcp.server.mcpserver import MCPServer
from mcp.server.runner import modern_on_request
from mcp.shared.direct_dispatcher import create_direct_dispatcher_pair
from mcp.shared.exceptions import MCPError

from hi2.jobs import TERMINAL, JobStore
from hi2.mcp_server import JobHandle, create_server, job_handle
from hi2.mcp_tasks import (
    EXTENSION_ID,
    PROTOCOL_VERSION,
    GetTaskRequest,
    JobTasks,
    TaskParams,
)
from hi2.workspace import WorkspacePolicy


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def offline_cpu(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "")


class WireClient:
    def __init__(self, dispatcher):
        self.dispatcher = dispatcher

    async def request(self, method, params=None, *, tasks=False):
        params = dict(params or {})
        params["_meta"] = {
            "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION,
            "io.modelcontextprotocol/clientInfo": {
                "name": "hi2-tasks-test",
                "version": "1",
            },
            "io.modelcontextprotocol/clientCapabilities": {
                "extensions": {EXTENSION_ID: {}} if tasks else {},
            },
        }
        result = await self.dispatcher.send_raw_request(
            method,
            json.loads(json.dumps(params)),
            {"timeout": 30},
        )
        return json.loads(json.dumps(result))

    async def call(self, name, arguments, *, tasks=False):
        return await self.request(
            "tools/call", {"name": name, "arguments": arguments}, tasks=tasks
        )


async def _unexpected_request(ctx, method, params):
    raise AssertionError(f"unexpected server-to-client request: {method}")


async def _unexpected_notification(ctx, method, params):
    raise AssertionError(f"unexpected notification: {method}")


@asynccontextmanager
async def wire(server):
    # SDK 2.3.0's Client unwraps MCPServer the same way; no public accessor exists.
    lowlevel = server._lowlevel_server
    async with lowlevel.lifespan(lowlevel) as lifespan:
        client, peer = create_direct_dispatcher_pair()
        async with anyio.create_task_group() as group:
            await group.start(
                peer.run,
                modern_on_request(lowlevel, lifespan),
                _unexpected_notification,
            )
            await group.start(client.run, _unexpected_request, _unexpected_notification)
            try:
                yield WireClient(client)
            finally:
                client.close()
                peer.close()


def _records(jobs):
    with sqlite3.connect(jobs.database) as db:
        db.row_factory = sqlite3.Row
        return [
            dict(row) for row in db.execute("SELECT * FROM jobs ORDER BY created_at")
        ]


async def _terminal(jobs, job_id):
    with anyio.fail_after(60):
        while True:
            record = await anyio.to_thread.run_sync(jobs.status, job_id)
            if record["state"] in TERMINAL:
                return record
            await anyio.sleep(0.05)


@pytest.fixture
async def jobs(tmp_path):
    state = tmp_path / "workspace"
    state.mkdir(mode=0o700)
    store = JobStore(state / "jobs")
    try:
        yield store
    finally:
        # These are real detached processes. A failed assertion must not leak one.
        with anyio.CancelScope(shield=True):
            for row in _records(store):
                await anyio.to_thread.run_sync(store.cancel, row["job_id"])
                await _terminal(store, row["job_id"])


def _validator_server(jobs, *, enabled=True):
    """Controlled tools with real JobStore submission and real worker execution."""
    server = MCPServer(
        "hi2-task-process-fixture", extensions=[JobTasks(jobs)] if enabled else []
    )

    @server.tool()
    async def hi2_evaluate(path: str, idempotency_key: str) -> JobHandle:
        """Run the real CPU artifact validator for the Tasks process fixture."""
        return job_handle(
            await anyio.to_thread.run_sync(
                jobs.submit,
                "validate",
                {"path": path},
                idempotency_key,
            )
        )

    @server.tool()
    async def hi2_job_status(job_id: str) -> JobHandle:
        """Read an existing job; this is deliberately not a task-creating tool."""
        return job_handle(await anyio.to_thread.run_sync(jobs.status, job_id))

    return server


def _assert_task_shape(value, job_id, *, creation=False):
    assert value["resultType"] == ("task" if creation else "complete")
    assert value["taskId"] == job_id
    assert "task" not in value  # This is not the obsolete nested 2025 task shape.
    assert "ttlMs" in value and value["ttlMs"] is None
    assert value["pollIntervalMs"] == 1000
    created = datetime.fromisoformat(value["createdAt"])
    updated = datetime.fromisoformat(value["lastUpdatedAt"])
    assert created.tzinfo is not None and updated.tzinfo is not None
    assert updated >= created
    assert value["status"] in {"working", "completed", "cancelled"}
    assert "inputRequests" not in value


def _assert_one_execution(jobs, job_id):
    rows = _records(jobs)
    assert [row["job_id"] for row in rows] == [job_id]
    events = jobs.status(job_id)["events"]
    assert (
        sum(
            event.get("phase") == "worker" and event.get("status") == "started"
            for event in events
        )
        == 1
    )


@pytest.mark.anyio
async def test_production_quantize_opt_in_same_job_completion_and_reconnect(
    tmp_path, jobs
):
    import torch
    from safetensors.torch import save_file

    source = tmp_path / "models" / "tiny"
    source.mkdir(parents=True)
    (source / "config.json").write_text(
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
                64,
                64,
                generator=torch.Generator().manual_seed(17),
            )
        },
        source / "model.safetensors",
    )
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    policy = WorkspacePolicy(
        model_roots=[source.parent],
        calibration_roots=[],
        output_roots=[outputs],
        state_root=tmp_path / "workspace",
    )
    arguments = {
        "recipe": {
            "schema_version": 1,
            "config": {
                "src": str(source),
                "dst": str(outputs / "quantized"),
                "devices": ["cpu"],
                "cfg": {
                    "effort": "fast",
                    "rotation": "off",
                    "rounding": "nearest",
                    "candidates": [[2, 32], [4, 32]],
                },
                "moe_layout": "per-expert",
            },
        },
        "idempotency_key": "same-work-with-or-without-tasks",
    }

    # Server opt-in is required even when the client already declares support.
    async with wire(create_server(policy, jobs)) as client:
        discovered = await client.request("server/discover", tasks=True)
        assert EXTENSION_ID not in (discovered["capabilities"].get("extensions") or {})
        ordinary = await client.call("hi2_quantize", arguments, tasks=True)
        assert ordinary["resultType"] == "complete" and not ordinary["isError"]
        job_id = ordinary["structuredContent"]["job_id"]
        assert "taskId" not in ordinary
        with pytest.raises(MCPError) as error:
            await client.request("tasks/get", {"taskId": job_id}, tasks=True)
        assert error.value.code == types.METHOD_NOT_FOUND

    async with wire(create_server(policy, jobs, enable_tasks=True)) as client:
        discovered = await client.request("server/discover")
        assert discovered["capabilities"]["extensions"][EXTENSION_ID] == {}
        ordinary = await client.call("hi2_quantize", arguments)
        assert ordinary["resultType"] == "complete" and "taskId" not in ordinary
        assert ordinary["structuredContent"]["job_id"] == job_id
        task = await client.call("hi2_quantize", arguments, tasks=True)
        _assert_task_shape(task, job_id, creation=True)
        assert "result" not in task and "error" not in task
        _assert_task_shape(
            await client.request("tasks/get", {"taskId": job_id}, tasks=True), job_id
        )

        # Negotiation is per request, not sticky after an opted-in call.
        unadvertised = await client.call("hi2_quantize", arguments)
        assert unadvertised["resultType"] == "complete" and "taskId" not in unadvertised
        assert unadvertised["structuredContent"]["job_id"] == job_id
        with pytest.raises(MCPError) as error:
            await client.request("tasks/get", {"taskId": job_id})
        assert error.value.code == -32021
        assert error.value.data["requiredCapabilities"]["extensions"] == {
            EXTENSION_ID: {}
        }

    # Work survives the connection and is retrieved from a newly opened registry.
    record = await _terminal(jobs, job_id)
    assert record["state"] == "succeeded", record["error"]
    assert record["result"] is not None
    assert (outputs / "quantized" / "hi2_report.json").is_file()
    restored = JobStore(jobs.root)
    async with wire(create_server(policy, restored, enable_tasks=True)) as client:
        completed = await client.request("tasks/get", {"taskId": job_id}, tasks=True)
        _assert_task_shape(completed, job_id)
        assert completed["status"] == "completed" and "error" not in completed
        result = completed["result"]
        assert result["resultType"] == "complete" and result["isError"] is False
        assert result["structuredContent"] == asdict(job_handle(record))
        assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
        assert result["structuredContent"]["state"] == "succeeded"
        assert result["structuredContent"]["resources"]
        replay = await client.call("hi2_quantize", arguments, tasks=True)
        _assert_task_shape(replay, job_id, creation=True)
        assert replay["status"] == "completed"
        ack = await client.request("tasks/cancel", {"taskId": job_id}, tasks=True)
        assert {key: value for key, value in ack.items() if key != "_meta"} == {
            "resultType": "complete"
        }
        assert (
            await client.request("tasks/get", {"taskId": job_id}, tasks=True)
            == completed
        )
    assert restored.status(job_id)["result"] == record["result"]
    _assert_one_execution(jobs, job_id)


@pytest.mark.anyio
@pytest.mark.parametrize("method", ["tasks/get", "tasks/cancel", "tasks/update"])
async def test_task_methods_require_capability_and_validate_ids(jobs, method):
    async with wire(_validator_server(jobs)) as client:
        extra = {"inputResponses": {}} if method == "tasks/update" else {}
        for task_id in ("", "../job", "job_" + "0" * 32):
            params = {"taskId": task_id, **extra}
            with pytest.raises(MCPError) as error:
                await client.request(method, params)
            assert error.value.code == -32021
            assert error.value.data["requiredCapabilities"]["extensions"] == {
                EXTENSION_ID: {}
            }
            with pytest.raises(MCPError) as error:
                await client.request(method, params, tasks=True)
            assert error.value.code == types.INVALID_PARAMS
        with pytest.raises(MCPError) as error:
            await client.request(method, {"taskId": 123, **extra}, tasks=True)
        assert error.value.code == types.INVALID_PARAMS
    assert _records(jobs) == []


@pytest.mark.anyio
async def test_failed_worker_returns_persisted_tool_error_not_protocol_failure(
    tmp_path, jobs
):
    artifact = tmp_path / "invalid-artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    arguments = {"path": str(artifact), "idempotency_key": "invalid-checkpoint"}
    async with wire(_validator_server(jobs)) as client:
        task = await client.call("hi2_evaluate", arguments, tasks=True)
        job_id = task["taskId"]
        _assert_task_shape(task, job_id, creation=True)
    record = await _terminal(jobs, job_id)
    assert record["state"] == "failed"
    assert record["error"]["type"] == "ValueError"
    assert "quantization config" in record["error"]["message"]

    async with wire(_validator_server(JobStore(jobs.root))) as client:
        terminal = await client.request("tasks/get", {"taskId": job_id}, tasks=True)
        _assert_task_shape(terminal, job_id)
        assert terminal["status"] == "completed" and "error" not in terminal
        assert terminal["result"]["isError"] is True
        assert terminal["result"]["structuredContent"] == asdict(job_handle(record))
        assert terminal["result"]["structuredContent"]["error"]
        # A read-only job tool must remain ordinary even for a Tasks client.
        ordinary = await client.call("hi2_job_status", {"job_id": job_id}, tasks=True)
        assert ordinary["resultType"] == "complete" and "taskId" not in ordinary
        assert ordinary["structuredContent"]["job_id"] == job_id
        # No input keys are outstanding; ignore arbitrary valid responses safely.
        ack = await client.request(
            "tasks/update",
            {
                "taskId": job_id,
                "inputResponses": {
                    "never-issued": {
                        "action": "accept",
                        "content": {"choice": "ignored"},
                    }
                },
            },
            tasks=True,
        )
        assert {key: value for key, value in ack.items() if key != "_meta"} == {
            "resultType": "complete"
        }
        assert (
            await client.request("tasks/get", {"taskId": job_id}, tasks=True)
            == terminal
        )
        duplicate = await client.call("hi2_evaluate", arguments, tasks=True)
        assert duplicate["taskId"] == job_id and duplicate["status"] == "completed"
    assert JobStore(jobs.root).status(job_id)["error"] == record["error"]
    _assert_one_execution(jobs, job_id)


@pytest.mark.anyio
async def test_rejected_submission_remains_ordinary_tool_error(tmp_path, jobs):
    async with wire(_validator_server(jobs)) as client:
        result = await client.call(
            "hi2_evaluate",
            {
                "path": str(tmp_path),
                "idempotency_key": "",
            },
            tasks=True,
        )
        assert result["resultType"] == "complete" and result["isError"] is True
        assert "taskId" not in result and "task" not in result
    assert _records(jobs) == []


async def _open_waiting_fifo(path, jobs, job_id):
    with anyio.fail_after(30):
        while True:
            try:
                return os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            except OSError as error:
                if error.errno != errno.ENXIO:
                    raise
            record = await anyio.to_thread.run_sync(jobs.status, job_id)
            assert record["state"] not in TERMINAL, record
            await anyio.sleep(0.05)


@pytest.mark.anyio
async def test_cancellation_ack_is_intent_until_actual_worker_exit(tmp_path, jobs):
    artifact = tmp_path / "blocked-artifact"
    artifact.mkdir()
    fifo = artifact / "config.json"
    os.mkfifo(fifo)
    async with wire(_validator_server(jobs)) as client:
        task = await client.call(
            "hi2_evaluate",
            {
                "path": str(artifact),
                "idempotency_key": "blocked-validation",
            },
            tasks=True,
        )
        job_id = task["taskId"]
        writer = await _open_waiting_fifo(fifo, jobs, job_id)
        supervisor = _records(jobs)[0]["supervisor_pid"]
        stopped = False
        try:
            # Freeze the actual supervisor, not the registry. Its live worker
            # retains ownership while blocked reading the real FIFO checkpoint.
            os.kill(supervisor, signal.SIGSTOP)
            stopped = True
            waited, state = await anyio.to_thread.run_sync(
                os.waitpid, supervisor, os.WUNTRACED
            )
            assert waited == supervisor and os.WIFSTOPPED(state)
            before = jobs.status(job_id)
            assert before["state"] == "running" and before["ownership"]["worker"]

            # Missing capabilities must not cause a cancellation side effect.
            with pytest.raises(MCPError) as error:
                await client.request("tasks/cancel", {"taskId": job_id})
            assert error.value.code == -32021
            assert jobs.status(job_id)["cancel_requested_at"] is None
            ack = await client.request(
                "tasks/update",
                {
                    "taskId": job_id,
                    "inputResponses": {"unknown": {"action": "decline"}},
                },
                tasks=True,
            )
            assert "status" not in ack and "taskId" not in ack
            assert jobs.status(job_id)["state"] == "running"
            ack = await client.request("tasks/cancel", {"taskId": job_id}, tasks=True)
            assert {key: value for key, value in ack.items() if key != "_meta"} == {
                "resultType": "complete"
            }
            cancelling = jobs.status(job_id)
            assert cancelling["state"] == "cancelling"
            assert cancelling["cancel_requested_at"] is not None
            assert cancelling["ownership"]["worker"]
            pending = await client.request("tasks/get", {"taskId": job_id}, tasks=True)
            _assert_task_shape(pending, job_id)
            assert pending["status"] == "working"
            assert "Cancellation requested" in pending["statusMessage"]
            assert "result" not in pending and "error" not in pending
        finally:
            if stopped:
                os.kill(supervisor, signal.SIGCONT)
            os.close(writer)
        record = await _terminal(jobs, job_id)
        assert record["state"] == "cancelled"
        assert record["ownership"]["worker"] is False
        cancelled = await client.request("tasks/get", {"taskId": job_id}, tasks=True)
        _assert_task_shape(cancelled, job_id)
        assert cancelled["status"] == "cancelled"
        assert "result" not in cancelled and "error" not in cancelled
        await client.request("tasks/cancel", {"taskId": job_id}, tasks=True)
        assert (
            await client.request("tasks/get", {"taskId": job_id}, tasks=True)
            == cancelled
        )
    _assert_one_execution(jobs, job_id)


@pytest.mark.anyio
async def test_legacy_client_keeps_job_handle_even_with_extension_advertised(
    tmp_path, jobs
):
    artifact = tmp_path / "legacy-invalid-artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    async with Client(
        _validator_server(jobs), mode="legacy", extensions=[advertise(EXTENSION_ID)]
    ) as client:
        assert client.protocol_version != PROTOCOL_VERSION
        result = await client.call_tool(
            "hi2_evaluate",
            {
                "path": str(artifact),
                "idempotency_key": "legacy-validation",
            },
        )
        assert isinstance(result, types.CallToolResult)
        job_id = result.structured_content["job_id"]
        assert "taskId" not in result.model_dump(by_alias=True)
        with pytest.raises(MCPError) as error:
            await client.session.send_request(
                GetTaskRequest(params=TaskParams(task_id=job_id)), types.Result
            )
        assert error.value.code == types.METHOD_NOT_FOUND
    await _terminal(jobs, job_id)
    _assert_one_execution(jobs, job_id)


@pytest.mark.anyio
@pytest.mark.skipif(
    os.name != "posix", reason="worker interruption requires POSIX FIFOs and SIGKILL"
)
async def test_interrupted_worker_is_completed_task_with_persisted_tool_error(
    tmp_path, jobs
):
    artifact = tmp_path / "interrupted-artifact"
    artifact.mkdir()
    fifo = artifact / "config.json"
    os.mkfifo(fifo)
    async with wire(_validator_server(jobs)) as client:
        task = await client.call(
            "hi2_evaluate",
            {"path": str(artifact), "idempotency_key": "interrupted-validation"},
            tasks=True,
        )
        job_id = task["taskId"]
        writer = await _open_waiting_fifo(fifo, jobs, job_id)
        try:
            row = _records(jobs)[0]
            assert row["worker_started_at"] is not None
            assert row["cancel_requested_at"] is None
            os.kill(row["worker_pid"], signal.SIGKILL)
            record = await _terminal(jobs, job_id)
        finally:
            os.close(writer)
    assert record["state"] == "interrupted"
    assert record["returncode"] == -signal.SIGKILL
    assert record["error"]["type"] == "WorkerExited"
    assert record["result"] is None and record["cancel_requested_at"] is None
    assert record["ownership"]["worker"] is False

    async with wire(_validator_server(JobStore(jobs.root))) as client:
        terminal = await client.request("tasks/get", {"taskId": job_id}, tasks=True)
        _assert_task_shape(terminal, job_id)
        assert terminal["status"] == "completed"
        assert "error" not in terminal
        result = terminal["result"]
        assert result["resultType"] == "complete" and result["isError"] is True
        assert result["structuredContent"] == asdict(job_handle(record))
        assert json.loads(result["content"][0]["text"]) == result["structuredContent"]
    _assert_one_execution(jobs, job_id)


@pytest.mark.anyio
async def test_task_ids_cannot_read_or_mutate_another_registry(tmp_path, jobs):
    artifact = tmp_path / "private-artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    async with wire(_validator_server(jobs)) as client:
        task = await client.call(
            "hi2_evaluate",
            {"path": str(artifact), "idempotency_key": "private-registry-job"},
            tasks=True,
        )
        job_id = task["taskId"]
    record = await _terminal(jobs, job_id)
    assert record["state"] == "failed"
    before = _records(jobs)
    other = JobStore(tmp_path / "other-workspace" / "jobs")
    async with wire(_validator_server(other)) as client:
        for method in ("tasks/get", "tasks/cancel", "tasks/update"):
            params = {"taskId": job_id}
            if method == "tasks/update":
                params["inputResponses"] = {"not-issued": {"action": "decline"}}
            with pytest.raises(MCPError) as error:
                await client.request(method, params, tasks=True)
            assert error.value.code == types.INVALID_PARAMS
    assert _records(other) == []
    assert not (other.root / job_id).exists()
    assert _records(jobs) == before
    assert jobs.status(job_id)["error"] == record["error"]
    _assert_one_execution(jobs, job_id)
