# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Real SDK stdio/client contracts: local CPU fixtures, no model downloads."""

import base64
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path

import pytest

pytest.importorskip("mcp")
import anyio
from mcp import Client, MCPError, StdioServerParameters, stdio_client
from mcp.types import CallToolResult

from hi2.jobs import MAX_EVENT_BYTES, JobStore
from hi2.mcp_server import (
    MAX_CURSOR,
    MAX_EVENTS,
    MAX_PAGE_BYTES,
    MAX_RESOURCE_BYTES,
    _file_page,
)

assert version("mcp") == "2.3.0", (
    "The MCP integration gate requires the pinned mcp==2.3.0 extra."
)

REPO = Path(__file__).resolve().parents[1]
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


@pytest.fixture(scope="module")
def local_inputs(tmp_path_factory):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    root = tmp_path_factory.mktemp("mcp-cpu-inputs")
    model = root / "models" / "tiny"
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        LlamaForCausalLM(config).eval().save_pretrained(model, safe_serialization=True)
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "hello": 1,
                "world": 2,
                **{f"word{i}": i for i in range(3, 16)},
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]"
    ).save_pretrained(model)
    calibration = root / "calibration"
    calibration.mkdir()
    text = calibration / "held-out.txt"
    text.write_text("hello world " * 128)
    return {
        "model_root": model.parent,
        "model": model,
        "calibration": calibration,
        "text": text,
    }


@pytest.fixture
def workspace(tmp_path, local_inputs):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    return {
        **local_inputs,
        "outputs": outputs,
        "state": tmp_path / "state",
        "output": outputs / "quantized",
    }


def recipe(workspace, **changes):
    config = {
        "src": str(workspace["model"]),
        "dst": str(workspace["output"]),
        "devices": ["cpu"],
        "include": r"model\.layers\.0\.mlp\.up_proj\.weight$",
        "moe_layout": "per-expert",
        "cfg": {"effort": "fast", "rotation": "off", "rounding": "nearest"},
    }
    config.update(changes)
    return {"schema_version": 1, "config": config}


def parameters(workspace, *, tasks=False, launcher=None):
    args = ["-m", "hi2.mcp_server"] if launcher is None else ["-c", launcher]
    args += [
        "--model-root",
        str(workspace["model_root"]),
        "--calibration-root",
        str(workspace["calibration"]),
        "--output-root",
        str(workspace["outputs"]),
        "--state-root",
        str(workspace["state"]),
    ]
    if tasks:
        args.append("--tasks")
    return StdioServerParameters(
        command=sys.executable,
        args=args,
        cwd=str(REPO),
        env={
            "PYTHONPATH": str(REPO),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    )


@asynccontextmanager
async def connected(workspace, *, tasks=False, launcher=None, errlog=None):
    params = parameters(workspace, tasks=tasks, launcher=launcher)
    transport = stdio_client(params, errlog=errlog) if errlog is not None else params
    with anyio.fail_after(180):
        async with Client(transport) as client:
            yield client


async def call(client, name, arguments):
    result = await client.call_tool(name, arguments)
    assert isinstance(result, CallToolResult)
    assert not result.is_error, result.content
    assert isinstance(result.structured_content, dict)
    return result.structured_content


async def finished(client, job_id):
    with anyio.fail_after(120):
        while True:
            record = await call(client, "hi2_status", {"job_id": job_id})
            if record["state"] in TERMINAL:
                return record
            await anyio.sleep(0.05)


async def resource(client, uri):
    result = await client.read_resource(uri)
    assert len(result.contents) == 1
    text = result.contents[0].text
    assert len(text.encode()) <= MAX_RESOURCE_BYTES
    return json.loads(text)


def references(handle):
    return {item["name"]: item["uri"] for item in handle["resources"]}


def test_import_is_optional_and_missing_extra_is_actionable(tmp_path):
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        class NoSdk(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'mcp' or fullname.startswith('mcp.'):
                    raise ModuleNotFoundError('SDK intentionally unavailable', name='mcp')
        sys.meta_path.insert(0, NoSdk())
        from hi2 import mcp_server
        assert 'mcp' not in sys.modules and 'torch' not in sys.modules
        mcp_server.main()
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, "--state-root", str(tmp_path / "state")],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert "hi2[mcp]" in result.stderr and "mcp==2.3.0" in result.stderr


def test_discovery_and_plan_do_not_submit_or_quantize(workspace):
    async def scenario():
        async with connected(workspace) as client:
            assert client.protocol_version == "2026-07-28"
            tools = {item.name: item for item in (await client.list_tools()).tools}
            assert set(tools) == {
                "hi2_inspect",
                "hi2_plan",
                "hi2_quantize",
                "hi2_evaluate",
                "hi2_status",
                "hi2_cancel",
                "hi2_validate",
                "hi2_portable",
            }
            for tool in tools.values():
                assert tool.input_schema["additionalProperties"] is False
                assert tool.output_schema
            definitions = tools["hi2_quantize"].input_schema["$defs"]
            assert definitions["Recipe"]["additionalProperties"] is False
            assert definitions["RunConfigInput"]["additionalProperties"] is False
            assert definitions["ConfigInput"]["additionalProperties"] is False
            assert definitions["QadConfigInput"]["additionalProperties"] is False
            templates = await client.list_resource_templates()
            assert any(
                "/events" in item.uri_template for item in templates.resource_templates
            )
            inspected = await call(
                client, "hi2_inspect", {"path": str(workspace["model"])}
            )
            assert inspected["resource_id"].startswith("model:")
            assert inspected["summary"]["tensor_count"] > 0
            planned = await call(client, "hi2_plan", {"recipe": recipe(workspace)})
            assert planned["summary"]["rounding"] == "nearest"
            assert planned["summary"]["phases"] == [
                "data_free_tables",
                "allocation",
                "encode",
                "validation",
            ]
            assert planned["summary"]["native_runtime"] == "not_measured"
            assert planned["output_ids"][0].startswith("output:")
            # Defaults are derived from service dataclasses, not just the explicit fixture settings.
            defaults = {
                "schema_version": 1,
                "config": {
                    "src": inspected["resource_id"],
                    "dst": str(workspace["outputs"] / "default-plan"),
                },
            }
            default_plan = await call(client, "hi2_plan", {"recipe": defaults})
            assert default_plan["summary"]["rounding"] == "nearest"
            assert not workspace["output"].exists()
            assert not (workspace["outputs"] / "default-plan").exists()
            with sqlite3.connect(workspace["state"] / "jobs" / "jobs.sqlite3") as db:
                assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0

    anyio.run(scenario)


def test_cpu_jobs_reconnect_validate_evaluate_and_artifact_ids(workspace):
    async def scenario():
        original = recipe(workspace)
        async with connected(workspace) as client:
            handle = await call(
                client,
                "hi2_quantize",
                {"recipe": original, "idempotency_key": "quantize-once"},
            )
            assert handle["state"] in {"queued", "running"}
            assert handle["operation"] == "quantize"
            assert set(references(handle)) == {"status", "events", "log"}
            # Disconnect before completion: the detached supervisor owns the work.
        async with connected(workspace) as client:
            replay = await call(
                client,
                "hi2_quantize",
                {"recipe": original, "idempotency_key": "quantize-once"},
            )
            assert replay["job_id"] == handle["job_id"]
            done = await finished(client, handle["job_id"])
            assert done["state"] == "succeeded", done
            assert (workspace["output"] / "hi2_report.json").is_file()
            changed = recipe(workspace, max_error_ratio=7)
            conflict = await client.call_tool(
                "hi2_quantize", {"recipe": changed, "idempotency_key": "quantize-once"}
            )
            assert conflict.is_error
            with sqlite3.connect(workspace["state"] / "jobs" / "jobs.sqlite3") as db:
                assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
            inspection = await call(
                client,
                "hi2_inspect",
                {"kind": "artifact", "path": str(workspace["output"])},
            )
            artifact_id = inspection["resource_id"]
            assert artifact_id.startswith("artifact:")
            assert {"manifest", "report", "support"} <= set(references(inspection))
            validation = await call(
                client,
                "hi2_validate",
                {"path": artifact_id, "idempotency_key": "validate-once"},
            )
            assert validation["operation"] == "validate"
            checked = await finished(client, validation["job_id"])
            assert checked["state"] == "succeeded", checked
            evaluation = await call(
                client,
                "hi2_evaluate",
                {
                    "request": {
                        "src": str(workspace["model"]),
                        "dst": artifact_id,
                        "text": str(workspace["text"]),
                        "samples": 1,
                        "seq_len": 8,
                        "device": "cpu",
                        "dtype": "float32",
                    },
                    "idempotency_key": "quality-once",
                },
            )
            assert evaluation["operation"] == "quality"
            evaluated = await finished(client, evaluation["job_id"])
            assert evaluated["state"] == "succeeded", evaluated
            quality = json.loads((workspace["output"] / "hi2_quality.json").read_text())
            assert quality["status"] == "measured"
            assert quality["context"] == "dense_numerical"
            assert quality["native_runtime"] == "not_measured"
            portable_path = workspace["outputs"] / "portable"
            portable = await call(
                client,
                "hi2_portable",
                {
                    "src": artifact_id,
                    "out": str(portable_path),
                    "assets_from": str(workspace["model"]),
                    "idempotency_key": "portable-once",
                },
            )
            assert portable["operation"] == "portable"
            exported = await finished(client, portable["job_id"])
            assert exported["state"] == "succeeded", exported
            assert (portable_path / "hi2_portable_manifest.json").is_file()
            portable_info = await call(
                client, "hi2_inspect", {"kind": "artifact", "path": str(portable_path)}
            )
            assert "manifest" in references(portable_info)
            portable_quality = await call(
                client,
                "hi2_evaluate",
                {
                    "request": {
                        "src": str(workspace["model"]),
                        "dst": portable_info["resource_id"],
                        "text": str(workspace["text"]),
                        "samples": 1,
                        "seq_len": 8,
                        "device": "cpu",
                        "dtype": "float32",
                    },
                    "idempotency_key": "portable-quality-once",
                },
            )
            measured = await finished(client, portable_quality["job_id"])
            assert measured["state"] == "succeeded", measured
            portable_report = json.loads(
                (portable_path / "hi2_quality.json").read_text()
            )
            assert portable_report["reference"]["matches_quantization_source"] is True
            portable_validation = await call(
                client,
                "hi2_validate",
                {
                    "path": portable_info["resource_id"],
                    "idempotency_key": "portable-validate-after-quality",
                },
            )
            checked_portable = await finished(client, portable_validation["job_id"])
            assert checked_portable["state"] == "succeeded", checked_portable
        async with connected(workspace) as client:
            reconnected = await call(
                client, "hi2_inspect", {"kind": "artifact", "path": artifact_id}
            )
            assert reconnected["resource_id"] == artifact_id
            assert "quality" in references(reconnected)
            report = await resource(client, references(inspection)["report"])
            assert report["artifact_id"] == artifact_id
            assert b'"tensors"' in base64.b64decode(report["data"])
            support = await resource(client, references(inspection)["support"])
            assert support["native_runtime"] == "not_measured"
            status = await resource(client, references(handle)["status"])
            assert (
                status["job_id"] == handle["job_id"] and status["state"] == "succeeded"
            )

    anyio.run(scenario)


def test_tasks_enabled_without_client_opt_in_returns_ordinary_job(workspace):
    async def scenario():
        async with connected(workspace, tasks=True) as client:
            assert (
                "io.modelcontextprotocol/tasks" in client.server_capabilities.extensions
            )
            response = await client.call_tool(
                "hi2_quantize",
                {"recipe": recipe(workspace), "idempotency_key": "no-opt-in"},
            )
            assert isinstance(response, CallToolResult)
            assert not response.is_error
            assert response.structured_content["job_id"].startswith("job_")
            assert response.model_dump(by_alias=True).get("resultType") != "task"
            job_id = response.structured_content["job_id"]
            cancelled = await call(client, "hi2_cancel", {"job_id": job_id})
            if cancelled["state"] not in TERMINAL:
                assert cancelled["state"] == "cancelling"
                assert cancelled["cancellation"] == "requested"
            terminal = await finished(client, job_id)
            assert terminal["state"] in {"cancelled", "succeeded"}, terminal
            # A cancellation that races completion may truthfully report success.
            if terminal["state"] == "cancelled":
                assert terminal["cancellation"] == "cancelled"
            again = await call(client, "hi2_cancel", {"job_id": job_id})
            assert again["state"] == terminal["state"]

    anyio.run(scenario)


def test_unknown_fields_bounds_and_root_escapes_are_denied(workspace, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("not-a-tool-resource")
    escape = workspace["outputs"] / "escape"
    escape.symlink_to(outside, target_is_directory=True)

    async def scenario():
        async with connected(workspace) as client:
            bad_recipes = []
            for key, value in (
                ("environment", {"TOKEN": "do-not-echo-credential"}),
                ("cache_dir", str(outside)),
                ("calib_ids", [[1, 2]]),
                ("unknown_setting", True),
                ("seq_len", 0),
                ("devices", []),
                ("src", str(outside)),
                ("dst", str(escape / "result")),
            ):
                invalid = recipe(workspace)
                invalid["config"][key] = value
                bad_recipes.append(invalid)
            nested = recipe(workspace)
            nested["config"]["cfg"]["shell"] = "do-not-echo-credential"
            bad_recipes.append(nested)
            version = recipe(workspace)
            version["schema_version"] = True
            bad_recipes.append(version)
            for invalid in bad_recipes:
                response = await client.call_tool("hi2_plan", {"recipe": invalid})
                assert response.is_error, invalid
                assert "do-not-echo-credential" not in str(response.content)
            response = await client.call_tool(
                "hi2_inspect",
                {"path": str(workspace["model"]), "shell": "do-not-echo-credential"},
            )
            assert response.is_error
            assert "do-not-echo-credential" not in str(response.content)
            for field, value in (
                ("samples", 0),
                ("samples", 4097),
                ("seq_len", 1),
                ("max_kld", -1),
                ("device", "shell:command"),
                ("credential", "hidden"),
            ):
                request = {
                    "src": str(workspace["model"]),
                    "dst": str(workspace["output"]),
                    "text": str(workspace["text"]),
                    field: value,
                }
                response = await client.call_tool(
                    "hi2_evaluate", {"request": request, "idempotency_key": "denied"}
                )
                assert response.is_error
            for uri in (
                outside.as_uri() + "/secret.txt",
                "hi2://artifacts/../../secret.txt/report",
                "hi2://artifacts/artifact:" + "0" * 32 + "/report",
                "hi2://jobs/../../secret.txt/log",
                "hi2://jobs/job_" + "0" * 32 + "/status",
            ):
                with pytest.raises(MCPError):
                    await client.read_resource(uri)
            with sqlite3.connect(workspace["state"] / "jobs" / "jobs.sqlite3") as db:
                assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0

    anyio.run(scenario)


def test_resources_are_bounded_pageable_and_named_only(workspace, tmp_path):
    async def scenario():
        async with connected(workspace) as client:
            handle = await call(
                client,
                "hi2_quantize",
                {"recipe": recipe(workspace), "idempotency_key": "resource-job"},
            )
            done = await finished(client, handle["job_id"])
            assert done["state"] == "succeeded", done
            log_path = workspace["state"] / "jobs" / handle["job_id"] / "worker.log"
            # Append after witnessed worker exit, so the byte-pagination assertion
            # covers a large log without manufacturing a fake job or executor.
            with log_path.open("ab") as stream:
                stream.write(("diagnostic λ\n" * 8000).encode())
            expected = log_path.read_bytes()
            uri = references(handle)["log"]
            cursor = 0
            chunks = []
            while True:
                page = await resource(client, f"{uri}?cursor={cursor}&limit=1024")
                chunk = base64.b64decode(page["data"], validate=True)
                assert len(chunk) <= 1024
                assert page["cursor"] == cursor
                assert page["next_cursor"] == cursor + len(chunk)
                chunks.append(chunk)
                cursor = page["next_cursor"]
                if page["eof"]:
                    break
            assert b"".join(chunks) == expected
            # Exercise the byte cap independently of the event-count cap. These
            # are persisted events on the real completed job, not a fake store.
            registry = JobStore(workspace["state"] / "jobs")
            with registry._transaction() as db:
                for index in range(MAX_EVENTS):
                    registry._event(
                        db,
                        handle["job_id"],
                        {
                            "phase": "fixture",
                            "status": "large-event",
                            "index": index,
                            "message": "x" * (MAX_EVENT_BYTES - 128),
                        },
                    )
            event_uri = references(handle)["events"]
            cursor = 0
            seen = []
            while True:
                page = await resource(client, f"{event_uri}?cursor={cursor}&limit=1")
                assert len(page["events"]) <= 1
                if not page["events"]:
                    break
                seen += [event["cursor"] for event in page["events"]]
                assert page["next_cursor"] > cursor
                cursor = page["next_cursor"]
            assert seen and seen == sorted(set(seen))
            cursor = 0
            capped_seen = []
            page_count = 0
            while cursor < seen[-1]:
                page = await resource(
                    client, f"{event_uri}?cursor={cursor}&limit={MAX_EVENTS}"
                )
                assert 1 <= len(page["events"]) <= MAX_EVENTS
                assert page["next_cursor"] > cursor
                capped_seen.extend(event["cursor"] for event in page["events"])
                cursor = page["next_cursor"]
                page_count += 1
            assert capped_seen == seen
            assert page_count > (len(seen) + MAX_EVENTS - 1) // MAX_EVENTS
            inspected = await call(
                client,
                "hi2_inspect",
                {"kind": "artifact", "path": str(workspace["output"])},
            )
            artifact = inspected["resource_id"]
            for denied in (
                f"{uri}?limit={MAX_PAGE_BYTES + 1}",
                f"{uri}?cursor=-1",
                f"{uri}?limit=0",
                f"{event_uri}?limit={MAX_EVENTS + 1}",
                f"hi2://artifacts/{artifact}/config.json",
                f"hi2://artifacts/{artifact}/worker.log",
                f"hi2://artifacts/{artifact}/%2e%2e%2fsecret.txt",
            ):
                with pytest.raises(MCPError):
                    await client.read_resource(denied)
            # Even an originally registered artifact is rechecked after reconnect
            # or a filesystem change; a named evidence symlink is not authority.
            secret = tmp_path / "secret.json"
            secret.write_text('{"credential":"not-a-tool-resource"}')
            report = workspace["output"] / "hi2_report.json"
            report.unlink()
            report.symlink_to(secret)
            with pytest.raises(MCPError):
                await client.read_resource(references(inspected)["report"])

    anyio.run(scenario)


def test_native_python_import_and_exit_noise_cannot_corrupt_stdio(workspace):
    launcher = textwrap.dedent(r"""
        import atexit
        import os
        from hi2 import mcp_server
        original_create = mcp_server.create_server
        def noisy_create(*args, **kwargs):
            print('python-before-sdk', flush=True)
            os.write(1, b'native-before-sdk\n')
            from hi2 import service
            original_plan = service.plan_run
            def noisy_plan(*args, **kwargs):
                print('python-plan-buffered')
                os.write(1, b'native-plan\n')
                return original_plan(*args, **kwargs)
            service.plan_run = noisy_plan
            def goodbye():
                print('python-exit-buffered')
                os.write(1, b'native-exit\n')
            atexit.register(goodbye)
            return original_create(*args, **kwargs)
        mcp_server.create_server = noisy_create
        mcp_server.main()
    """)

    async def scenario(log):
        async with connected(workspace, launcher=launcher, errlog=log) as client:
            assert (await client.list_tools()).tools
            planned = await call(client, "hi2_plan", {"recipe": recipe(workspace)})
            assert planned["summary"]["rounding"] == "nearest"
            assert (await client.list_tools()).tools

    with tempfile.TemporaryFile(mode="w+") as log:
        anyio.run(scenario, log)
        log.seek(0)
        diagnostics = log.read()
    for expected in (
        "python-before-sdk",
        "native-before-sdk",
        "python-plan-buffered",
        "native-plan",
        "python-exit-buffered",
        "native-exit",
    ):
        assert expected in diagnostics


def test_manifest_provenance_cannot_grant_calibration_access(workspace, tmp_path):
    workspace["output"].mkdir()
    secret = tmp_path / "private.txt"
    secret.write_text("not an authorized calibration resource")
    (workspace["output"] / "hi2_run.json").write_text(
        json.dumps(
            {
                "calibration": {"paths": [{"path": str(secret)}]},
            }
        )
    )

    async def scenario():
        async with connected(workspace) as client:
            response = await client.call_tool(
                "hi2_evaluate",
                {
                    "request": {
                        "src": str(workspace["model"]),
                        "dst": str(workspace["output"]),
                        "text": str(workspace["text"]),
                        "samples": 1,
                        "seq_len": 8,
                    },
                    "idempotency_key": "untrusted-provenance",
                },
            )
            assert response.is_error
            assert secret.read_text() not in str(response.content)
            with sqlite3.connect(workspace["state"] / "jobs" / "jobs.sqlite3") as db:
                assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0

    anyio.run(scenario)


@pytest.mark.parametrize("contents", [b"", b"\x00\xff\xe2\x82\xac\r\n" * 5])
def test_file_pages_preserve_binary_bytes_and_end_offsets(tmp_path, contents):
    (tmp_path / "worker.log").write_bytes(contents)
    cursor = 0
    chunks = []
    while cursor < len(contents):
        page = _file_page(tmp_path, "worker.log", cursor, 2)
        assert page["encoding"] == "base64"
        chunk = base64.b64decode(page["data"], validate=True)
        assert chunk == contents[cursor : cursor + 2]
        assert page["cursor"] == cursor
        assert page["next_cursor"] == cursor + len(chunk)
        assert page["size_bytes"] == len(contents)
        cursor = page["next_cursor"]
        assert page["eof"] is (cursor == len(contents))
        chunks.append(chunk)
    assert b"".join(chunks) == contents
    for offset in (len(contents), len(contents) + 1, MAX_CURSOR):
        page = _file_page(tmp_path, "worker.log", offset, MAX_PAGE_BYTES)
        assert page["data"] == ""
        assert page["cursor"] == page["next_cursor"] == offset
        assert page["size_bytes"] == len(contents)
        assert page["eof"] is True


@pytest.mark.parametrize(
    ("cursor", "limit"),
    [(True, 1), (1.5, 1), (MAX_CURSOR + 1, 1), (0, True), (0, 1.5)],
)
def test_file_page_bounds_are_checked_before_filesystem_access(tmp_path, cursor, limit):
    with pytest.raises(ValueError):
        _file_page(tmp_path / "missing-directory", "worker.log", cursor, limit)


@pytest.mark.skipif(
    os.name != "posix", reason="resource policy requires POSIX descriptors and FIFOs"
)
@pytest.mark.parametrize(
    "kind", ["symlink", "hardlink", "fifo", "directory", "symlink-parent"]
)
def test_file_pages_reject_aliased_and_nonregular_resources(tmp_path, kind):
    directory = tmp_path / "job"
    directory.mkdir()
    external = tmp_path / "private.log"
    secret = b"private bytes must not be returned"
    external.write_bytes(secret)
    entry = directory / "worker.log"
    if kind == "symlink":
        entry.symlink_to(external)
    elif kind == "hardlink":
        os.link(external, entry)
    elif kind == "fifo":
        os.mkfifo(entry)
    elif kind == "directory":
        entry.mkdir()
    else:
        entry.write_bytes(secret)
        alias = tmp_path / "job-alias"
        alias.symlink_to(directory, target_is_directory=True)
        directory = alias
    assert (directory / "worker.log").exists()

    # A regression that blocks opening the FIFO must fail with a timeout, not
    # hang pytest. subprocess.run kills and reaps this child on timeout.
    program = textwrap.dedent("""
        import sys
        from pathlib import Path
        from hi2.mcp_server import _file_page
        try:
            _file_page(Path(sys.argv[1]), "worker.log", 0, 128)
        except (OSError, ValueError):
            pass
        else:
            raise AssertionError("unsafe resource was readable")
    """)
    result = subprocess.run(
        [sys.executable, "-c", program, str(directory)],
        cwd=REPO,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b""
    assert external.read_bytes() == secret
