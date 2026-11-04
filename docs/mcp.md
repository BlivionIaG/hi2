# Local MCP adapter

`hi2-mcp` is an optional, local **stdio** adapter over the same service and durable
job registry used by hi2. It does not accept shell commands, Python functions,
remote model IDs, credentials, or arbitrary resource filenames. The baseline
workflow uses ordinary MCP tool results containing job IDs; **Tasks is not
required**.

Inspection and planning read checkpoint metadata but do not quantize. Quantization,
held-out quality evaluation, structural validation, and portable export return a
job handle; detached local supervisors own the work after the client disconnects.
A successful structural check or dense numerical evaluation is **not** evidence
of native vLLM/kernel execution, native memory use, throughput, or platform support.

## Installation and compatibility

Follow the [installation guide](installation.md#checkout-and-virtual-environment),
including its optional MCP extra, in the Python environment that will launch the
server. Use the combined test/MCP extras when running adapter tests.

The extra pins the **official `mcp==2.3.0` Python SDK**, using
`mcp.server.MCPServer` rather than the legacy `FastMCP` API. The same package supplies
the Python `Client` below.
Importing `hi2.mcp_server` does not import the SDK or Torch; invoking the adapter
without the extra exits with an installation command on stderr. Other SDK
versions are rejected rather than silently using unverified transport internals.

| Component | Intended compatibility | Actual smoke-test status |
| --- | --- | --- |
| Python SDK/server | `mcp==2.3.0` | Editable extra installation and `hi2-mcp --help` passed on Python 3.12.13, Darwin arm64 |
| Python client | `mcp==2.3.0`, `mcp.Client` with `StdioServerParameters` | Standalone discover → plan → submit → disconnect → reconnect → validate → evaluate → portable workflow passed |
| MCP protocol | `2026-07-28`, negotiated by the SDK client | Observed in the standalone stdio session; negotiated Tasks tests passed in the CPU suite |
| CPU fixture | Locally generated, one-layer Llama safetensors and local tokenizer; no downloads | Queued job survived reconnect with the same ID; quality gate passed; cancellation acknowledged `cancelling`, then reached `cancelled` |
| Native accelerator/runtime | No claim | Not measured |

Verified on **2026-11-04**: the complete CPU/MCP suite passed **941 tests**, with
61 GPU/HIP, multi-GPU, GPU-pinned-memory or unavailable-Triton skips. The MCP extra
was installed, so protocol tests executed rather than skipping. The documented
discover → plan → submit → disconnect → reconnect → validate client also passed
again on November 4. The expanded standalone evaluate/export/cancellation
observations in the table were recorded on November 3; those behaviors also have
coverage in the current suite. Native execution remains unmeasured. Rerun the
focused protocol suite with:

```sh
python -m pytest -q tests/test_mcp_server.py tests/test_mcp_tasks.py
```

The tests require the extra and CPU PyTorch/Transformers dependencies. With no
MCP extra installed, optional adapter tests are skipped; that skip is not MCP
acceptance evidence. No GPU, downloaded model, network service, or credentials
are needed. The adapter targets local POSIX hosts (Linux/macOS): fd isolation,
`fcntl.flock`, SQLite, inherited ownership descriptors, detached child sessions
and Unix signals are required. Keep the launching interpreter/checkout available
until jobs finish. The process tests also exercise FIFOs and stop/continue/kill
signals, so they need a host that permits real child-process lifecycle operations.
This is not a Windows or remotely exposed service. See
[focused regression suites](usage.md#development-tests) and
[durable-job prerequisites](usage.md#durable-local-jobs).

## Operator-owned roots

The operator grants access when launching the process, not through tool arguments:

```sh
hi2-mcp \
  --model-root /srv/hi2/models \
  --calibration-root /srv/hi2/calibration \
  --output-root /srv/hi2/outputs \
  --state-root /srv/hi2/private-state
```

Equivalently use `python -m hi2.mcp_server` with those arguments. Paths must be
absolute, without parent traversal. Each `--model-root`, `--calibration-root`, and
`--output-root` flag may be repeated. An omitted root category grants no access to
that category. `--state-root` is required and must be outside the input/output resource trees.
The process creates a missing state directory privately; an existing one must be
operator-owned, mode **0700**, and not a symlink. Do not use a shared writable
state directory. Jobs live only in `<state-root>/jobs`.

WorkspacePolicy rechecks paths and registered IDs against the configured roots,
including symlink escapes. Model and calibration inputs must already exist;
outputs must be subdirectories of output roots and must not overlap inputs or
private state. Safetensors metadata is readable; untrusted pickle/cache payloads
are not accepted as tool resources. Recipes cannot nominate work/cache paths:
those are private, deterministic, application-managed directories under state.
Planning may create these private directories and persist registrations, but it
does not create a checkpoint or submit a job.

Quantizer-created ancillary links may point into configured model roots: tokenizer
and template assets remain read-only model capabilities. Checkpoint weights,
configuration and run evidence cannot use that exception. Portable export
materializes its recognized assets. Model loading explicitly disables remote code;
manifest calibration paths are checked against calibration roots before evaluation.

This is a local capability boundary, not an OS sandbox against other hostile
processes running as the same user. Protect the input roots and private state
from concurrent modification by untrusted users. Client MCP roots do not expand
these operator grants. Changing roots on reconnect may deliberately invalidate
previous registrations. Never delete active job state to "reset" a client;
process ownership and leases depend on it.

## Tool contracts

Discover current input/output JSON Schemas with `list_tools()`. The SDK validates
the typed signatures and results; unknown arguments, unknown recipe fields,
unknown nested encoder/distillation fields, invalid enums, nonfinite numbers,
and out-of-range pagination/evaluation settings are rejected. Recipe fields are
derived from the shared service dataclasses rather than a second configuration
language. WorkspacePolicy and service preflight additionally enforce path
capabilities and semantic constraints before submitting work.

Recipes are service configuration, not a serialization of CLI flags. `devices` is
a list of `cpu`/`cuda`/`cuda:N` strings, never `"auto"` or a comma-separated string;
`calib` is one rooted path/ID or a list (up to 32). Encoding settings are nested in
`cfg`; `qad` is a distillation-settings object or `null`. Omitting `qad` leaves QAD
off, unlike calibrated CLI `quantize`, which resolves a preset by default. For a
CLI-planned recipe, omit `workdir`, `cache_dir`, and `calib_ids` before submission:
they are excluded from the MCP input schema even if null. Keep the remaining
resolved settings explicit. `--target-kld` is a CLI tuning workflow, not a recipe
field. `hi2_plan.summary` is compact planning output, not a replayable recipe.

All successful tool responses have `schema_version: 1`. Check `result.is_error`
before using `result.structured_content`; a protocol-level rejection instead
raises `MCPError` in the SDK client.

| Tool | Arguments | Structured result / effect |
| --- | --- | --- |
| `hi2_inspect` | `path`, optional `kind: "model"` or `"artifact"` | Safetensors metadata preview, persisted `resource_id`, available evidence URIs. No tensor validation or model execution. |
| `hi2_plan` | `recipe: {schema_version: 1, config: {...}}` | Compact resolved phases/settings/estimates and `output_ids`. No job submitted. |
| `hi2_quantize` | `recipe`, `idempotency_key` | `JobHandle`, operation `quantize`. |
| `hi2_evaluate` | `request`, `idempotency_key` | `JobHandle`, operation `quality`; held-out dense numerical evaluation. |
| `hi2_status` | `job_id` | Compact durable `JobHandle`, including result/error previews and evidence URIs. |
| `hi2_cancel` | `job_id` | Current `JobHandle` after requesting cancellation; not a promise of immediate termination. |
| `hi2_validate` | artifact `path` or registered artifact ID, `idempotency_key` | `JobHandle`, operation `validate`; structural checkpoint validation runs in the worker. |
| `hi2_portable` | artifact `src`, new `out`, `idempotency_key`, optional model `assets_from` | `JobHandle`, operation `portable`; export/checking occurs in the worker. |

`hi2_evaluate.request` requires `src` (model root/ID), `dst` (artifact root/ID),
and `text` (calibration root/ID). Optional fields are `excluded` (at most 32 rooted
text paths), `max_kld` (nonnegative), `max_ppl_ratio` (positive), `samples`
(1–4096, default 8), `seq_len` (2–131072, default 2048), `device` (default `cpu`),
and `dtype` (`float32`, `float16`, or `bfloat16`, default `float32`). `device` accepts
one `cpu`, `cuda`, or `cuda:N`, not the CLI's comma-separated device list. CPU is
explicit in the example; the adapter does not alter architecture-specific dtype
support. Use held-out text separate from calibration/tuning text. The service checks
exact document overlap against recorded calibration and explicit `excluded` paths,
not semantic deduplication or unknown training data. Recorded paths must still
exist and pass the calibration-root policy.

An idempotency key contains 1–512 characters and is unique across operations in
one registry. Persist it with the recipe and job ID. Repeating the same normalized
operation/request and key returns the same job, even if it is terminal; reusing it
for different settings fails rather than starting a second computation. Keep the
same roots, state and device-visibility environment on reconnect. Workspace path
checks still apply before submission retries: a key does not bypass root policy or
restore removed inputs. Use `hi2_status` with a saved job ID to retrieve existing
job evidence. Do not put secrets into keys. Different keys still obey shared
resource leases; a retry of interrupted work needs a new key and an
identity-compatible recipe, not changed settings in an old output/workspace.

A job handle contains `job_id`, `state`, `operation`, timestamps, request digest,
`cancellation`, bounded `result`/`error` previews, and `resources` (a list of
`{name, uri}`). States are `queued`, `running`, `cancelling`, `succeeded`, `failed`,
`cancelled`, and `interrupted`. Previews may be truncated or omitted; they are not
full reports. An `interrupted` job is not a successful artifact and must not be
silently treated as resumable success.

Quality reports use `context: "dense_numerical"` and one of `target_met`,
`target_not_met`, `not_measured`, or `measured`. `measured` means measurements exist
without acceptance criteria. Rejected quality criteria can leave a readable
`hi2_quality.json` even though the job failed. Read that evidence before making a
quality decision. `native_runtime: "not_measured"` remains explicit.

## Python client: discover, inspect, plan, start, reconnect, validate

Install the same optional extra in the client environment. Save this as a local
client script, after replacing the example operator paths with your own. The
model must already be a local safetensors checkpoint; this script does not fetch
anything. The example uses data-free CPU quantization and therefore does not
need calibration text to start.

```python
import json
from pathlib import Path
import sys

import anyio
from mcp import Client, StdioServerParameters

ROOT = Path("/srv/hi2")
MODEL = ROOT / "models" / "local-model"
OUTPUT = ROOT / "outputs" / "cpu-run"
RECEIPT = Path("hi2-job-receipt.json")  # Client-owned; not a server resource.

server = StdioServerParameters(
    command=sys.executable,  # This interpreter must have hi2[mcp] installed.
    args=[
        "-m",
        "hi2.mcp_server",
        "--model-root",
        str(ROOT / "models"),
        "--calibration-root",
        str(ROOT / "calibration"),
        "--output-root",
        str(ROOT / "outputs"),
        "--state-root",
        str(ROOT / "private-state"),
    ],
    env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
)
recipe = {
    "schema_version": 1,
    "config": {
        "src": str(MODEL),
        "dst": str(OUTPUT),
        "devices": ["cpu"],
        "cfg": {"effort": "fast", "rounding": "nearest", "rotation": "off"},
        "moe_layout": "per-expert",
    },
}
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


async def invoke(client, name, arguments):
    result = await client.call_tool(name, arguments)
    if result.is_error:
        raise RuntimeError(result.content)
    return result.structured_content


async def wait_for_job(client, job_id):
    while True:
        status = await invoke(client, "hi2_status", {"job_id": job_id})
        print(status["state"], status["resources"])
        if status["state"] in TERMINAL:
            return status
        await anyio.sleep(1)


async def main():
    async with Client(server) as client:
        print("Negotiated protocol:", client.protocol_version)
        tools = await client.list_tools()
        print([(tool.name, tool.input_schema) for tool in tools.tools])
        inspected = await invoke(client, "hi2_inspect", {"path": str(MODEL)})
        print(inspected)
        print(await invoke(client, "hi2_plan", {"recipe": recipe}))
        key = "cpu-run-001"
        handle = await invoke(
            client,
            "hi2_quantize",
            {
                "recipe": recipe,
                "idempotency_key": key,
            },
        )
        RECEIPT.write_text(
            json.dumps({"job_id": handle["job_id"], "key": key, "recipe": recipe})
        )
    # Exiting Client closes only this stdio server. Durable work is independent.

    receipt = json.loads(RECEIPT.read_text())
    async with Client(server) as client:  # Fresh process, same roots/state.
        result = await wait_for_job(client, receipt["job_id"])
        if result["state"] != "succeeded":
            raise RuntimeError(result)
        artifact = await invoke(
            client,
            "hi2_inspect",
            {
                "kind": "artifact",
                "path": str(OUTPUT),
            },
        )
        print("Persist this artifact ID and its URIs:", artifact)
        validation = await invoke(
            client,
            "hi2_validate",
            {
                "path": artifact["resource_id"],
                "idempotency_key": "cpu-run-001-validation",
            },
        )
        checked = await wait_for_job(client, validation["job_id"])
        if checked["state"] != "succeeded":
            raise RuntimeError(checked)
        for reference in artifact["resources"]:
            print(
                reference["name"],
                (await client.read_resource(reference["uri"])).contents,
            )


if __name__ == "__main__":
    anyio.run(main)
```

If a client loses the submit response, reopen `Client(server)` and repeat
`hi2_quantize` with the **same saved recipe and key**, then save the returned ID.
Do not make up a fresh key merely because the connection closed. The reference
script stores the receipt after receiving the handle; applications should persist
the intended key/recipe before submission when recovery from that specific
client-crash window is required.

A plain desktop MCP host can launch exactly the same executable and arguments.
For example, adapt this host configuration to absolute local paths (configuration
key names may differ between hosts):

```json
{
  "mcpServers": {
    "hi2": {
      "command": "/srv/hi2/venv/bin/hi2-mcp",
      "args": [
        "--model-root", "/srv/hi2/models",
        "--calibration-root", "/srv/hi2/calibration",
        "--output-root", "/srv/hi2/outputs",
        "--state-root", "/srv/hi2/private-state"
      ]
    }
  }
}
```

This is a launch example, not a claim of testing a particular desktop host/version.
The Python SDK client is the explicit compatibility target. No credentials or
shell wrappers are necessary.

## Cancellation and optional evaluation/export

Cancellation is a separate durable action, not transport-request cancellation.
Closing the client, cancelling `call_tool`, or killing its stdio server is **not**
`hi2_cancel`. From a fresh client connected with the same state root:

```python
async with Client(server) as client:
    receipt = json.loads(RECEIPT.read_text())
    response = await invoke(client, "hi2_cancel", {"job_id": receipt["job_id"]})
    print(response["state"], response["cancellation"])
    terminal = await wait_for_job(client, receipt["job_id"])
    print(terminal)
```

`cancelling` with `cancellation: "requested"` means the supervisor has received
the request, **not** that the worker stopped or its leases were released. Poll to
a terminal state. Cancellation may race success; `succeeded`/`already_finished`
is truthful in that case. Partial artifacts from cancelled/failed/interrupted
work are not automatically accepted.

For quantization, the worker binds its supervisor token and work directory to
both the recipe's deterministic `run_id` and a fresh per-invocation `attempt_id`.
The attempt is persisted in `run.json` and the committed `hi2_run.json`; reusing
or resuming the same recipe keeps its `run_id` but creates a new attempt. A
validated checkpoint committed by the bound attempt remains `succeeded` even
if cancellation arrives before the worker writes its receipt or during teardown.
This is recognized only after worker ownership ends; leases are not released
early. A prior or later success of the same recipe cannot complete a different
attempt's job. Orphan recovery also retains a matching failed/interrupted receipt
and its original error, including quality measurements. Explicit cancellation
takes precedence over an uncommitted failure and retains that worker error.
The actual process return code is retained (and can be nonzero after a successful
commit); orphan recovery reports an unknown return code as `null`.

Ownership uses inherited file locks, not a PID or heartbeat timeout. A stopped
orphan retains leases until its ownership descriptor closes; do not delete locks
or active state to force progress. Status can restart an unstarted queue entry but
does not automatically replay an interrupted operation. Separate state registries
do not arbitrate shared devices with one another.
Detached supervisors are reaped by a background waiter when their submitter
remains alive; submission does not wait for completion, and exiting the submitter
does not stop its jobs.

After a successful run, submit held-out quality evaluation with a new key:

```python
async with Client(server) as client:
    evaluation = await invoke(
        client,
        "hi2_evaluate",
        {
            "request": {
                "src": str(MODEL),
                "dst": str(OUTPUT),
                "text": str(ROOT / "calibration" / "held-out.txt"),
                "samples": 8,
                "seq_len": 128,
                "device": "cpu",
                "dtype": "float32",
                "max_kld": 0.1,
                "max_ppl_ratio": 1.1,
            },
            "idempotency_key": "cpu-run-001-held-out",
        },
    )
    print(await wait_for_job(client, evaluation["job_id"]))
    artifact = await invoke(
        client, "hi2_inspect", {"kind": "artifact", "path": str(OUTPUT)}
    )
    print(artifact["resources"])  # Includes quality when the report exists.
```

Those criteria are illustrative application choices, not promised model quality.
For export, call `hi2_portable` with `src=artifact["resource_id"]`, an unused
`out` under an output root, a fresh idempotency key, and optionally
`assets_from=str(MODEL)`. Poll its job, then inspect/register the new output.

Portable export retains the artifact's own run/quality evidence and byte-identical
HI2 weights, with a checksum manifest for copied files. Assets supplied through
`assets_from` do not replace that evidence. It does not run quality evaluation,
bundle reference-model weights/calibration corpora, rewrite original provenance
paths, or qualify a native runtime. Later evaluation needs the reference model and
recorded calibration files, still available within the configured roots; missing
files fail rather than disabling contamination checks. A report without recorded
source identity cannot establish that the reference matches the quantization source.

Re-evaluating a portable artifact updates its quality report and checksums under
the output lock. These are separate atomic replacements, not jointly hard-kill
atomic: a kill between them can leave a bundle rejected by structural/checksum
validation. Submit `hi2_validate` after an abrupt termination before trusting or
distributing it; do not remove evidence to bypass rejection. See
[quality/portable provenance](usage.md#quality-and-portable-provenance) and
[distribution guidance](operations.md#distribute-a-checkpoint).

## Read-only resources and limits

Resource templates are discoverable with `list_resource_templates()`. Tools
return concrete URIs. Resource reads cannot register new IDs or take a filesystem
path; only `hi2_inspect` registers an existing model/artifact after policy checks.
The registry persists IDs across server restarts with the same state root.

| URI | Meaning | Paging |
| --- | --- | --- |
| `hi2://jobs/{job_id}/status` | Compact durable status, identical handle shape | One bounded JSON document |
| `hi2://jobs/{job_id}/events{?cursor,limit}` | Retained progress events | Exclusive monotonic event cursor; default/max 16 events |
| `hi2://jobs/{job_id}/log{?cursor,limit}` | Private worker log only | Byte cursor; default/max 16384 bytes |
| `hi2://artifacts/{artifact_id}/manifest{?cursor,limit}` | `hi2_run.json`, or portable manifest when no run manifest exists | Byte cursor; default/max 16384 bytes |
| `hi2://artifacts/{artifact_id}/report{?cursor,limit}` | `hi2_report.json` | Byte cursor; default/max 16384 bytes |
| `hi2://artifacts/{artifact_id}/quality{?cursor,limit}` | `hi2_quality.json`, when present | Byte cursor; default/max 16384 bytes |
| `hi2://artifacts/{artifact_id}/support` | Evidence inventory, limits, quality context, explicit unmeasured-native status | One compact document; cursor must be 0 |

Artifact IDs include their `artifact:` prefix, for example
`hi2://artifacts/artifact:0123456789abcdef0123456789abcdef/report` (illustrative,
not an actual registration). No `file://` resources, weight bytes, config filenames,
private cache files, or client-selected log paths are served. Supported evidence
names are exactly `manifest`, `report`, `quality`, and `support`; other names fail.
Every read rechecks the registration and roots. Symlink/device/hardlink evidence
files are refused rather than followed.

Every resource content payload is capped at **32768 UTF-8 bytes**, including its
JSON envelope. File pages return `encoding: "base64"`, `data`, `cursor`,
`next_cursor`, `size_bytes`, and `eof`. Base64 preserves exact UTF-8/multibyte
boundaries and arbitrary log bytes; decode each page before interpreting the
concatenated file. `limit` is a raw-byte count, not encoded length. Cursors must
be integers in `[0, 2**53 - 1]`. Support/status are compact documents, not file
pages. Missing reports are errors, not fabricated empty reports. A log not yet
created for queued/cancelling/cancelled work reports `available: false`.
An allowed cursor at or beyond the current file size returns an empty page with
`eof: true` and the requested cursor unchanged. It does not seek beyond EOF, so
the protocol's cursor range does not depend on a filesystem's maximum seek offset.

```python
import base64


async def print_log(client, log_uri):
    cursor = 0
    while True:
        result = await client.read_resource(f"{log_uri}?cursor={cursor}&limit=4096")
        page = json.loads(result.contents[0].text)
        print(base64.b64decode(page["data"]).decode("utf-8", errors="replace"), end="")
        cursor = page["next_cursor"]
        if page["eof"]:
            return cursor
```

For a running log, `eof` means "caught up at this read", not that the worker is
finished; poll job status and resume from the saved byte cursor. Event pages
contain `next_cursor`, `oldest_cursor`, `last_cursor`, and `events_truncated`.
The job registry retains at most 1000 events per job, so an old cursor may have
missed history; do not claim it is a full audit trail. The byte cap may reduce an
event page below the requested count. An oversized individual event becomes an
explicit omission marker rather than an over-limit response. Real worker progress
is also appended as JSON lines to `worker.log`, alongside stdout/stderr and
tracebacks, so event rollover does not erase early diagnostics. Worker logs remain
on disk until operator cleanup; bounded responses are not a disk-retention quota.

## Optional negotiated Tasks

Add `--tasks` only if needed. The server then advertises
`io.modelcontextprotocol/tasks`; it does not require a client to use it. On
protocol `2026-07-28`, only callers that explicitly negotiate the extension in
per-request client capabilities may receive `resultType: "task"` from
`hi2_quantize` or `hi2_evaluate`. Ordinary clients, including the script above,
still receive ordinary `CallToolResult`/`JobHandle` responses even with the flag.

Tasks wraps the **same already-submitted durable job**; `taskId == job_id`. There
is no extra job store, executor, or second worker. `tasks/get` projects queued,
running, and cancelling jobs as `working`; succeeded jobs become `completed`.
Tool/domain failures (`failed`/`interrupted` jobs) become completed task results
with `isError: true`, rather than being mislabelled JSON-RPC task failures.
Cancelled jobs become `cancelled`. Completed results contain the same compact
job-handle schema and evidence URIs as ordinary tools.

Poll at the suggested `pollIntervalMs: 1000`. `ttlMs: null` means no automatic
expiry: receipts/registry remain until the operator removes state. Reconnect
with the same roots/state and task ID. The extension is polling-only here; it
has no input requests or task notification stream. `tasks/update` ignores
unknown/already-satisfied input response keys for existing jobs. `tasks/cancel`
acknowledges cancellation **intent**, not worker termination; keep polling.
Do not interpret a cancellation acknowledgement as released resource ownership.

See the [official Tasks overview](https://modelcontextprotocol.io/extensions/tasks/overview.md)
and [official Python SDK client documentation](https://py.sdk.modelcontextprotocol.io/client/index.md)
for client extension negotiation. Legacy experimental Tasks APIs are not this
adapter's contract.

## Troubleshooting

- **Missing extra or wrong SDK:** launch with the same interpreter used for
  installation; install `hi2[mcp]` and confirm the pinned `mcp==2.3.0`. The
  `hi2-mcp` entrypoint has no dependency on an unrelated third-party FastMCP package.
- **No output when launched in a terminal:** stdio waits for an MCP client on
  stdin. Do not type shell commands into it. Help/errors go to stderr.
- **Protocol corruption:** launch the entrypoint directly; do not place an
  `echo`, model import, or banner-writing wrapper before it. The entrypoint saves
  the original stdout wire before importing the SDK/model code, permanently
  redirects real fd 1 plus Python stdout to stderr, and passes the saved wire
  explicitly to the SDK. This includes native writes and shutdown-time flushes;
  worker output is independently captured in the job log. Code that runs before
  the entrypoint is outside that boundary.
- **Root denial / unknown ID:** use absolute local paths under the matching
  operator root; IDs are kind-scoped. Model, output, and artifact IDs are not
  interchangeable. Register a completed output using
  `hi2_inspect(kind="artifact", path=...)`. Reuse the same state root on reconnect.
- **Invalid recipe:** use the versioned `schema_version/config` envelope and
  discovered schema. Do not paste CLI-only flags, credentials, remote-code
  options, shell expressions, or `workdir`/`cache_dir`/`calib_ids` into it.
- **Busy / cancelling job:** inspect status/events/logs. Shared leases prevent
  conflicting writers; cancelling does not release ownership before actual exit.
  Never bypass a lease by deleting live state.
- **No quality evidence:** metadata inspection and structural validation do not
  measure quality. Submit `hi2_evaluate` with appropriate held-out text, then
  inspect the status and quality report. Native serving support remains unmeasured.
- **Resource page rejected:** retain the ID, use a supported resource kind, and
  keep cursor/limit within the advertised bounds. A report may not exist yet or
  may have been removed; a URI is not a capability to read other files instead.
