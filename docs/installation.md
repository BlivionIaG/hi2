# Installation and backend selection

Install `hi2` to quantize a local model. This does not install or qualify a native
inference runtime, such as a vLLM loader for `rdna_hint_flex`.

**Before you start:** use a POSIX environment, Python >=3.10 and a local filesystem
with working advisory locks. The commands below use Python 3.12. Native Windows
is not supported by the run lifecycle. See [backend boundaries](#backend-boundaries).

**Next:** [create a virtual environment](#checkout-and-virtual-environment),
select Torch for your backend, and run the
[installation check](#first-use-compilation-and-smoke).
Then follow the [no-download CPU tutorial](getting-started.md#executed-no-download-cpu-smoke).
These checks establish local functionality, not model quality or native serving.

## Recorded environment and evidence

Functional release tip: `bee522da50c11f609e26832b660222ba5f0516e9`
(feature series based on `dc1e8599faf6fdbdfc4554bc095dc1a8a9d5c3b5`).
The CPU environment below was exercised on **2026-11-03–04** against that feature
tree with the accompanying documentation and CI-formatting changes:
dependency imports/checks, editable `.[mcp]`
installation, both console entry points, and a no-download tiny-model workflow.
The final CPU/MCP gate and documented CLI/reconnect workflows ran on November 4.
The model was randomly initialized; this is functional coverage, not real-model
quality, native-runtime, or accelerator qualification.

| Environment | Python | Torch | Transformers | safetensors | Evidence boundary |
| --- | --- | --- | --- | --- | --- |
| CPU, Darwin arm64 integration environment | 3.12.13 | 2.15.0 | 5.22.1 | 0.8.0 | 2026-11-04 gate: **941 passed, 61 skipped** in 200.09 s. Skips require GPU/HIP, multiple GPUs, GPU-pinned memory or unavailable Triton. Ruff check/format passed (153 files). Documented CLI calibration, QAD, final-KL, validated publication, recipe replay, final quality, portable export and MCP reconnect succeeded. CUDA unavailable. |
| Historical gfx1030 V620 experiments | Not fully recorded in the study | 2.12 (study notation, not an exact wheel pin) | Exact version not recorded; MoE notes refer to Transformers 5 | Not recorded | [QAD dtype study](qad.md#the-half-precision-math-measured-2026-10-28) records ROCm 7.14 and Torch 2.12; it is not a reproducible dependency lock or a new GPU qualification. |

Packaging requires Python >=3.10 and declares `torch`, `transformers`,
`safetensors` and `regex` ([pyproject.toml](../pyproject.toml)). Unconstrained
requirements do not establish that every version combination works. Record the
actual repository revision, package versions and backend with any published run;
do not use the CPU versions above as a validated ROCm/CUDA lock.

The isolated environment was installed with `uv pip install --python <venv-python>
-e '.[mcp]'`; the optional SDK resolved to `mcp==2.3.0`.
`uv pip check --python <venv-python>` reported all 58 installed packages compatible.
The [no-download smoke](getting-started.md#executed-no-download-cpu-smoke) records
the tiny-model recipe. These observations do not turn unconstrained dependencies
into a lockfile or qualify a GPU wheel.

## Checkout and virtual environment

These are installation instructions, not a transcript of commands run for this
page. Use an existing clone if you have one; do not reset or discard local work.
For a fresh checkout:

```bash
git clone https://github.com/BlivionIaG/hi2.git
cd hi2
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Confirm that `.venv` is active with
`python -c 'import sys; print(sys.executable)'`. The path must point inside `.venv`.
If `python3.12` is unavailable, install Python 3.12 through your platform's normal
installation method.

**Select Torch before installing the project.** Use the
[PyTorch installation selector](https://pytorch.org/get-started/locally/) for your
OS and CPU/CUDA/ROCm backend. On macOS, choose the platform wheel and use
`--devices cpu` for this documented path; the quantizer rejects `mps` devices.
On Linux, use the CPU wheel index for a deliberately CPU-only
installation, or the CUDA/ROCm wheel matched to your driver/toolchain. Do not
install a CUDA wheel and expect it to access a ROCm GPU.

For the recorded macOS CPU dependency versions, the package requests are:

```bash
python -m pip install 'torch==2.15.0' 'transformers==5.22.1' 'safetensors==0.8.0'
python -m pip install -e .
```

These requests select the listed versions; they are not a complete dependency
lock. Linux CPU and GPU wheels can use a different version suffix or package
index. Select the correct wheel instead of copying the macOS command unchanged.
`hi2` and `python -m hi2` use the same entry point.

For the optional Model Context Protocol (MCP) adapter:

```bash
python -m pip install -e '.[mcp]'
```

This adds the pinned `mcp==2.3.0` software development kit (SDK), not a native
inference runtime. Contributors should use the combined `.[test,mcp]` setup
[below](#contributor-tests-and-checks) to include the optional protocol tests.
For benchmarks, use the separate `bench` extra and setup in
[bench/README.md](../bench/README.md). There is no installed `hi2 bench` command.

## Backend boundaries

The current lifecycle implementation requires a **POSIX environment**: run/output
ownership uses `fcntl.flock`; durable jobs additionally use inherited descriptors,
process signals and SQLite. Native Windows is not supported by these paths.
Use local filesystems with working advisory locks; network/shared-filesystem
locking and multi-host scheduling are not qualified here. The recorded CPU
environment is macOS; Linux is the intended ROCm/CUDA host, not an additional
qualification result. MCP workspace capabilities are local single-user controls,
not an OS sandbox or a multi-tenant security boundary.

| Selection | Quantizer behavior | Qualification boundary |
| --- | --- | --- |
| `--devices cpu` | Torch reference encoder; no HIP extension needed | Suitable for small compatibility checks. Full-model RAM/time must still be planned. |
| `--devices cuda:0` with CUDA Torch | Torch reference GPTQ path | The custom fused HIP GPTQ kernel is not a CUDA kernel; no CUDA performance qualification is supplied here. |
| `--devices cuda:0` with ROCm Torch on gfx10/gfx11 | Architecture-gated fused HIP GPTQ/refinement when available, otherwise Torch reference | Published runs mainly cover gfx1030 V620. A family gate is implementation eligibility, not proof for every GPU/toolchain. |
| Other HIP architectures | Reference path for this fused GPTQ backend | Not qualified by the RDNA measurements; do not remove architecture checks or force RDNA-specific kernels. |

ROCm also uses PyTorch's `cuda:N` device spelling. Prefer explicit devices on a
shared machine; `auto` can select every visible GPU. A working device import is
not a model-fit or native-serving test.

## First-use compilation and smoke

The optional fused GPTQ path needs a compatible ROCm Torch build, HIP compiler
(`hipcc`), C++ build environment and the build tooling required by
`torch.utils.cpp_extension` (including Ninja). It builds on first use into
`$HI2_KERNEL_DIR` or `~/.cache/hi2/kernels`. The implementation defaults
`PYTORCH_ROCM_ARCH` to `gfx1030`; on another eligible GPU, configure the actual
architecture rather than treating that default as a universal binary.

An automatic GPTQ extension build failure falls back to the Torch reference.
`HI2_GPTQ_BACKEND=torch` forces that path;
`HI2_GPTQ_BACKEND=kernel` makes an eligible-device build failure an error.
`HI2_REFINE_BACKEND=torch` selects reference refinement. Neither setting adds
native inference support. Historical V620 extension builds took roughly 40-50 s;
model-specific Triton backward autotuning can take much longer
([operations](operations.md#first-use-and-slow-starts)).

A lightweight, no-model-download installation check to run in the activated
environment is:

```bash
python -m pip check
python -c 'import sys, torch, transformers, safetensors; print(sys.version); print("torch", torch.__version__, "transformers", transformers.__version__, "safetensors", safetensors.__version__); print("cuda", torch.version.cuda, "hip", torch.version.hip, "visible_gpu", torch.cuda.is_available())'
python -m hi2 --help
python -m hi2 quantize --help
python -m hi2 doctor
```

Expected outcomes:

- `pip check` reports no dependency conflict.
- Imports print the package versions and backend fields.
- Help lists commands and options without loading a model.
- `doctor` reports the local dependency and backend environment.

These checks do not run model forward or backward passes. They do not qualify
native kernels or GPU inference. Save the outputs, `git rev-parse HEAD` and the
exact model revision before a
[local-model smoke](getting-started.md#first-model-walkthrough).

## Contributor tests and checks

From the repository root, after selecting Torch in the activated environment:

```bash
python -m pip install -e '.[test,mcp]' ruff
python -m pip check
```

The `test` extra supplies pytest; Ruff is a separate development tool. The `mcp`
extra is needed for `test_mcp_server.py` and `test_mcp_tasks.py`, which otherwise
skip at import time. Installing only `.[test]` does not cover the complete adapter
surface. The benchmark extra is for actual dataset-backed benchmark runs, not a
requirement to download datasets for these local tests.

Use an offline CPU environment for the following commands:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Focused numerical/unit coverage (a subset, not the full gate).
python -m pytest -q -rs tests/test_format.py tests/test_encode.py \
  tests/test_hessian.py tests/test_allocate_fast.py tests/test_calib.py

# Local-model and process/protocol integration coverage (also a subset).
python -m pytest -q -rs tests/test_pipeline.py tests/test_service.py \
  tests/test_checkpoint_safety.py tests/test_quality.py tests/test_cli_contracts.py \
  tests/test_workspace.py tests/test_jobs.py tests/test_mcp_server.py tests/test_mcp_tasks.py

# Full contributor gate; includes the subsets above.
python -m pytest -q -rs
python -m ruff check .
python -m ruff format --check .
```

These are instructions, not newly executed results or a claim of current CI
success. File selections distinguish useful test groups; there are no configured
`unit`/`integration` markers to substitute for them. The integration fixtures use
local tiny models, temporary state and real subprocesses/protocol exchanges; no
model download or native inference server is required. Run the complete gate
before handoff rather than presenting a focused subset as full coverage.

Inspect skip reasons (`-rs`): accelerator-dependent skips are expected on CPU;
missing MCP dependencies are not evidence that MCP passed. Preserve architecture
gates. A deliberately selected GPU run needs a separate backend/toolchain setup
and visibility settings; HIP kernel tests require eligible gfx10/gfx11 hardware,
and some tests need two GPUs. A skipped kernel test, a Torch fallback, or a green
CPU suite does not qualify GPU kernels, full-model memory fit or native serving.
The dated **941 passed / 61 skipped** record is the observed CPU/MCP gate, not a
promise that future dependency versions or accelerator environments are qualified.
