#!/usr/bin/env bash
# Build upstream exllamav3 (>= 1.6, which has an RDNA/ROCm path) for gfx1030 in a scratch
# directory, without touching the frozen venv: the JIT extension is built into <work_dir>/ext_build
# and the package is imported from <work_dir>/upstream via PYTHONPATH.
# usage: exl3_build.sh <work_dir>
set -eu
W=$1
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final; SL=$V/lib/python3.12/site-packages
mkdir -p "$W"
[ -d "$W/upstream" ] || git clone --depth 1 https://github.com/turboderp-org/exllamav3 "$W/upstream"
# gfx1030 has no bf16 dot instruction, so Triton cannot compile the vendored gated-delta-net kernels
# with bf16 inputs ("Cannot select: intrinsic llvm.amdgcn.fdot2.bf16.bf16"). On ROCm, run that
# one entry point in fp16 and cast the result back. Idempotent.
python3 - "$W/upstream/exllamav3/vendor/fla/__init__.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
if "HI2-PATCH" not in s:
    s = s.replace(
        "@input_guard\ndef chunk_gated_delta_rule(",
        "# HI2-PATCH: fp16 wrapper for ROCm RDNA2 (no bf16 dot2)\n"
        "def chunk_gated_delta_rule(q, k, v, g, beta, *args, **kwargs):\n"
        "    if torch.version.hip and q.dtype == torch.bfloat16:\n"
        "        dt = q.dtype\n"
        "        o, st = _chunk_gated_delta_rule(\n"
        "            q.half(), k.half(), v.half(), g, beta.half() if beta.dtype == torch.bfloat16 else beta,\n"
        "            *args, **kwargs)\n"
        "        return o.to(dt), st\n"
        "    return _chunk_gated_delta_rule(q, k, v, g, beta, *args, **kwargs)\n\n\n"
        "@input_guard\ndef _chunk_gated_delta_rule(",
        1,
    )
    open(p, "w").write(s)
PY
export LD_LIBRARY_PATH=$SL/_rocm_sdk_libraries/lib:$SL/_rocm_sdk_core/lib/host-math/lib:$SL/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SL/_rocm_sdk_core/lib/core/lib:$SL/torch/lib
export ROCM_HOME=/opt/rocm/core-7.14 ROCM_PATH=/opt/rocm/core-7.14 HIP_PATH=/opt/rocm/core-7.14
export PATH=/opt/rocm/core-7.14/bin:$PATH
export PYTORCH_ROCM_ARCH=gfx1030 TORCH_EXTENSIONS_DIR=$W/ext_build MAX_JOBS=${MAX_JOBS:-24}
export PYTHONPATH=$W/upstream
env HIP_VISIBLE_DEVICES='' $V/bin/python -c "
import exllamav3.ext as e
print('EXL3 EXT OK', e.exllamav3_ext)
"
