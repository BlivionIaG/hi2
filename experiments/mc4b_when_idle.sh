#!/usr/bin/env bash
# Dense-export an HI2 4B checkpoint on the CPU, then score it with vllm_mc.sh on the first of
# our GPUs that is really idle (when_idle.sh). Same recipe as chain_mc4b_hi2.sh.
# usage: mc4b_when_idle.sh <run tag> <port> ["<hip indices>"]   (box paths)
set -u
B=/home/chenco_adm/w4a8_runs/hadamard-int2
TAG=$1; PORT=$2; CANDS=${3:-2 3 4 5}
HERE=$(cd "$(dirname "$0")" && pwd)
D=$B/mc_data/ikawrakow__validation-datasets-for-llama.cpp
M4=$(cat "$B/m4b.path")
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final; SL=$V/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SL/_rocm_sdk_libraries/lib:$SL/_rocm_sdk_core/lib/host-math/lib:$SL/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SL/_rocm_sdk_core/lib/core/lib:$SL/torch/lib
HIP_VISIBLE_DEVICES='' PYTHONPATH=$HERE/.. "$V/bin/python" -m hi2 export-dense "$M4" "$B/runs4b/$TAG" "$B/mc4b/dense-$TAG" > "$B/mc4b/export-$TAG.log" 2>&1
bash "$HERE/when_idle.sh" "$CANDS" bash "$B/hi2/experiments/vllm_mc.sh" "$B/mc4b/dense-$TAG" "hi2-4b-$TAG" "$PORT" "$D" "$B/mc4b"
rm -rf "$B/mc4b/dense-$TAG"
