#!/usr/bin/env bash
# Re-export an HI2 4B checkpoint as dense fp16 and score it with vllm_mc.sh.
# usage: HIP_VISIBLE_DEVICES=N chain_mc4b_hi2.sh <run tag> <port>
B=/home/chenco_adm/w4a8_runs/hadamard-int2
TAG=$1; PORT=$2
D=$B/mc_data/ikawrakow__validation-datasets-for-llama.cpp
M4=$(cat "$B/m4b.path")
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final; SL=$V/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SL/_rocm_sdk_libraries/lib:$SL/_rocm_sdk_core/lib/host-math/lib:$SL/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SL/_rocm_sdk_core/lib/core/lib:$SL/torch/lib
GPU=${HIP_VISIBLE_DEVICES}
HIP_VISIBLE_DEVICES='' PYTHONPATH=$B/hi2 "$V/bin/python" -m hi2 export-dense "$M4" "$B/runs4b/$TAG" "$B/mc4b/dense-$TAG" > "$B/mc4b/export-$TAG.log" 2>&1
HIP_VISIBLE_DEVICES=$GPU bash "$B/hi2/experiments/vllm_mc.sh" "$B/mc4b/dense-$TAG" "hi2-4b-$TAG" "$PORT" "$D" "$B/mc4b"
rm -rf "$B/mc4b/dense-$TAG"
