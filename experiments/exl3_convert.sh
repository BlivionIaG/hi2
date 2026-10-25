#!/usr/bin/env bash
# Convert a model with upstream exllamav3 (built by exl3_build.sh) on one gfx1030 GPU.
# usage: HIP_VISIBLE_DEVICES=5 exl3_convert.sh <hf_model_dir> <bits> <out_dir> [work_dir] [extra convert.py args...]
# Uses the mul1 codebook and a 6-bit head (exllamav3 defaults) and its own calibration data.
set -eu
M=$1; BITS=$2; OUT=$3; WORK=${4:-$OUT.work}; shift $(( $# > 3 ? 4 : $# ))
W=${EXL3_DIR:-/home/chenco_adm/w4a8_runs/hadamard-int2/exl3}
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final; SL=$V/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SL/_rocm_sdk_libraries/lib:$SL/_rocm_sdk_core/lib/host-math/lib:$SL/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SL/_rocm_sdk_core/lib/core/lib:$SL/torch/lib
export PYTHONFAULTHANDLER=1 ROCM_HOME=/opt/rocm/core-7.14 ROCM_PATH=/opt/rocm/core-7.14 HIP_PATH=/opt/rocm/core-7.14
export PATH=/opt/rocm/core-7.14/bin:$PATH PYTORCH_ROCM_ARCH=gfx1030 TORCH_EXTENSIONS_DIR=$W/ext_build PYTHONPATH=$W/upstream
rm -rf "$OUT" "$WORK"
date +%s > "$OUT.t0"
cd "$W/upstream"
$V/bin/python convert.py -i "$M" -w "$WORK" -o "$OUT" -b "$BITS" -hb 6 -cb mul1 -d 0 "$@" > "$OUT.log" 2>&1
echo "convert_exit=$?" >> "$OUT.log"
date +%s > "$OUT.t1"
rm -rf "$WORK"   # intermediate states are large and not needed once the output is written
