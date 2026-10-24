#!/usr/bin/env bash
# Quantize a model with hi2 and score it with `hi2 eval` (held-out text, HF evaluator).
# usage: MODEL=<hf dir> TEXT=<text file> OUT=<runs dir> [EVAL_SKIP=800000] \
#        HIP_VISIBLE_DEVICES=2 run_hi2_model.sh TAG DEVICES [hi2 quantize args...]
set -u
TAG=$1; DEVS=$2; shift 2
HERE=$(cd "$(dirname "$0")/.." && pwd)
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final; SL=$V/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SL/_rocm_sdk_libraries/lib:$SL/_rocm_sdk_core/lib/host-math/lib:$SL/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SL/_rocm_sdk_core/lib/core/lib:$SL/torch/lib
export PYTHONPATH=$HERE OMP_NUM_THREADS=32
mkdir -p "$OUT"; rm -rf "${OUT:?}/$TAG"
date +%s > "$OUT/$TAG.t0"
$V/bin/python -m hi2 quantize "$MODEL" "$OUT/$TAG" --workdir "$OUT/$TAG.work" --devices "$DEVS" "$@" > "$OUT/$TAG.quant.log" 2>&1
echo "quant_exit=$?" >> "$OUT/$TAG.quant.log"
date +%s > "$OUT/$TAG.t1"
EVD=cpu; case "$DEVS" in cuda*) EVD=${DEVS%%,*};; esac
$V/bin/python -m hi2 eval "$MODEL" "$OUT/$TAG" --text "$TEXT" --skip-tokens "${EVAL_SKIP:-800000}" --samples 8 --seq-len 2048 --device "$EVD" > "$OUT/$TAG.eval.log" 2>&1
echo "eval_exit=$?" >> "$OUT/$TAG.eval.log"
date +%s > "$OUT/$TAG.t2"
