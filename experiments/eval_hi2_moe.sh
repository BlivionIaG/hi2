#!/usr/bin/env bash
# `hi2 eval` of one OLMoE HI2 checkpoint on the two held-out texts (fp32 on CPU by default).
# usage: eval_hi2_moe.sh TAG   (env: MODEL OUT EVAL_DEVICE EVAL_DTYPE)
set -u
TAG=$1
HERE=$(cd "$(dirname "$0")/.." && pwd)
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final; SL=$V/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SL/_rocm_sdk_libraries/lib:$SL/_rocm_sdk_core/lib/host-math/lib:$SL/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SL/_rocm_sdk_core/lib/core/lib:$SL/torch/lib
export PYTHONPATH=$HERE OMP_NUM_THREADS=32 PYTHONUNBUFFERED=1
H=/home/chenco_adm/w4a8_runs/hadamard-int2
MODEL=${MODEL:-/home/chenco_adm/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5}
OUT=${OUT:-$H/runsmoe}
for t in "study:$H/study_text_all.txt:800000" "mix:$H/eval_mix.txt:0"; do
  IFS=: read -r name text skip <<< "$t"
  "$V/bin/python" -m hi2 eval "$MODEL" "$OUT/$TAG" --text "$text" --skip-tokens "$skip" \
    --samples 8 --seq-len 2048 --device "${EVAL_DEVICE:-cpu}" --dtype "${EVAL_DTYPE:-float32}" \
    > "$OUT/$TAG.eval-$name.log" 2>&1
  echo "eval_exit=$?" >> "$OUT/$TAG.eval-$name.log"
done
