#!/usr/bin/env bash
# OLMoE (or another MoE) with hi2: one calibration (phase 1 + sensitivity probe) shared
# by two allocations (quality target and a bpw target), then `hi2 eval` of both on two
# held-out texts. Takes the box's cooperative GPU lock ($LOCKS/qad.gpuN.lock).
# usage: GPU=<hip index> run_hi2_moe.sh
#   env: MODEL CALIB OUT SAMPLES RATIO BPW EVAL_DEVICE EVAL_DTYPE
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final; SL=$V/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SL/_rocm_sdk_libraries/lib:$SL/_rocm_sdk_core/lib/host-math/lib:$SL/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SL/_rocm_sdk_core/lib/core/lib:$SL/torch/lib
export PYTHONPATH=$HERE OMP_NUM_THREADS=32 PYTHONUNBUFFERED=1
H=/home/chenco_adm/w4a8_runs/hadamard-int2
MODEL=${MODEL:-/home/chenco_adm/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5}
CALIB=${CALIB:-$H/calib_mix.txt}
OUT=${OUT:-$H/runsmoe}
SAMPLES=${SAMPLES:-178}  # calib_mix.txt holds 366k OLMoE tokens = 178 x 2048
RATIO=${RATIO:-4}
BPW=${BPW:-3.0}
EVAL_DEVICE=${EVAL_DEVICE:-cpu}
EVAL_DTYPE=${EVAL_DTYPE:-float32}
LOCK=$H/qad.gpu$GPU.lock
WORK=$OUT/olmoe.work
mkdir -p "$OUT"
until mkdir "$LOCK" 2>/dev/null; do sleep 60; done
trap 'rmdir "$LOCK"' EXIT
export HIP_VISIBLE_DEVICES=$GPU
COMMON=(--workdir "$WORK" --devices cuda:0 --calib "$CALIB" --calib-samples "$SAMPLES"
  --seq-len 2048 --sensitivity probe --sens-dtype float16)  # gfx1030 has no bf16 GEMM
quant() {  # tag, allocation flags...
  local tag=$1; shift
  rm -rf "${OUT:?}/$tag"
  date +%s > "$OUT/$tag.t0"
  /usr/bin/time -v "$V/bin/python" -m hi2 quantize "$MODEL" "$OUT/$tag" "${COMMON[@]}" "$@" \
    > "$OUT/$tag.quant.log" 2>&1
  echo "quant_exit=$?" >> "$OUT/$tag.quant.log"
  date +%s > "$OUT/$tag.t1"
}
quant "olmoe-r$RATIO" --max-error-ratio "$RATIO"
quant "olmoe-b$BPW" --target-bpw "$BPW"
rm -rf "$WORK/hessian"  # about 20 GB of per-expert Hessians, only needed by quantize
rmdir "$LOCK"; trap - EXIT
if [ "$EVAL_DEVICE" != cpu ]; then
  until mkdir "$LOCK" 2>/dev/null; do sleep 60; done
  trap 'rmdir "$LOCK"' EXIT
fi
for tag in "olmoe-r$RATIO" "olmoe-b$BPW"; do
  for t in "study:$H/study_text_all.txt:800000" "mix:$H/eval_mix.txt:0"; do
    IFS=: read -r name text skip <<< "$t"
    "$V/bin/python" -m hi2 eval "$MODEL" "$OUT/$tag" --text "$text" --skip-tokens "$skip" \
      --samples 8 --seq-len 2048 --device "$EVAL_DEVICE" --dtype "$EVAL_DTYPE" \
      > "$OUT/$tag.eval-$name.log" 2>&1
    echo "eval_exit=$?" >> "$OUT/$tag.eval-$name.log"
  done
done
date +%s > "$OUT/done"
