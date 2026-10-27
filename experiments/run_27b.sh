#!/usr/bin/env bash
# Qwen3.8-27B (BF16) -> rdna_hint_flex at about 3.0 bpw with QAD on several GPUs.
# Phase 1 and the encode use every listed GPU (calibration split, rows sharded); QAD trains on
# the first. No `hi2 eval`: the 27B original does not fit one GPU, so score the result in the
# fork's vLLM (see docs/results.md).
# usage: MODEL=<hf dir> CALIB=<calibration text> run_27b.sh TAG [hi2 quantize args...]
#   re-run the same command to resume. HIPS=0,1 (default) the visible devices to use;
#   OUT=<runs dir> (default runs27b); DRY=1 prints the plan only.
# Defaults: --target-bpw 3.0 --qad-preset fast; later flags win.
# Logs: $OUT/TAG.quant.log, TAG.t0 (start) and TAG.t1 (end).
set -u
TAG=$1; shift
MODEL=${MODEL:?set MODEL to the Hugging Face model directory}
CALIB=${CALIB:?set CALIB to the calibration text (see docs/usage.md, hi2 calib)}
HIPS=${HIPS:-0,1}
OUT=${OUT:-runs27b}
PY=${PY:-python}
mkdir -p "$OUT"
export HIP_VISIBLE_DEVICES=$HIPS OMP_NUM_THREADS=${OMP_NUM_THREADS:-32}
ndev=$(awk -F, '{print NF}' <<< "$HIPS")
devs=$(seq -s, 0 $((ndev - 1)) | sed 's/\([0-9]\+\)/cuda:\1/g')
date +%s > "$OUT/$TAG.t0"
rm -f "$OUT/$TAG.t1"
EXTRA=()
[ "${DRY:-0}" = 1 ] && EXTRA=(--dry-run)
"$PY" -m hi2 quantize "$MODEL" "$OUT/$TAG" --workdir "$OUT/$TAG.work" --devices "$devs" \
  --calib "$CALIB" --calib-samples 128 --seq-len 2048 \
  --target-bpw 3.0 --qad-preset fast "${EXTRA[@]}" "$@" > "$OUT/$TAG.quant.log" 2>&1
echo "quant_exit=$?" >> "$OUT/$TAG.quant.log"
date +%s > "$OUT/$TAG.t1"
