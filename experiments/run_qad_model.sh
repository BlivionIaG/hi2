#!/usr/bin/env bash
# Quantize a model with hi2 (any args, e.g. --qad) and score it with `hi2 eval` on the repo-docs
# held-out text (skip 800000) and on the mixed held-out text (skip 0).
# usage: MODEL=<hf dir> OUT=<runs dir> HIP_VISIBLE_DEVICES=N run_qad_model.sh TAG [hi2 quantize args...]
# Logs: $OUT/TAG.quant.log, TAG.eval.log (docs), TAG.evalmix.log (mix); TAG.t0/t1/t2 timestamps.
# KEEP_WORK=1 keeps the workdir (QAD per-layer artifacts, tables); KEEP_HESSIAN=1 also keeps the
# Hessians (so other runs can share phase 1). SHARE_FROM=<workdir of a finished run with the same
# calibration settings> hard-links its phase-1 tables, statistics, Hessians and sensitivity first, so
# only the QAD stage runs (the fingerprint check in hi2 still decides whether they are reused).
set -u
TAG=$1; shift
HERE=$(cd "$(dirname "$0")/.." && pwd)
B=/home/chenco_adm/w4a8_runs/hadamard-int2
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final; SL=$V/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SL/_rocm_sdk_libraries/lib:$SL/_rocm_sdk_core/lib/host-math/lib:$SL/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SL/_rocm_sdk_core/lib/core/lib:$SL/torch/lib
export PYTHONPATH=$HERE OMP_NUM_THREADS=32
mkdir -p "$OUT"
if [ -n "${SHARE_FROM:-}" ]; then
  mkdir -p "$OUT/$TAG.work"
  for f in tables stats hessian phase1.json sensitivity.json sensitivity.key.json; do
    [ -e "$SHARE_FROM/$f" ] && cp -al "$SHARE_FROM/$f" "$OUT/$TAG.work/"
  done
fi
rm -f "$OUT/$TAG.t1" "$OUT/$TAG.t2" "$OUT/$TAG.eval.log" "$OUT/$TAG.evalmix.log"  # stale from a killed run
date +%s > "$OUT/$TAG.t0"
$V/bin/python -m hi2 quantize "$MODEL" "$OUT/$TAG" --workdir "$OUT/$TAG.work" --devices cuda:0 "$@" > "$OUT/$TAG.quant.log" 2>&1
echo "quant_exit=$?" >> "$OUT/$TAG.quant.log"
date +%s > "$OUT/$TAG.t1"
[ "${KEEP_HESSIAN:-0}" = 1 ] || rm -rf "$OUT/$TAG.work/hessian"
[ "${KEEP_WORK:-0}" = 1 ] || [ "${KEEP_HESSIAN:-0}" = 1 ] || rm -rf "$OUT/$TAG.work"
$V/bin/python -m hi2 eval "$MODEL" "$OUT/$TAG" --text "$B/study_text_all.txt" --skip-tokens 800000 --samples 8 --seq-len 2048 --device cuda:0 > "$OUT/$TAG.eval.log" 2>&1
echo "eval_exit=$?" >> "$OUT/$TAG.eval.log"
$V/bin/python -m hi2 eval "$MODEL" "$OUT/$TAG" --text "$B/eval_mix.txt" --skip-tokens 0 --samples 8 --seq-len 2048 --device cuda:0 > "$OUT/$TAG.evalmix.log" 2>&1
echo "eval_exit=$?" >> "$OUT/$TAG.evalmix.log"
date +%s > "$OUT/$TAG.t2"
