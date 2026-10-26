#!/usr/bin/env bash
# One PTQ run (quantize + both held-out evals) with the code of CODE (default: this
# copy), on HIP device HIP, under runsfast/. Times go to <tag>.t0/.t1/.t2.
# usage: CODE=<hi2 dir> fast_ptq.sh HIP TAG MODEL CALIB [hi2 quantize args...]
set -u
HIP=$1; TAG=$2; MODEL=$3; CALIB=$4; shift 4
B=/home/chenco_adm/w4a8_runs/hadamard-int2
HERE=$(cd "$(dirname "$0")/.." && pwd)
CODE=${CODE:-$HERE}
. "$HERE/experiments/box_env.sh"
export PYTHONPATH=$CODE OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} HIP_VISIBLE_DEVICES=$HIP
O=$B/runsfast; mkdir -p "$O"
LOCK=$B/hifast.gpu$HIP.lock
for l in "$B"/*.gpu"$HIP".lock; do  # someone else holds this GPU: refuse
  [ -e "$l" ] && [ "$l" != "$LOCK" ] && { echo "HIP $HIP is locked by $l" >&2; exit 3; }
done
# atomic: two of our own launchers can race for the same GPU too
( set -C; echo "$TAG $$ $(date +%s)" > "$LOCK" ) 2>/dev/null || { echo "HIP $HIP is busy ($LOCK)" >&2; exit 3; }
trap 'rm -f "$LOCK"' EXIT
rm -rf "${O:?}/$TAG" "$O/$TAG.work"
cd "$CODE" || exit 1  # python -m puts the cwd first on sys.path: run from the code under test
date +%s > "$O/$TAG.t0"
"$PY" -m hi2 quantize "$MODEL" "$O/$TAG" --workdir "$O/$TAG.work" --devices cuda:0 \
  --calib "$CALIB" --calib-samples "${CALIB_SAMPLES:-128}" --seq-len 2048 "$@" > "$O/$TAG.quant.log" 2>&1
echo "quant_exit=$?" >> "$O/$TAG.quant.log"
date +%s > "$O/$TAG.t1"
[ -n "${NO_EVAL:-}" ] && exit 0
"$PY" -m hi2 eval "$MODEL" "$O/$TAG" --text "$B/study_text_all.txt" --skip-tokens 800000 \
  --samples 8 --seq-len 2048 --device cuda:0 > "$O/$TAG.eval.log" 2>&1
"$PY" -m hi2 eval "$MODEL" "$O/$TAG" --text "$B/eval_mix.txt" --skip-tokens 0 \
  --samples 8 --seq-len 2048 --device cuda:0 > "$O/$TAG.evalmix.log" 2>&1
date +%s > "$O/$TAG.t2"
