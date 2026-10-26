#!/usr/bin/env bash
# Restack the OLMoE HI2 checkpoints (per-expert -> stacked, no re-encode), prove the
# weights bit-identical, round-trip back to per-expert, and score the stacked copies
# and the originals with `hi2 eval` (fp32 CPU) on both held-out texts.
# usage: restack_olmoe.sh [restack|eval|all]   (from the hi2-stack copy on the box)
set -u
STEP=${1:-all}
HERE=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=/dev/null
. "$HERE/experiments/box_env.sh"
export HIP_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
H=/home/chenco_adm/w4a8_runs/hadamard-int2
RUNS=$H/runsmoe
LOG=$H/hi2-stack-logs
MODEL=/home/chenco_adm/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5
mkdir -p "$LOG"
cd "$HERE" || exit 1
if [ "$STEP" = restack ] || [ "$STEP" = all ]; then
  export OMP_NUM_THREADS=32
  for t in olmoe-r4 olmoe-b3.0; do
    rm -rf "${RUNS:?}/$t-stacked" "${LOG:?}/$t-back"
    /usr/bin/time -v "$PY" -m hi2 restack "$RUNS/$t" "$RUNS/$t-stacked" --layout stacked \
      > "$LOG/$t.restack.json" 2> "$LOG/$t.restack.log"
    echo "restack_exit=$?" >> "$LOG/$t.restack.log"
    "$PY" experiments/compare_layouts.py "$RUNS/$t" "$RUNS/$t-stacked" \
      > "$LOG/$t.compare.json" 2> "$LOG/$t.compare.log"
    echo "compare_exit=$?" >> "$LOG/$t.compare.log"
    # stacked -> per-expert must give back the original arrays
    "$PY" -m hi2 restack "$RUNS/$t-stacked" "$LOG/$t-back" --layout per-expert \
      > "$LOG/$t.back.json" 2>> "$LOG/$t.restack.log"
    "$PY" experiments/compare_layouts.py "$RUNS/$t" "$LOG/$t-back" \
      > "$LOG/$t.back-compare.json" 2> "$LOG/$t.back-compare.log"
    echo "back_compare_exit=$?" >> "$LOG/$t.back-compare.log"
    rm -rf "${LOG:?}/$t-back"
  done
fi
if [ "$STEP" = eval ] || [ "$STEP" = all ]; then
  export OMP_NUM_THREADS=16
  for tag in olmoe-r4 olmoe-r4-stacked olmoe-b3.0 olmoe-b3.0-stacked; do
    (
      for spec in "study:$H/study_text_all.txt:800000" "mix:$H/eval_mix.txt:0"; do
        IFS=: read -r name text skip <<< "$spec"
        "$PY" -m hi2 eval "$MODEL" "$RUNS/$tag" --text "$text" --skip-tokens "$skip" \
          --samples 8 --seq-len 2048 --device cpu --dtype float32 \
          > "$LOG/$tag.eval-$name.log" 2>&1
        echo "eval_exit=$?" >> "$LOG/$tag.eval-$name.log"
      done
    ) &
  done
  wait
fi
