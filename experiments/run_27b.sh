#!/usr/bin/env bash
# Qwen3.8-27B (BF16) -> rdna_hint_flex at about 3.0 bpw with QAD, on two V620s.
# Phase 1 and the encode use both GPUs (calibration split, rows sharded); QAD trains on the
# first. Holds one lock file per GPU. No `hi2 eval`: the 27B original does not fit one GPU, so
# score the result in the fork's vLLM (see docs/results.md).
# usage: run_27b.sh TAG [hi2 quantize args...]       (re-run the same command to resume)
#   HIPS=2,3 (default) the HIP devices to use; OUT=<runs dir> (default runs27b);
#   WAIT=1 waits for the GPUs instead of failing; DRY=1 prints the plan only.
# Defaults: --target-bpw 3.0 --qad-preset fast on the mixed calibration text; later flags win.
# Logs: $OUT/TAG.quant.log, TAG.t0 (start) and TAG.t1 (end).
set -u
TAG=$1; shift
B=/home/chenco_adm/w4a8_runs/hadamard-int2
. "$(dirname "$0")/box_env.sh"
HIPS=${HIPS:-2,3}
OUT=${OUT:-$B/runs27b}
MODEL=${MODEL:-$(ls -d /home/chenco_adm/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/* | head -1)}
mkdir -p "$OUT"
LOCKS=()
for h in ${HIPS//,/ }; do LOCKS+=("$B/hi27b.gpu$h.lock"); done
busy() {  # another job's lock on a GPU, or rocm-smi shows use or VRAM (rocm-smi GPU[i] = HIP i-2)
  local h l r use mem
  for h in ${HIPS//,/ }; do
    r=$((h + 2))
    for l in "$B"/*.gpu"$h".lock; do
      [ -e "$l" ] && [ "$l" != "$B/hi27b.gpu$h.lock" ] && return 0
    done
    use=$(rocm-smi -d "$r" --showuse | awk -F: '/GPU use/{gsub(/ /,"",$NF); print $NF}')
    mem=$(rocm-smi -d "$r" --showmemuse | awk -F: '/Allocated/{gsub(/ /,"",$NF); print $NF}')
    { [ "${use:-100}" != 0 ] || [ "${mem:-100}" -ge 5 ]; } && return 0
  done
  return 1
}
acquire() {  # all GPU locks or none (atomic per file; our own names, other jobs use theirs)
  local got=() l
  for l in "${LOCKS[@]}"; do
    if ( set -C; echo "$TAG $$ $(date +%s)" > "$l" ) 2> /dev/null; then
      got+=("$l")
    else
      [ ${#got[@]} -gt 0 ] && rm -f "${got[@]}"
      return 1
    fi
  done
}
if [ "${DRY:-0}" != 1 ]; then
  until ! busy && acquire; do
    [ "${WAIT:-0}" = 1 ] || { echo "HIP $HIPS is busy or locked" >&2; exit 3; }
    sleep 60
  done
  trap 'rm -f "${LOCKS[@]}"' EXIT
fi
export HIP_VISIBLE_DEVICES=$HIPS OMP_NUM_THREADS=32
ndev=$(awk -F, '{print NF}' <<< "$HIPS")
devs=$(seq -s, 0 $((ndev - 1)) | sed 's/\([0-9]\+\)/cuda:\1/g')
date +%s > "$OUT/$TAG.t0"
rm -f "$OUT/$TAG.t1"
EXTRA=()
[ "${DRY:-0}" = 1 ] && EXTRA=(--dry-run)
"$PY" -m hi2 quantize "$MODEL" "$OUT/$TAG" --workdir "$OUT/$TAG.work" --devices "$devs" \
  --calib "$B/calib_qad.txt" --calib-samples 128 --seq-len 2048 \
  --target-bpw 3.0 --qad-preset fast "${EXTRA[@]}" "$@" > "$OUT/$TAG.quant.log" 2>&1
echo "quant_exit=$?" >> "$OUT/$TAG.quant.log"
date +%s > "$OUT/$TAG.t1"
