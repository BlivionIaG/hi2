#!/usr/bin/env bash
# Low-bit study runner (docs/lowbit.md): quantize with hi2 on one of our GPUs under an
# atomic lock file, then score with `hi2 eval` on the docs held-out text (skip 800000) and the
# mixed held-out text (skip 0). Refuses a GPU that another job locks or uses (rocm-smi).
# usage: lowbit_run.sh HIP TAG [hi2 quantize args...]   (from the rsynced lowbit copy)
#   MODEL=<hf dir> (default Qwen3.5-0.8B), OUT=<runs dir> (default runslow), KEEP_WORK=1 keeps the
#   workdir, NO_EVAL=1 skips the two evals, WAIT=1 waits for the GPU instead of failing,
#   SHARE_QAD=<workdir of a finished QAD run> copies (not links: hi2 rewrites stale artifacts in
#   place) its per-layer QAD artifacts first, so a run that differs only in the final logit-KL
#   phase reuses the layer-wise phase (hi2's fingerprint still decides).
# Logs: $OUT/TAG.quant.log, TAG.eval.log (docs), TAG.evalmix.log (mix), TAG.t0/t1/t2.
set -u
HIP=$1; TAG=$2; shift 2
B=/home/chenco_adm/w4a8_runs/hadamard-int2
. "$(dirname "$0")/box_env.sh"
MODEL=${MODEL:-/home/chenco_adm/models/Qwen3.5-0.8B}
OUT=${OUT:-$B/runslow}
mkdir -p "$OUT"
LOCK=$B/hilowbit.gpu$HIP.lock
busy() {  # another lock on this GPU, or rocm-smi shows use or VRAM (rocm-smi GPU[i] = HIP i-2)
  local l r=$((HIP + 2)) use mem
  for l in "$B"/*.gpu"$HIP".lock; do
    [ -e "$l" ] && [ "$l" != "$LOCK" ] && return 0
  done
  use=$(rocm-smi -d "$r" --showuse | awk -F: '/GPU use/{gsub(/ /,"",$NF); print $NF}')
  mem=$(rocm-smi -d "$r" --showmemuse | awk -F: '/Allocated/{gsub(/ /,"",$NF); print $NF}')
  [ "${use:-100}" != 0 ] || [ "${mem:-100}" -ge 5 ]
}
# atomic: our own queued runs race for the same lock name, other jobs use other names
until ! busy && ( set -C; echo "$TAG $$ $(date +%s)" > "$LOCK" ) 2>/dev/null; do
  [ "${WAIT:-0}" = 1 ] || { echo "HIP $HIP is busy or locked" >&2; exit 3; }
  sleep 60
done
trap 'rm -f "$LOCK"' EXIT
export HIP_VISIBLE_DEVICES=$HIP OMP_NUM_THREADS=32
rm -f "$OUT/$TAG.t1" "$OUT/$TAG.t2" "$OUT/$TAG.eval.log" "$OUT/$TAG.evalmix.log"
date +%s > "$OUT/$TAG.t0"
if [ -n "${SHARE_QAD:-}" ]; then
  mkdir -p "$OUT/$TAG.work"
  rm -rf "$OUT/$TAG.work/qad"
  cp -r "$SHARE_QAD/qad" "$OUT/$TAG.work/qad"
  rm -f "$OUT/$TAG.work/qad/final_kl.pt"
fi
"$PY" -m hi2 quantize "$MODEL" "$OUT/$TAG" --workdir "$OUT/$TAG.work" --devices cuda:0 "$@" > "$OUT/$TAG.quant.log" 2>&1
echo "quant_exit=$?" >> "$OUT/$TAG.quant.log"
date +%s > "$OUT/$TAG.t1"
[ "${KEEP_WORK:-0}" = 1 ] || rm -rf "$OUT/$TAG.work"
if [ "${NO_EVAL:-0}" != 1 ]; then
  "$PY" -m hi2 eval "$MODEL" "$OUT/$TAG" --text "$B/study_text_all.txt" --skip-tokens 800000 --samples 8 --seq-len 2048 --device cuda:0 > "$OUT/$TAG.eval.log" 2>&1
  echo "eval_exit=$?" >> "$OUT/$TAG.eval.log"
  "$PY" -m hi2 eval "$MODEL" "$OUT/$TAG" --text "$B/eval_mix.txt" --skip-tokens 0 --samples 8 --seq-len 2048 --device cuda:0 > "$OUT/$TAG.evalmix.log" 2>&1
  echo "eval_exit=$?" >> "$OUT/$TAG.evalmix.log"
fi
date +%s > "$OUT/$TAG.t2"
