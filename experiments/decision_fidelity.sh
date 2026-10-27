#!/usr/bin/env bash
# Serve each model in the fork's vLLM on one GPU, run the decision-fidelity prompt set against
# it (decision_fidelity.py), and keep the decisions as $ROOT/TAG.json.
# usage: GPU=4 PORT=18731 decision_fidelity.sh TAG=MODEL_DIR [TAG=MODEL_DIR ...]
#   TREE=<fork tree> (default the rebased rdna_hint_flex tree), ROOT=<results dir>.
# Compare afterwards: decision_fidelity.py compare $ROOT/dense-a.json $ROOT/dense-b.json ...
set -u
B=/home/chenco_adm/w4a8_runs/hadamard-int2
GPU=${GPU:?HIP device (2-5) required}
PORT=${PORT:-18731}
TREE=${TREE:-/home/chenco_adm/vllm-rdna-hi2-rebase}
ROOT=${ROOT:-$B/decision}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$ROOT"
LOCK=$B/hidecision.gpu$GPU.lock
( set -C; echo "$$ $(date +%s)" > "$LOCK" ) 2> /dev/null || { echo "HIP $GPU is locked" >&2; exit 3; }
trap 'rm -f "$LOCK"' EXIT
wait_vram_free() {  # a boot right after a teardown sees the old process's VRAM
  local u
  for _ in $(seq 1 36); do
    u=$(rocm-smi -d $((GPU + 2)) --showmeminfo vram 2> /dev/null | awk -F': ' '/Total Used Memory/ {print $NF}')
    ((${u:-0} < 1073741824)) && return 0
    sleep 5
  done
}
for arm in "$@"; do
  tag=${arm%%=*}
  model=${arm#*=}
  wait_vram_free
  echo "[$(date +%H:%M:%S)] arm $tag $model GPU=$GPU"
  VENV=${VENV:-$HOME/Apps/vllm/venv-7.14.0_0.31.0-final} RECIPE=full MODEL=$model \
    GPUS=$GPU PORT=$PORT TAG=$tag OUT=$ROOT/serve-$tag CACHE=$ROOT/cache-$tag \
    CELLS=none CONC_PROBE=0 PREFIX_PROBE=0 LOAD_HOOK="$HERE/decision_fidelity_hook.sh" \
    bash "$TREE/tools/rdna/port_v031/serve_validate.sh" TP=1 MAXLEN=32768 SEQS=4 \
    KV=6000000000 TUNABLEOP=1
  cp "$ROOT/serve-$tag/decisions.json" "$ROOT/$tag.json" 2> /dev/null
  echo "[$(date +%H:%M:%S)] arm $tag done: $(tail -1 "$ROOT/serve-$tag/load_hook.txt" 2> /dev/null)"
done
