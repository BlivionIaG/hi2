#!/usr/bin/env bash
# Multiple-choice accuracy of an HI2 checkpoint, same recipe as mc4b_when_idle.sh (dense fp16
# export on the CPU, then vllm_mc.sh: the fork's vLLM, one GPU, eager), but holding our lock file
# hilowbit.gpuN.lock for the GPU part and waiting until no other job locks or uses that GPU.
# usage: mc_lowbit.sh <hi2 checkpoint dir> <tag> <port> <hip> [source model dir]   (box paths)
#   source model: default the Qwen3.5-4B snapshot in m4b.path. Results: mc4b/hi2-lb-<tag>.mc.json
set -u
B=/home/chenco_adm/w4a8_runs/hadamard-int2
CK=$1; TAG=$2; PORT=$3; HIP=$4; SRC=${5:-$(cat "$B/m4b.path")}
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/box_env.sh"
D=$B/mc_data/ikawrakow__validation-datasets-for-llama.cpp
DENSE=$B/mc4b/dense-lb-$TAG
HIP_VISIBLE_DEVICES='' "$PY" -m hi2 export-dense "$SRC" "$CK" "$DENSE" > "$B/mc4b/export-lb-$TAG.log" 2>&1
LOCK=$B/hilowbit.gpu$HIP.lock
busy() {
  local l r=$((HIP + 2)) use mem
  for l in "$B"/*.gpu"$HIP".lock; do
    [ -e "$l" ] && [ "$l" != "$LOCK" ] && return 0
  done
  use=$(rocm-smi -d "$r" --showuse | awk -F: '/GPU use/{gsub(/ /,"",$NF); print $NF}')
  mem=$(rocm-smi -d "$r" --showmemuse | awk -F: '/Allocated/{gsub(/ /,"",$NF); print $NF}')
  [ "${use:-100}" != 0 ] || [ "${mem:-100}" -ge 5 ]
}
until ! busy && ( set -C; echo "mc-$TAG $$ $(date +%s)" > "$LOCK" ) 2>/dev/null; do sleep 60; done
trap 'rm -f "$LOCK"' EXIT
HIP_VISIBLE_DEVICES=$HIP bash "$HERE/vllm_mc.sh" "$DENSE" "hi2-lb-$TAG" "$PORT" "$D" "$B/mc4b"
rm -rf "$DENSE"
