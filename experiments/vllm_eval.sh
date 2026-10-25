#!/usr/bin/env bash
# Serve one model with the fork's vLLM (one GPU, eager) and score it with vllm_ppl.py.
# usage: HIP_VISIBLE_DEVICES=4 vllm_eval.sh <model_dir> <tag> <port> <text_file> <out_dir> [KEY=value serve overrides...]
set -u
MODEL=$1; TAG=$2; PORT=$3; TEXT=$4; OUT=$5; shift 5
TREE=${VLLM_TREE:-/home/chenco_adm/vllm-rdna-0.31.0-exl3}
VENV=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
# serve_rdna.sh kills every vLLM process that shares VLLM_CACHE_ROOT at launch; a private root per
# server lets several servers run at once without killing each other
export VLLM_CACHE_ROOT="$OUT/cache-$TAG/vllm"
mkdir -p "$VLLM_CACHE_ROOT"
cd "$TREE" || exit 1
VENV=$VENV setsid bash tools/rdna/serve_rdna.sh RECIPE=27b-exl3 MODEL="$MODEL" SERVED_NAME="$TAG" TP=1 EAGER=1 MAXLEN=4096 SEQS=2 PORT="$PORT" GMEM=0.85 KVCAP=3000000000 "$@" > "$OUT/$TAG.server.log" 2>&1 < /dev/null &
SPID=$!
echo "$SPID" > "$OUT/$TAG.server.pid"
up=0
for _ in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
  kill -0 "$SPID" 2> /dev/null || break
  sleep 5
done
if [ "$up" = 1 ]; then
  $VENV/bin/python "$HERE/vllm_ppl.py" --port "$PORT" --model-dir "$MODEL" --served-name "$TAG" --text "$TEXT" > "$OUT/$TAG.ppl.log" 2>&1
else
  echo "server did not come up" > "$OUT/$TAG.ppl.log"
fi
kill -- "-$SPID" 2> /dev/null || kill "$SPID" 2> /dev/null
sleep 3
rm -rf "$OUT/cache-$TAG"
rm -rf "$OUT/cache-$TAG"
echo "done" > "$OUT/$TAG.done"
