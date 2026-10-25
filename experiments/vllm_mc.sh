#!/usr/bin/env bash
# Serve one model with the fork's vLLM (one GPU, eager) and run mc_eval.py against it.
# usage: HIP_VISIBLE_DEVICES=3 vllm_mc.sh <model_dir> <tag> <port> <data_dir> <out_dir> [tokenizer_dir]
set -u
MODEL=$1; TAG=$2; PORT=$3; DATA=$4; OUT=$5; TOKDIR=${6:-$1}
TREE=${VLLM_TREE:-/home/chenco_adm/vllm-rdna-0.31.0-exl3}
VENV=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
cd "$TREE" || exit 1
VENV=$VENV setsid bash tools/rdna/serve_rdna.sh RECIPE=27b-exl3 MODEL="$MODEL" SERVED_NAME="$TAG" TP=1 EAGER=1 MAXLEN=4096 SEQS=16 PORT="$PORT" GMEM=0.85 KVCAP=4000000000 > "$OUT/$TAG.server.log" 2>&1 < /dev/null &
SPID=$!
up=0
for _ in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
  kill -0 "$SPID" 2> /dev/null || break
  sleep 5
done
if [ "$up" = 1 ]; then
  $VENV/bin/python "$HERE/mc_eval.py" --port "$PORT" --model-dir "$TOKDIR" --served-name "$TAG" --data "$DATA" --out "$OUT/$TAG.mc.json" > "$OUT/$TAG.mc.log" 2>&1
else
  echo "server did not come up" > "$OUT/$TAG.mc.log"
fi
kill -- "-$SPID" 2> /dev/null || kill "$SPID" 2> /dev/null
sleep 3
echo "done" > "$OUT/$TAG.mc.done"
