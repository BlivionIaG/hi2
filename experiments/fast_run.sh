#!/usr/bin/env bash
# Run one hi2 command on one of our GPUs with a lock file, logging under runsfast/.
# usage: fast_run.sh HIP TAG -- <python args...>   (from the rsynced hi2-fast copy)
set -u
HIP=$1; TAG=$2; shift 3
B=/home/chenco_adm/w4a8_runs/hadamard-int2
. "$(dirname "$0")/box_env.sh"
mkdir -p "$B/runsfast/logs"
LOCK=$B/hifast.gpu$HIP.lock
for l in "$B"/*.gpu"$HIP".lock; do  # someone else holds this GPU: refuse
  [ -e "$l" ] && [ "$l" != "$LOCK" ] && { echo "HIP $HIP is locked by $l" >&2; exit 3; }
done
# atomic: two of our own launchers can race for the same GPU too
( set -C; echo "$TAG $$ $(date +%s)" > "$LOCK" ) 2>/dev/null || { echo "HIP $HIP is busy ($LOCK)" >&2; exit 3; }
trap 'rm -f "$LOCK"' EXIT
date +%s > "$B/runsfast/logs/$TAG.t0"
HIP_VISIBLE_DEVICES=$HIP OMP_NUM_THREADS=32 "$PY" "$@" > "$B/runsfast/logs/$TAG.log" 2>&1
echo "exit=$?" >> "$B/runsfast/logs/$TAG.log"
date +%s > "$B/runsfast/logs/$TAG.t1"
