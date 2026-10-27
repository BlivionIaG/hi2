#!/usr/bin/env bash
# serve_validate.sh LOAD_HOOK: run the bench suite against the server it booted (PORT, SERVED,
# OUT and MODEL are exported by serve_validate.sh). TASKS, LIMIT, WORKERS and EXTRA come from
# serve_run.sh. Ends with PASS or FAIL.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
if "$PY" "$HERE/run.py" --url "http://127.0.0.1:$PORT" --model "$SERVED" \
  --tasks "${TASKS:-all}" ${LIMIT:+--limit "$LIMIT"} --workers "${WORKERS:-16}" \
  --out "$OUT/bench.json" ${EXTRA:-}; then
  echo PASS
else
  echo FAIL
fi
