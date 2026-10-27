#!/usr/bin/env bash
# serve_validate.sh LOAD_HOOK: run the bench suite against the server it booted (PORT, SERVED,
# OUT and MODEL are exported by serve_validate.sh). TASKS, LIMIT, WORKERS and EXTRA come from
# serve_run.sh. Ends with PASS or FAIL.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
# REF_RESULTS (the reference run's bench.json; default: this run's own, for the reference arm)
# also gives the teacher-forced domain KLD: lp.json next to bench.json.
if "$PY" "$HERE/run.py" --url "http://127.0.0.1:$PORT" --model "$SERVED" \
  --tasks "${TASKS:-all}" ${LIMIT:+--limit "$LIMIT"} --workers "${WORKERS:-16}" \
  --out "$OUT/bench.json" ${EXTRA:-}; then
  "$PY" "$HERE/tfkld.py" score --url "http://127.0.0.1:$PORT" --model "$SERVED" \
    --ref-results "${REF_RESULTS:-$OUT/bench.json}" --limit "${TF_LIMIT:-200}" \
    --out "$OUT/lp.json" && echo PASS || echo FAIL
else
  echo FAIL
fi
