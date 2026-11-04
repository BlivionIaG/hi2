#!/usr/bin/env bash
# serve_validate.sh LOAD_HOOK: run the bench suite against the server it booted (PORT, SERVED,
# OUT and MODEL are exported by serve_validate.sh). TASKS, LIMIT, WORKERS and EXTRA come from
# serve_run.sh. Ends with PASS or FAIL.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
stage=benchmark
finish() {
  rc=$?
  printf '%s %s %s\n' "${BENCH_ATTEMPT:-manual}" "$rc" "$stage" > "$OUT/bench.exit"
  if ((rc == 0)); then echo PASS; else echo FAIL; fi
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# REF_RESULTS (the reference run's bench.json; default: this run's own, for the reference arm)
# also supplies teacher-forced log-probability scores: lp.json next to bench.json.
"$PY" "$HERE/run.py" --url "http://127.0.0.1:$PORT" --model "$SERVED" \
  --tasks "${TASKS:-all}" ${LIMIT:+--limit "$LIMIT"} --workers "${WORKERS:-16}" \
  --out "$OUT/bench.json" ${EXTRA:-}
if [[ ${TF_LOGPROB_GAP:-1} != 0 ]]; then
  stage=teacher_forced
  "$PY" "$HERE/logprob_gap.py" score --url "http://127.0.0.1:$PORT" --model "$SERVED" \
    --ref-results "${REF_RESULTS:-$OUT/bench.json}" --limit "${TF_LIMIT:-200}" \
    ${TF_TASKS:+--tasks "$TF_TASKS"} --out "$OUT/lp.json"
fi
