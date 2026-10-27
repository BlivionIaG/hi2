#!/usr/bin/env bash
# serve_validate.sh LOAD_HOOK: run the decision-fidelity prompt set against the server it
# booted (PORT, SERVED and OUT are exported by serve_validate.sh; its URL is the completions
# endpoint, so the base URL is built from PORT). Ends with PASS or FAIL.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final/bin/python}
if "$PY" "$HERE/decision_fidelity.py" run --url "http://127.0.0.1:$PORT" --model "$SERVED" --out "$OUT/decisions.json"; then
  echo PASS
else
  echo FAIL
fi
