#!/usr/bin/env bash
# serve_validate.sh LOAD_HOOK: run the decision-fidelity prompt set against the server it
# booted (PORT, SERVED and OUT are exported by serve_validate.sh; its URL is the completions
# endpoint, so the base URL is built from PORT). Also scores the perplexity of the served model
# on the docs and mix held-out texts (vllm_ppl.py), the same slices as `hi2 eval`, to
# $OUT/ppl_docs.txt and $OUT/ppl_mix.txt. Ends with PASS or FAIL.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final/bin/python}
B=/home/chenco_adm/w4a8_runs/hadamard-int2
"$PY" "$HERE/vllm_ppl.py" --port "$PORT" --model-dir "$MODEL" --served-name "$SERVED" \
  --text "${PPL_DOCS:-$B/study_text_all.txt}" --skip-tokens 800000 > "$OUT/ppl_docs.txt" 2>&1
"$PY" "$HERE/vllm_ppl.py" --port "$PORT" --model-dir "$MODEL" --served-name "$SERVED" \
  --text "${PPL_MIX:-$B/eval_mix.txt}" --skip-tokens 0 > "$OUT/ppl_mix.txt" 2>&1
if "$PY" "$HERE/decision_fidelity.py" run --url "http://127.0.0.1:$PORT" --model "$SERVED" --out "$OUT/decisions.json"; then
  echo PASS
else
  echo FAIL
fi
