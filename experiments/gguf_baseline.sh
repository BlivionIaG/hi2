#!/usr/bin/env bash
# GGUF baselines (k-quants, i-quants, ROCmFPX) on the same model and held-out text as `hi2 eval`.
# usage: HIP_VISIBLE_DEVICES=3 gguf_baseline.sh <hf_model_dir> <text_file> <out_dir>
# Calibration (imatrix) uses the first 32 x 2048 tokens, evaluation the 8 x 2048 tokens at
# offset 800000, matching the hi2 runs. Perplexity is llama.cpp's own (its tokenizer, ctx 2048).
set -u
M=$1; TEXT=$2; OUT=$3
D=/home/chenco_adm/rocmfpx-rdna2
BIN=$D/build-gfx1030/bin
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final
export LD_LIBRARY_PATH=$BIN:/opt/rocm/core-7.14/lib
mkdir -p "$OUT"
$V/bin/python - "$M" "$TEXT" "$OUT" <<'PY'
import sys
from transformers import AutoTokenizer
m, text, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(m)
ids = tok(open(text).read(), add_special_tokens=False).input_ids
open(f"{out}/calib.txt", "w").write(tok.decode(ids[:32 * 2048]))
open(f"{out}/eval.txt", "w").write(tok.decode(ids[800000:800000 + 8 * 2048]))
PY
PYTHONPATH=$D/gguf-py $V/bin/python $D/convert_hf_to_gguf.py "$M" --outtype f16 --outfile "$OUT/f16.gguf" > "$OUT/convert.log" 2>&1
$BIN/llama-imatrix -m "$OUT/f16.gguf" -f "$OUT/calib.txt" -c 2048 --chunks 32 -ngl 99 -o "$OUT/imat.dat" > "$OUT/imatrix.log" 2>&1
ppl() { $BIN/llama-perplexity -m "$1" -f "$OUT/eval.txt" -c 2048 --chunks 8 -ngl 99 > "$2" 2>&1; }
ppl "$OUT/f16.gguf" "$OUT/f16.ppl.log"
for T in Q8_0 Q6_K Q5_K_M Q4_K_M IQ4_XS Q3_K_M Q3_K_S IQ3_XS IQ3_XXS Q2_K IQ2_M IQ2_S IQ2_XS IQ2_XXS Q2_0_ROCMFPX; do
  $BIN/llama-quantize --imatrix "$OUT/imat.dat" "$OUT/f16.gguf" "$OUT/$T.gguf" "$T" > "$OUT/$T.quant.log" 2>&1 || { echo "$T quantize failed" >> "$OUT/failures.txt"; continue; }
  ppl "$OUT/$T.gguf" "$OUT/$T.ppl.log"
done
# imatrix effect at one point
$BIN/llama-quantize "$OUT/f16.gguf" "$OUT/Q3_K_M-noimat.gguf" Q3_K_M > "$OUT/Q3_K_M-noimat.quant.log" 2>&1 && ppl "$OUT/Q3_K_M-noimat.gguf" "$OUT/Q3_K_M-noimat.ppl.log"
echo "done" > "$OUT/DONE"
