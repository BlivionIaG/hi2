#!/usr/bin/env bash
# Qwen3.5-4B GGUF reference points scored with gguf_suite.sh (box paths). Waits for the vLLM accuracy
# chain to free the GPU, then: f16 + base logits, then each quant type (imatrix from the earlier run).
B=/home/chenco_adm/w4a8_runs/hadamard-int2
S=$B/hi2/experiments/gguf_suite.sh
D=$B/mc_data/ikawrakow__validation-datasets-for-llama.cpp
M4=$(cat $B/m4b.path)
OUT=$B/gguf4b_suite
G=$B/gguf4b
BIN=/home/chenco_adm/rocmfpx-rdna2/build-gfx1030/bin
export LD_LIBRARY_PATH=$BIN:/opt/rocm/core-7.14/lib
while [ ! -e $B/mc4b/hi2-4b-3.0-sens.mc.done ]; do sleep 30; done
mkdir -p "$OUT"
bash $S base "$M4" "$G/f16.gguf" "$G/eval.txt" "$D" "$OUT"
for T in Q2_K Q3_K_S Q3_K_M IQ4_XS Q4_K_M; do
  $BIN/llama-quantize --imatrix "$G/imat.dat" "$G/f16.gguf" "$G/$T.gguf" "$T" > "$OUT/$T.quant.log" 2>&1 || continue
  bash $S score "$G/$T.gguf" "$T" "$G/eval.txt" "$D" "$OUT"
  rm -f "$G/$T.gguf"
done
echo "done" > "$OUT/ALL_DONE"
