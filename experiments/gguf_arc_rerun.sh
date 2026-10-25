#!/usr/bin/env bash
# The first GGUF suite run scored ARC-Easy/Challenge without -np, so llama-perplexity stopped at the first
# task with more than 4 choices. Re-score just those two sets for the f16 file and each quant type.
B=/home/chenco_adm/w4a8_runs/hadamard-int2
D=$B/mc_data/ikawrakow__validation-datasets-for-llama.cpp
G=$B/gguf4b
OUT=$B/gguf4b_suite
BIN=/home/chenco_adm/rocmfpx-rdna2/build-gfx1030/bin
export LD_LIBRARY_PATH=$BIN:/opt/rocm/core-7.14/lib
arc() { # gguf tag
  for name in arc-easy-validation arc-challenge-validation; do
    "$BIN/llama-perplexity" -m "$1" --multiple-choice -bf "$D/$name.bin" --multiple-choice-tasks 600 -c 2048 -ngl 99 -np 8 > "$OUT/$2.mc2-$name.log" 2>&1
  done
}
arc "$G/f16.gguf" f16
for T in Q2_K Q3_K_S Q3_K_M IQ4_XS Q4_K_M; do
  "$BIN/llama-quantize" --imatrix "$G/imat.dat" "$G/f16.gguf" "$G/$T.gguf" "$T" > "$OUT/$T.quant2.log" 2>&1 || continue
  arc "$G/$T.gguf" "$T"
  rm -f "$G/$T.gguf"
done
echo "done" > "$OUT/ARC_DONE"
