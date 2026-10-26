#!/usr/bin/env bash
# i-quant GGUF reference points for Qwen3.5-4B (same imatrix, base logits and flags as chain_gguf4b.sh).
# The Qwen3.5 MTP layer (blk.32) is never run by the imatrix pass, and very low-bit types refuse to
# quantize without importance data, so that layer is kept at q8_0 (it is not used for scoring).
# usage: HIP_VISIBLE_DEVICES=N chain_gguf_iq.sh [types...]   (default: IQ2_M IQ3_XXS IQ3_XS IQ3_S IQ3_M)
B=/home/chenco_adm/w4a8_runs/hadamard-int2
S=$B/hi2-q/experiments/gguf_suite.sh
D=$B/mc_data/ikawrakow__validation-datasets-for-llama.cpp
OUT=$B/gguf4b_suite
G=$B/gguf4b
BIN=/home/chenco_adm/rocmfpx-rdna2/build-gfx1030/bin
export LD_LIBRARY_PATH=$BIN:/opt/rocm/core-7.14/lib
LOCK=$B/gguf-iq.gpu${HIP_VISIBLE_DEVICES:-x}.lock
echo "chain_gguf_iq.sh $$ $(date +%s)" > "$LOCK"
types=("$@")
[ ${#types[@]} -gt 0 ] || types=(IQ2_M IQ3_XXS IQ3_XS IQ3_S IQ3_M)
for T in "${types[@]}"; do
  "$BIN/llama-quantize" --imatrix "$G/imat.dat" --tensor-type "blk\\.32\\.=q8_0" "$G/f16.gguf" "$G/$T.gguf" "$T" > "$OUT/$T.quant.log" 2>&1 || continue
  bash "$S" score "$G/$T.gguf" "$T" "$G/eval.txt" "$D" "$OUT"
  rm -f "$G/$T.gguf"
done
rm -f "$LOCK"
echo "done" > "$OUT/IQ_DONE"
