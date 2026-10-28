#!/usr/bin/env bash
# The KLD of a GGUF against its own BF16 base, the way llama.cpp measures it:
# KL(BF16 || quantized) over the whole vocabulary, in nats/token, on a held-out text.
#
# The same quantity and the same window as `hi2 eval` (the HF tokenization of the text,
# the tokens after --skip-tokens, 8 x 2048), so the numbers can be put in one table with
# the HI2 ones. llama.cpp scores the second half of each 2048-token chunk (8 192 positions
# for 8 chunks) where `hi2 eval` scores every position (16 376).
#
# usage: gguf_kld.sh <llama-bin-dir> <base.gguf> <quant.gguf> <text> <out-dir> [threads]
#
# Two passes: the base model writes its log-probs (--save-all-logits), the quantized model
# is scored against them (--kl-divergence). The base logits are 4 GB for a 27B over
# 8 chunks, and the second pass reuses the tokens stored in them, so both passes see
# exactly the same tokens.
#
# The pass runs on the CPU with "-dev none" when the GPUs are busy with a quantization
# run: a 27B is about 110 s per chunk at 32 threads, so both passes take half an hour.
set -u
BIN=${1:?llama bin dir}
BASE=${2:?base gguf}
QUANT=${3:?quantized gguf}
TEXT=${4:?text file}
OUT=${5:?out dir}
TH=${6:-32}
mkdir -p "$OUT"
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
cd "$OUT" || exit 1
"$BIN/llama-perplexity" -m "$BASE" -f "$TEXT" -c 2048 --chunks 8 \
  -t "$TH" -tb "$TH" -dev none --no-warmup --save-all-logits "$OUT/base.logits" \
  > "$OUT/base.log" 2>&1 || { echo "base pass failed"; tail -5 "$OUT/base.log"; exit 1; }
"$BIN/llama-perplexity" -m "$QUANT" -f "$TEXT" -c 2048 --chunks 8 \
  -t "$TH" -tb "$TH" -dev none --no-warmup \
  --kl-divergence --kl-divergence-base "$OUT/base.logits" > "$OUT/kld.log" 2>&1 || {
  echo "the KL pass failed"; tail -5 "$OUT/kld.log"; exit 1; }
grep -E "Mean    KLD|Mean PPL\(Q\)|Mean PPL\(base\)" "$OUT/kld.log"
