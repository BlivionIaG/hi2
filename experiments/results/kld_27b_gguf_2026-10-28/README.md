# GGUF Q4_K_M KLD against its own BF16 base (Qwen3.8-27B)

`llama-perplexity --kl-divergence`, CPU only (`-dev none`), on the same 16k-token windows the
`hi2 eval` slices use: the study text at offset 800 000 for docs and the mix text for mix,
8 × 2048 tokens each.

| text | file | Mean KLD (nats/token) | PPL x | Same top p |
|---|---|---:|---:|---:|
| docs | `docs.kld.log` | **0.014241 ± 0.000840** | 1.008766 | 95.39 % |
| mix | `mix.kld.log` | **0.077601 ± 0.007484** | 1.065524 | 95.05 % |

Models: `Qwen3.8-27B-BF16-00001-of-00002.gguf` (unsloth's BF16 conversion, base logits saved to
`base.logits` and reused) against `Qwen3.8-27B-UD-Q4_K_M.gguf`. Re-run on 2026-10-31 from the
2026-10-28 driver and reproduced to every printed digit.

Two differences from `hi2 eval` to keep in mind: llama.cpp scores the second half of each
2048-token chunk (8 192 positions) while `hi2 eval` scores every position (16 376), and the two
run in different engines (llama.cpp against transformers). The KLD is the same quantity in both.
