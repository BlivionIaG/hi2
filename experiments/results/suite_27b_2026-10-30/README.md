# The 27B suite + domain KLD, 2026-10-30/31

The reduced suite (`bench/run.py`, 100 items per family) and the teacher-forced domain KLD
(`bench/tfkld.py score`, 200 items per task) for the two candidates that were missing from the
2026-10-28 anchor: the `default` QAD checkpoint (`q27b-b3.0-qad-default`) and the AWQ INT4
reference re-run. Both were served one V620 each (`GPU=9` and `GPU=2`, TP=1, `MAXLEN=32768`,
`SEQS=4`, `KV=6000000000`), against the BF16 reference run in `bench27/serve-bf16/`.

| file | what |
|---|---|
| `compare-hi2.md`, `compare-hi2.json` | `bench/compare.py` paired against BF16: per-family accuracy, diff with paired-bootstrap CI, McNemar p, flip rates, decision agreement |
| `compare-awq.md`, `compare-awq.json` | the same for the AWQ re-run |
| `kld_domain_27b.json` | `tfkld.py compare` output: KL(BF16 ‖ candidate) per family, nats/token |

Headline numbers (BF16 macro 86.2 %):

| | HI2 `default` 3.0 bpw | AWQ INT4 4.625 bpw |
|---|---:|---:|
| macro | 82.5 % | 84.2 % |
| pooled vs BF16 | -3.4 [-5.6, -1.2] | -1.9 [-3.6, -0.3] |
| tool calls | 55.7 % (BF16 60.8) | 49.5 % |
| decisions | 74.2 % (BF16 87.9) | 83.3 % |
| domain KLD gsm8k / humaneval / mbpp / ifeval / toolcall / decision | 0.0404 / 0.0212 / 0.0521 / 0.1740 / 0.0325 / 0.0770 | 0.00346 / 0.00191 / 0.00731 / 0.01977 / 0.00983 / 0.02333 |

The AWQ re-run's per-family counts move by one item against the 2026-10-28 run (humaneval 96
against 95, mbpp 89 against 90, mc 68 against 67, decision 81.8 against 83.3) while its pooled
difference and every domain KLD reproduce. That is the run-to-run noise documented in
`docs/results.md` ("The 27B noise floor"), not a different checkpoint.

Raw per-item log-probabilities (`lp.json`, 2.4 MB each) and `bench.json` stay on the build
server under `~/w4a8_runs/hadamard-int2/bench27/serve-{bf16,hi2,awq}/`.
