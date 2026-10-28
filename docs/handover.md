# Handover (2026-10-27)

State of the project at the end of this session, and what comes next. Detail lives in the
linked documents. Machine-specific paths and hosts are deliberately not written here.

## Where things are

* Repository: `BlivionIaG/hi2` (private, Apache-2.0), branch `main`. History was rewritten once
  on 2026-10-27 (a trailer removed): re-clone or `git fetch && git reset --hard origin/main`
  any other clone.
* Long jobs run on the GPU build box (8 Radeon PRO V620; HIP devices 2-5 are ours) from a
  frozen copy of the repository, never from a directory that is re-synced; nothing under
  `/tmp`; stop only your own processes, by explicit pid.
* Checkpoints on the box: Qwen3.8-27B at 3.0 bpw (PTQ and PTQ + QAD, 14.9 GiB each), Qwen3.5-4B
  at 3.0, 3.5 and 4.0 bpw (PTQ and QAD), 0.8B references. The 27B Hessian cache (48 GB) is
  kept for reruns.

## Done this session

* Final logit-KL phase is on by default (`default` and `thorough` presets, 200 steps) and
  splits teacher and student layer-wise over the devices, so the 27B can train it.
* Data-parallel QAD step over all `--devices` (`data_parallel`, on by default). Same quality as
  one device on the 0.8B (docs KLD 0.099, mix 0.254); 1.7x per step at 3 GPUs, 10 % over the
  whole phase there. Not yet measured on big layers.
* `hi2 eval` reports EAR and a `lossless` flag (KLD < 0.01) and splits the model over several
  GPUs (`--device cuda:0,cuda:1,...`). 27B KLD against BF16 is in [results.md](results.md).
* `quantize --target-kld K --eval-text T` picks the error ratio from a measured KLD
  (`hi2/tune.py`). Smoke-tested on the 0.8B, PTQ only (target 0.05 gave ratio 2.2, 4.1 bpw).
* `bench/`: task-level evaluation suite (GSM8K, HumanEval, MBPP, instruction following, ToolACE
  tool calls, long-context retrieval, multiple choice, decisions) with paired comparison, and
  `bench/tfkld.py`, the teacher-forced domain KLD. Drivers `bench/serve_run.sh`,
  `bench/hook.sh`. The smoke run is verified: a served 4B as the reference and a 3.0 bpw PTQ
  checkpoint as the candidate, scored against the reference's own completions, 20 items per
  task. The teacher-forced pass works now. The domain KLD of that checkpoint is 0.162 (gsm8k),
  0.051 (toolcall) and 0.161 (decision). Pooled task accuracy falls from 76.2 % to 62.5 %
  (-13.8 points, McNemar p = 0.000977), and the reference arm reproduced the earlier run
  item for item.
* Experiment scripts cut to three drivers with no host paths; the AWQ INT4 column added to
  the 27B comparison (AWQ is clearly better: 81.8 % first-tool agreement against 66.7 %).
* Calibration study on the 4B: a four-point PTQ ladder (one shared phase 1, probe and encode)
  plus the four checkpoints already on the box, eight points on the full 4765-item suite with
  paired statistics, the teacher-forced domain KLD per task family, a target table per use case
  and the noise floor. The tables and the reasoning are in [results.md](results.md).
* The 27B anchor on the task suite (reduced to 100 items per family, so not comparable with the
  4B numbers): against a BF16 27B at TP=4, the 3.0 bpw PTQ model loses 6.2 points pooled and the
  QAD one 8.2, almost all of it in tool calls (-23.7 and -32.0) and the agent decisions
  (-13.6 and -9.1); the AWQ INT4 checkpoint loses 1.9. First tool call as BF16: 53.0 % (PTQ),
  66.7 % (QAD), 81.8 % (AWQ).
* The same count applied to the other formats on the 27B (every byte the loader reads over the
  quantized weights, divided by the weights): HI2 3.0 bpw **3.012** (15 290 MiB file), AWQ INT4
  **4.625** (20 044 MiB, not the 4.5 estimated before), GGUF Q4_K_M **4.738** (15 702 MiB, and
  it also quantizes the embedding, the head and the MTP layer). HI2 is the smallest file of the
  three: 2.6 % below the GGUF and 23.8 % below the AWQ.
* The quality half of that comparison, measured on the GGUF side with llama.cpp's own KLD on the
  same held-out window the HI2 numbers use: **GGUF Q4_K_M 0.0142 nats/token** against 0.0901 for
  HI2 at 3.0 bpw with QAD and 0.0975 for PTQ (perplexity ratio 1.009 against 1.043 and 1.049;
  same top-1 95.0 % against 88.8 % and 88.1 %). The GGUF buys that with 1.57 bpw more; at equal
  file size HI2 is ahead. The AWQ INT4 checkpoint still has no KLD: it is a compressed-tensors
  pack-quantized checkpoint, not an HI2 one, so `hi2 eval` cannot read it.

## Findings to keep in mind

* 27B QAD (`fast`) improves docs KLD (0.090 against 0.097) but worsens mix KLD (0.295 against
  0.217). Judge any rerun on the mix KLD as well as the decisions.
* On the 4B, task loss follows KLD, not the method: top-1 disagreement is about 0.42 x sqrt(KLD)
  for PTQ and QAD alike. So the calibration can use cheap PTQ points.
* Token-level KLD understates agentic loss: the 27B decisions are far below what its top-1
  agreement suggests.
* The domain KLD of one checkpoint is not one number: the 3.0 bpw PTQ 4B checkpoint scores
  0.16 on gsm8k and decision but 0.05 on toolcall. A target table needs a KLD per task family.
* The 0.42 x sqrt(KLD) rule held on the smoke run as well: 18.8 % of the items flipped against
  16.8 % predicted from the mean KLD of 0.16.
* The first Triton autotune of the linear-attention backward takes about 10 minutes.
* **The suite is stable enough to read small differences.** A second run of the same BF16 model,
  on another GPU and another server, reproduces the first: under half a point on every family,
  a fifth of a point pooled, the same first tool call on all 66 decision prompts, and a
  teacher-forced KLD of 2e-05 or less. Every difference in the calibration tables is therefore a
  property of the checkpoint.
* **On this 4B nothing below 4.2 bpw is free.** The families that break break early: tool calls
  lose at a domain KLD of 0.0219 (4.0 bpw), math at 0.0592, instruction following at 0.2166,
  short programs at 0.0442 (4.2 bpw). The agent-decision family is not ordered by KLD at all:
  two checkpoints 0.2 bpw apart land 44 points apart on it.
* **On the 27B, tool calling is the loss.** At 3 bpw the format loses 24 points on the
  tool-call family and 14 on the agent decisions while math, instruction following, multiple
  choice and long-context retrieval stay flat. The AWQ INT4 checkpoint at about 4.5 bpw loses
  1.9 points pooled against the same-size formats' 6 to 8, so the extra 1.5 bpw buy most of
  the loss on agentic work.
* A single "default ratio" is the wrong knob: the budget depends on the domain, so the default
  quality rule has to be picked per use case (tool calls need a KLD below 0.01 on their own
  text, instruction following tolerates 0.13).

## Next steps

1. Calibration study: **done**. PTQ ladder on the 4B (ratios 2-16 in one shared pass) plus the
   existing 4B checkpoints, the full suite on eight points with paired statistics, the domain
   KLD per task family, a target table (KLD per use case) and the noise floor, then the 27B
   anchor on the reduced suite. In [results.md](results.md).
2. 27B rerun with the `default` preset, final-KL and four-GPU data-parallel QAD (cached
   Hessians); score docs and mix KLD, decisions and the new suite. **Running** (restarted
   2026-10-28 11:33 after a GPU page fault in the probe): the statistics and tables caches hit,
   the sensitivity probe finished in fp16 because the 27B does not fit a shard of a 30 GiB card
   in fp32, and the QAD phase is at 300 steps per layer over 64 layers (about 350 s a layer,
   ETA around 20:00). A chain is armed to score the result as soon as it lands: a second BF16
   27B reference run **with** a teacher-forced pass on the first run's completions, the domain
   KLD of every 27B candidate against it, the 27B noise floor, `hi2 eval` of the new checkpoint
   on the docs and mix windows, and the new checkpoint on the suite.
3. The teacher-forced domain KLD of the 27B anchor is still missing: the BF16 reference arm
   carries no teacher-forced pass, so the reference's own log-probs on its completions are
   absent. It needs a second BF16 27B serve at TP=4, which needs all four cards. **Armed** in
   the chain above (the reference arm is scored on the *first* run's completions, so it pairs
   exactly with the candidates' log-probs).
4. `--target-kld` with QAD on the 4B; decide the default quality rule from step 1, per use case
   rather than as one global ratio.
5. Optional: fp16 matrix math in QAD (layer forward and backward is 94 % of a step), INT3 rows,
   a KLD path for AWQ checkpoints (they are compressed-tensors pack-quantized, so `hi2 eval`
   cannot read them; the cheapest path is the teacher-forced pass the chain already runs).
6. Before sharing v1: README note that checkpoints need the vLLM fork, check old commits for
   internal paths, decide whether the 27B table is public.
