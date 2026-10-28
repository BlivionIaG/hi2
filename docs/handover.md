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
  same held-out windows the HI2 numbers use: **GGUF Q4_K_M 0.0142 nats/token** on the docs text
  and 0.0776 on the mix text, against 0.0901 / 0.2171 for HI2 at 3.0 bpw with QAD and
  0.0975 / 0.2947 for PTQ (perplexity ratio 1.009 against 1.043 and 1.049; same top-1 95.0 %
  against 88.8 % and 88.1 %). At the same bit rate the GGUF's own types are worse: Q2_K
  (3.175 bpw) 0.1395, and the GGUF needs Q3_K_M at 3.95 bpw (47 % more body bytes) to match
  HI2's 0.0901. The AWQ INT4 checkpoint still has no KLD: it is a compressed-tensors
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
* **The layers a restarted run trains fresh start from a worse held-out nMSE - and the cause was
  a data race in the pinned staging.** In the 27B run of 2026-10-28, the layers the run trained
  itself (54 to 63) started at 1.9e-01 to 2.6e-01 where the run of 2026-10-27 had 1.2e-01 to
  1.6e-01 for the same layers, while the layers reused from the cached artifacts tracked that
  run to within 2.5 %. The cause: the encoders shard a tensor across the devices and run the
  shards in parallel threads, and they all shared one pinned staging window, so a shard could be
  encoded from another shard's weights. The fingerprint is layer 63's initialisation: a held-out
  nMSE of **1.0088**, an output uncorrelated with its target. The training was not affected (it
  starts from the weights, not from the PTQ encoding, and the PTQ encoding is only the fallback
  whose loss is reported), and the exported checkpoint was checked: all 496 tensors, no
  anomalous error, the mean relative error 0.0446 against 0.0443 for the run of 2026-10-27. The
  window keeps its buffers per thread now, and `tests/test_pinned.py` fails against the previous
  code (one thread gets another thread's data) and passes against this one.
* **The 27B's GPU page faults are the copy engine on host pages - not a bad card, not our
  kernels.** The faulting client is the copy engine (SDMA0 or SDMA1), so no custom kernel is
  involved; the faulting address is a process virtual address, and the faults walk over
  consecutive 4 KiB pages of it. The translation succeeds (`MAPPING_ERROR: 0x0`,
  `WALKER_ERROR: 0x0`) and the protection check refuses the access, which is what the runtime's
  "Page not present or supervisor privilege" means. The four crashes are the two directions of a
  bulk transfer between VRAM and **pageable** host memory: a read during the probe's
  host-to-device weight loads, a write when a layer's artifact is copied back to the host, and
  two reads in the QAD phase. The box is not short of memory (about 328 GiB free, no swap
  pressure, no OOM kill) and no other process has ever faulted on those cards - the serving arms
  ran for hours over the same four. Staging the hidden states alone did not stop it (the fourth
  crash came after); the weights had to be staged as well.
* **The QAD's host-device copies now go through a pinned window** (`hi2/pinned.py`,
  `PinnedWindow`). The QAD moves about 1.6 TB of host memory per run (40 MiB of hidden state
  per sequence per step, 264 sequences per layer), all of it pageable, and the copy engine
  addresses those pages directly. The staged copies are bit-identical to the direct ones in
  both directions, with and without a dtype cast, and for tensors larger than the window
  (`experiments/pinned_copy_check.py`). The cost is 7.5 ms a transfer (a 40 MiB host copy
  into the pinned window) against a 1 s step, and the device-to-host direction is actually
  *faster* than the direct copy (14.2 ms against 26.9 ms), because the window is reused
  instead of a fresh pageable tensor being allocated each time.

## Next steps

1. Calibration study: **done**. PTQ ladder on the 4B (ratios 2-16 in one shared pass) plus the
   existing 4B checkpoints, the full suite on eight points with paired statistics, the domain
   KLD per task family, a target table (KLD per use case) and the noise floor, then the 27B
   anchor on the reduced suite. In [results.md](results.md).
2. 27B rerun with the `default` preset, final-KL and four-GPU data-parallel QAD (cached
   Hessians); score docs and mix KLD, decisions and the new suite. **Crashed twice**: a GPU
   page fault in the probe (2026-10-28 11:28), one in the QAD phase (14:59, at layer 19 of 64,
   after 2 h) and a third in the QAD phase of the fast recipe (18:36, at layer 48 of 64, after
   2.4 h). All are device-memory faults with no host-side error ("Page not present or supervisor
   privilege", `PERMISSION_FAULTS` in the kernel log), and they moved between cards, so they are
   not one bad card.

   The kernel log says what they are (see the finding above): the copy engine addressing
   **pageable host memory**, in both directions. The fix for it is in: the copies go through a
   pinned window (`hi2/pinned.py`), the QAD's tests are green with it, and the run has been
   restarted with it. The run is 2 h of a 5.3 h phase each time, the probe cache is complete and
   the per-layer artifacts are cached, so a restart is cheap: the probe is skipped and the
   layers that were already distilled are reused.

   Still worth doing:

   a. **Watch the restarted run's copy counters.** The pinned window makes the copy engine
      address mapped memory, but only the crash rate says whether that was the whole story.
   b. **Serialize the copies against the optimizer step**, so a copy cannot still be in flight
      when its host buffer is released.
   c. **Reproduce it outside the pipeline**: a loop of pageable 40 MiB copies in both
      directions under a 150 GiB RSS process, on one card, for a few hours. That is the
      cheapest way to tell the runtime's direct-DMA path from our own use of it, and it does
      not need the 27B.
3. The teacher-forced domain KLD of the 27B anchor is still missing: the BF16 reference arm
   carries no teacher-forced pass, so the reference's own log-probs on its completions are
   absent. It needs a second BF16 27B serve at TP=4, which needs all four cards. **Armed** in
   a chain (the reference arm is scored on the *first* run's completions, so it pairs
   exactly with the candidates' log-probs).
4. `--target-kld` with QAD on the 4B; decide the default quality rule from step 1, per use case
   rather than as one global ratio.
5. **The QAD's speed, in the order the measurements put it** (the modes - recipe, math
   dtype, buffer dtype - and what each one costs are in
   [qad.md](qad.md#the-modes); the measurement behind this list is in
   [qad.md](qad.md#the-27b-at-four-gpus-measured-on-the-2026-10-28-rerun)):
   a. **fp16 matrix math for the QAD's layer forward and backward - measured, and it
      does not work on this hardware.** `--qad-dtype float16` runs the layer's matrix
      products in half precision (the latents, the ladder bytes and the loss stay
      float32). It is 1.43x faster (2019 ms a step against 1414 ms on one card, the
      27B's layer 10) and not 3x, because a float16 product of the MLP's shape is only
      1.47x faster than a float32 one on this stack (`experiments/fp16_gemm_check.py`),
      not the 4x the V_DOT2 peak suggests. Worse, the QAD stops learning: the held-out
      nMSE falls 1.8 % in float16 against 45 % in float32, and the full-attention layers
      (every fourth layer) do not run at all - the model reaches the attention with q
      and k in float32 and v in float16. Bfloat16 is not an option either: RDNA2 has no
      `fdot2.bf16`. The option stays in the code (default float32) with the measurement
      in [qad.md](qad.md#the-half-precision-math-measured-2026-10-28); a usable version
      would keep the attention and the recurrence in float32 and put only the MLP in
      float16.
   b. **Stop the layers that stopped gaining - a flag, and running.** `--qad-preset fast`
      (budget 0.5, early stop under 1 %) or `--qad-config '{"budget": ..., "min_gain":
      ...}'`. The 27B rerun of 2026-10-28 (after 15:00) runs it with the float32
      buffers, so the early stop is not firing on the bf16 rounding.
      `experiments/qad_budget_sim.py` replays a measured run through both rules and prices
      the recipes: on the first 19 layers of the 27B, the `fast` recipe spends 38 % of the
      steps for +0.3 % of the layer-wise held-out nMSE. The layer-wise nMSE is a proxy
      that did **not** predict the one end-to-end result we have (the `fast` checkpoint at
      the 27B came out worse than PTQ on the mix KLD), so a faster recipe has to be
      judged on the mix KLD.
   c. **The copies.** Once the math is faster, the per-step PCIe traffic (about 9.6 GB a
      step: the decoded weights to the three workers and the gradients back) becomes the
      next bottleneck. The fix is to decode on each worker from the codes and the small
      surrogate state instead of shipping the decoded weights.
   d. **The hidden-state buffers.** 168 MB a tensor, re-uploaded every step; the float32
      ones need 41 GiB of RAM. Keep them device-resident or use bf16.
6. Before sharing v1: README note that checkpoints need the vLLM fork, check old commits for
   internal paths, decide whether the 27B table is public.
7. The 27B decode path does not win at this size (batch 16: HI2 7.26 s against AWQ INT4
   5.86 s on one card, BF16 3.25 s on four), unlike the 4B (78.4 against 38.1 tok/s at c=1).
   There is no c=1 measurement at the 27B yet, and the hybrid linear-attention recurrence is
   the suspect: it does not shrink with the quantized weights. **Deferred** (2026-10-28): there
   are no spare resources for it right now, and it does not touch the quantizer. What it would
   need when it is picked up: the same harness at c=1 on one card, and the dequantize-and-GEMM
   step of the serving path measured against AWQ's, to tell the layout from the recurrence.
8. Also open: INT3 rows, and a KLD path for AWQ checkpoints (they are compressed-tensors
   pack-quantized, so `hi2 eval` cannot read them; the cheapest path is the teacher-forced
   pass the chain already runs).
