# Experiments

These scripts made the numbers in [../docs/results.md](../docs/results.md) and the other
documents. They are records, not a library. They ran on one build box (8 Radeon PRO V620,
gfx1030). The shell drivers read their paths from environment variables (see the header of
each); the Python scripts take their inputs on the command line.

Raw output of the studies is in [`results/`](results/). `results/lowbit_runs_2026-10-26.json`
holds the perplexity, KLD and top-1 numbers of every low-bit study run.

## Drivers

| Script | Job |
| --- | --- |
| `run_27b.sh` | Qwen3.8-27B (BF16) to about 3.0 bpw with QAD on several GPUs. |
| `decision_fidelity.sh`, `decision_fidelity_hook.sh` | Serve each model in the fork's vLLM and run the decision-fidelity set (spec 12.3). |

## Evaluation

Task-level evaluation (math, code, instruction following, tool calls, long context, and
the MC and decision sets below as paired tasks) is in [`../bench/`](../bench/README.md).

| Script | Job |
| --- | --- |
| `vllm_ppl.py` | Score the perplexity of a model served in the fork's vLLM. |
| `gguf_bpw.py` | Count the bpw of a GGUF the way `hi2 size` counts an HI2 checkpoint (every byte the loader reads over the quantized weights). Pure python, no gguf-py. |
| `gguf_kld.sh` | The KLD of a GGUF against its own BF16 base with llama.cpp (`KL(BF16 ‖ quantized)` over the whole vocabulary), on the same window `hi2 eval` uses. Runs on the CPU when the GPUs are busy. |
| `mc_eval.py` | Score multiple-choice tasks against a served model. |
| `decision_prompts.py`, `decision_fidelity.py`, `data/decision_prompts.json` | The decision-fidelity prompt set, the client and the comparison. |
| `summarize_gguf.py`, `summarize_qad.py`, `ratio_table.py`, `int8_usage.py` | Collect run logs into tables. |
| `check_tables.py`, `compare_layouts.py`, `compare_qad_init.py`, `qad_layers.py` | Consistency checks between runs and layouts. |

## Studies of the format

| Script | Question |
| --- | --- |
| `scale_study.py`, `scope_study.py` | Which scale system and which scope of scales work best? |
| `rotation_study.py`, `block_rotation_study.py` | Where does the Hadamard rotation help? |
| `candidate_usage.py` | Which (bits, block) candidates does the allocator use? |
| `refine_study.py` | How much does coordinate descent on the codes gain over GPTQ? 6-9 % of the objective (spec 8.9). |
| `refine_cost.py` | What `--refine-passes` costs (seconds per sweep against the GPTQ encode, with the fused kernel or `HI2_REFINE_BACKEND=torch`) and what two sweeps gain on whole models (spec 8.9). |

## Inkling-2-Mini, the stock NVFP4 / MXFP8 checkpoint

Read by `hi2/inkling_source.py` ([../docs/inkling-source.md](../docs/inkling-source.md)). The
numbers they wrote on 2026-11-23 are in [`results/inkling2_mini_2026-11-23/`](results/inkling2_mini_2026-11-23/).

| Script | Job |
| --- | --- |
| `inkling_inventory.py` | Tensor classes, elements, bytes and stock bits per weight from the safetensors headers (main shard, MTP, vision); checks the design-review numbers and says which are definitions. |
| `inkling_reference_check.py` | `hi2.inkling_source` against the vLLM fork's reference decoders (`tiny_reference.py`) on every expert of every layer: the float32 bit patterns and the largest absolute difference. |
| `inkling_teacher_check.py` | The stock model assembled from the reader on the CPU (`Inkling2LoopForConditionalGeneration` built on the meta device, every text tensor filled by `InklingSource`) and run on a prompt: the top-8 against the fork's CPU oracle, the largest activation and the non-finite count of every decoder layer, and the logits of a float16 or bfloat16 model against a float32 one (KL, top-1 agreement). Needs the fork's transformers; 105 GB of host memory in float32. |
| `inkling_unfuse_check.py` | `hi2.moe.unfuse_experts` on a tiny real `Inkling2LoopForConditionalGeneration` (the fork's transformers and tiny config): does the unfused forward run, and equal the fused one? Exit 1 while `hi2.moe` called `act_fn(gate) * up` (the experts' activation takes `(gate, up)`); 0 since `UnfusedExperts` runs a two-argument activation. |
| `inkling_hf_names_check.py` | The reader's names and shapes against the real `Inkling2LoopForConditionalGeneration` built on the meta device (needs the fork's transformers): every text-model tensor, both ways. |
| `inkling_hf_load_check.py` | `AutoModelForCausalLM.from_pretrained` of the stock directory (the fork's transformers, the class registered by `hi2.inkling_source.register_auto_class`), in float16, bfloat16 or float32, against the reader: dense tensors and `message_input.query` exactly, the experts of every layer elementwise, and transformers' own dequantization op with its sources cast or as stored. |
| `inkling_scale_stats.py` | Histograms of the UE8M0 / E4M3 block scales and the NVFP4 `amax`; the weights that overflow, underflow or are subnormal in float16 and bfloat16, and the energy of each cast, for every layer. |
| `inkling_alloc_dryrun.py` | HI2's data-free mode (`build_table(stats=None)`) and its allocators on sampled real experts and attention linears, tiled to the whole population: INT2/INT4/INT8 and block mix, effective bpw and implied size at 3.0 / 3.5 / 4.0 bpw. First-order: no activations, no Hessian. |
| `inkling_dequant_probe.py` | The device decode against the CPU decode, and what decoding one layer costs (read, host-to-device, decode, peak memory). |
| `inkling_quantize_dryrun.py` | `hi2 quantize` on the real stock directory as far as a CPU goes (the decoded source view of `hi2.checkpoint.source_map` / `open_source`): the `--dry-run` plan, a census of every tensor against the default `--exclude`, a data-free run on a sample of experts and attention linears, `_load_model` with `check_unfused` and a forward against saved float32 logits, and `hi2` on a tiny real Inkling model: a data-free run (works), the layer-wise calibration driver and a calibrated run (fail on `block_states`: the expected result), the sensitivity probe (equals its no-replay twin). Needs the fork's transformers and `tools/rdna/inkling2`. |

## Calibration text

| Script | Job |
| --- | --- |
| `build_calib_mix.py` | The multi-domain calibration file. |
| `make_calib_qad.py`, `make_calib_tasks.py` | The QAD text and the task-style text. |

## Profiling and instruction probes

| Script | Job |
| --- | --- |
| `profile_encode.py`, `profile_new.py`, `profile_probe.py` | Where the quantizer spends its time. |
| `profile_qad.py` | Where a QAD optimizer step spends its time on one real layer (decode, layer forward and backward, backward through the decode, optimizer). |
| `qad_dtype_check.py` | The same real layer in float32, float16 and bfloat16: the time of a step, the held-out nMSE and the difference of the gradient the optimizer sees (does the QAD still learn the same in half precision?). |
| `qad_budget_sim.py` | Replay a QAD log through the step budget and the early stop of `hi2/qad.py` and price the recipes: the steps, the time and the held-out nMSE a different recipe would have cost. |
| `fp16_gemm_check.py` | The same matrix product and the same backward in float32 and float16: the time and the relative difference of the result and of the two gradients. |
| `pinned_copy_check.py` | The host-device copies through the pinned window (`hi2/pinned.py`) against the direct ones: are the numbers the same, and what does the staging cost? |
| `staging_rate_check.py` | What the pinned staging costs on the *weights*: a 27B layer and one of its largest linears, direct against staged, and the staged copy without the per-chunk device temporary. |
| `isa/probe.hip`, `isa/run.hip` | Check RDNA2 instruction facts (spec section 7) on a V620. |
