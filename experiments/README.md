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
| `isa/probe.hip`, `isa/run.hip` | Check RDNA2 instruction facts (spec section 7) on a V620. |
