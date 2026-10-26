# Experiments

These scripts made the numbers in [../docs/results.md](../docs/results.md) and the other
documents. They are records, not a library. They ran on one build box (8 Radeon PRO V620,
gfx1030). Paths such as `/home/chenco_adm/w4a8_runs/...` and the GPU rules (HIP devices
2-5, lock files) belong to that box. Change `box_env.sh` and the `B=` lines to run them
elsewhere.

Raw output of the studies is in [`results/`](results/).

## Run drivers

One quantize run, scored, with a GPU lock.

| Script | Job |
| --- | --- |
| `box_env.sh` | Source it first: the venv, the ROCm libraries, and `PYTHONPATH`. |
| `free_gpu.sh`, `when_idle.sh` | Pick an idle GPU, or wait for one, before a run. |
| `run_hi2_model.sh` | Quantize (PTQ) and score with `hi2 eval`. |
| `run_qad_model.sh` | The same with QAD. |
| `run_hi2_moe.sh`, `eval_hi2_moe.sh`, `restack_olmoe.sh` | OLMoE runs, evaluation, and the per-expert to stacked conversion. |
| `fast_ptq.sh`, `fast_run.sh` | One timed run, one command under a lock. |
| `lowbit_run.sh`, `restamp_qad.py` | A low-bit study run; reuse QAD layer artifacts of an older run. |
| `download_hf.sh` | Download a model into the Hugging Face cache. |

## Run chains

Fixed sequences of runs for one question.

| Script | Question |
| --- | --- |
| `chain_opt1.sh`, `chain_opt2.sh` | Does the calibration domain or act-order help? |
| `chain_sweep08.sh` | How does the error ratio trade quality and size on the 0.8B model? |
| `chain_qad08.sh`, `chain_budget08.sh` | QAD settings; the effect of the step budget. |
| `chain_fast.sh` | Before and after timing of the speed work ([speed.md](../docs/speed.md)). |
| `chain_lowbit.sh` | The low-bit study ([lowbit.md](../docs/lowbit.md)). |
| `chain_mc4b_hi2.sh`, `mc4b_when_idle.sh`, `mc_lowbit.sh` | Multiple-choice scoring of 4B checkpoints. |

## Evaluation

| Script | Job |
| --- | --- |
| `vllm_eval.sh`, `vllm_ppl.py` | Serve a model in the fork's vLLM and score perplexity. |
| `vllm_mc.sh`, `mc_eval.py` | Serve a model and score multiple-choice tasks. |
| `summarize_gguf.py`, `summarize_qad.py`, `ratio_table.py`, `int8_usage.py` | Collect run logs into tables. |
| `check_tables.py`, `compare_layouts.py`, `compare_qad_init.py`, `qad_layers.py` | Consistency checks between runs and layouts. |

## Baselines from other formats

| Script | Job |
| --- | --- |
| `gguf_baseline.sh`, `gguf_suite.sh`, `gguf_arc_rerun.sh`, `gguf_eval_dense.sh` | GGUF files (k-quants, i-quants) and dense exports scored in llama.cpp with the same flags. |
| `chain_gguf4b.sh`, `chain_gguf_iq.sh` | The Qwen3.5-4B reference points. |
| `exl3_build.sh`, `exl3_convert.sh` | Build upstream exllamav3 and convert a model for the EXL3 reference. |

## Studies of the format

| Script | Question |
| --- | --- |
| `scale_study.py`, `scope_study.py` | Which scale system and which scope of scales work best? |
| `rotation_study.py`, `block_rotation_study.py` | Where does the Hadamard rotation help? |
| `candidate_usage.py` | Which (bits, block) candidates does the allocator use? |

## Calibration text

| Script | Job |
| --- | --- |
| `build_calib_mix.py` | The multi-domain calibration file. |
| `make_calib_qad.py`, `make_calib_tasks.py` | The QAD text and the task-style text. |

## Profiling and instruction probes

| Script | Job |
| --- | --- |
| `profile_encode.py`, `profile_new.py`, `profile_probe.py` | Where the quantizer spends its time. |
| `isa/probe.hip`, `isa/run.hip` | Check RDNA2 instruction facts (spec section 7) on a V620. |
