# Documentation index

Start with three concepts:

1. **Inputs and quantization:** use eligible local model weights and calibration
   text to produce a low-bit checkpoint.
2. **Validation scope:** check structure, dense numerical quality and native
   runtime behavior separately. One does not establish the others.
3. **Execution and recovery:** choose devices, plan resources and keep the source,
   data and work paths needed to resume.

## User journey

| Step | Read |
| --- | --- |
| Start | [Installation and platform boundaries](installation.md), [no-download CPU tutorial](getting-started.md#executed-no-download-cpu-smoke), then [first model and support matrix](getting-started.md) |
| Prepare data | [Separate calibration, training, tuning and final-test inputs](evaluation.md#prepare-independent-data) |
| Quantize | [First-model walkthrough](getting-started.md#first-model-walkthrough), [canonical commands and defaults](usage.md) |
| Validate | [Coverage, size and metric interpretation](evaluation.md#read-the-results), [task benchmark setup and safety](../bench/README.md) |
| Recover | [Phase resources, artifacts and troubleshooting](operations.md) |
| Serve or distribute | [Dense export versus native serving](getting-started.md#export-and-native-serving), [portability](operations.md#distribute-a-checkpoint) |
| Automate locally | [Model Context Protocol (MCP) setup, workspace access and durable jobs](mcp.md) |
| Contribute | [Dependency extras, unit/integration tests and full checks](installation.md#contributor-tests-and-checks) |

Use the walkthrough for explicit recipe settings. Use [usage.md](usage.md) for
command-line options and defaults. Research tables describe the stated runs;
they do not replace current help or guarantee installation compatibility.

## Reference and research

| Document | Content |
| --- | --- |
| [architecture.md](architecture.md) | Phases, module map, execution boundaries |
| [overview.html](overview.html) | Interactive orientation, format anatomy and educational size estimate; not a quantization result |
| [format-spec.md](format-spec.md) | Published v3.2 serialization reference: section 2 is normative for the tensor representation; experimental/runtime discussion is separately scoped |
| [moe.md](moe.md) | Mixture-of-experts (MoE) eligibility, current stacked layout, dated OLMoE measurements |
| [qad.md](qad.md) | Quantization-aware distillation (QAD), exceptions, dated quality and runtime studies |
| [speed.md](speed.md) | Cache identities, fused encoder and measured speed |
| [results.md](results.md) | Model/corpus-specific numerical, task and runtime results |
| [surfaces.md](surfaces.md) | 27B decision guide and completed 2026-10-29/31 measurements |
| [lowbit.md](lowbit.md) | 8-bit integer (INT8) quantization and limited-parameter final-KL studies |
| [design-notes.md](design-notes.md) | Rotation and comparisons with GGUF/EXL3 |
| [experiments/README.md](../experiments/README.md) | Scripts and provenance for reported numbers |

## Historical maintainer material

[Handover, 2026-10-29](handover.md) records the state of one session, including
then-current runtime blockers and superseded next steps. It is not current
installation, recovery or serving guidance; later results are linked from its header.
