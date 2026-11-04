# Documentation index

## User journey

| Step | Read |
| --- | --- |
| Start | [Installation and platform boundaries](installation.md), [no-download CPU tutorial](getting-started.md#executed-no-download-cpu-smoke), then [first model and support matrix](getting-started.md) |
| Prepare data | [Independent PTQ, QAD, tuning and final-test inputs](evaluation.md#prepare-independent-data) |
| Quantize | [First-model walkthrough](getting-started.md#first-model-walkthrough), [canonical commands and defaults](usage.md) |
| Validate | [Coverage, size and metric interpretation](evaluation.md#read-the-results), [task benchmark setup and safety](../bench/README.md) |
| Recover | [Phase resources, artifacts and troubleshooting](operations.md) |
| Serve or distribute | [Dense export versus native serving](getting-started.md#export-and-native-serving), [portability](operations.md#distribute-a-checkpoint) |
| Automate locally | [MCP setup, workspace capabilities and durable jobs](mcp.md) |
| Contribute | [Dependency extras, unit/integration tests and full checks](installation.md#contributor-tests-and-checks) |

The walkthrough uses explicit recipe settings; [usage.md](usage.md) remains the
CLI reference. A research table is evidence for the stated run, not a replacement
for current help or an installation guarantee.

## Reference and research

| Document | Content |
| --- | --- |
| [architecture.md](architecture.md) | Phases, module map, execution boundaries |
| [overview.html](overview.html) | Interactive format anatomy and bit budget |
| [format-spec.md](format-spec.md) | Published v3.2 serialization reference: section 2 is normative for the tensor representation; experimental/runtime discussion is separately scoped |
| [moe.md](moe.md) | MoE eligibility, current stacked layout, dated OLMoE measurements |
| [qad.md](qad.md) | Implemented QAD behavior, exceptions, dated quality and runtime studies |
| [speed.md](speed.md) | Cache identities, fused encoder and measured speed |
| [results.md](results.md) | Model/corpus-specific numerical, task and runtime results |
| [surfaces.md](surfaces.md) | 27B decision guide and completed 2026-10-29/31 measurements |
| [lowbit.md](lowbit.md) | INT8 and limited-parameter final-KL studies |
| [design-notes.md](design-notes.md) | Rotation and comparisons with GGUF/EXL3 |
| [experiments/README.md](../experiments/README.md) | Scripts and provenance for reported numbers |

## Historical maintainer material

[Handover, 2026-10-29](handover.md) records the state of one session, including
then-current runtime blockers and superseded next steps. It is not current
installation, recovery or serving guidance; later results are linked from its header.
