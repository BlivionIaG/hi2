# Documentation index

Start with [usage.md](usage.md) to run the tool. Start with [architecture.md](architecture.md)
to learn how it works.

| Document | Content | Audience |
| --- | --- | --- |
| [usage.md](usage.md) | Commands, flags, defaults, examples | User |
| [architecture.md](architecture.md) | Phases, module map, diagrams | Reader of the code |
| [overview.html](overview.html) | Interactive explainer: format anatomy, bit budget | Newcomer |
| [format-spec.md](format-spec.md) | The rdna_hint_flex format, normative, v3.2 | Kernel and loader authors |
| [design-notes.md](design-notes.md) | Rotation, and where the format stands against GGUF and EXL3 | Researcher |
| [results.md](results.md) | Measured quality and speed, all models | Everyone |
| [qad.md](qad.md) | Quantization-aware distillation: design and measurements | User, researcher |
| [moe.md](moe.md) | Mixture-of-experts models and the stacked layout | User |
| [speed.md](speed.md) | Quantizer speed, the cache, the fused kernel | User |
| [lowbit.md](lowbit.md) | INT8 rows, final logit-KL, QAD data | Researcher |

The scripts that produced the numbers are described in
[../experiments/README.md](../experiments/README.md).
