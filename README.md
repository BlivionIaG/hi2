# hi2: HadamardINT2-FP16 v3.2 quantizer (flex)

Standalone (torch, safetensors, regex, transformers for calibration). Implements
`docs/rdna2/hadamard-int2-v3.2-spec.md` only. Run from `tools/hi2`:

```bash
python -m hi2 quantize <hf_model_dir> <out_dir> --calib text.txt --target-bpw 3.0
python -m hi2 verify   <hf_model_dir> <out_dir>
python -m hi2 eval     <hf_model_dir> <out_dir> --text text.txt --skip-tokens 800000
```

## How it works

1. **Tabulate (calibration).** Layer by layer: one decoder layer is copied to each
   compute device, the calibration sequences run through it, and hooks accumulate a
   16x16 second-moment block per column group for every Linear. Each Linear then
   gets a table of the activation-weighted squared error of every row under every
   candidate `(bits in {2,4}) x (block in {16,32,64})`.
2. **Allocate (global).** Every row of the whole model picks its configuration to
   minimise the sum over tensors of relative weighted error under one model-wide
   `--target-bpw`.
3. **Encode (write).** The checkpoint is streamed again and each tensor is encoded
   with its picks (spec 2.4 layout, tensors `<name>.hi2_*`), everything else copied.

Without `--calib` it runs data-free (column weights of 1) and streams tensors
without loading the model.

## Placement

| Goal | Flags |
| --- | --- |
| CPU / RAM only | `--devices cpu` |
| One GPU | `--devices cuda:0` (pick the physical GPU with `HIP_VISIBLE_DEVICES`) |
| Several GPUs (2, 4, ...) | `--devices cuda:0,cuda:1,...`: calibration sequences are split across devices and every tensor's rows are sharded across them. Results are identical to one device |
| GPU + RAM, model larger than VRAM | automatic: the model stays in RAM, one layer at a time is on the GPU(s), hidden states stay in RAM |

"Several GPUs" is row-sharded quantization plus data-parallel calibration, not a
tensor-parallel forward pass (a single layer fits one GPU).

## Notes

* `--rotation` is **off** by default: with calibration data it measured worse end to
  end (spec 4.5). `auto`/`on` exist for data-free use and experiments.
* `--bits` / `--blocks` restrict the candidate set (for example `--bits 4 --blocks 64`
  for a uniform INT4/B64 reference). `--effort fast|normal|max`.
* Activations are propagated from the unquantized model (the final configuration is
  unknown until the global allocation), so there is no quantized-propagation mode.
* `workdir` (default `<out>.work`) holds tables, statistics and shard markers;
  finished output shards are kept on re-run.
* Calibration needs `--calib-samples x --seq-len` tokens of text. Evaluate on text
  that is not in the calibration region (`--skip-tokens`).
* `eval` dequantizes back to dense weights and runs the original architecture: it
  checks the format numerically, it is not a kernel test.
* Tests: `cd tools/hi2 && pytest tests`.
