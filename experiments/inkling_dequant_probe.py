# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Exactness and cost of decoding one Inkling-2-Mini expert layer on a device.

For every layer asked for (an NVFP4 layer and an MXFP8 layer by default):

* exactness: the decode of a few experts on ``--device`` against the decode on the CPU
  (``hi2.inkling_source``; the CPU decode is itself compared with the reference decoders of
  the vLLM fork by ``tests/test_inkling_source.py``): ``torch.equal`` and max abs diff;
* split cost of one chunk of experts: the host copy of the packed bytes (mapped file to
  RAM), their staging plus host-to-device copy, the decode kernels on a device-resident
  chunk;
* the whole layer, per chunk size and output dtype: ``fused_layer`` (read, stage, decode,
  store into the fused ``[E, 2I, H]`` and ``[E, H, I]`` tensors) in ms, the peak device
  memory (``torch.cuda.max_memory_allocated``), and the streaming variant that keeps one
  chunk at a time (``iter_experts``, nothing materialised) in ms.

The checkpoint is read through the page cache, so the numbers are of a warm cache (the
first repetition is reported apart). Run it on a GPU only when the device is yours.

    HIP_VISIBLE_DEVICES=0 python experiments/inkling_dequant_probe.py CKPT \\
        --device cuda:0 --out dequant_probe.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from hi2.inkling_source import InklingSource

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def timed(fn, dev, repeat: int):
    """``(first ms, median ms of the rest or of the first, last result)``."""
    ts = []
    out = None
    for _ in range(repeat):
        out = None  # drop the previous result first: it would count in the peak memory
        sync(dev)
        t0 = time.perf_counter()
        out = fn()
        sync(dev)
        ts.append(1e3 * (time.perf_counter() - t0))
    rest = ts[1:] or ts
    return ts[0], statistics.median(rest), out


def peak_reset(dev):
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)


def peak_mib(dev) -> float | None:
    return torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else None


def device_info(dev: torch.device) -> dict:
    if dev.type != "cuda":
        return {"device": str(dev)}
    p = torch.cuda.get_device_properties(dev)
    info = {
        "device": str(dev),
        "name": p.name,
        "total_gib": p.total_memory / 2**30,
        "arch": getattr(p, "gcnArchName", None),
        "torch": torch.__version__,
        "hip": torch.version.hip,
    }
    for k in ("pci_bus_id", "pci_device_id", "pci_domain_id"):
        if hasattr(p, k):
            info[k] = getattr(p, k)
    return info


def exactness(src: InklingSource, layer: int, experts, dev) -> dict:
    out = {}
    with InklingSource(src.path, "cpu") as cpu:
        for which, getter in (
            ("gate_up", lambda s, d: s.gate_up(layer, experts, d)),
            ("down", lambda s, d: (s.down(layer, experts, d),)),
        ):
            ref = getter(cpu, "cpu")
            got = getter(src, dev)
            out[which] = {
                "equal": all(torch.equal(r, g.cpu()) for r, g in zip(ref, got)),
                "max_abs_diff": max(
                    float((r - g.cpu()).abs().max()) for r, g in zip(ref, got)
                ),
            }
    return out


def split_cost(
    src: InklingSource, cpu: InklingSource, layer: int, chunk: int, dev, repeat
):
    """ms of the host copy of the packed bytes, of their staging plus host-to-device copy,
    and of the decode of one chunk of experts (device-resident bytes)."""
    ex = list(range(chunk))

    def host_copy():  # the mapped bytes into RAM: what the staging has to read at least
        return [
            tuple(t.clone() for t in cpu.raw(layer, w, ex, "cpu")[:2])
            for w in ("w13", "w2")
        ]

    host1, host, _ = timed(host_copy, dev, repeat)
    stage1, stage, raws = timed(
        lambda: [src.raw(layer, w, ex, dev) for w in ("w13", "w2")], dev, repeat
    )
    dec1, dec, _ = timed(lambda: [src.decode(r) for r in raws], dev, repeat)
    peak_reset(dev)
    sync(dev)
    base = torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0
    _ = [src.decode(r) for r in raws]
    sync(dev)
    raw_bytes = sum(t.numel() * t.element_size() for r in raws for t in r[:2])
    return {
        "experts": chunk,
        "raw_mib": raw_bytes / 2**20,
        "host_copy_ms_first": host1,
        "host_copy_ms": host,
        "stage_h2d_ms_first": stage1,
        "stage_h2d_ms": stage,
        "decode_ms_first": dec1,
        "decode_ms": dec,
        "decode_extra_peak_mib": (
            (peak_mib(dev) - base / 2**20) if dev.type == "cuda" else None
        ),
        "decode_gelem_per_s": chunk * 3 * src.inter * src.hidden / (dec * 1e-3) / 1e9,
    }


def layer_cost(src: InklingSource, layer: int, chunk: int, dtype, dev, repeat) -> dict:
    peak_reset(dev)
    first, med, fused = timed(
        lambda: src.fused_layer(layer, chunk, dev, dtype), dev, repeat
    )
    fused_peak = peak_mib(dev)
    nbytes = sum(t.numel() * t.element_size() for t in fused)
    del fused
    peak_reset(dev)

    def stream():
        n = 0
        for _e, g, u, d in src.iter_experts(layer, None, chunk, dev):
            n += g.shape[0] + u.shape[0] + d.shape[0]
        return n

    s_first, s_med, _ = timed(stream, dev, repeat)
    return {
        "layer": layer,
        "format": src.expert_format(layer),
        "chunk": chunk,
        "dtype": str(dtype).removeprefix("torch."),
        "fused_layer_bytes": nbytes,
        "fused_ms_first": first,
        "fused_ms": med,
        "fused_peak_mib": fused_peak,
        "stream_ms_first": s_first,
        "stream_ms": s_med,
        "stream_peak_mib": peak_mib(dev),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layers", default="0,20")
    ap.add_argument("--chunks", default="8,16,32")
    ap.add_argument("--dtypes", default="float32,float16,bfloat16")
    ap.add_argument("--check-experts", default="0,17,64,127")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("inkling_dequant_probe.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    layers = [int(x) for x in a.layers.split(",")]
    chunks = [int(x) for x in a.chunks.split(",")]
    dtypes = [DTYPES[x] for x in a.dtypes.split(",")]
    result = {"device": device_info(dev), "repeat": a.repeat, "layers": []}
    print(json.dumps(result["device"]), flush=True)
    with InklingSource(a.ckpt, dev) as src, InklingSource(a.ckpt, "cpu") as cpu:
        for layer in layers:
            entry = {"layer": layer, "format": src.expert_format(layer)}
            entry["exactness"] = exactness(
                src, layer, [int(x) for x in a.check_experts.split(",")], dev
            )
            print(f"layer {layer} exactness vs CPU: {entry['exactness']}", flush=True)
            entry["split"] = [
                split_cost(src, cpu, layer, c, dev, a.repeat) for c in chunks
            ]
            for s in entry["split"]:
                print(
                    f"layer {layer} chunk {s['experts']:3d}: host copy {s['host_copy_ms']:.1f} ms, "
                    f"stage+h2d {s['stage_h2d_ms']:.1f} ms ({s['raw_mib']:.0f} MiB packed), "
                    f"decode {s['decode_ms']:.1f} ms ({s['decode_gelem_per_s']:.2f} G elem/s)",
                    flush=True,
                )
            entry["whole_layer"] = []
            for c in chunks:
                for dt in dtypes:
                    r = layer_cost(src, layer, c, dt, dev, a.repeat)
                    entry["whole_layer"].append(r)
                    print(
                        f"layer {layer} chunk {c:3d} {r['dtype']:<8}: fused {r['fused_ms']:.0f} ms "
                        f"(peak {r['fused_peak_mib']} MiB), streamed {r['stream_ms']:.0f} ms "
                        f"(peak {r['stream_peak_mib']} MiB)",
                        flush=True,
                    )
            result["layers"].append(entry)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
