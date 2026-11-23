# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Optional source reader for the stock Inkling-2-Mini checkpoint.

The published checkpoint is a *serving* layout (``model.llm.*``): the routed experts of
layers 0-19 are NVFP4 (``w13_weight`` U8 ``[E, 2I, H/2]`` + E4M3 block scales + one amax
per expert), those of layers 20-23 are MXFP8 (``w13_weight`` E4M3 ``[E, 2I, H]`` + UE8M0
block scales), everything else is bf16 or fp32. Nothing here is on a default path: the
module is imported only by code that asks for this source (docs/inkling-source.md).

What it does

* decodes the experts to float32, **one chunk of experts at a time**, on the CPU or on a
  device (GPU), with the arithmetic of the reference decoders
  (``tools/rdna/inkling2/tiny_reference.py`` of the vLLM fork): every element is a
  single fp32 product of exactly decoded factors, so the result is bit-identical on any
  device. Raw bytes reach a device through :class:`hi2.pinned.PinnedWindow`;
* de-interleaves ``w13`` (rows ``[g0, u0, g1, u1, ...]`` in the weights *and* in the
  scales) into ``gate`` ``[I, H]`` and ``up`` ``[I, H]``; ``down`` is ``w2`` as stored;
* names the per-expert tensors the way :mod:`hi2.moe` has them after ``unfuse_experts``
  (``model.language_model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight``) and maps
  the dense serving names to the Hugging Face model names (the mapping of the fork's
  ``ref_checkpoint.build_name_map``).

Formats (value of one element)

* NVFP4: ``e2m1(code) * e4m3(block scale) * (amax[e] / 2688)``, blocks of 16 along K.
  ``amax / 2688`` is divided in float64 and rounded once to float32; the block scale
  times that factor is rounded to float32, then the element product: two fp32 roundings,
  in this order (as the reference).
* MXFP8: ``e4m3(code) * 2^(u - 127)``, blocks of 32 along K (``u`` = UE8M0 byte).
"""

import json
from collections.abc import Iterator
from pathlib import Path

import regex as re
import torch
import torch.nn.functional as F

from .checkpoint import weight_map
from .pinned import PinnedWindow

HF_PREFIX = "model.language_model"
SERVING_PREFIX = "model.llm"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
NVFP4_BLOCK = 16
MXFP8_BLOCK = 32
# e2m1 code -> value: sign bit 3, exponent bits 2-1, mantissa bit 0 (index 8 is -0.0)
_FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_VALUES = _FP4_MAGNITUDES + tuple(-v for v in _FP4_MAGNITUDES)
_UE8M0_NAN = 255  # 0xFF is NaN in the OCP MX spec; no real checkpoint stores it
_UE8M0_ZERO_BITS = 0x00400000  # 2^-127 is a float32 subnormal: mantissa bit 22

_LUTS: dict[tuple[str, torch.device], torch.Tensor] = {}


def _lut(kind: str, dev: torch.device) -> torch.Tensor:
    """``[256, D]`` float32 decode table on ``dev``: ``e4m3`` (D=1) or ``fp4`` (D=2, the
    low nibble first)."""
    key = (kind, torch.device(dev))
    t = _LUTS.get(key)
    if t is None:
        codes = torch.arange(256, dtype=torch.uint8)
        if kind == "e4m3":
            t = codes.view(torch.float8_e4m3fn).float()[:, None]
        elif kind == "fp4":
            fp4 = torch.tensor(FP4_VALUES, dtype=torch.float32)
            c = codes.long()
            t = torch.stack((fp4[c & 0xF], fp4[c >> 4]), dim=-1)
        else:
            raise ValueError(kind)
        t = _LUTS[key] = t.contiguous().to(dev)
    return t


def _bytes(x: torch.Tensor) -> torch.Tensor:
    """The same storage as uint8 (E4M3 weights and scales travel as bytes)."""
    return x if x.dtype == torch.uint8 else x.view(torch.uint8)


def e4m3_to_float32(x: torch.Tensor) -> torch.Tensor:
    """Float32 value of E4M3 (``float8_e4m3fn``) codes, by table lookup: the same on every
    device and independent of the device's fp8 conversion kernels."""
    return F.embedding(_bytes(x).to(torch.int32), _lut("e4m3", x.device)).squeeze(-1)


def ue8m0_to_float32(x: torch.Tensor) -> torch.Tensor:
    """``2^(u - 127)`` as float32 for UE8M0 bytes ``u``, built from the exponent field
    (exact on every device; ``u = 0`` is the subnormal ``2^-127`` and ``u = 255`` is
    ``+inf``, as ``exp2(128)`` is)."""
    u = _bytes(x).to(torch.int32)
    bits = torch.where(u == 0, torch.full_like(u, _UE8M0_ZERO_BITS), u << 23)
    return bits.view(torch.float32)


def dequant_nvfp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    amax: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    """U8 ``[E, N, K/2]`` + E4M3 ``[E, N, K/16]`` (+ ``amax[E]``, ``denominator[1]``) ->
    float32 ``[E, N, K]``. Bit-identical to ``tiny_reference.dequant_nvfp4``."""
    e, n, kh = packed.shape
    k = 2 * kh
    if (
        k % NVFP4_BLOCK
        or tuple(scale.shape) != (e, n, k // NVFP4_BLOCK)
        or amax.numel() != e
    ):
        raise ValueError(
            f"NVFP4 shapes: packed {tuple(packed.shape)}, scale {tuple(scale.shape)}, "
            f"amax {tuple(amax.shape)}"
        )
    dev = packed.device
    # the global factor is divided in float64 on the host, as the reference does
    den = denominator.detach().cpu().double().reshape(-1)[0]
    glob = (amax.detach().cpu().double() / den).float().to(dev)
    block = e4m3_to_float32(scale) * glob[:, None, None]
    vals = F.embedding(packed.to(torch.int32), _lut("fp4", dev))  # [E, N, K/2, 2]
    vals = vals.reshape(e, n, k // NVFP4_BLOCK, NVFP4_BLOCK)
    vals.mul_(block[..., None])
    return vals.reshape(e, n, k)


def dequant_mxfp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """E4M3 ``[E, N, K]`` + UE8M0 ``[E, N, K/32]`` -> float32 ``[E, N, K]``. Bit-identical
    to ``tiny_reference.dequant_mxfp8``."""
    e, n, k = weight.shape
    if k % MXFP8_BLOCK or tuple(scale.shape) != (e, n, k // MXFP8_BLOCK):
        raise ValueError(
            f"MXFP8 shapes: weight {tuple(weight.shape)}, scale {tuple(scale.shape)}"
        )
    if bool((_bytes(scale) == _UE8M0_NAN).any()):
        raise ValueError("UE8M0 scale 0xFF (NaN) in an MXFP8 tensor")
    vals = e4m3_to_float32(weight).reshape(e, n, k // MXFP8_BLOCK, MXFP8_BLOCK)
    vals.mul_(ue8m0_to_float32(scale)[..., None])
    return vals.reshape(e, n, k)


# ---------------------------------------------------------------- names


def expert_name(layer: int, e: int, proj: str) -> str:
    """Name of one expert projection after ``hi2.moe.unfuse_experts``."""
    if proj not in PROJECTIONS:
        raise ValueError(f"{proj!r} is not one of {PROJECTIONS}")
    return f"{HF_PREFIX}.layers.{layer}.mlp.experts.{e}.{proj}.weight"


def name_map(num_layers: int, full_layers) -> dict[str, str]:
    """Serving name -> Hugging Face model name of every dense tensor (the routed experts
    are fused 3-D tensors in the serving layout and are not in this map). Spans every
    layer, as the fork's ``ref_checkpoint.build_name_map`` does: a key a layer does not
    have (loop controllers on a non-loop layer, ...) is simply absent from the file.
    ``full_layers``: the full-attention layers, whose k/v short convs are dynamic."""
    full = set(full_layers)
    s, h = SERVING_PREFIX, HF_PREFIX
    m = {
        f"{s}.embed.weight": f"{h}.embed_tokens.weight",
        f"{s}.embed_norm.weight": f"{h}.embed_tokens.embed_norm.weight",
        f"{s}.norm.weight": f"{h}.norm.weight",
        f"{s}.unembed.weight": "lm_head.weight",
        f"{s}.attn_res.output_query": f"{h}.final_input.query",
    }
    for i in range(num_layers):
        m[f"{s}.attn_res.mlp_queries.{i}"] = f"{h}.layers.{i}.mlp_input.query"
        if i + 1 < num_layers:
            m[f"{s}.attn_res.attn_queries.{i}"] = (
                f"{h}.layers.{i + 1}.attention_input.query"
            )
        src, dst = f"{s}.layers.{i}", f"{h}.layers.{i}"
        attn = f"{dst}.self_attn"
        m.update(
            {
                f"{src}.attn_norm.weight": f"{dst}.input_layernorm.weight",
                f"{src}.q_input_norm.weight": f"{dst}.q_input_layernorm.weight",
                f"{src}.k_input_norm.weight": f"{dst}.k_input_layernorm.weight",
                f"{src}.v_input_norm.weight": f"{dst}.v_input_layernorm.weight",
                f"{src}.mlp_norm.weight": f"{dst}.post_attention_layernorm.weight",
                f"{src}.attn_sconv.weight": (
                    f"{dst}.post_attention_convolution.conv1d.weight"
                ),
                f"{src}.mlp_sconv.weight": f"{dst}.post_mlp_convolution.conv1d.weight",
                f"{src}.attn.wq_du.weight": f"{attn}.q_proj.linear.weight",
                f"{src}.attn.q_l2_clip.scale": f"{attn}.q_proj.norm.scale",
                f"{src}.attn.wk_dv.weight": f"{attn}.k_proj.linear.weight",
                f"{src}.attn.wv_dv.weight": f"{attn}.v_proj.linear.weight",
                f"{src}.attn.wo_ud.weight": f"{attn}.o_proj.linear.weight",
                f"{src}.attn.gate_attn.weight": f"{attn}.o_proj.gate_proj.weight",
                f"{src}.attn.attn_sink_proj.weight": f"{attn}.sink_proj.weight",
                f"{src}.attn.wr_du.weight": (
                    f"{attn}.relative_position_bias.linear.weight"
                ),
                f"{src}.attn.rel_l2_clip.scale": (
                    f"{attn}.relative_position_bias.norm.scale"
                ),
                f"{src}.attn.rel_logits_proj.proj": (
                    f"{attn}.relative_position_bias.proj"
                ),
                f"{src}.mlp.gate.weight": f"{dst}.mlp.gate.weight",
                f"{src}.mlp.gate.bias": f"{dst}.mlp.gate.e_score_correction_bias",
            }
        )
        for side in ("k", "v"):
            base_s, base_d = f"{src}.attn.{side}_sconv", f"{attn}.{side}_proj.sconv"
            if i in full:
                m[f"{base_s}.static_w"] = f"{base_d}.static_weight"
                m[f"{base_s}.weight_proj"] = f"{base_d}.weight_proj.weight"
            else:
                m[f"{base_s}.weight"] = f"{base_d}.conv1d.weight"
        for ctl, gate in (
            ("attn", "q_message_gate"),
            ("k", "k_message_gate"),
            ("v", "v_message_gate"),
            ("mlp", "mlp_message_gate"),
        ):
            for part, mod in (
                ("down", "down_proj"),
                ("forget", "forget_gate_proj"),
                ("gain", "message_gate_proj"),
            ):
                m[f"{src}.controllers.{ctl}.{part}_weight"] = (
                    f"{dst}.{gate}.{mod}.weight"
                )
    return m


# Roles of the serving tensors, first match wins. ``expert`` and ``attn_linear`` are the
# HI2 candidates of the working proposal; everything else stays bf16/fp32.
_ROLES = (
    (r"^model\.visual\.", "vision"),
    (r"\.mlp\.experts\.", "expert"),
    (r"\.attn\.(wq_du|wk_dv|wv_dv|wo_ud|gate_attn|wr_du)\.weight$", "attn_linear"),
    (r"\.attn\.attn_sink_proj\.weight$", "attn_sink"),
    (
        r"\.attn\.(q_l2_clip\.scale|rel_l2_clip\.scale|rel_logits_proj\.proj)$",
        "attn_aux",
    ),
    (r"\.attn\.(k_sconv|v_sconv)\.", "kv_sconv"),
    (r"\.(attn_sconv|mlp_sconv)\.weight$", "sconv"),
    (r"\.controllers\.", "controller"),
    (r"\.mlp\.gate\.weight$", "router"),
    (r"\.mlp\.gate\.bias$", "router_bias"),
    (r"^model\.llm\.attn_res\.", "attn_res_query"),
    (r"^model\.llm\.middle_cycle_cross_pass_", "cross_pass_query"),
    (r"^model\.llm\.embed\.weight$", "embed"),
    (r"^model\.llm\.unembed\.weight$", "unembed"),
    (r"norm\.weight$", "norm"),
)
_ROLE_RES = [(re.compile(p), r) for p, r in _ROLES]


def tensor_role(name: str) -> str:
    """Class of a serving tensor of ``model.safetensors`` (``other`` if unknown)."""
    for rx, role in _ROLE_RES:
        if rx.search(name):
            return role
    return "other"


# ---------------------------------------------------------------- the reader


class InklingSource:
    """Reader of an Inkling-2-Mini serving checkpoint directory, an expert chunk at a time.

    ``device`` is where the decoded float32 tensors live (``cpu`` or ``cuda:N``; HIP on
    ROCm). The raw bytes are read with ``safetensors`` slices (only the requested experts
    are read) and reach a device through a pinned window. Use as a context manager or
    call :meth:`close`.
    """

    def __init__(
        self,
        path: str | Path,
        device: str | torch.device = "cpu",
        stager: PinnedWindow | None = None,
    ):
        self.path = Path(path)
        self.device = torch.device(device)
        self._stager = stager if stager is not None else PinnedWindow()
        self._wmap = weight_map(self.path)
        self._files: dict = {}
        self._hf_to_serving: dict[str, str] | None = None
        cfg = json.loads((self.path / "config.json").read_text())
        text = cfg.get("text_config", cfg)
        self.num_layers: int = text["num_hidden_layers"]
        self.n_experts: int = text["n_routed_experts"]
        self.hidden: int = text["hidden_size"]
        self.inter: int = text["intermediate_size"]
        self.unpadded_vocab: int = text["unpadded_vocab_size"]
        self.full_layers = frozenset(
            set(range(self.num_layers)) - set(text["local_layer_ids"])
        )
        self._check_layout()

    def __enter__(self) -> "InklingSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._files.clear()  # the safetensors handles unmap when dropped

    # -- raw access
    def _file(self, name: str):
        from safetensors import safe_open

        shard = self._wmap[name]
        f = self._files.get(shard)
        if f is None:
            f = self._files[shard] = safe_open(self.path / shard, "pt")
        return f

    def names(self) -> list[str]:
        """The serving names of the index (the MTP and vision files are not read here)."""
        return list(self._wmap)

    def header(self, name: str) -> tuple[str, tuple[int, ...]]:
        """``(safetensors dtype string, shape)`` of a tensor, without reading it."""
        sl = self._file(name).get_slice(name)
        return sl.get_dtype(), tuple(sl.get_shape())

    def tensor(self, name: str) -> torch.Tensor:
        """A whole serving tensor, as stored (CPU)."""
        return self._file(name).get_tensor(name)

    def _rows(self, name: str, experts: list[int]) -> torch.Tensor:
        """Leading-dimension entries ``experts`` of a stacked tensor (CPU); consecutive
        experts are one slice."""
        sl = self._file(name).get_slice(name)
        runs: list[list[int]] = []
        for e in experts:
            if runs and e == runs[-1][1]:
                runs[-1][1] += 1
            else:
                runs.append([e, e + 1])
        parts = [sl[a:b] for a, b in runs]
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    # -- layout
    def _expert_names(self, layer: int, which: str) -> tuple[str, str, str, str]:
        base = f"{SERVING_PREFIX}.layers.{layer}.mlp.experts.{which}_weight"
        return base, f"{base}_scale", f"{base}_amax", f"{base}_outer_scale_denominator"

    def expert_format(self, layer: int) -> str:
        """``nvfp4`` or ``mxfp8``, from the dtypes in the header."""
        w, s, a, d = self._expert_names(layer, "w13")
        wd, sd = self.header(w)[0], self.header(s)[0]
        if wd == "U8" and sd == "F8_E4M3" and a in self._wmap and d in self._wmap:
            return "nvfp4"
        if wd == "F8_E4M3" and sd == "U8":
            return "mxfp8"
        raise ValueError(
            f"layer {layer}: unknown expert format (weight {wd}, scale {sd})"
        )

    def formats(self) -> list[str]:
        return [self.expert_format(i) for i in range(self.num_layers)]

    def _check_layout(self) -> None:
        """Every layer has both stacks in a known format, with the shapes the config gives."""
        e, h, i = self.n_experts, self.hidden, self.inter
        for layer in range(self.num_layers):
            width = 2 if self.expert_format(layer) == "nvfp4" else 1
            for which, rows, cols in (("w13", 2 * i, h), ("w2", h, i)):
                shape = self.header(self._expert_names(layer, which)[0])[1]
                if shape != (e, rows, cols // width):
                    raise ValueError(
                        f"layer {layer} {which}: shape {shape}, expected "
                        f"{(e, rows, cols // width)}"
                    )

    # -- experts
    def _experts(self, experts) -> list[int]:
        if experts is None:
            return list(range(self.n_experts))
        if isinstance(experts, int):
            experts = [experts]
        elif isinstance(experts, slice):
            experts = range(*experts.indices(self.n_experts))
        out = [int(e) for e in experts]
        if not out or min(out) < 0 or max(out) >= self.n_experts:
            raise IndexError(f"experts {out[:4]}... outside [0, {self.n_experts})")
        return out

    def raw(self, layer: int, which: str, experts=None, device=None):
        """``(weight, scale, amax, denominator)`` of ``which`` (``w13`` or ``w2``) for
        ``experts`` (all by default): weight and scale as bytes on ``device`` (one staged
        copy), amax and denominator as host tensors (None for MXFP8). :meth:`decode` turns
        them into float32; the two halves are public so a profile can time them apart."""
        if which not in ("w13", "w2"):
            raise ValueError(which)
        dev = torch.device(device) if device is not None else self.device
        ex = self._experts(experts)
        wn, sn, an, dn = self._expert_names(layer, which)
        stage = self._stager.to_device
        w = stage(_bytes(self._rows(wn, ex)), dev)
        s = stage(_bytes(self._rows(sn, ex)), dev)
        if self.expert_format(layer) == "mxfp8":
            return w, s, None, None
        return w, s, self._rows(an, ex), self.tensor(dn)

    @staticmethod
    def decode(raw, rows: slice | None = None) -> torch.Tensor:
        """Float32 ``[e, N', K]`` of the staged bytes ``raw``; with ``rows`` only those
        rows of the stack (``slice(0, None, 2)``: the gate rows of ``w13``)."""
        w, s, amax, den = raw
        if rows is not None:
            w, s = w[:, rows], s[:, rows]
        if amax is None:
            return dequant_mxfp8(w, s)
        return dequant_nvfp4(w, s, amax, den)

    def stack(self, layer: int, which: str, experts=None, device=None) -> torch.Tensor:
        """float32 ``[e, N, K]`` of ``w13`` (rows interleaved ``[g0, u0, ...]`` as stored)
        or ``w2``, for ``experts`` (all by default): the decoders' output before any
        de-interleave, which is what the reference decoders return."""
        return self.decode(self.raw(layer, which, experts, device))

    def gate_up(
        self, layer: int, experts=None, device=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(gate, up)`` float32 ``[e, I, H]`` of ``experts`` (all by default): the even
        and odd rows of the interleaved ``w13``, each contiguous."""
        raw = self.raw(layer, "w13", experts, device)
        return self.decode(raw, slice(0, None, 2)), self.decode(raw, slice(1, None, 2))

    def down(self, layer: int, experts=None, device=None) -> torch.Tensor:
        """``down`` float32 ``[e, H, I]`` of ``experts`` (all by default)."""
        return self.decode(self.raw(layer, "w2", experts, device))

    def expert(
        self, layer: int, e: int, device=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(gate [I, H], up [I, H], down [H, I])`` float32 of one expert."""
        gate, up = self.gate_up(layer, [e], device)
        return gate[0], up[0], self.down(layer, [e], device)[0]

    def iter_experts(
        self, layer: int, experts=None, chunk: int = 16, device=None
    ) -> Iterator[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """``(e, gate, up, down)`` per expert, ``chunk`` experts decoded at a time (about
        ``chunk * 80 MB`` of float32 and temporaries on the device for Inkling)."""
        ex = self._experts(experts)
        for a in range(0, len(ex), chunk):
            part = ex[a : a + chunk]
            gate, up = self.gate_up(layer, part, device)
            down = self.down(layer, part, device)
            for j, e in enumerate(part):
                yield e, gate[j], up[j], down[j]

    def fused_layer(
        self,
        layer: int,
        chunk: int = 16,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(gate_up [E, 2I, H], down [E, H, I])`` of one layer in ``dtype`` on
        ``device``: the fused 3-D layout of transformers 5 (rows ``[gate; up]``) that
        :func:`hi2.moe.unfuse_experts` takes. The whole layer is materialised (4.1 GB in
        float32 for Inkling); the decode itself is chunked."""
        dev = torch.device(device) if device is not None else self.device
        e, i, h = self.n_experts, self.inter, self.hidden
        gate_up = torch.empty(e, 2 * i, h, dtype=dtype, device=dev)
        down = torch.empty(e, h, i, dtype=dtype, device=dev)
        for a in range(0, e, chunk):
            part = list(range(a, min(a + chunk, e)))
            gate, up = self.gate_up(layer, part, dev)
            gate_up[a : a + len(part), :i] = gate.to(dtype)
            gate_up[a : a + len(part), i:] = up.to(dtype)
            down[a : a + len(part)] = self.down(layer, part, dev).to(dtype)
        return gate_up, down

    def sample_experts(self, n: int) -> list[int]:
        """``n`` experts spread evenly over ``[0, E)``, the first and the last included."""
        if not 1 <= n <= self.n_experts:
            raise ValueError(f"n must be in [1, {self.n_experts}]")
        if n == 1:
            return [0]
        return sorted({round(i * (self.n_experts - 1) / (n - 1)) for i in range(n)})

    # -- Hugging Face names
    def hf_names(self) -> dict[str, str]:
        """Serving name -> HF name of the dense tensors present in the file."""
        return {
            s: d
            for s, d in name_map(self.num_layers, self.full_layers).items()
            if s in self._wmap
        }

    def hf_expert_names(self, layer: int, experts=None) -> Iterator[str]:
        """The per-expert projection names of ``layer``, expert-major (gate, up, down)."""
        for e in self._experts(experts):
            for proj in PROJECTIONS:
                yield expert_name(layer, e, proj)

    def load(
        self, hf_name: str, dtype: torch.dtype | None = None, device=None
    ) -> torch.Tensor:
        """A tensor by its Hugging Face name: a dense tensor as stored (the head trimmed
        to the unpadded vocabulary), or one dequantized expert projection."""
        dev = torch.device(device) if device is not None else self.device
        m = re.match(
            rf"^{re.escape(HF_PREFIX)}\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)\.weight$",
            hf_name,
        )
        if m:
            layer, e, proj = int(m.group(1)), int(m.group(2)), m.group(3)
            if proj == "down_proj":
                t = self.down(layer, [e], dev)[0]
            elif proj in PROJECTIONS:
                gate, up = self.gate_up(layer, [e], dev)
                t = (gate if proj == "gate_proj" else up)[0]
            else:
                raise KeyError(hf_name)
        else:
            if self._hf_to_serving is None:
                self._hf_to_serving = {d: s for s, d in self.hf_names().items()}
            if hf_name not in self._hf_to_serving:
                raise KeyError(hf_name)
            t = self.tensor(self._hf_to_serving[hf_name])
            if hf_name == "lm_head.weight":
                t = t[: self.unpadded_vocab]
            t = t.to(dev)
        return t if dtype is None else t.to(dtype)
