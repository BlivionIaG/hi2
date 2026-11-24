# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""The encodes of an Inkling-2 run side by side: what they weigh, what they cost in KL, and whether
the layers that got the fewest bits are the ones that lose the most.

    python experiments/inkling_target_table.py --runs stage4/runs --evals stage4/eval \\
        [--targets 3.0 3.5 4.0] [--stock ~/hfcache/inkling2-mini] [--json OUT.json]

``--runs/bpwT/`` holds the output of ``hi2 quantize`` (``hi2_report.json``, ``model*.safetensors``),
``--evals/bpwT_SET.json`` the output of ``inkling_kl_eval.py`` for ``SET`` in mixed, wiki, pydocs and
fine (what is missing prints as ``-``). Three tables:

1. **Targets**: the experts' size, the whole directory's, the allocation's bits and shares, and the KL,
   top-1 agreement and perplexity ratio of every evaluated set.
2. **Bits per layer**: the bits per weight of each layer's experts, by target.
3. **Groups of the fine evaluation**: the KL of every group of layers, its bits per weight by target, its
   share of INT2 rows, the allocator's own predicted relative error of its experts and the weight error the
   evaluation measured, with a rank correlation of each against the KL. Bits and KL that disagree, or a
   predicted error that does not follow the KL, show up there.
"""

import argparse
import json
import math
import sys
from pathlib import Path

GIB = 2**30
EXPERT_NAME = "layers.{}.mlp.experts."
SETS = ("mixed", "wiki", "pydocs")


def _sibling(name: str):
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    return __import__(name)


def layer_stats(report: dict) -> dict[int, dict]:
    """Per layer: expert weights, bits, INT2 weights and the allocator's predicted relative error
    (``pred_rel_err`` of each expert tensor, weighted by the tensor's weights; the root mean square
    as well)."""
    mod = _sibling("inkling_alloc_report")
    out: dict[int, dict] = {}
    for name, t in report["tensors"].items():
        m = mod.EXPERT.search(name)
        if m is None or "shape" not in t:
            continue
        layer = int(m.group(1))
        rows, cols = t["shape"]
        w = rows * cols
        a = out.setdefault(
            layer, {"weights": 0, "bits": 0.0, "w2": 0.0, "err": 0.0, "err2": 0.0}
        )
        f4, f8 = t.get("frac_int4", 0.0), t.get("frac_int8", 0.0)
        a["weights"] += w
        a["bits"] += t["bpw"] * w
        a["w2"] += max(0.0, 1.0 - f4 - f8) * w
        if t.get("pred_rel_err") is not None:
            a["err"] += t["pred_rel_err"] * w
            a["err2"] += t["pred_rel_err"] ** 2 * w
    return out


def parse_group(spec: str) -> tuple[int, int]:
    lo, _, hi = spec.partition("-")
    return int(lo), int(hi or lo)


def group_stats(layers: dict[int, dict], lo: int, hi: int) -> dict:
    sel = [layers[i] for i in range(lo, hi + 1) if i in layers]
    w = sum(a["weights"] for a in sel)
    if not w:
        return {}
    return {
        "weights_g": w / 1e9,
        "bpw": sum(a["bits"] for a in sel) / w,
        "int2": sum(a["w2"] for a in sel) / w,
        "pred_err": sum(a["err"] for a in sel) / w,
        "pred_err_rms": math.sqrt(sum(a["err2"] for a in sel) / w),
    }


def ranks(xs: list[float]) -> list[float]:
    """1-based ranks, ties averaged."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if not sxx or not syy:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sxx * syy)


def expert_load(report: dict) -> dict[int, dict]:
    """Per layer, from the ``gate_proj`` of every expert (its gate, up and down see the same tokens):
    the tokens per expert, how skewed they are (``cv``: the standard deviation over the mean; ``hot``:
    the share of all tokens that go to the hottest eighth of the experts, 16 of 128), the bits per
    weight of the layer, of its least routed quarter of experts (``bpw_cold``) and of its most routed
    quarter (``bpw_hot``), and the Spearman correlation of an expert's tokens with its bits (``rho``)."""
    mod = _sibling("inkling_alloc_report")
    per: dict[int, list[tuple[float, float, int]]] = {}
    for name, t in report["tensors"].items():
        m = mod.EXPERT.search(name)
        if m is None or m.group(2) != "gate_proj" or t.get("n_tokens") is None:
            continue
        rows, cols = t["shape"]
        per.setdefault(int(m.group(1)), []).append(
            (t["n_tokens"], t["bpw"], rows * cols)
        )

    def bits(part):
        return sum(b * w for _, b, w in part) / sum(w for _, _, w in part)

    out: dict[int, dict] = {}
    for layer, experts in per.items():
        n = len(experts)
        tokens = [x[0] for x in experts]
        mean = sum(tokens) / n
        by_load = sorted(experts, key=lambda x: x[0])
        quarter = max(1, n // 4)
        out[layer] = {
            "experts": n,
            "mean_tokens": mean,
            "cv": math.sqrt(sum((x - mean) ** 2 for x in tokens) / n) / mean
            if mean
            else 0.0,
            "hot": sum(x[0] for x in by_load[-max(1, n // 8) :]) / sum(tokens)
            if mean
            else 0.0,
            "bpw": bits(experts),
            "bpw_cold": bits(by_load[:quarter]),
            "bpw_hot": bits(by_load[-quarter:]),
            "rho": spearman(tokens, [x[1] for x in experts]),
        }
    return out


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _dir_gib(path: Path, pattern: str) -> float | None:
    files = sorted(path.glob(pattern))
    return sum(f.stat().st_size for f in files) / GIB if files else None


def collect(runs: Path, evals: Path, targets: list[str], stock: Path | None) -> dict:
    mod = _sibling("inkling_alloc_report")
    out: dict = {"targets": {}, "stock": None}
    if stock is not None:
        out["stock"] = {
            "files_gib": _dir_gib(stock, "*.safetensors"),
            "mtp_gib": _dir_gib(stock, "mtp*.safetensors"),
        }
    for t in targets:
        run = runs / f"bpw{t}"
        report = _load(run / "hi2_report.json")
        entry: dict = {"evals": {}}
        if report is not None:
            alloc = mod.allocation_by_group(report)
            entry["experts"] = alloc.get("all")
            entry["layers"] = layer_stats(report)
            entry["load"] = expert_load(report)
            entry["total_gib"] = _dir_gib(run, "model*.safetensors")
        for s in (*SETS, "fine"):
            ev = _load(evals / f"bpw{t}_{s}.json")
            if ev is not None:
                entry["evals"][s] = ev
        out["targets"][t] = entry
    return out


def _f(x, spec=".4f") -> str:
    return "-" if x is None else format(x, spec)


def render(data: dict) -> str:
    targets = list(data["targets"])
    L: list[str] = []
    head = (
        "| target | experts GiB | other GiB | total GiB | bpw | INT2 / INT4 / INT8 % |"
    )
    head += "".join(f" {s}: KL / top-1 / ppl |" for s in SETS)
    L += ["#### Targets", "", head, "| " + " | ".join(["---"] * (6 + len(SETS))) + " |"]
    stock = data.get("stock")
    if stock and stock.get("files_gib") is not None:
        note = f"stock files {stock['files_gib']:.3f} GiB"
        if stock.get("mtp_gib") is not None:
            note += f" (MTP {stock['mtp_gib']:.3f})"
        L.insert(0, f"Stock directory: {note}; stock experts 14.594 GiB.\n")
    for t in targets:
        e = data["targets"][t]
        x = e.get("experts")
        cells = [t]
        if x:
            tot = e.get("total_gib")
            cells += [
                f"{x['gib']:.3f}",
                _f(None if tot is None else tot - x["gib"], ".3f"),
                _f(tot, ".3f"),
                f"{x['bpw']:.3f}",
                f"{100 * x['int2']:.1f} / {100 * x['int4']:.1f} / {100 * x['int8']:.1f}",
            ]
        else:
            cells += ["-"] * 5
        for s in SETS:
            ev = e["evals"].get(s)
            st = ev and ev.get("student")
            cells.append(
                "-"
                if not st
                else f"{st['kl_mean']:.4f} +- {_f(st.get('kl_sem'))} / {st['top1_agreement']:.3f} / x{st['ppl_ratio']:.4f}"
            )
        L.append("| " + " | ".join(cells) + " |")

    layers = sorted({i for e in data["targets"].values() for i in e.get("layers", {})})
    if layers:
        L += ["", "#### Bits per weight of each layer's experts", ""]
        L += [
            "| layer | " + " | ".join(targets) + " |",
            "| " + " | ".join(["---"] * (1 + len(targets))) + " |",
        ]
        for i in layers:
            row = [f"{i:02d}"]
            for t in targets:
                a = data["targets"][t].get("layers", {}).get(i)
                row.append(_f(a and a["bits"] / a["weights"], ".3f"))
            L.append("| " + " | ".join(row) + " |")

    loads = {t: e["load"] for t, e in data["targets"].items() if e.get("load")}
    if loads:
        first = next(iter(loads.values()))
        L += [
            "",
            "#### Expert load against bits (the gate of each expert; tokens counted by the calibration)",
            "",
            "| layer | tokens per expert | CV | hottest eighth of the experts: % of the tokens | "
            + " | ".join(
                f"{t}: bpw of the coldest / hottest quarter (rho)" for t in targets
            )
            + " |",
            "| " + " | ".join(["---"] * (4 + len(targets))) + " |",
        ]
        for i in sorted(first):
            a = first[i]
            cells = [
                f"{i:02d}",
                f"{a['mean_tokens']:.0f}",
                f"{a['cv']:.2f}",
                f"{100 * a['hot']:.0f}",
            ]
            for t in targets:
                b = loads.get(t, {}).get(i)
                cells.append(
                    "-"
                    if not b
                    else f"{b['bpw_cold']:.2f} / {b['bpw_hot']:.2f} ({_f(b['rho'], '+.2f')})"
                )
            L.append("| " + " | ".join(cells) + " |")
        for t, load in loads.items():
            ls = sorted(load)
            rhos = [load[i]["rho"] for i in ls if load[i]["rho"] is not None]
            bpw = [load[i]["bpw"] for i in ls]
            L += [
                "",
                f"At {t}: a layer's bits per weight against the CV of its load "
                f"{_f(spearman(bpw, [load[i]['cv'] for i in ls]), '+.2f')} and against the share of "
                f"its hottest eighth {_f(spearman(bpw, [load[i]['hot'] for i in ls]), '+.2f')} "
                f"(Spearman over {len(ls)} layers); an expert's bits against its tokens, within a "
                f"layer: mean {_f(sum(rhos) / len(rhos) if rhos else None, '+.2f')}.",
            ]

    for t in targets:
        fine = data["targets"][t]["evals"].get("fine")
        if not fine or not fine.get("groups"):
            continue
        L += ["", f"#### Groups of layers at {t} bpw (fine evaluation)", ""]
        wer = fine.get("weight_error_per_layer") or {}
        rows = []
        for spec, g in fine["groups"].items():
            lo, hi = parse_group(spec)
            gs = {
                tt: group_stats(data["targets"][tt].get("layers", {}), lo, hi)
                for tt in targets
            }
            own = gs[t]
            rms = [
                wer[str(i)]["relative_rms"]
                for i in range(lo, hi + 1)
                if str(i) in wer and wer[str(i)].get("relative_rms") is not None
            ]
            kl = g["kl_mean"]
            prms = own.get("pred_err_rms")
            wrms = sum(rms) / len(rms) if rms else None
            rows.append(
                {
                    "group": spec,
                    "kl": kl,
                    "sem": g.get("kl_sem"),
                    "top1": g.get("top1_agreement"),
                    "ppl": g.get("ppl_ratio"),
                    "bpw": {tt: gs[tt].get("bpw") for tt in targets},
                    "int2": own.get("int2"),
                    "pred_err": own.get("pred_err"),
                    "pred_err_rms": prms,
                    "weight_rms": wrms,
                    "weights_g": own.get("weights_g"),
                    "kl_per_pred2": kl / prms**2 if prms else None,
                    "kl_per_weight2": kl / wrms**2 if wrms else None,
                }
            )
        L += [
            "| layers | weights G | KL +- sem | top-1 | "
            + " | ".join(f"bpw {tt}" for tt in targets)
            + " | INT2 % | predicted rel. err (mean / rms) | weight RMS err"
            + " | KL / predicted rms^2 | KL / weight RMS^2 |",
            "| " + " | ".join(["---"] * (10 + len(targets))) + " |",
        ]
        for r in rows:
            L.append(
                f"| {r['group']} | {_f(r['weights_g'], '.2f')} | {r['kl']:.4f} +- {_f(r['sem'])} "
                f"| {_f(r['top1'], '.3f')} | "
                + " | ".join(_f(r["bpw"][tt], ".3f") for tt in targets)
                + f" | {_f(None if r['int2'] is None else 100 * r['int2'], '.1f')} "
                f"| {_f(r['pred_err'], '.4f')} / {_f(r['pred_err_rms'], '.4f')} "
                f"| {_f(r['weight_rms'], '.3f')} "
                f"| {_f(r['kl_per_pred2'], '.1f')} | {_f(r['kl_per_weight2'], '.1f')} |"
            )
        kls = [r["kl"] for r in rows]
        corr = {
            "bits per weight": spearman(kls, [r["bpw"][t] or 0.0 for r in rows]),
            "INT2 share": spearman(kls, [r["int2"] or 0.0 for r in rows]),
            "predicted relative error": spearman(
                kls, [r["pred_err"] or 0.0 for r in rows]
            ),
            "weight RMS error": spearman(kls, [r["weight_rms"] or 0.0 for r in rows]),
        }
        spread = []
        for key, what in (
            ("kl_per_pred2", "predicted error"),
            ("kl_per_weight2", "weight error"),
        ):
            v = [r[key] for r in rows if r[key]]
            if len(v) > 1:
                spread.append(
                    f"{min(v):.1f} to {max(v):.1f} (x{max(v) / min(v):.1f}) per squared {what}"
                )
        L += [
            "",
            "Spearman rank correlation of the groups' KL with: "
            + ", ".join(f"{k} {_f(v, '+.2f')}" for k, v in corr.items())
            + f" ({len(rows)} groups).",
            "KL per squared error, "
            + "; ".join(spread)
            + ": the same value in every group"
            " would mean that the error alone sets the KL.",
        ]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--runs", type=Path, required=True, help="directory of the bpwT outputs"
    )
    ap.add_argument(
        "--evals", type=Path, required=True, help="directory of the bpwT_SET.json"
    )
    ap.add_argument("--targets", nargs="+", default=["3.0", "3.5", "4.0"])
    ap.add_argument(
        "--stock", type=Path, default=None, help="the stock directory (its file sizes)"
    )
    ap.add_argument(
        "--json", type=Path, default=None, help="also write the collected numbers"
    )
    a = ap.parse_args(argv)
    data = collect(a.runs, a.evals, a.targets, a.stock)
    if not any(e.get("experts") for e in data["targets"].values()):
        print(
            f"{a.runs}: no hi2_report.json under bpw{{{','.join(a.targets)}}}",
            file=sys.stderr,
        )
        return 1
    if a.json is not None:
        slim = {
            "stock": data["stock"],
            "targets": {
                t: {
                    "experts": e.get("experts"),
                    "total_gib": e.get("total_gib"),
                    "layers": {
                        str(i): {**v, "bpw": v["bits"] / v["weights"]}
                        for i, v in e.get("layers", {}).items()
                    },
                    "load": {str(i): v for i, v in e.get("load", {}).items()},
                    "evals": {
                        s: {
                            "student": ev.get("student"),
                            "groups": {
                                k: g.get("kl_mean")
                                for k, g in (ev.get("groups") or {}).items()
                            },
                        }
                        for s, ev in e["evals"].items()
                    },
                }
                for t, e in data["targets"].items()
            },
        }
        a.json.write_text(json.dumps(slim, indent=1) + "\n")
    print(render(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
