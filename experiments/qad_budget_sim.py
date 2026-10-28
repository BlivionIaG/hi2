# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the hi2 project
"""Price a faster QAD recipe by replaying a measured run through the step budget and the
early stop.

usage: qad_budget_sim.py <quant.log> [--preset fast|default|thorough]
                          [--budget F] [--min-gain G] [--min-steps N] [--eval-every E]
                          [--steps N] [--layers N] [--json]

The QAD phase of a run spends ``steps`` optimizer steps on every layer (``hi2/qad.py``:
``StepBudget`` spreads ``budget x steps x layers`` steps over the layers, weighted by the
gain the previous layers showed; ``train_layer`` stops a layer early when one evaluation
interval improves the held-out nMSE by less than ``min_gain``). This script takes the log
of a run that already happened, replays its measured per-layer validation curves through
those two rules, and reports what a different recipe would have cost:

* the optimizer steps it would have spent (and the optimizer time, from the measured
  per-step times of the run);
* the held-out nMSE it would have left on the table, as a fraction of the total held-out
  nMSE of all layers (the layer-wise error of the model, a proxy that ignores the
  compounding of the errors from layer to layer).

The validation curves are the measured ones of the run (one point per evaluation), and the
curve of a layer that would have run fewer steps is interpolated from them. The curves
also depend on the cosine schedule, which a shorter allotment makes steeper, so the
simulated loss at a given step is slightly pessimistic (the real one would be a little
lower): the nMSE given up is an upper bound.

Pure python: no torch, so it runs anywhere the log does.
"""

import argparse
import json
import re
import sys

# The presets of hi2/qad.py (PRESETS), with the QadConfig defaults they do not override.
PRESETS = {
    "fast": dict(steps=300, budget=0.5, min_gain=0.01, min_steps=75, eval_every=50),
    "default": dict(steps=300, budget=1.0, min_gain=0.0, min_steps=0, eval_every=0),
    "thorough": dict(steps=600, budget=1.0, min_gain=0.0, min_steps=0, eval_every=0),
}
DEFAULTS = dict(steps=300, budget=1.0, min_gain=0.0, min_steps=0, eval_every=0, evals=4)

RE_LAYERS = re.compile(r"^QAD: (\d+) layers \((\d+) trained\)")
RE_STEP = re.compile(r"^  L(\d+) step (\d+)/(\d+) train (\S+) val (\S+)\s*$")
RE_SUMMARY = re.compile(
    r"^QAD layer (\d+)/(\d+): (\d+) linears, held-out nMSE (\S+) -> (\S+) "
    r"\((-?[\d.]+)%[^)]*\), (\d+)/(\d+) steps, (\d+) s"
)


class Budget:
    """The step budget of hi2/qad.py:StepBudget (the same arithmetic)."""

    def __init__(self, steps: int, budget: float, min_steps: int, n_layers: int):
        self.cap = steps
        self.floor = min(max(min_steps, 1), steps)
        self.full = budget >= 1
        self.n = n_layers
        self.total = steps * n_layers if self.full else round(budget * steps * n_layers)
        self.left, self.spent = n_layers, 0
        self.gains: list[float] = []

    def allot(self) -> int:
        if self.full:
            return self.cap
        base = (self.total - self.spent) / max(1, self.left)
        w = 2.0
        if self.gains:
            mean = sum(self.gains) / len(self.gains)
            w = min(2.0, max(0.5, self.gains[-1] / mean)) if mean > 0 else 0.5
        return int(max(self.floor, min(self.cap, round(base * w))))

    def record(self, steps: int, gain: float) -> None:
        self.left -= 1
        self.spent += steps
        self.gains.append(max(0.0, gain))


def parse(path: str):
    """The layers of a QAD log: index, linears, the validation curve and the time."""
    layers: dict[int, dict] = {}
    n_layers = n_trained = 0
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = RE_LAYERS.match(line)
            if m:
                n_layers, n_trained = int(m.group(1)), int(m.group(2))
                continue
            m = RE_STEP.match(line)
            if m:
                idx, step, _tot = int(m.group(1)), int(m.group(2)), int(m.group(3))
                layers.setdefault(idx, {"curve": []})["curve"].append(
                    (step, float(m.group(5)))
                )
                continue
            m = RE_SUMMARY.match(line)
            if m:
                idx = int(m.group(1))
                d = layers.setdefault(idx, {"curve": []})
                d["linears"] = int(m.group(3))
                d["before"] = float(m.group(4))
                d["after"] = float(m.group(5))
                d["gain"] = float(m.group(6)) / 100.0
                d["steps"] = int(m.group(7))
                d["seconds"] = float(m.group(9))
                d["summary"] = True
    out = []
    for idx in sorted(layers):
        d = layers[idx]
        if not d.get("summary"):
            continue  # the run was still going when the log was cut
        d["curve"].sort()
        out.append(d)
    return n_layers, n_trained, out


def val_at(curve, before: float, step: int) -> float:
    """The held-out nMSE at ``step``, linearly interpolated from the measured curve."""
    pts = [(0, before)] + list(curve)
    if step <= pts[0][0]:
        return pts[0][1]
    for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
        if step <= s1:
            if s1 == s0:
                return v1
            return v0 + (v1 - v0) * (step - s0) / (s1 - s0)
    return pts[-1][1]


def simulate(layers, cfg):
    """Replay the layers through the budget and the early stop."""
    budget = Budget(cfg["steps"], cfg["budget"], cfg["min_steps"], len(layers))
    rows = []
    for d in layers:
        allot = budget.allot()
        every = cfg["eval_every"] or max(1, allot // max(1, cfg["evals"]))
        before = d["before"]
        # the trace of hi2/qad.py:train_layer: (step, train loss, val loss)
        trace = [(0, before)]
        stopped_at = None
        for step in range(1, allot + 1):
            if step % every == 0 or step == allot:
                vl = val_at(d["curve"], before, step)
                prev = trace[-1][1]
                trace.append((step, vl))
                if (
                    cfg["min_gain"] > 0
                    and step >= cfg["min_steps"]
                    and step < allot
                    and prev - vl < cfg["min_gain"] * prev
                ):
                    stopped_at = step
                    break
        spent = stopped_at if stopped_at is not None else allot
        final = min(v for _, v in trace)  # keep_best keeps the best evaluation
        gain = max(0.0, 1.0 - final / before)
        budget.record(spent, gain)
        rows.append(
            dict(
                index=len(rows),
                allot=allot,
                spent=spent,
                stopped=stopped_at is not None,
                gain=gain,
                val_final=final,
                val_full=d["after"],
                per_step=d["seconds"] / max(1, d["steps"]),
                linears=d["linears"],
            )
        )
    return budget, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--min-gain", type=float, default=None)
    ap.add_argument("--min-steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--layers", type=int, default=0, help="only replay the first N layers")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    cfg = dict(DEFAULTS)
    if args.preset:
        cfg.update(PRESETS[args.preset])
    for key in ("budget", "min_gain", "min_steps", "eval_every", "steps"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    n_layers, n_trained, layers = parse(args.log)
    if args.layers:
        layers = layers[: args.layers]
    if not layers:
        print(f"no completed layers in {args.log}", file=sys.stderr)
        return 1

    budget, rows = simulate(layers, cfg)

    nominal_steps = cfg["steps"] * len(rows)
    spent_steps = budget.spent
    # the optimizer time, from the measured per-step times of the run: the measured layer
    # times are steps x per-step, so the baseline is the measured layer phase and the
    # simulation scales every layer by the steps it would have run
    sec_base = sum(d["seconds"] for d in layers)
    sec_sim = sum(r["spent"] * r["per_step"] for r in rows)
    before_sum = sum(d["before"] for d in layers)
    full_sum = sum(r["val_full"] for r in rows)
    sim_sum = sum(r["val_final"] for r in rows)
    given_up = sim_sum - full_sum
    n_stopped = sum(1 for r in rows if r["stopped"])

    out = dict(
        layers=len(rows),
        trained_layers=n_trained,
        recipe=cfg,
        nominal_steps=nominal_steps,
        spent_steps=spent_steps,
        spent_fraction=spent_steps / max(1, nominal_steps),
        stopped_layers=n_stopped,
        optimizer_seconds_base=sec_base,
        optimizer_seconds_sim=sec_sim,
        nMSE_before=before_sum,
        nMSE_full=full_sum,
        nMSE_sim=sim_sum,
        nMSE_given_up=given_up,
        nMSE_given_up_fraction=given_up / max(1.0, before_sum),
    )
    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print(f"layers replayed: {len(rows)} of {n_trained} trained")
    print(
        f"recipe: steps={cfg['steps']} budget={cfg['budget']:g} min_gain={cfg['min_gain']:g} "
        f"min_steps={cfg['min_steps']} eval_every={cfg['eval_every'] or 'auto'}"
    )
    print(
        f"optimizer steps: {spent_steps} of {nominal_steps} "
        f"({100 * out['spent_fraction']:.0f} %), {n_stopped} layers stopped early"
    )
    print(
        f"optimizer time: {sec_base / 3600:.2f} h as measured -> {sec_sim / 3600:.2f} h "
        f"(the fixed per-layer cost is not in the model)"
    )
    print(
        f"held-out nMSE over the layers: {full_sum:.4e} as measured -> {sim_sum:.4e} "
        f"({given_up:+.2e}, {100 * given_up / max(1.0, before_sum):+.1f} % of the total "
        f"held-out nMSE)"
    )
    print()
    print("| layer | allotment | spent | early stop | gain | nMSE given up |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r, d in zip(rows, layers):
        up = r["val_final"] - r["val_full"]
        print(
            f"| {r['index']} | {r['allot']} | {r['spent']} | "
            f"{'yes' if r['stopped'] else 'no'} | {100 * r['gain']:.1f} % | {up:+.2e} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
