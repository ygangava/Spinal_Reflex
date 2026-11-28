"""Model-mismatch sweep: the true physics changes, every controller keeps the nominal model.

  python -m scripts.mismatch_sweep

Two numbers per episode, because a wrong mass does two different things:
  calm_mm      steady error before the push: gravity is mis-compensated, so the arm sags
  push_peak_mm peak of the push-induced deviation, ||e(t) - e(onset)||: recovery quality on top of the sag
"""
import csv
from pathlib import Path

import numpy as np

from scripts.mlp import parallel, development_cases
from scripts.noise_sweep import read_rows, parse_only

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/mlp_history.npz"
SNN_PATH = ROOT / "models/snn.npz"
OUT = ROOT / "results/mismatch"
# (label, plant_mismatch kwargs). Franka's rated payload is 3 kg.
CONDITIONS = [("nominal", {}),
              ("payload 0.5 kg", {"payload_kg": 0.5}),
              ("payload 1 kg", {"payload_kg": 1.0}),
              ("payload 2 kg", {"payload_kg": 2.0}),
              ("links x0.8", {"link_mass_scale": 0.8}),
              ("links x1.2", {"link_mass_scale": 1.2})]
CONTROLLERS = ["OSC", "Observer K=100", "Observer K=1000", "MLP", "Delta MLP k=0.01", "SNN", "Oracle"]


def make_reflex(name):
    from osc import DT
    from reflex import NoReflex, OracleReflex
    from mlp_reflex import MLPModel, MLPReflex
    from delta_mlp_reflex import DeltaMLPReflex
    from observer_reflex import MomentumObserver
    from snn import SNNModel, SNNReflex
    if name == "OSC":
        return NoReflex(), 0
    if name == "SNN":
        return SNNReflex(SNNModel.load(SNN_PATH)), 1   # +1 tick, same as the MLPs
    if name == "Oracle":
        return OracleReflex(), 2   # +1 tick: the oracle sees the force a tick before sensors can (as in scripts/mlp.py)
    if name.startswith("Observer"):
        return MomentumObserver(float(name.split("=")[1])), 1
    model = MLPModel.load(MODEL_PATH)
    if name == "MLP":
        return MLPReflex(model, DT), 1
    return DeltaMLPReflex(model, float(name.split("=")[1])), 1


def run_one(job):
    from osc import run_episode, plant_mismatch, DT
    kind, p, name, label, kw = job
    r, delay = make_reflex(name)
    with plant_mismatch(**kw):
        log = run_episode(p, t_end=(p.end_tick + round(0.6 / DT)) * DT, reflex=r, reflex_delay=delay)
    e_vec = 1000 * log["e_p"]
    e = np.linalg.norm(e_vec, axis=1)

    # Calm = 0.5 s after start (sag has settled) to the push
    calm = slice(round(0.5 / DT), p.onset_tick)
    push = slice(p.onset_tick, p.end_tick)
    deviation = np.linalg.norm(e_vec[p.onset_tick:] - e_vec[p.onset_tick], axis=1)
    # SNN: the lazy count (adds only, no leak)
    ops = np.array(getattr(r, "ops_lazy", getattr(r, "ops", np.full(len(e), np.nan))))
    return {"kind": kind, "controller": name, "condition": label,
            "calm_mm": e[calm].mean(), "push_peak_mm": deviation.max(), "peak_mm": e[p.onset_tick:].max(),
            "calm_ops": ops[calm].mean(), "push_ops": ops[push].mean()}


def main():
    from osc import DT
    only = parse_only(CONTROLLERS)
    cases = development_cases([100], DT)
    jobs = [(kind, p, name, label, kw) for _, kind, p in cases
            for name in (only or CONTROLLERS) for label, kw in CONDITIONS]
    rows = parallel(run_one, jobs, 4)
    if only:
        rows = [r for r in read_rows(OUT / "mismatch_sweep.csv", ("kind", "controller", "condition"))
                if r["controller"] not in only] + rows

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "mismatch_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [label for label, _ in CONDITIONS]
    for kind in ("sustained", "impact"):
        print(f"\n{kind}: calm sag mm / push peak mm (delta: + calm/push MACs)")
        print(f"{'controller':18}" + "".join(f"{l:>22}" for l in labels))
        for name in CONTROLLERS:
            cells = []
            for label in labels:
                rr = [r for r in rows if r["kind"] == kind and r["controller"] == name and r["condition"] == label]
                mean = lambda key: np.mean([r[key] for r in rr])
                ops = "" if np.isnan(mean("calm_ops")) else f" {mean('calm_ops'):.0f}/{mean('push_ops'):.0f}"
                cells.append(f"{mean('calm_mm'):.2f}/{mean('push_peak_mm'):.2f}{ops}")
            print(f"{name:18}" + "".join(f"{c:>22}" for c in cells))
    print(f"\nSaved {OUT / 'mismatch_sweep.csv'}")


if __name__ == "__main__":
    main()
