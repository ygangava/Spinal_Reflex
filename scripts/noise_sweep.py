"""Sensor-noise sweep: every reflex sees noisy q / dq, on the validation seed.

  python -m scripts.noise_sweep

dq noise is the knob. q noise is tied to it as if dq were a finite difference of q:
dq_std = sqrt(2) * q_std / DT, so q_std = dq_std * DT / sqrt(2).
"""
import csv
from pathlib import Path

import numpy as np

from scripts.mlp import parallel, development_cases

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/mlp_history.npz"
SNN_PATH = ROOT / "models/snn.npz"
OUT = ROOT / "results/noise"
DQ_STDS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]   # rad/s
CONTROLLERS = ["OSC", "Observer K=100", "Observer K=1000", "MLP", "Delta MLP k=0.01", "Delta MLP k=0.1",
               "Delta MLP k=0.01 floor", "SNN", "SNN floor"]
FLOOR_SIGMAS = 6   # thresholds >= 6 sigma of each feature's noise: calm noise almost never fires


def make_reflex(name, q_std, dq_std):
    from osc import DT
    from reflex import NoReflex
    from mlp_reflex import MLPModel, MLPReflex
    from delta_mlp_reflex import DeltaMLPReflex, feature_noise
    from observer_reflex import MomentumObserver
    from snn import SNNModel, SNNReflex
    if name == "OSC":
        return NoReflex()
    if name.startswith("Observer"):
        return MomentumObserver(float(name.split("=")[1]))
    if name.startswith("SNN"):
        # Same floor rule as the delta MLP; the SNN's k (0.1) was fixed at training
        floor = FLOOR_SIGMAS * feature_noise(q_std, dq_std, DT) if name.endswith("floor") else None
        return SNNReflex(SNNModel.load(SNN_PATH), floor)
    model = MLPModel.load(MODEL_PATH)
    if name == "MLP":
        return MLPReflex(model, DT)
    k = float(name.split("=")[1].split()[0])
    floor = FLOOR_SIGMAS * feature_noise(q_std, dq_std, DT) if name.endswith("floor") else None
    return DeltaMLPReflex(model, k, floor)


def read_rows(path, text_keys):
    """Saved CSV rows back as dicts, with numbers as floats (so --only can reuse other controllers' rows)."""
    with path.open() as f:
        return [{k: (v if k in text_keys else float(v)) for k, v in row.items()} for row in csv.DictReader(f)]


def parse_only(controllers):
    # --only NAME ...: rerun just these controllers and reuse the saved CSV rows for the rest.
    # Only valid while the other controllers' code is unchanged since that CSV was written.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=controllers, help="rerun only these; reuse saved rows for the rest")
    return ap.parse_args().only


def run_one(job):
    from osc import run_episode, DT
    from reflex import NoisyObs
    kind, p, name, dq_std, seed = job
    q_std = dq_std * DT / np.sqrt(2)
    inner = make_reflex(name, q_std, dq_std)
    r = NoisyObs(inner, q_std=q_std, dq_std=dq_std, seed=seed)
    log = run_episode(p, t_end=(p.end_tick + round(0.6 / DT)) * DT, reflex=r, reflex_delay=1)
    e = 1000 * np.linalg.norm(log["e_p"], axis=1)

    # Calm = before the push, skipping the first 200 ms so start-up transients don't count
    calm = slice(round(0.2 / DT), p.onset_tick)
    push = slice(p.onset_tick, p.end_tick)
    # Ops/tick: only the delta MLP and the SNN vary; the others cost the same every tick.
    # SNN: the lazy count (adds only, no leak), comparable to the delta MLP counting work only on event ticks.
    ops = np.array(getattr(inner, "ops_lazy", getattr(inner, "ops", np.full(len(e), np.nan))))
    return {"kind": kind, "controller": name, "dq_std": dq_std,
            "peak_mm": e[p.onset_tick:].max(), "auc_mm_s": e[p.onset_tick:].sum() * DT,
            "calm_jitter_mm": e[calm].mean(), "calm_ops": ops[calm].mean(), "push_ops": ops[push].mean()}


def main():
    from osc import DT
    only = parse_only(CONTROLLERS)
    cases = development_cases([100], DT)
    # Noise seed differs per episode but is the same across controllers: paired comparison
    jobs = [(kind, p, name, s, 1000 * i) for i, (_, kind, p) in enumerate(cases)
            for name in (only or CONTROLLERS) for s in DQ_STDS]
    rows = parallel(run_one, jobs, 4)
    if only:
        rows = [r for r in read_rows(OUT / "noise_sweep.csv", ("kind", "controller"))
                if r["controller"] not in only] + rows

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "noise_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for kind in ("sustained", "impact"):
        print(f"\n{kind}: peak mm / calm jitter mm / calm MACs / push MACs")
        print(f"{'controller':18}" + "".join(f"{s:>22g}" for s in DQ_STDS))
        for name in CONTROLLERS:
            cells = []
            for s in DQ_STDS:
                rr = [r for r in rows if r["kind"] == kind and r["controller"] == name and r["dq_std"] == s]
                mean = lambda key: np.mean([r[key] for r in rr])
                ops = "" if np.isnan(mean("calm_ops")) else f" {mean('calm_ops'):.0f}/{mean('push_ops'):.0f}"
                cells.append(f"{mean('peak_mm'):.2f}/{mean('calm_jitter_mm'):.2f}{ops}")
            print(f"{name:18}" + "".join(f"{c:>22}" for c in cells))
    print(f"\nSaved {OUT / 'noise_sweep.csv'}")


if __name__ == "__main__":
    main()
