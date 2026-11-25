"""Phase B check: random-weight SNNReflex on one sustained push. No training.

  python -m scripts.snn_check

Checks plumbing and op counting only. Random weights mean the torque is meaningless.
"""
import numpy as np

from osc import run_episode, DT
from disturbance import sustained_pushes
from mlp_reflex import MLPModel
from snn import SNNModel, SNNReflex


def random_model(seed=0, k=0.01, n=4):
    # Borrow the MLP's feature and torque scales so θ = k · x_scale is the same as the delta MLP's.
    mlp = MLPModel.load("models/mlp_history.npz")
    rng = np.random.default_rng(seed)
    return SNNModel(
        w_in=rng.normal(0, 0.3, (56, 64)),
        w_out=rng.normal(0, 0.01, (64, 7)),   # small: random torques stay gentle
        beta=1 - 2.0 ** -n, k=k,
        x_scale=mlp.x_scale, y_scale=mlp.y_scale, torque_limit=mlp.torque_limit,
        metadata={"dt": DT, "features": "history"},
    )


def peak_mm(log, p):
    # Hand error from onset on (before onset it's ~0), same as scripts/delta_mlp.py
    return 1000 * np.linalg.norm(log["e_p"], axis=1)[p.onset_tick:].max()


def main():
    p = sustained_pushes(0, DT)[0]
    t_end = (p.end_tick + round(0.6 / DT)) * DT

    base = run_episode(p, t_end=t_end)
    r = SNNReflex(random_model())
    log = run_episode(p, t_end=t_end, reflex=r, reflex_delay=1)

    eager, lazy = np.array(r.ops), np.array(r.ops_lazy)
    calm, push = slice(1, p.onset_tick), slice(p.onset_tick, p.end_tick)
    print(f"push: onset tick {p.onset_tick}, {p.dur_ticks} ticks long")
    print(f"ops/tick   calm: lazy {lazy[calm].mean():7.1f}  eager {eager[calm].mean():7.1f}")
    print(f"           push: lazy {lazy[push].mean():7.1f}  eager {eager[push].mean():7.1f}")
    print(f"peak hand error: OSC {peak_mm(base, p):.2f} mm   SNN (random) {peak_mm(log, p):.2f} mm")
    print(f"max |reflex torque| per joint (N·m): {np.round(np.abs(log['tau_res']).max(axis=0), 3)}")
    print(f"limit                                : {r.model.torque_limit}")


if __name__ == "__main__":
    main()
