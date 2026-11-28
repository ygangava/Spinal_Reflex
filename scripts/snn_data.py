"""Phase C: sequence training data for the SNN.

  python -m scripts.snn_data            # train (seeds 0,1,2) + val (seed 100), OSC-only + oracle
  python -m scripts.snn_data --summary  # just print shapes and event counts of saved files
  python -m scripts.snn_data --dagger r1  # DAgger round: current models/snn.npz IN the loop -> snn_{train,val}_r1.npz

Saves FEATURES, not events: the encoder runs at train time (`encode` below), so k and
sensor noise can change without rerunning the sim. Unlike the MLP's rows, each episode is
kept as one contiguous window (onset - 50 ... end + 300 ticks), because the SNN has state.
"""
import argparse
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np

from encoder import SendOnDelta
from scripts.mlp import development_cases, parallel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/snn"                  # /runs is gitignored: data files never get committed
MLP_PATH = ROOT / "models/mlp_history.npz"
SNN_PATH = ROOT / "models/snn.npz"         # the SNN that drives the arm in a DAgger round
PRE, POST = 50, 300                     # window = [onset - PRE, end + POST)
SPLITS = {"train": (0, 1, 2), "val": (100,)}


def collect_episode(job):
    # Imports inside: each worker process starts fresh ("spawn" pool).
    from osc import run_episode, DT
    from reflex import OracleReflex
    from mlp_reflex import features_from_log
    seed, kind, p, controller = job
    kw = {"reflex": OracleReflex(), "reflex_delay": 5} if controller == "oracle" else {}
    if controller == "snn":
        # DAgger: the student drives, so the data shows the states ITS mistakes lead to (e.g. not letting go
        # after release). The label is still the teacher -JᵀF, which is 0 once the push is over.
        from snn import SNNModel, SNNReflex
        kw = {"reflex": SNNReflex(SNNModel.load(SNN_PATH)), "reflex_delay": 1}
    log = run_episode(p, t_end=(p.end_tick + POST + 10) * DT, **kw)

    x = features_from_log(log, "history", DT)                    # (T, 28), identical to online features
    y = np.vstack([np.zeros((1, 7)), log["tau_teacher"][:-1]])   # 1-tick-shifted label, same as the MLP
    lo, hi = p.onset_tick - PRE, p.end_tick + POST

    # The encoder's first sample sets its ref. Online, that's tick 0, not the window start.
    # Before the push the hold is perfectly still, so x[lo] == x[0] and starting the
    # encoder from x[0] reproduces the online events exactly. Check instead of assuming.
    drift = np.abs(x[1:lo] - x[0]).max()
    assert drift < 1e-9, f"features moved before the window ({drift:.2e}); encode from tick 0 instead"

    return dict(x0=x[0], x=x[lo:hi], y=y[lo:hi], seed=seed, kind=kind, push_id=p.push_id,
                controller=controller, onset=PRE, end=PRE + p.dur_ticks)


def save(name, episodes):
    # Ragged sequences stored flat: episode i is rows start[i] : start[i] + length[i].
    lengths = np.array([len(e["x"]) for e in episodes])
    np.savez_compressed(
        OUT / f"snn_{name}.npz",
        x=np.concatenate([e["x"] for e in episodes]).astype(np.float32),   # (sum T, 28)
        y=np.concatenate([e["y"] for e in episodes]).astype(np.float32),   # (sum T, 7) N·m
        x0=np.stack([e["x0"] for e in episodes]).astype(np.float32),       # (N, 28) encoder start
        start=np.r_[0, np.cumsum(lengths)[:-1]], length=lengths,
        onset=np.array([e["onset"] for e in episodes]),                    # push start, within window
        end=np.array([e["end"] for e in episodes]),                        # push end, within window
        seed=np.array([e["seed"] for e in episodes]),
        push_id=np.array([e["push_id"] for e in episodes]),
        kind=np.array([e["kind"] for e in episodes]),
        controller=np.array([e["controller"] for e in episodes]))


def load(name):
    """-> list of episode dicts (x0, x, y, onset, end, kind, controller)."""
    with np.load(OUT / f"snn_{name}.npz", allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    return [dict(x0=d["x0"][i], x=d["x"][s:s + n], y=d["y"][s:s + n], onset=int(d["onset"][i]),
                 end=int(d["end"][i]), kind=str(d["kind"][i]), controller=str(d["controller"][i]))
            for i, (s, n) in enumerate(zip(d["start"], d["length"]))]


def encode(x0, x, theta):
    """Features (T, 28) -> event counts (T, 56) = [on, off], exactly as SNNReflex sees them online."""
    enc = SendOnDelta(theta)
    enc.step(x0)                          # first sample only sets ref (no events), like tick 0 online
    out = np.empty((len(x), 2 * x.shape[1]), np.int32)
    for t, xt in enumerate(x):
        on, off = enc.step(xt)
        out[t] = np.r_[on, off]
    return out


def summary():
    from mlp_reflex import MLPModel
    x_scale = MLPModel.load(MLP_PATH).x_scale
    for name in SPLITS:
        eps = load(name)
        lengths = [len(e["x"]) for e in eps]
        print(f"{name}: {len(eps)} episodes, {sum(lengths)} ticks, window {min(lengths)}–{max(lengths)} ticks; "
              f"x (T, {eps[0]['x'].shape[1]}), y (T, {eps[0]['y'].shape[1]})")
        for k in (0.01, 0.1):
            ev = [encode(e["x0"], e["x"], k * x_scale)[e["onset"]:e["end"]].sum(axis=1).mean() for e in eps]
            print(f"   k = {k:<5} mean events/tick during push: {np.mean(ev):6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--dagger", metavar="TAG", help="collect with the current SNN in the loop, save as snn_<split>_TAG")
    args = ap.parse_args()
    if args.dagger:
        from osc import DT
        for name, seeds in SPLITS.items():
            jobs = [(seed, kind, p, "snn") for seed, kind, p in development_cases(seeds, DT)]
            save(f"{name}_{args.dagger}", parallel(collect_episode, jobs, args.workers))
            print(f"saved {OUT / f'snn_{name}_{args.dagger}.npz'} ({len(jobs)} episodes, SNN = {SNN_PATH.name})")
        return
    if not args.summary:
        from osc import DT
        OUT.mkdir(parents=True, exist_ok=True)
        for name, seeds in SPLITS.items():
            jobs = [(seed, kind, p, c) for seed, kind, p in development_cases(seeds, DT)
                    for c in ("none", "oracle")]
            save(name, parallel(collect_episode, jobs, args.workers))
    summary()


if __name__ == "__main__":
    main()
