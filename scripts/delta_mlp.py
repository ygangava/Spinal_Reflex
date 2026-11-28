
"""Sweep the delta MLP's event threshold k on the validation seed.

  python -m scripts.delta_mlp
"""
import numpy as np

from scripts.mlp import parallel, development_cases

MODEL_PATH = "models/mlp_history.npz"
SNN_PATH = "models/snn.npz"
KS = [None, 0.001, 0.003, 0.01, 0.03, 0.1, "SNN"]   # None = periodic MLP; "SNN" = trained SNN (its k = 0.1, fixed)


def run_one(job):
    # imports inside: each worker process starts fresh
    from osc import run_episode, DT
    from mlp_reflex import MLPModel, MLPReflex
    from delta_mlp_reflex import DeltaMLPReflex
    from snn import SNNModel, SNNReflex

    kind, p, k = job
    # load model; pick MLPReflex if k is None, the SNN if "SNN", else DeltaMLPReflex(model, k)
    if k == "SNN":
        r = SNNReflex(SNNModel.load(SNN_PATH))
    else:
        model = MLPModel.load(MODEL_PATH)
        r = MLPReflex(model, DT) if k is None else DeltaMLPReflex(model, k)
    
    # Same settings as the MLP evaluation in scripts/mlp.py: +1 tick output delay, run 600 ms past the push
    log = run_episode(p, t_end=(p.end_tick + round(0.6 / DT)) * DT, reflex=r, reflex_delay=1)

    # Hand error in mm, from push onset on (before onset it's ~0 and only dilutes the numbers)
    e = 1000 * np.linalg.norm(log["e_p"], axis=1)[p.onset_tick:]
    
    # MACs on every tick. Periodic does the full 28x64 + 64x7 = 2240 every tick.
    # SNN: lazy count = adds only (64 per input event count + 7 per hidden spike); eager adds 64 leak updates/tick.
    n = len(log["e_p"])
    ops = np.full(n, 2240) if k is None else np.array(r.ops_lazy if k == "SNN" else r.ops)

    # Split by phase (the script may use push timing, the reflex may not).
    # Skip tick 0: the delta's one-time setup isn't a per-tick cost.
    calm = ops[1:p.onset_tick].mean()
    push = ops[p.onset_tick:p.end_tick].mean()
    after = ops[p.end_tick:].mean()
    return {"kind": kind, "k": k, "peak_mm": e.max(), "auc_mm_s": e.sum() * DT,
            "calm": calm, "push": push, "after": after}

def main():
    from osc import DT
    cases = development_cases([100], DT)        # validation seed, not test
    jobs = [(kind, p, k) for _, kind, p in cases for k in KS]
    rows = parallel(run_one, jobs, 4)

    print(f"{'kind':10} {'k':>6} {'peak mm':>8} {'auc mm*s':>9} {'calm':>6} {'push':>6} {'after':>6}"
          "  (ops/tick: MLPs = MACs, SNN = adds, lazy leak)")
    for kind in ("impact", "sustained"):
        for k in KS:
            rr = [r for r in rows if r["kind"] == kind and r["k"] == k]
            mean = lambda key: np.mean([r[key] for r in rr])
            print(f"{kind:10} {'MLP' if k is None else k:>6} {mean('peak_mm'):8.3f} {mean('auc_mm_s'):9.4f} "
                  f"{mean('calm'):6.0f} {mean('push'):6.0f} {mean('after'):6.0f}")
    print("observer: ~112 MACs/tick, every tick (docs/OBSERVER.md)")
    print("SNN eager leak: + 64 adds/shifts every tick on top of the SNN row")



if __name__ == "__main__":
    main()