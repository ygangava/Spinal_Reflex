# Baseline matrix, no learning: OSC, high-KP OSC, and the oracle at 0/5/25 ms on impacts, sustained pushes
# and slow presses. Writes runs/baseline/results.csv (one row per episode) and renders/baseline.png.
# Run from the project root: python -m scripts.baseline        (add --plot-only to re-plot from the CSV)
import csv
import os
import sys
from multiprocessing import Pool

# One BLAS thread per worker: our matrices are 7x7, so multithreading only adds overhead,
# and 10 workers x N threads each fight over the cores. Must be set before numpy is imported
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np

SEED = 0
OUT_CSV = "runs/baseline/results.csv"
OUT_PNG = "renders/baseline.png"

# Controller name -> run_episode kwargs. Oracle delays are in ms, converted to ticks in the worker
CONTROLLERS = {
    "OSC": {},
    "High-KP OSC": {"gains": "high"},
    "Oracle 0 ms": {"oracle_ms": 0},
    "Oracle 5 ms": {"oracle_ms": 5},
    "Oracle 25 ms": {"oracle_ms": 25},
}


def episode(job):
    # One (push set, push, controller) episode -> metrics. Imports here so each worker loads its own MuJoCo model
    from osc import run_episode, DT, TAU_MAX, HIGH_KP_GAINS
    from reflex import OracleReflex
    set_name, p, ctrl = job
    spec = CONTROLLERS[ctrl]
    kw = {}
    if spec.get("gains") == "high":
        kw["gains"] = HIGH_KP_GAINS
    if "oracle_ms" in spec:
        kw["reflex"], kw["reflex_delay"] = OracleReflex(), round(spec["oracle_ms"] / 1000 / DT)

    # Run 1 s past the end of the push so recovery is captured
    log = run_episode(p, t_end=(p.end_tick + round(1.0 / DT)) * DT, **kw)
    e = 1000 * np.linalg.norm(log["e_p"], axis=1)[p.onset_tick:]          # mm, from onset
    row = {"set": set_name, "controller": ctrl, "push_id": p.push_id,
           "dir_deg": round(float(np.degrees(np.arctan2(p.direction[1], p.direction[0]))) % 360),
           "impulse": p.impulse, "peak_force": p.peak_force,
           "dur_ms": round(p.dur_ticks * DT * 1000), "rise_ms": round(p.rise_ticks * DT * 1000),
           "peak_mm": e.max(), "auc_mm_s": e.sum() * DT,
           "sat_frac": float((np.abs(log["tau_cmd"]) > TAU_MAX).any(axis=1).mean())}

    # Presses: steady deflection = mean over the last quarter of the hold (force constant, arm settled)
    if set_name == "press":
        h1 = p.dur_ticks - p.rise_ticks
        h0 = h1 - (p.dur_ticks - 2 * p.rise_ticks) // 4
        row["hold_mm"] = e[h0:h1].mean()
    return row


def compute():
    from osc import DT
    from disturbance import grid_pushes, sustained_pushes, press_pushes
    sets = {"impact": grid_pushes(SEED, DT), "sustained": sustained_pushes(SEED, DT), "press": press_pushes(SEED, DT)}
    jobs = [(s, p, c) for s, ps in sets.items() for p in ps for c in CONTROLLERS]
    with Pool() as pool:
        rows = pool.map(episode, jobs, chunksize=4)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV}: {len(rows)} episodes")


# ---- Plot ----
# Colors: OSC and high-KP are categorical (orange, aqua); the oracle delays are one ordinal blue ramp,
# dark = fastest. Validated with the dataviz palette script (categorical all-pairs; ordinal ramp)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {
    "OSC":          dict(color="#eb6834", marker="o"),
    "High-KP OSC":  dict(color="#1baf7a", marker="s"),
    "Oracle 0 ms":  dict(color="#184f95", marker="D"),
    "Oracle 5 ms":  dict(color="#2a78d6", marker="^"),
    "Oracle 25 ms": dict(color="#6da7ec", marker="v"),
}


def load():
    with open(OUT_CSV) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def means(rows, set_name, xkey, ykey, ctrl):
    # Mean of ykey per x value (averaged over directions and the set's other factors)
    xs = sorted({r[xkey] for r in rows if r["set"] == set_name})
    ys = [np.mean([r[ykey] for r in rows if r["set"] == set_name and r["controller"] == ctrl and r[xkey] == x])
          for x in xs]
    return np.array(xs), np.array(ys)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = load()

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), facecolor=SURFACE)
    panels = [
        ("impact", "impulse", "peak_mm", "Short impacts (20 & 80 ms)", "Push impulse (N·s)", "Peak hand error (mm)"),
        ("sustained", "dur_ms", "peak_mm", "Sustained pushes (10 ms rise)", "Push duration (ms)", "Peak hand error (mm)"),
        ("press", "rise_ms", "hold_mm", "Slow presses (30 N): does the arm yield?", "Force rise time (ms, log)",
         "Deflection during hold (mm)"),
    ]
    for ax, (s, xk, yk, title, xl, yl) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for ctrl, st in STYLE.items():
            x, y = means(rows, s, xk, yk, ctrl)
            ax.plot(x, y, lw=2, ms=6, mec=SURFACE, mew=1.5, label=ctrl, **st)
            # Direct label at the line's right end (relief for the low-contrast aqua / light blue).
            # On presses the oracle lines sit on top of each other at 0, so they share one label
            text = ctrl
            if s == "press" and ctrl.startswith("Oracle"):
                if ctrl != "Oracle 0 ms":
                    continue
                text = "Oracle (all delays)"
            nudge = {"Oracle 0 ms": -4, "Oracle 5 ms": 4}.get(ctrl, 0) if s == "sustained" else 0
            ax.annotate(text, (x[-1], y[-1]), xytext=(6, nudge), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.25)
        if s == "press":
            ax.set_xscale("log")
            ax.set_xlim(12, 6000)

    # Press panel: what each end of the sweep should look like for the reflex we want
    ax = axes[2]
    ax.text(0.03, 0.62, "higher = yields (compliant)\n0 = rigid", transform=ax.transAxes, va="top",
            fontsize=8, color=INK2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.005, f"Hand pushes, home pose, DT 1 ms, 87/12 N·m + 1000 N·m/s limits. Mean over 4 directions "
             f"(±10° jitter) and the set's other factors; seed {SEED}. Oracle = perfect −JᵀF, delayed.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, facecolor=SURFACE)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        compute()
    plot()
