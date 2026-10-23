"""Momentum observer: ablation, K_O sweep, and comparison against OSC / high-KP / MLP / oracle.

  python -m scripts.observer ablate     # step 7: what each known-torque term is worth
  python -m scripts.observer sweep      # step 8: K_O sweep on the validation seed (100)
  python -m scripts.observer compare    # held-out seeds 200/201: same cases + metrics as scripts.mlp evaluate

Timing convention (same as scripts.mlp): learned/observer reflexes see a push one tick late (sensing) and get
`delay` more ticks of output delay; the oracle sees it instantly, so it gets delay + 1. Same force -> torque latency.
"""
import argparse
import csv
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/spinal-reflex-matplotlib")

import numpy as np

from disturbance import Push, sustained_pushes
from scripts.mlp import development_cases, heldout_cases, parallel, MODEL_PATH

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/observer"

# Colors: baselines recessive gray, one hue per reflex family (observer variants share orange, K=100 dashed)
COLORS = {"OSC": "#a3a29b", "High-KP OSC": "#6b6a64", "MLP": "#2a78d6",
          "Observer K=100": "#eb6834", "Observer K=1000": "#eb6834", "Oracle": "#1baf7a"}
DASHED = {"Observer K=100"}


def controller(label, kind, delay=1, k_o=None):
    return {"label": label, "kind": kind, "delay": delay, "k_o": k_o}


def main_controllers(delay=1):
    return [controller("OSC", "osc"), controller("High-KP OSC", "highkp"),
            controller("MLP", "mlp", delay), controller("Observer K=100", "observer", delay, 100),
            controller("Observer K=1000", "observer", delay, 1000),
            controller("Oracle", "oracle", delay + 1)]


def episode_kwargs(c):
    from osc import DT, HIGH_KP_GAINS
    from reflex import OracleReflex
    from mlp_reflex import MLPModel, MLPReflex
    from observer_reflex import MomentumObserver
    if c["kind"] == "highkp":
        return {"gains": HIGH_KP_GAINS}
    if c["kind"] == "mlp":
        return {"reflex": MLPReflex(MLPModel.load(MODEL_PATH), DT), "reflex_delay": c["delay"]}
    if c["kind"] == "observer":
        return {"reflex": MomentumObserver(c["k_o"]), "reflex_delay": c["delay"]}
    if c["kind"] == "oracle":
        return {"reflex": OracleReflex(), "reflex_delay": c["delay"]}
    return {}


def run_case(job):
    # Metrics are defined exactly as in scripts.mlp.evaluate_episode, so numbers are comparable
    from osc import run_episode, DT, TAU_MAX
    seed, kind, p, c, keep_trace = job
    log = run_episode(p, t_end=(p.end_tick + round(1.0 / DT)) * DT, **episode_kwargs(c))
    e = 1000 * np.linalg.norm(log["e_p"], axis=1)
    post = e[p.onset_tick:]
    row = {"seed": seed, "set": kind, "push_id": p.push_id, "controller": c["label"],
           "output_delay": c["delay"] if c["kind"] in ("mlp", "observer", "oracle") else None,
           "peak_mm": float(post.max()), "auc_mm_s": float(post.sum() * DT),
           "saturation_fraction": float((np.abs(log["tau_cmd"]) > TAU_MAX).any(axis=1).mean()),
           "torque_limited_fraction": float((np.abs(log["tau_cmd"] - log["tau_applied"]) > 1e-6).any(axis=1).mean()),
           "final_mm": float(e[-1]), "rise_s": p.rise_ticks * DT}
    if kind == "press":
        end = p.end_tick - p.rise_ticks
        start = end - (p.dur_ticks - 2 * p.rise_ticks) // 4
        row["hold_mm"] = float(e[start:end].mean())
    trace = None
    if keep_trace:
        trace = {"t": (np.arange(len(e)) - p.onset_tick) * DT, "error_mm": e,
                 "est": log["reflex_est"], "teacher": log["tau_teacher"]}
    return row, trace


def mean_by(rows, *keys, metric):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r[metric])
    return {k: float(np.mean(v)) for k, v in groups.items()}


def write_csv(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b5b4ad")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
    ax.set_axisbelow(True)


def plt_module():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                         "axes.labelcolor": "#52514e", "text.color": "#0b0b0b"})
    return plt


def observer_op_counts(n=7):
    # Observer's own arithmetic per tick, given M, b, passive (the dynamics model is shared with the OSC,
    # which already evaluates M and qfrc_bias every tick). A length-n dot = n MACs.
    return {"macs": 2 * n * n + 2 * n,           # p = M dq; Mdot_dq = dM @ dq_prev; dt * (...); k_o * (...)
            "add_sub": n * n + 5 * n + 2 * n,    # M - M_prev; 5 terms into the integrand; p - p0 - I
            "div": n * n,                        # (M - M_prev) / dt (foldable into a constant)
            "neg": n,
            "note": "excludes M(q), qfrc_bias, qfrc_passive evaluation (shared with the OSC)"}


# ---------------------------------------------------------------- step 7: ablation
def ablate(args):
    from osc import run_episode, DT
    from observer_reflex import MomentumObserver
    p = sustained_pushes(0, DT)[13]
    t_end = (p.end_tick + 600) * DT
    after = slice(p.end_tick + 100, None)
    rows = []
    for label, kw in (("full observer", {}), ("no damping", {"damping": False}),
                      ("no Mdot*dq", {"mdot": False}), ("neither", {"damping": False, "mdot": False})):
        watch = run_episode(p, reflex=MomentumObserver(100, **kw), reflex_gain=0.0, t_end=t_end)
        act = run_episode(p, reflex=MomentumObserver(100, **kw), reflex_delay=1, t_end=t_end)
        rows.append({"variant": label,
                     "false_estimate_after_release_nm": float(np.abs(watch["reflex_est"][after]).max()),
                     "peak_mm_gain1": float(1000 * np.linalg.norm(act["e_p"], axis=1)[p.onset_tick:].max())})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ablation.json").write_text(json.dumps({"push": "sustained_pushes(0)[13]", "k_o": 100,
                                                        "rows": rows}, indent=2))
    print("| Variant | False estimate after release (N·m) | Peak error, gain 1 (mm) |\n|---|---|---|")
    for r in rows:
        print(f"| {r['variant']} | {r['false_estimate_after_release_nm']:.3f} | {r['peak_mm_gain1']:.2f} |")


# ---------------------------------------------------------------- step 8: K_O sweep (validation seed)
SWEEP_K = (25, 50, 100, 200, 500, 1000)


def sweep(args):
    from osc import DT
    cases = [c for c in development_cases([100], DT)]
    ctrls = ([controller(f"K={k}", "observer", 1, k) for k in SWEEP_K]
             + [controller("MLP", "mlp", 1), controller("Oracle", "oracle", 2)])
    jobs = [(s, kind, p, c, False) for s, kind, p in cases for c in ctrls]
    print(f"Sweep: {len(jobs)} validation episodes", flush=True)
    rows = [r for r, _ in parallel(run_case, jobs, args.workers)]
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / "sweep.csv")
    peak = mean_by(rows, "set", "controller", metric="peak_mm")
    print("| K_O | lag 1/K_O | Sustained peak (mm) | Impact peak (mm) |\n|---|---|---|---|")
    for k in SWEEP_K:
        print(f"| {k} | {1000 / k:.0f} ms | {peak[('sustained', f'K={k}')]:.2f} | {peak[('impact', f'K={k}')]:.2f} |")
    for ref in ("MLP", "Oracle"):
        print(f"| {ref} | | {peak[('sustained', ref)]:.2f} | {peak[('impact', ref)]:.2f} |")

    plt = plt_module()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, kind, title in zip(axes, ("sustained", "impact"), ("Sustained pushes", "Impacts")):
        ys = [peak[(kind, f"K={k}")] for k in SWEEP_K]
        ax.plot(SWEEP_K, ys, color=COLORS["Observer K=1000"], linewidth=2, marker="o", markersize=6,
                label="Observer")
        for ref, ls in (("MLP", "--"), ("Oracle", ":")):
            ax.axhline(peak[(kind, ref)], color=COLORS[ref], linewidth=1.5, linestyle=ls, label=ref)
        ax.set_xscale("log")
        ax.set_xticks(SWEEP_K, [str(k) for k in SWEEP_K])
        ax.minorticks_off()
        ax.set_xlabel("Observer gain K_O (1/s)   [lag = 1/K_O]")
        ax.set_ylabel("Mean peak hand error (mm)")
        ax.set_ylim(0, max(ys) * 1.1)
        ax.set_title(title, loc="left")
        style(ax)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("K_O sweep | validation seed 100, +1 tick output delay", fontsize=12)
    fig.savefig(args.out / "ko_sweep.png", dpi=160)
    plt.close(fig)
    print(f"Saved {args.out / 'ko_sweep.png'}")


# ---------------------------------------------------------------- held-out comparison
DELAYS = (0, 1, 2, 5, 10)


def compare(args):
    from osc import DT
    seeds = [200, 201]
    cases = heldout_cases(seeds, DT)
    ctrls = main_controllers(1)
    trace_case = next((s, kind, p) for s, kind, p in cases if kind == "sustained")
    jobs = [(s, kind, p, c, (s, kind, p) == trace_case) for s, kind, p in cases for c in ctrls]
    # Latency sweep on held-out sustained pushes: every reflex at the same total latency (delay + 1 ticks)
    sus = [(s, kind, p) for s, kind, p in cases if kind == "sustained"]
    lat_ctrls = [controller(lbl, kind, d + (kind == "oracle"), k)
                 for d in DELAYS for lbl, kind, k in (("MLP", "mlp", None), ("Observer K=100", "observer", 100),
                                                     ("Observer K=1000", "observer", 1000), ("Oracle", "oracle", None))]
    lat_jobs = [(s, kind, p, c, False) for s, kind, p in sus for c in lat_ctrls]
    print(f"Compare: {len(jobs)} held-out episodes + {len(lat_jobs)} latency-sweep episodes", flush=True)
    results = parallel(run_case, jobs, args.workers)
    lat_rows = [r for r, _ in parallel(run_case, lat_jobs, args.workers)]
    rows = [r for r, _ in results]
    traces = {r["controller"]: t for r, t in results if t is not None}

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / "results.csv")
    write_csv(lat_rows, args.out / "latency.csv")
    labels = [c["label"] for c in ctrls]
    summary = {"evaluation_seeds": seeds, "output_delay_ticks": 1, "oracle_delay_ticks": 2,
               "observer_op_counts_per_tick": observer_op_counts(), "mlp_dense_macs_per_tick": 2240,
               "means": {}}
    for kind in ("impact", "sustained", "press", "quiet"):
        summary["means"][kind] = {}
        for lbl in labels:
            rr = [r for r in rows if r["set"] == kind and r["controller"] == lbl]
            vals = {m: float(np.mean([r[m] for r in rr])) for m in
                    ("peak_mm", "auc_mm_s", "saturation_fraction", "torque_limited_fraction", "final_mm")}
            if kind == "press":
                vals["hold_mm"] = float(np.mean([r["hold_mm"] for r in rr]))
            vals["n"] = len(rr)
            summary["means"][kind][lbl] = vals
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    m = summary["means"]
    print("\n| Metric | " + " | ".join(labels) + " |\n|---|" + "---|" * len(labels))
    for kind, metric, name in (("sustained", "peak_mm", "Sustained peak (mm)"), ("sustained", "auc_mm_s", "Sustained AUC (mm·s)"),
                               ("impact", "peak_mm", "Impact peak (mm)"), ("impact", "auc_mm_s", "Impact AUC (mm·s)"),
                               ("press", "hold_mm", "Press hold deflection (mm)"),
                               ("quiet", "peak_mm", "Quiet max error (mm)")):
        print(f"| {name} | " + " | ".join(f"{m[kind][l][metric]:.3g}" for l in labels) + " |")

    plt = plt_module()
    # Figure 1: held-out bars
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    for ax, (kind, metric, title, ylabel) in zip(axes, (
            ("sustained", "peak_mm", "Sustained pushes", "Mean peak hand error (mm)"),
            ("impact", "peak_mm", "Impacts", "Mean peak hand error (mm)"),
            ("press", "hold_mm", "Slow presses (should yield like OSC)", "Mean hold deflection (mm)"))):
        vals = [m[kind][l][metric] for l in labels]
        ax.bar(range(len(labels)), vals, color=[COLORS[l] for l in labels], width=0.62,
               edgecolor="white", linewidth=2, hatch=None)
        for i, (l, v) in enumerate(zip(labels, vals)):
            if l in DASHED:
                ax.patches[i].set_hatch("////")
                ax.patches[i].set_edgecolor("white")
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color="#52514e")
        ax.set_xticks(range(len(labels)), [l.replace(" K=", "\nK=").replace("High-KP ", "High-KP\n")
                                           for l in labels], fontsize=8.5)
        ax.set_ylim(0, max(vals) * 1.15)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        style(ax)
    fig.suptitle("Held-out pushes (seeds 200, 201) | home pose, noise-free | reflexes at 2-tick total latency",
                 fontsize=12)
    fig.savefig(args.out / "comparison.png", dpi=160)
    plt.close(fig)

    # Figure 2: latency sweep
    lat = mean_by(lat_rows, "controller", "output_delay", metric="peak_mm")
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    x = [d + 1 for d in DELAYS]   # total force -> torque latency in ticks (ms)
    for lbl in ("MLP", "Observer K=100", "Observer K=1000", "Oracle"):
        ys = [lat[(lbl, d + (lbl == "Oracle"))] for d in DELAYS]
        # Oracle drawn last, thin and dotted, so the identical Observer K=1000 line stays visible under it
        oracle = lbl == "Oracle"
        ax.plot(x, ys, color=COLORS[lbl], linewidth=1.5 if oracle else 2.5, marker="o",
                markersize=4 if oracle else 8, linestyle=":" if oracle else "--" if lbl in DASHED else "-",
                label=lbl + (" (overlaps K=1000)" if oracle else ""))
    ax.set_xticks(x)
    ax.set_xlim(0.5, 11.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Total latency, push to cancelling torque (ms)")
    ax.set_ylabel("Mean peak hand error (mm)")
    ax.set_title("Held-out sustained pushes vs reflex latency", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)
    fig.savefig(args.out / "latency.png", dpi=160)
    plt.close(fig)

    # Figure 3: press yielding vs rise time
    press = mean_by([r for r in rows if r["set"] == "press"], "controller", "rise_s", metric="hold_mm")
    rises = sorted({k[1] for k in press})
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    for lbl in labels:
        ys = [press[(lbl, r)] for r in rises]
        ax.plot(rises, ys, color=COLORS[lbl], linewidth=2, marker="o", markersize=5,
                linestyle="--" if lbl in DASHED else "-", label=lbl)
    ax.set_xscale("log")
    ax.set_xticks(rises, [f"{r:g}" for r in rises])
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Press rise time (s)   [fast onset → slow, deliberate]")
    ax.set_ylabel("Mean hold deflection (mm)")
    ax.set_title("Slow presses: every reflex is rigid at every rise time", loc="left")
    ax.annotate("MLP ≈ 0.3 mm; both observers and oracle ≈ 0 mm (lines overlap)", (rises[0], 1.2),
                fontsize=8.5, color="#52514e")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="center right", bbox_to_anchor=(1, 0.62))
    style(ax)
    fig.savefig(args.out / "press.png", dpi=160)
    plt.close(fig)

    # Figure 4: one held-out sustained push - hand error, and estimate vs teacher on the most-loaded joint
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    t = traces["Oracle"]["t"]
    j = int(np.argmax(np.abs(traces["OSC"]["teacher"]).max(axis=0)))
    for lbl in labels:
        tr = traces[lbl]
        ls = "--" if lbl in DASHED else "-"
        axes[0].plot(tr["t"] * 1000, tr["error_mm"], color=COLORS[lbl], linewidth=2, linestyle=ls, label=lbl)
        if lbl in ("MLP", "Observer K=100", "Observer K=1000"):
            axes[1].plot(tr["t"] * 1000, tr["est"][:, j], color=COLORS[lbl], linewidth=2, linestyle=ls, label=lbl)
    axes[1].plot(t * 1000, traces["Observer K=1000"]["teacher"][:, j], color="#0b0b0b", linewidth=1,
                 linestyle=":", label="Teacher −JᵀF")
    axes[0].set_xlim(-20, 400)
    axes[1].set_xlim(-10, 60)
    axes[0].set_ylabel("Hand error (mm)")
    axes[1].set_ylabel(f"Joint {j + 1} torque (N·m)")
    axes[0].set_title("Hand error", loc="left")
    axes[1].set_title("Push onset, zoomed: estimate vs teacher", loc="left")
    for ax in axes:
        ax.set_xlabel("Time from push onset (ms)")
        ax.legend(frameon=False, fontsize=8.5)
        style(ax)
    fig.suptitle("First held-out sustained push (seed 200)", fontsize=12)
    fig.savefig(args.out / "trace.png", dpi=160)
    plt.close(fig)
    print(f"Saved results, summary and figures to {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("ablate", "sweep", "compare"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    {"ablate": ablate, "sweep": sweep, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
