"""Train and evaluate the small periodic MLP, using episode-level splits.

  python -m scripts.mlp train
  python -m scripts.mlp evaluate

Test episodes are constructed only by evaluate, after training/model selection.
All dynamics stay in osc.run_episode; inference is the ordinary Reflex slot.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
from itertools import product
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/spinal-reflex-matplotlib")

import numpy as np

from disturbance import Push, grid_pushes, heldout_pushes, jittered, sustained_pushes, press_pushes
from mlp_reflex import MLPModel, MLPReflex, features_from_log

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/mlp_history.npz"
OUT = ROOT / "runs/mlp"


def parallel(function, jobs, workers):
    if workers == 1:
        return [function(job) for job in jobs]
    with get_context("spawn").Pool(workers) as pool:
        return pool.map(function, jobs, chunksize=4)


def development_cases(seeds, dt):
    return [(seed, kind, p) for seed in seeds for kind, ps in
            (("impact", grid_pushes(seed, dt)), ("sustained", sustained_pushes(seed, dt)))
            for p in ps]


def heldout_cases(seeds, dt, include_press=True):
    cases = []
    for seed in seeds:
        cases += [(seed, "impact", p) for p in heldout_pushes(seed, dt)]
        rng = np.random.default_rng(seed)
        # Forces, durations, and nominal directions are absent from training.
        for i, (force, angle, duration) in enumerate(product(
                (20.0, 40.0), (45, 135, 225, 315), (0.15, 0.25))):
            onset, direction = jittered(rng, angle, dt)
            p = Push(i, "hand", onset, round(duration / dt), None, direction, dt,
                     "trap", round(0.01 / dt), force)
            cases.append((seed, "sustained", p))
        if include_press:
            # Diagnostic only: slow-press yielding is NOT a training objective.
            cases += [(seed, "press", p) for p in press_pushes(seed, dt)]
        cases.append((seed, "quiet", Push(0, "hand", round(1.0 / dt),
                                          20, 0.0, (1.0, 0.0, 0.0), dt)))
    return cases


def collect_episode(job):
    from osc import run_episode, DT
    from reflex import OracleReflex
    seed, kind, p, controller, model_path, mode = job
    kw = {}
    if controller == "oracle":
        kw = {"reflex": OracleReflex(), "reflex_delay": 5}
    elif controller == "mlp":
        kw = {"reflex": MLPReflex(MLPModel.load(model_path), DT)}
    log = run_episode(p, t_end=(p.end_tick + round(0.6 / DT)) * DT, **kw)
    x = features_from_log(log, mode, DT)
    # State[k] reflects force[k-1]. Learn the identifiable one-tick-old
    # cancellation torque, rather than requiring an impossible onset prediction.
    y = np.vstack([np.zeros((1, 7)), log["tau_teacher"][:-1]])
    # Keep all disturbance ticks + first 100 ms after release; subsample calm
    # and later recovery so predict-zero is not rewarded by a huge class imbalance.
    lo, hi = p.onset_tick, min(len(x), p.end_tick + round(0.1 / DT))
    rest = np.r_[np.arange(1, lo), np.arange(hi, len(x))]
    rng = np.random.default_rng(seed * 10000 + p.push_id)
    chosen = np.sort(np.r_[np.arange(lo, hi),
                           rng.choice(rest, min(128, len(rest)), replace=False)])
    return x[chosen].astype(np.float32), y[chosen].astype(np.float32)


def collect(cases, controllers, model_path, mode, workers):
    jobs = [(seed, kind, p, c, str(model_path), mode)
            for seed, kind, p in cases for c in controllers]
    rows = parallel(collect_episode, jobs, workers)
    return np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows])


def provenance():
    def git(*args, cwd=ROOT):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                              text=True, check=True).stdout.strip()
    files = ("osc.py", "disturbance.py", "reflex.py", "mlp_reflex.py", "scripts/mlp.py")
    return {"base_commit": git("rev-parse", "HEAD"),
            "menagerie_commit": git("rev-parse", "HEAD", cwd=ROOT / "mujoco_menagerie"),
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}}


def fit(train_data, val_data, args, metadata):
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    x, y = train_data
    vx, vy = val_data
    floors = np.r_[np.full(7, 1e-3), np.full(7, 1e-2)]
    if args.features == "history":
        floors = np.r_[floors, np.full(7, 1.0), np.full(7, 0.5)]
    xs = np.maximum(np.sqrt(np.mean(x.astype(float) ** 2, axis=0)), floors).astype(np.float32)
    ys = np.maximum(np.sqrt(np.mean(y.astype(float) ** 2, axis=0)), 1.0).astype(np.float32)
    tx, ty = torch.from_numpy(x / xs), torch.from_numpy(y / ys)
    vtx, vty = torch.from_numpy(vx / xs), torch.from_numpy(vy / ys)

    class AnchoredMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(x.shape[1], args.hidden)
            self.output = nn.Linear(args.hidden, 7, bias=False)

        def forward(self, inputs):
            h = torch.tanh(self.hidden(inputs)) - torch.tanh(self.hidden.bias)
            return self.output(h)

    net = AnchoredMLP()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    best_loss, best, stale, history = float("inf"), None, 0, []
    for epoch in range(args.epochs):
        net.train()
        indices = torch.randperm(len(tx))
        total = 0.0
        for ix in indices.split(args.batch_size):
            optimizer.zero_grad()
            loss = (net(tx[ix]) - ty[ix]).square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(ix)
        net.eval()
        with torch.no_grad():
            val_loss = (net(vtx) - vty).square().mean().item()
        history.append({"epoch": epoch + 1, "train_mse_scaled": total / len(tx),
                        "validation_mse_scaled": val_loss})
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1}: train={total/len(tx):.5f}, val={val_loss:.5f}", flush=True)
        if stale >= 20:
            break
    net.load_state_dict(best)
    model = MLPModel(net.hidden.weight.detach().numpy().T.copy(),
                     net.hidden.bias.detach().numpy().copy(),
                     net.output.weight.detach().numpy().T.copy(), xs, ys,
                     np.array([87., 87, 87, 87, 12, 12, 12]),
                     {**metadata, "best_epoch": min(history, key=lambda r: r["validation_mse_scaled"])["epoch"],
                      "parameters": sum(p.numel() for p in net.parameters()),
                      "dense_macs_per_tick": x.shape[1] * args.hidden + args.hidden * 7,
                      "torch_version": torch.__version__})
    report = {"train_rows": len(x), "validation_rows": len(vx), "history": history,
              "validation_rmse_nm": float(np.sqrt(np.mean((model.predict(vx) - vy) ** 2))),
              "predict_zero_rmse_nm": float(np.sqrt(np.mean(vy ** 2)))}
    with torch.no_grad():
        expected = net(vtx[:128]).numpy() * ys
    expected = np.clip(expected, -model.torque_limit, model.torque_limit)
    if not np.allclose(model.predict(vx[:128]), expected, atol=2e-4, rtol=2e-4):
        raise RuntimeError("NumPy inference disagrees with the PyTorch training model")
    return model, report


def train(args):
    from osc import DT
    if set(args.train_seeds) & set(args.val_seeds):
        raise ValueError("Training and validation seeds must be disjoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    tc, vc = development_cases(args.train_seeds, DT), development_cases(args.val_seeds, DT)
    # Base-controller and near-cancelled demonstrations cover small-error states
    # absent from OSC-only logs. The oracle never runs inside the learned reflex.
    print(f"Collecting {2 * len(tc)} training and {2 * len(vc)} validation episodes", flush=True)
    td = collect(tc, ("none", "oracle"), args.output, args.features, args.workers)
    vd = collect(vc, ("none", "oracle"), args.output, args.features, args.workers)
    meta = {"format_version": 1, "features": args.features, "dt": DT,
            "hidden": args.hidden, "activation": "tanh", "label_delay_ticks": 1,
            "label": "previous tick -J^T F", "train_seeds": args.train_seeds,
            "validation_seeds": args.val_seeds, "training_seed": args.seed,
            "scope": "home pose; hand pushes; noise-free simulation",
            "training_config": {"epochs_per_round": args.epochs, "rounds": args.rounds,
                                "batch_size": args.batch_size, "learning_rate": args.lr,
                                "optimizer": "Adam", "loss": "scaled torque MSE",
                                "teacher_demonstration_delay_ticks": 5},
            "gain": 1.0, "provenance": provenance()}
    reports = []
    for round_id in range(args.rounds):
        if round_id:
            print(f"Collecting closed-loop student data for round {round_id+1}", flush=True)
            new_t = collect(tc, ("mlp",), args.output, args.features, args.workers)
            new_v = collect(vc, ("mlp",), args.output, args.features, args.workers)
            td = tuple(np.concatenate([a, b]) for a, b in zip(td, new_t))
            vd = tuple(np.concatenate([a, b]) for a, b in zip(vd, new_v))
        model, report = fit(td, vd, args, {**meta, "data_rounds": round_id + 1})
        model.save(args.output)
        reports.append(report)
        print(f"round {round_id+1}: validation torque RMSE {report['validation_rmse_nm']:.3f} Nm "
              f"vs zero {report['predict_zero_rmse_nm']:.3f} Nm", flush=True)
    (args.out / "training.json").write_text(json.dumps({"model": model.metadata, "rounds": reports}, indent=2))
    print(f"Saved {args.output}", flush=True)


def evaluate_episode(job):
    from osc import run_episode, DT, HIGH_KP_GAINS, TAU_MAX
    from reflex import OracleReflex
    seed, kind, p, controller, model_path, delay, keep_trace = job
    kw = {}
    if controller == "High-KP OSC":
        kw["gains"] = HIGH_KP_GAINS
    elif controller == "MLP":
        model = MLPModel.load(model_path)
        kw = {"reflex": MLPReflex(model, DT), "reflex_gain": model.metadata.get("gain", 1.0),
              "reflex_delay": delay}
    elif controller == "Oracle":
        kw = {"reflex": OracleReflex(), "reflex_delay": delay + 1}
    log = run_episode(p, t_end=(p.end_tick + round(1.0 / DT)) * DT, **kw)
    e = 1000 * np.linalg.norm(log["e_p"], axis=1)
    post = e[p.onset_tick:]
    row = {"seed": seed, "set": kind, "push_id": p.push_id, "controller": controller,
           "peak_mm": float(post.max()), "auc_mm_s": float(post.sum() * DT),
           "pre_push_peak_mm": float(e[:p.onset_tick].max()),
           "saturation_fraction": float((np.abs(log["tau_cmd"]) > TAU_MAX).any(axis=1).mean()),
           "torque_limited_fraction": float((np.abs(log["tau_cmd"] - log["tau_applied"]) > 1e-6).any(axis=1).mean()),
           "final_mm": float(e[-1]), "push": json.dumps(asdict(p), sort_keys=True)}
    if kind == "press":
        end = p.end_tick - p.rise_ticks
        start = end - (p.dur_ticks - 2 * p.rise_ticks) // 4
        row["hold_mm"] = float(e[start:end].mean())
    trace = {"time_from_onset_s": np.arange(len(e)) * DT - p.onset_tick * DT,
             "error_mm": e, "force_n": np.linalg.norm(log["f_push"], axis=1)} if keep_trace else None
    return row, trace


def write_results(results, model, args):
    rows = [r for r, _ in results]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (args.out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    seeds = args.val_seeds if args.command == "validate" else args.test_seeds
    summary = {"model": model.metadata, "evaluation_split": args.command,
               "evaluation_seeds": seeds,
               "additional_output_delay_ticks": args.delay,
               "oracle_delay_ticks": args.delay + 1, "episodes": len(rows),
               "controllers": {}, "provenance": provenance()}
    for kind in ("impact", "sustained", "press", "quiet"):
        summary["controllers"][kind] = {}
        for ctrl in ("OSC", "High-KP OSC", "MLP", "Oracle"):
            rr = [r for r in rows if r["set"] == kind and r["controller"] == ctrl]
            if not rr:
                continue
            metrics = ("peak_mm", "auc_mm_s", "pre_push_peak_mm", "final_mm",
                       "saturation_fraction", "torque_limited_fraction")
            values = {k: float(np.mean([r[k] for r in rr])) for k in metrics}
            if kind == "press":
                values["hold_mm"] = float(np.mean([r["hold_mm"] for r in rr]))
            values["n"] = len(rr)
            summary["controllers"][kind][ctrl] = values
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    for kind, ctrls in summary["controllers"].items():
        print(kind + ": " + "; ".join(f"{c} {v['peak_mm']:.3f} mm" for c, v in ctrls.items()), flush=True)
    plot_results(results, summary, args.out)


def plot_results(results, summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    controllers = ["OSC", "High-KP OSC", "MLP", "Oracle"]
    colors = ["#8b9299", "#d17b24", "#1769aa", "#28805a"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    prefix = "Validation" if summary["evaluation_split"] == "validate" else "Held-out"
    for ax, kind, title in zip(axes[:2], ("impact", "sustained"),
                               (f"{prefix} impacts", f"{prefix} sustained pushes")):
        values = summary["controllers"][kind]
        heights = [values[c]["peak_mm"] for c in controllers]
        ax.bar(range(4), heights, color=colors, width=0.65)
        for i, h in enumerate(heights):
            ax.text(i, h, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(4), controllers, rotation=15)
        ax.set_ylabel("Mean peak hand error (mm)")
        ax.set_title(title, loc="left")
        ax.set_ylim(0, max(heights) * 1.18 + 0.1)
    ax = axes[2]
    trace_payload = {}
    for c, color in zip(controllers, colors):
        trace = next((t for r, t in results if r["controller"] == c and t is not None), None)
        if trace is not None:
            ax.plot(trace["time_from_onset_s"], trace["error_mm"], label=c, color=color)
            trace_payload.update({c.replace(" ", "_") + "_" + k: v for k, v in trace.items()})
    ax.set_xlim(-0.03, 0.75)
    ax.set_xlabel("Time from push onset (s)")
    ax.set_ylabel("Hand error (mm)")
    ax.set_title("First sustained episode in the split", loc="left")
    ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    split = "validation grid" if summary["evaluation_split"] == "validate" else "held-out pushes"
    fig.suptitle(f"Periodic MLP reflex | home pose, noise-free | {split}", fontsize=13)
    fig.savefig(out / "comparison.png", dpi=180)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)
    np.savez_compressed(out / "example_trace.npz", **trace_payload)


def evaluate(args):
    from osc import DT
    model = MLPModel.load(args.model)
    seeds = args.val_seeds if args.command == "validate" else args.test_seeds
    forbidden = model.metadata["train_seeds"] + ([] if args.command == "validate" else model.metadata["validation_seeds"])
    if set(seeds) & set(forbidden):
        raise ValueError("Test seeds overlap training/validation")
    if args.command == "validate":
        cases = development_cases(seeds, DT)
        cases += [(s, "quiet", Push(0, "hand", 1000, 20, 0.0, (1., 0., 0.), DT)) for s in seeds]
    else:
        cases = heldout_cases(seeds, DT)
    args.out.mkdir(parents=True, exist_ok=True)
    controllers = ("OSC", "High-KP OSC", "MLP", "Oracle")
    first_seed = seeds[0]
    jobs = [(s, kind, p, c, str(args.model), args.delay,
             s == first_seed and kind == "sustained" and p.push_id == 0)
            for s, kind, p in cases for c in controllers]
    print(f"Evaluating {len(jobs)} paired {args.command} episodes; model frozen", flush=True)
    results = parallel(evaluate_episode, jobs, args.workers)
    write_results(results, model, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--features", choices=("state", "history"), default="history")
    tr.add_argument("--hidden", type=int, default=64)
    tr.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    tr.add_argument("--val-seeds", type=int, nargs="+", default=[100])
    tr.add_argument("--seed", type=int, default=17)
    tr.add_argument("--epochs", type=int, default=80)
    tr.add_argument("--rounds", type=int, default=2)
    tr.add_argument("--batch-size", type=int, default=1024)
    tr.add_argument("--lr", type=float, default=0.002)
    tr.add_argument("--output", type=Path, default=MODEL_PATH)
    ev = sub.add_parser("evaluate")
    va = sub.add_parser("validate")
    ev.add_argument("--test-seeds", type=int, nargs="+", default=[200, 201])
    va.add_argument("--val-seeds", type=int, nargs="+", default=[100])
    for p in (ev, va):
        p.add_argument("--model", type=Path, default=MODEL_PATH)
        p.add_argument("--delay", type=int, default=1,
                       help="Additional output delay in ticks; learned sensing already costs one tick")
    for p in (tr, ev, va):
        p.add_argument("--out", type=Path, default=OUT / "validation" if p is va else OUT)
        p.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or (args.command == "train" and
                            min(args.epochs, args.rounds, args.hidden, args.batch_size) < 1):
        parser.error("Worker count and training sizes must be positive")
    if args.command in ("evaluate", "validate") and args.delay < 0:
        parser.error("Delay must be nonnegative")
    (train if args.command == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
