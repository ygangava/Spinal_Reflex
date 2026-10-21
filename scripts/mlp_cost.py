"""Count periodic MLP work and measure single-tick host latency.

python -m scripts.mlp_cost

Counts are algorithmic scalar operations, not CPU instructions or energy.
Latency is measured on recorded, prebuilt observations, without batching and
without timing the simulator, model loading, logging, or communication.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np

from mlp_reflex import MLPModel, MLPReflex, features_from_log

ROOT = Path(__file__).resolve().parents[1]


def operation_counts(model):
    """Steady-state step, including history features and gain application.

    A dense dot of length n has n multiplies and n-1 additions. For the common
    2-FLOP/MAC convention we also report 2 * dense_macs, separately. tanh is
    counted as an evaluation, not assigned an arbitrary FLOP/energy equivalent.
    Memory copies, allocation, and dtype conversion are not arithmetic ops.
    """
    n, hidden = model.w1.shape
    outputs = model.w2.shape[1]
    history = model.metadata["features"] == "history"
    dense_macs = model.w1.size + model.w2.size
    dense_adds = hidden * (n - 1) + outputs * (hidden - 1)
    extra = {
        "feature_subtractions": 21 if history else 7,
        "feature_divisions": 7 if history else 0,
        "normalization_divisions": n,
        "hidden_bias_additions": hidden,
        "tanh_evaluations": hidden,
        "rest_output_subtractions": outputs,
        "output_scale_multiplications": outputs,
        "gain_multiplications": outputs,
        "clip_limit_negations": outputs,
        "clip_comparisons": 2 * outputs,
    }
    scalar = {
        "multiplications": dense_macs + 2 * outputs,
        "additions": dense_adds + hidden,
        "subtractions": extra["feature_subtractions"] + outputs,
        "divisions": n + extra["feature_divisions"],
        "negations": outputs,
        "comparisons": 2 * outputs,
        "tanh_evaluations": hidden,
    }
    dt = float(model.metadata["dt"])
    params = model.w1.size + model.b1.size + model.w2.size
    parameter_bytes = sum(a.nbytes for a in (model.w1, model.b1, model.w2))
    resident = {k: int(getattr(model, k).nbytes) for k in
                ("w1", "b1", "w2", "x_scale", "y_scale", "torque_limit", "rest_output")}
    return {
        "architecture": [n, hidden, outputs], "dt_s": dt, "ticks_per_s": 1 / dt,
        "dense_layers": [
            {"shape": list(model.w1.shape), "macs_per_tick": int(model.w1.size)},
            {"shape": list(model.w2.shape), "macs_per_tick": int(model.w2.size)},
        ],
        "dense_macs_per_tick": int(dense_macs), "dense_macs_per_s": dense_macs / dt,
        "dense_flops_per_tick_2_per_mac_convention": int(2 * dense_macs),
        "dense_flops_per_s_2_per_mac_convention": 2 * dense_macs / dt,
        "extra_operations_per_tick": extra,
        "full_step_scalar_operations_per_tick": scalar,
        "full_step_scalar_operations_per_s": {k: v / dt for k, v in scalar.items()},
        "parameters": int(params), "nonzero_weights": int(np.count_nonzero(model.w1) + np.count_nonzero(model.w2)),
        "parameter_bytes": int(parameter_bytes), "parameter_dtype": str(model.w1.dtype),
        "resident_numeric_array_bytes": sum(resident.values()), "resident_arrays": resident,
        "calm_dense_macs_per_s": dense_macs / dt,
        "pushed_dense_macs_per_s": dense_macs / dt,
        "scope": "MLPReflex.step only; excludes base OSC, physics, IO, actuator limiter, training, allocation and copies",
    }


def latency_summary(values):
    a = np.asarray(values, dtype=float)
    return {"samples": len(a), "mean_us": float(a.mean()),
            "median_us": float(np.median(a)), "p95_us": float(np.percentile(a, 95)),
            "p99_us": float(np.percentile(a, 99)), "observed_max_us": float(a.max())}


def benchmark(model, repeats=10, warmup=3):
    from osc import run_episode, DT
    from scripts.mlp import heldout_cases
    if not np.isclose(DT, model.metadata["dt"]):
        raise ValueError("Model timestep differs from simulator")
    # Same preselected held-out trajectory used in the results figure. The
    # model is frozen; labels are not read and no model selection occurs here.
    _, _, p = next(c for c in heldout_cases([200], DT, include_press=False) if c[1] == "sustained")
    log = run_episode(p, reflex=MLPReflex(model, DT), reflex_delay=1,
                      t_end=(p.end_tick + round(1 / DT)) * DT)
    x = features_from_log(log, model.metadata["features"], DT)
    observations = [{"q": log["q"][k], "dq": log["dq"][k],
                     "tau_base": log["tau_base"][k],
                     "tau_prev": log["tau_base"][0] if k == 0 else log["tau_applied"][k-1]}
                    for k in range(len(x))]
    phases = ["calm" if k < p.onset_tick else "pushed" if k < p.end_tick else "recovery"
              for k in range(len(x))]
    timer = time.perf_counter_ns
    timer_samples = []
    for _ in range(1000):
        start = timer()
        timer_samples.append((timer() - start) / 1000)
    rows = []
    all_summaries = {}
    for mode in ("network_only", "full_reflex_step"):
        reflex = MLPReflex(model, DT)
        inputs = x if mode == "network_only" else observations
        call = model.predict if mode == "network_only" else reflex.step
        for _ in range(warmup):
            reflex.reset()
            for item in inputs:
                call(item)
        for repeat in range(repeats):
            reflex.reset()
            for k, item in enumerate(inputs):
                start = timer()
                output = call(item)
                elapsed = (timer() - start) / 1000
                rows.append({"mode": mode, "repeat": repeat, "tick": k,
                             "phase": phases[k], "latency_us": elapsed})
        rr = [r for r in rows if r["mode"] == mode]
        all_summaries[mode] = {
            "all": latency_summary([r["latency_us"] for r in rr]),
            "phases": {phase: latency_summary([r["latency_us"] for r in rr if r["phase"] == phase])
                       for phase in ("calm", "pushed", "recovery")},
            "per_repeat_median_us": [float(np.median([r["latency_us"] for r in rr if r["repeat"] == i]))
                                     for i in range(repeats)],
        }
    return {"latency": all_summaries, "timer_pair_overhead": latency_summary(timer_samples),
            "warmup_passes": warmup, "repeats": repeats, "trajectory_ticks": len(x),
            "input_dtype": str(x.dtype), "synthetic_output_delay_ticks": 1,
            "method": "perf_counter_ns around individual calls; no batching; prebuilt observations; no timer-overhead subtraction",
            "limits": "Host wall-clock latency only; observed max is not a WCET bound; excludes simulator, sensor acquisition, communication and model loading"}, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/mlp_history.npz")
    parser.add_argument("--out", type=Path, default=ROOT / "results/mlp/cost")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup nonnegative")
    model = MLPModel.load(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    counts = operation_counts(model)
    timing, rows = benchmark(model, args.repeats, args.warmup)
    summary = {"counts": counts, "benchmark": timing,
               "environment": {"platform": platform.platform(), "architecture": platform.machine(),
                               "processor": platform.processor(), "python": platform.python_version(),
                               "numpy": np.__version__, "logical_cpus_reported": os.cpu_count(),
                               "blas_thread_environment": {k: os.environ.get(k) for k in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")}},
               "artifact_bytes": args.model.stat().st_size,
               "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
               "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "power_watts": None, "energy_joules_per_tick": None,
               "fpga_resources": None, "fpga_latency_us": None}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.out / "latency_samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"counts": counts, "latency": timing["latency"]}, indent=2))
    print(f"Saved cost summary and raw timing samples to {args.out}")


if __name__ == "__main__":
    main()
