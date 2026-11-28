"""Phase D: train the SNN's two weight matrices with BPTT + surrogate gradients.

  python -m scripts.snn_train                 # train, export models/snn.npz, check NumPy == torch
  python -m scripts.snn_train --epochs 2      # quick smoke test
  python -m scripts.snn_train --extra r1 --w-sustained 5 --w-after 3   # + DAgger round r1, reweighted loss

Only W_in (56, 64) and W_out (64, 7) are trained. beta, threshold, k, and the encoder are fixed.
The LIF step + surrogate come from snn_torch.py (yours); this file is the harness around them.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from delta_mlp_reflex import feature_noise
from mlp_reflex import MLPModel
from scripts.snn_data import encode, load
from snn import SNNModel, snn_step
from snn_torch import lif_step_torch

ROOT = Path(__file__).resolve().parents[1]
MLP_PATH = ROOT / "models/mlp_history.npz"
OUT_MODEL = ROOT / "models/snn.npz"
OUT_RUN = ROOT / "runs/snn"
HIDDEN = 64
DT = 0.001
FLOOR_SIGMAS = 6     # same floor rule as scripts/noise_sweep.py: θ ≥ 6σ of each feature's noise


def add_noise(e, dq_std, rng):
    """Copy of episode e with sensor noise in its features, the way NoisyObs + CausalFeatures make them online.

    Same q/dq tie as noise_sweep: q_std = dq_std·DT/√2. Features are [q − q_ref, dq, acc, tau_prev − ref]:
      q − q_ref: q_ref is the NOISY tick-0 sample, so the noise is nq[t] − nq[0]
      dq:        + ndq[t]
      acc:       (dq[t] − dq[t−1]) / DT, so + (ndq[t] − ndq[t−1]) / DT   (noise ×1400: the big one)
      tau_prev:  no noise (NoisyObs default tau_std = 0)
    Approximation: online, the encoder and the SNN also see noise during the ticks BEFORE the window. Here the
    window starts fresh. With the floor, calm noise almost never makes an event, so that state is ~0 anyway.
    """
    q_std = dq_std * DT / np.sqrt(2)
    n = len(e["x"])
    nq0 = rng.normal(0, q_std, 7)
    nq = rng.normal(0, q_std, (n, 7))
    ndq = rng.normal(0, dq_std, (n + 1, 7))      # row 0 = the tick just before the window (for the first acc)
    x = e["x"].astype(np.float64)                # astype copies: the clean episode is left alone
    x[:, 0:7] += nq - nq0
    x[:, 7:14] += ndq[1:]
    x[:, 14:21] += (ndq[1:] - ndq[:-1]) / DT
    x0 = e["x0"].astype(np.float64)
    x0[7:14] += rng.normal(0, dq_std, 7)         # tick 0: q − q_ref = 0 by definition, acc = 0 (dq_prev = dq)
    return {**e, "x0": x0, "x": x, "dq_std": dq_std}


def tensors(episodes, theta, y_scale, w_sustained=1.0, w_after=1.0):
    """Episodes -> padded (N, T, 56) scaled event counts, (N, T, 7) scaled targets, (N, T) per-tick loss weights.

    Events: an episode with sensor noise (from add_noise) is encoded at the FLOOR θ the SNN will run at, and each
    count is scaled by θ_floor/θ_k, which is exactly what SNNReflex(model, floor) folds into W_in. Clean episodes:
    ratio 1, plain counts.

    Weights (masked_mse is a weighted mean, so only their ratios matter):
      padding 0; sustained episodes x w_sustained (impacts dominate the plain MSE ~25:1 per tick);
      ticks after the push ends x w_after (teacher = 0 there: "let go", the closed-loop failure in Lesson 7.4).
    """
    T = max(len(e["x"]) for e in episodes)
    N = len(episodes)
    E, Y, M = np.zeros((N, T, 56), np.float32), np.zeros((N, T, 7), np.float32), np.zeros((N, T), np.float32)
    for i, e in enumerate(episodes):
        n = len(e["x"])
        dq_std = e.get("dq_std", 0.0)
        floor = FLOOR_SIGMAS * feature_noise(dq_std * DT / np.sqrt(2), dq_std, DT)
        theta_e = np.maximum(theta, floor)           # = theta when there's no noise
        ratio = theta_e / theta
        E[i, :n] = encode(e["x0"], e["x"], theta_e) * np.r_[ratio, ratio]   # same input SNNReflex sees online
        Y[i, :n] = e["y"] / y_scale                 # target in acc units: torque = acc · y_scale
        M[i, :n] = 1.0                              # padding ticks don't count in the loss
        M[i, e["end"]:n] *= w_after                 # after the push
        if e["kind"] == "sustained":
            M[i, :n] *= w_sustained
    return torch.from_numpy(E), torch.from_numpy(Y), torch.from_numpy(M)


class SNN(torch.nn.Module):
    def __init__(self, beta, acc_limit, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # Same scales as the random check: an event moves v by ~0.3, a spike moves acc by ~0.01 (bigger made epoch 1 worse than predict-zero).
        self.w_in = torch.nn.Parameter(0.3 * torch.randn(56, HIDDEN, generator=g))
        self.w_out = torch.nn.Parameter(0.01 * torch.randn(HIDDEN, 7, generator=g))
        self.beta = beta
        self.register_buffer("acc_limit", torch.as_tensor(acc_limit, dtype=torch.float32))

    def forward(self, E):
        # E: (B, T, 56). Unrolled in time, so autograd builds the BPTT graph.
        B, T, _ = E.shape
        v = E.new_zeros(B, HIDDEN)
        acc = E.new_zeros(B, 7)
        outs, rate = [], 0.0
        for t in range(T):
            v, s = lif_step_torch(v, E[:, t] @ self.w_in, self.beta)
            # clamp = the saturating acc, same as SNNReflex (no gradient while pinned at the limit)
            acc = torch.clamp(acc + s @ self.w_out, -self.acc_limit, self.acc_limit)
            outs.append(acc)
            rate = rate + s.mean()
        return torch.stack(outs, 1), rate / T     # (B, T, 7), mean spike prob per neuron-tick


def masked_mse(pred, Y, M):
    return ((pred - Y).square().mean(-1) * M).sum() / M.sum()


def numpy_forward(model, x0, x):
    """The deployed path: features -> encode -> snn.snn_step (your NumPy code) -> clip."""
    ev = encode(x0, x, model.k * model.x_scale)
    v, acc, lim = np.zeros(HIDDEN), np.zeros(7), model.torque_limit / model.y_scale
    out, spikes = [], []
    for t in range(len(x)):
        v, acc, s, _ = snn_step(v, acc, ev[t, :28], ev[t, 28:], model.w_in, model.w_out, model.beta)
        acc = np.clip(acc, -lim, lim)
        out.append(acc * model.y_scale)
        spikes.append(s)
    return np.array(out), np.array(spikes)


def agreement(model, episodes, n=128):
    """NumPy (float64) vs torch (float64) on the same weights. Spike ties at exactly v = 1 can differ."""
    net = SNN(model.beta, model.torque_limit / model.y_scale, 0).double()
    net.w_in.data, net.w_out.data = torch.from_numpy(model.w_in), torch.from_numpy(model.w_out)
    worst = 0.0
    for e in episodes[:n]:
        out_np, _ = numpy_forward(model, e["x0"], e["x"])
        E = torch.from_numpy(encode(e["x0"], e["x"], model.k * model.x_scale)[None].astype(np.float64))
        with torch.no_grad():
            out_t = net(E)[0][0].numpy() * model.y_scale
        worst = max(worst, float(np.abs(out_np - out_t).max()))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=4, help="beta = 1 - 2**-n")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam", type=float, default=1e-2, help="spike-rate penalty weight")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--extra", nargs="*", default=[], metavar="TAG",
                    help="add DAgger rounds snn_{train,val}_TAG.npz (from snn_data --dagger TAG)")
    ap.add_argument("--w-sustained", type=float, default=1.0, help="loss weight on sustained episodes")
    ap.add_argument("--w-after", type=float, default=1.0, help="loss weight on ticks after the push ends")
    ap.add_argument("--noise", nargs="*", type=float, default=[], metavar="DQ_STD",
                    help="add one noisy copy of every episode, dq noise drawn from these levels (rad/s), at the floor θ")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    mlp = MLPModel.load(MLP_PATH)                      # borrow the scales: same θ as the delta MLP
    x_scale, y_scale, limit = mlp.x_scale, mlp.y_scale, mlp.torque_limit
    beta = 1 - 2.0 ** -args.n
    # DAgger rounds go into val too, so early stopping also sees the closed-loop states (not letting go)
    train_eps = load("train") + [e for tag in args.extra for e in load(f"train_{tag}")]
    val_eps = load("val") + [e for tag in args.extra for e in load(f"val_{tag}")]
    if args.noise:
        # Noisy copies alongside the clean ones (the clean ones = the "dq 0" level). Fixed seeds: same data every run.
        for eps, seed in ((train_eps, 1), (val_eps, 2)):
            rng = np.random.default_rng(seed)
            eps += [add_noise(e, rng.choice(args.noise), rng) for e in list(eps)]
    w = dict(w_sustained=args.w_sustained, w_after=args.w_after)
    E, Y, M = tensors(train_eps, args.k * x_scale, y_scale, **w)
    vE, vY, vM = tensors(val_eps, args.k * x_scale, y_scale, **w)
    # The bar to beat: "predict zero", with the same weights (so not comparable to unweighted runs)
    zero_val = masked_mse(torch.zeros_like(vY), vY, vM).item()
    print(f"train {tuple(E.shape)}  val {tuple(vE.shape)}  beta {beta}  k {args.k}")
    print(f"predict-zero val MSE (scaled): {zero_val:.4f}")

    net = SNN(beta, limit / y_scale, args.seed)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best, best_val, stale, history = None, float("inf"), 0, []
    for epoch in range(args.epochs):
        t0, total, rates = time.time(), 0.0, []
        net.train()
        for ix in torch.randperm(len(E)).split(args.batch):
            opt.zero_grad()
            pred, rate = net(E[ix])
            mse = masked_mse(pred, Y[ix], M[ix])
            loss = mse + args.lam * rate            # accuracy + "spike less" (sparsity = cheaper)
            loss.backward()                          # BPTT through all T ticks
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)   # long unrolls can spike gradients
            opt.step()
            total += mse.item() * len(ix)
            rates.append(rate.item())
        net.eval()
        with torch.no_grad():
            vpred, vrate = net(vE)
            val = masked_mse(vpred, vY, vM).item()
        history.append({"epoch": epoch + 1, "train_mse": total / len(E), "val_mse": val,
                        "train_rate": float(np.mean(rates)), "val_rate": vrate.item()})
        print(f"epoch {epoch+1:3d}  train {total/len(E):.4f}  val {val:.4f}  "
              f"spike rate {vrate.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if val < best_val:
            best_val, stale = val, 0
            best = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break

    net.load_state_dict(best)
    model = SNNModel(net.w_in.detach().double().numpy(), net.w_out.detach().double().numpy(), beta, args.k,
                     x_scale, y_scale, limit,
                     {"dt": 0.001, "features": "history", "hidden": HIDDEN, "lam": args.lam,
                      "extra_rounds": args.extra, "w_sustained": args.w_sustained, "w_after": args.w_after,
                      "train_noise_dq": args.noise, "floor_sigmas": FLOOR_SIGMAS,
                      "best_epoch": min(history, key=lambda r: r["val_mse"])["epoch"]})
    model.save(OUT_MODEL)
    worst = agreement(model, val_eps)
    OUT_RUN.mkdir(parents=True, exist_ok=True)
    report = {"args": vars(args), "predict_zero_val_mse": zero_val, "best_val_mse": best_val,
              "numpy_vs_torch_max_abs_nm": worst, "history": history}
    (OUT_RUN / "training.json").write_text(json.dumps(report, indent=1))
    print(f"best val MSE {best_val:.4f} vs predict-zero {zero_val:.4f}")
    print(f"NumPy vs torch, 128 val episodes: max |Δ torque| = {worst:.2e} N·m")
    print(f"saved {OUT_MODEL}")


if __name__ == "__main__":
    main()
