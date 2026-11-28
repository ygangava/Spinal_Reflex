
from dataclasses import dataclass
import json

import numpy as np

from reflex import Reflex
from mlp_reflex import CausalFeatures, FEATURE_SIZES
from encoder import SendOnDelta


# leak+integrate, compare, subtract
def lif_step(v, current, beta, threshold = 1):
    
    # v_new = beta * v + current # Leak + integrate
    v_new = beta * np.asarray(v, dtype=float) + current
    spike = v_new >= threshold
    v_new = v_new - threshold* spike
    
    return v_new, spike

# B = 0.9375
# # example -- constant 0.3
# v = 0.0
# for k in range(20):
#     v, s = lif_step(v, 0.3, B)
#     if s: 
#         print(f"SPIKE + {v}")
#     else:
#         print(f"no spike + {v}")

# v, v_new, spike - (64,), on/off - (28,), w_in (56,64)

def layer_step(v, on, off, w_in, beta, threshold = 1.0):
    # (56,)  ON rows 0–27, OFF rows 28–55: same order as w_in
    # will let us multiply
    counts = np.concatenate([on,off])
    current = counts @ w_in
    
    v_new, spike = lif_step(v, current, beta, threshold)
    
    ops = 64 * int(counts.sum())
    return v_new, spike, ops


def output_step(acc, spike, w_out):
    # acc (7,), spike (64,) bool, w_out (64, 7). ops = 7 × number of spikes.
    # acc += W_out[spiked].sum(axis=0) 
    acc_new = acc + w_out[spike].sum(axis = 0)
    ops = 7*int((spike.sum()))
    return acc_new, ops

def snn_step(v, acc, on, off, w_in, w_out, beta):
    v_new, spike, ops1 = layer_step(v, on, off, w_in, beta)
    acc_new, ops2 = output_step(acc, spike, w_out)
    ops = ops1+ops2
    return v_new, acc_new, spike, ops


# ---------------------------------------------------------------- model (weights + scales, saved to .npz)
# similar to mlp_reflex.py one

@dataclass
class SNNModel:
    w_in: np.ndarray          # (56, 64)  rows 0–27 = ON channels, 28–55 = OFF channels
    w_out: np.ndarray         # (64, 7)   row i = torque nudge per joint when hidden neuron i spikes
    beta: float               # 1 - 2**-n
    k: float                  # encoder threshold θ = k · x_scale
    x_scale: np.ndarray       # (28,)  per-feature RMS (borrowed from the MLP for now)
    y_scale: np.ndarray       # (7,)   acc → N·m
    torque_limit: np.ndarray  # (7,)   87 / 12 N·m
    metadata: dict            # needs "dt" and "features" ("history")

    def __post_init__(self):
        # Fail loudly on wrong shapes now, not as a silent broadcast bug mid-episode.
        n = FEATURE_SIZES[self.metadata["features"]]
        h = self.w_in.shape[1]
        if (self.w_in.shape != (2 * n, h) or self.w_out.shape != (h, 7)
                or self.x_scale.shape != (n,) or self.y_scale.shape != (7,)
                or self.torque_limit.shape != (7,)):
            raise ValueError("Invalid SNN array shapes")
        if not 0 < self.beta < 1 or self.k <= 0:
            raise ValueError("Need 0 < beta < 1 and k > 0")

    def save(self, path):
        # Scalars stored as 0-d arrays; metadata as a JSON string (so load needs no pickle).
        np.savez_compressed(path, w_in=self.w_in, w_out=self.w_out,
                            beta=self.beta, k=self.k,
                            x_scale=self.x_scale, y_scale=self.y_scale,
                            torque_limit=self.torque_limit,
                            metadata=json.dumps(self.metadata, sort_keys=True))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            arrays = {k: f[k].copy() for k in ("w_in", "w_out", "x_scale", "y_scale", "torque_limit")}
            return cls(**arrays, beta=float(f["beta"]), k=float(f["k"]),
                       metadata=json.loads(str(f["metadata"])))


# ---------------------------------------------------------------- reflex (plugs into run_episode)

class SNNReflex(Reflex):
    name = "snn"

    def __init__(self, model, floor=None):
        self.model = model
        # theta = k · x_scale, raised to `floor` where given (same rule as DeltaMLPReflex)
        theta = model.k * model.x_scale if floor is None else np.maximum(model.k * model.x_scale, floor)
        self.event_size = theta / model.x_scale
        model_dt = float(model.metadata["dt"])

        # W_in was trained with one event = θ_k = k·x_scale. With a floor, one event = θ ≥ θ_k (a bigger change),
        # so scale each channel's row by θ/θ_k (the delta MLP's event_size idea). Folded into the weights once
        # here: no extra work per event, and in RTL it's just a different weight ROM per noise level.
        # Without a floor the ratio is exactly 1, so this changes nothing.
        ratio = theta / (model.k * model.x_scale)               # (28,)
        self.w_in = model.w_in * np.r_[ratio, ratio][:, None]   # ON rows 0–27 and OFF rows 28–55 get the same ratio

        self.features = CausalFeatures(model.metadata["features"], model_dt,
                                       model.metadata.get("torque_ref", "hold"))   # old models: "hold"
        self.enc = SendOnDelta(theta)

        # acc limit = torque_limit / y_scale   (clip acc itself → no windup; RTL = saturating register)
        acc_limit = model.torque_limit / model.y_scale
        self.acc_limit = acc_limit
        self.reset()

    def reset(self):
        self.features.reset()
        self.enc.reset()
        self.torque = np.zeros(7)

        self.v = np.zeros(64)
        self.acc = np.zeros(7)  # one val per joint
        
        self.ops = []  # (eager: + 64 leak per tick)
        self.ops_lazy = [] # (input + output adds only)
        
        pass

    def compute(self, obs, priv=None):
        x = self.features.step(obs)
        on, off = self.enc.step(x)
        self.v, self.acc, spike, ops = snn_step(self.v, self.acc, on, off, self.w_in, self.model.w_out, self.model.beta)
        # clip acc to ± its limit
        self.acc = np.clip(self.acc, -self.acc_limit, self.acc_limit)
        # torque = acc · y_scale   (already within torque_limit after the acc clip)
        self.torque = self.acc * self.model.y_scale
        
        # ops + 64 to self.ops, ops to self.ops_lazy
        self.ops.append(ops+ 64) 
        self.ops_lazy.append(ops)

        # no need to check fired here becuasue no events mean zero counts, and just leaks

        return self.torque, self.torque.copy()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    w_in = rng.normal(0, 0.3, (56, 64))
    v = np.zeros(64)
    on, off = np.zeros(28, int), np.zeros(28, int)

    beta = 0.9375
    # quiet tick: expect ops 0, no spikes
    v, s, ops = layer_step(v, on, off, w_in, beta)
    print("quiet:", ops, s.sum())

    # P4: channel 3 ON x2, channel 40 = OFF channel 12 (40 - 28) x1
    # on[3] = 2: joint 4's position error rose by 2 thresholds (2θ) this tick.
    # off[12] = 1: joint 6's velocity (index 12 − 7 = 5, so the 6th joint) fell by 1 threshold.
    on[3], off[12] = 2, 1  # on added twice, off12 once
    # on and off usually come from SendOnDelta
    v, s, ops = layer_step(v, on, off, w_in, beta)
    print("P4 ops:", ops, " spikes:", np.flatnonzero(s))
    
    