"""Small periodic ANN reflex. NumPy inference; PyTorch is needed only for training.

State mode: [q - q_initial, dq] (14 inputs).
History mode: add causal finite-difference acceleration and previous applied
motor torque minus the initial hold torque (28 inputs). Neither mode uses k,
force, onset, teacher labels, or future state. This first experiment holds the
home pose: q_initial is a posture reference, not a moving trajectory target.
"""
from dataclasses import dataclass
import json

import numpy as np

from reflex import Reflex


FEATURE_SIZES = {"state": 14, "history": 28}


# What the tau_prev channel is measured against (SNN_LEARN.md Q7):
#   "hold": the OSC torque at tick 0, a constant (today; only right while holding still)
#   "ff":   last tick's OSC feedforward tau_ff, the torque the plan needed then with no push (moving arm)
# At a hold with a perfect model the two agree at tick 0 and differ only by how gravity changes as the arm deflects.
TORQUE_REFS = ("hold", "ff")


class CausalFeatures:
    def __init__(self, mode="history", dt=0.001, torque_ref="hold"):
        if mode not in FEATURE_SIZES or dt <= 0:
            raise ValueError("Expected state/history features and positive dt")
        if torque_ref not in TORQUE_REFS:
            raise ValueError(f"torque_ref must be one of {TORQUE_REFS}")
        self.mode, self.dt, self.torque_ref = mode, dt, torque_ref
        self.reset()

    def reset(self):
        self.q_ref = self.tau_ref = self.dq_prev = self.tau_ff_prev = None

    def step(self, obs):
        q, dq = np.asarray(obs["q"]), np.asarray(obs["dq"])
        if self.q_ref is None:
            self.q_ref = q.copy()
            self.tau_ref = np.asarray(obs["tau_base"]).copy()
            self.dq_prev = dq.copy()
            if self.torque_ref == "ff":
                # Tick 0: tau_prev is this tick's tau_base, so compare it with this tick's tau_ff
                self.tau_ff_prev = np.asarray(obs["tau_ff"]).copy()
        parts = [q - self.q_ref, dq]
        if self.mode == "history":
            # tau_prev was applied LAST tick, so subtract what the plan needed last tick (same tick on both sides)
            ref = self.tau_ref if self.torque_ref == "hold" else self.tau_ff_prev
            parts += [(dq - self.dq_prev) / self.dt,
                      np.asarray(obs["tau_prev"]) - ref]
        self.dq_prev = dq.copy()
        if self.torque_ref == "ff":
            self.tau_ff_prev = np.asarray(obs["tau_ff"]).copy()
        return np.concatenate(parts)


def features_from_log(log, mode, dt, torque_ref="hold"):
    """Exactly the online feature definition, vectorized for training."""
    if mode not in FEATURE_SIZES:
        raise ValueError("Unknown feature mode")
    if torque_ref not in TORQUE_REFS:
        raise ValueError(f"torque_ref must be one of {TORQUE_REFS}")
    parts = [log["q"] - log["q"][0], log["dq"]]
    if mode == "history":
        acceleration = np.diff(log["dq"], axis=0, prepend=log["dq"][:1]) / dt
        previous = np.vstack([log["tau_base"][:1], log["tau_applied"][:-1]])
        if torque_ref == "hold":
            ref = log["tau_base"][0]
        else:
            ref = np.vstack([log["tau_ff"][:1], log["tau_ff"][:-1]])   # last tick's tau_ff (tick 0: its own)
        parts += [acceleration, previous - ref]
    return np.concatenate(parts, axis=1)


@dataclass
class MLPModel:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    x_scale: np.ndarray
    y_scale: np.ndarray
    torque_limit: np.ndarray
    metadata: dict

    def __post_init__(self):
        n = FEATURE_SIZES[self.metadata["features"]]
        h = self.b1.size
        if (self.w1.shape != (n, h) or self.w2.shape != (h, 7)
                or self.x_scale.shape != (n,) or self.y_scale.shape != (7,)
                or self.torque_limit.shape != (7,)):
            raise ValueError("Invalid MLP array shapes")
        arrays = (self.w1, self.b1, self.w2, self.x_scale, self.y_scale, self.torque_limit)
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("Non-finite MLP parameters")
        if any((a <= 0).any() for a in (self.x_scale, self.y_scale, self.torque_limit)):
            raise ValueError("Scales and limits must be positive")
        self.rest_output = np.tanh(self.b1) @ self.w2

    def predict(self, x):
        # Subtract the model's rest prediction: zero features -> exactly zero torque.
        z = np.asarray(x) / self.x_scale
        hidden = np.tanh(z @ self.w1 + self.b1)
        torque = (hidden @ self.w2 - self.rest_output) * self.y_scale
        return np.clip(torque, -self.torque_limit, self.torque_limit)

    def save(self, path):
        np.savez_compressed(path, w1=self.w1, b1=self.b1, w2=self.w2,
                            x_scale=self.x_scale, y_scale=self.y_scale,
                            torque_limit=self.torque_limit,
                            metadata=json.dumps(self.metadata, sort_keys=True))

    @classmethod
    def load(cls, path):
        # No pickle or executable model deserialization.
        with np.load(path, allow_pickle=False) as f:
            arrays = {k: f[k].copy() for k in
                      ("w1", "b1", "w2", "x_scale", "y_scale", "torque_limit")}
            return cls(**arrays, metadata=json.loads(str(f["metadata"])))


class MLPReflex(Reflex):
    name = "periodic_mlp"

    def __init__(self, model, dt=None):
        self.model = model
        model_dt = float(model.metadata["dt"])
        if dt is not None and not np.isclose(model_dt, dt):
            raise ValueError("Model timestep does not match the simulator")
        self.encoder = CausalFeatures(model.metadata["features"], model_dt,
                                      model.metadata.get("torque_ref", "hold"))   # old models: "hold"
        #encoder here is features

    def reset(self):
        self.encoder.reset()

    def compute(self, obs, priv=None):
        torque = self.model.predict(self.encoder.step(obs))
        # Same cancellation-torque sign convention as OracleReflex.
        return torque, torque.copy()
