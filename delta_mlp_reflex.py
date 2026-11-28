import numpy as np

from reflex import Reflex
from mlp_reflex import CausalFeatures, MLPModel
from encoder import SendOnDelta

def feature_noise(q_std, dq_std, dt, tau_std=0.0):
    # Sensor noise std -> std of each of the 28 features (q - q0, dq, acc, tau_prev - hold).
    # acc = (dq[k] - dq[k-1]) / dt: difference of two independent noises, divided by dt
    return np.r_[np.full(7, q_std), np.full(7, dq_std),
                 np.full(7, np.sqrt(2) * dq_std / dt), np.full(7, tau_std)]


class DeltaMLPReflex(Reflex):
    name = "delta_mlp"

    def __init__(self, model, k, floor=None):
        # store model, build features + encoder, then reset
        # theta = k * x_scale, raised to `floor` where noise is bigger (floor = c * feature_noise(...)).
        # The robot can measure its sensor noise at rest, so the floor isn't privileged.
        # Below theta, noise never moves the encoder's ref: the threshold is also a noise filter.
        self.model = model
        theta = k * model.x_scale if floor is None else np.maximum(k * model.x_scale, floor)
        # One event = this much change in scaled units, per channel (= k everywhere without a floor)
        self.event_size = theta / model.x_scale
        model_dt = float(model.metadata["dt"])

        self.features = CausalFeatures(model.metadata["features"], model_dt,
                                       model.metadata.get("torque_ref", "hold"))   # old models: "hold"
        self.enc = SendOnDelta(theta)
        
        # self.torque, ticks, compute_ticks are created inside reset, so need to call to make them exist now
        self.reset()

    def reset(self):
        # reset features and encoder; torque = zeros(7); ticks = compute_ticks = 0
        self.features.reset()
        self.enc.reset()
        self.torque = np.zeros(7)
        # self.ticks = 0
        # self.compute_ticks = 0
        self.pre = None
        self.ops = []

    def compute(self, obs, priv=None):
        # features -> encoder -> if any event: recompute torque from enc.ref
        # return torque, torque.copy()
        
        # get all 28 causal features from current sensor obs
        x = self.features.step(obs)
        
        # Pass through send on delta encoder to see if any channel passed theta
        on, off = self.enc.step(x)
        m = self.model

        # First tick: build pre once from the starting reference (a one-time full layer 1)
        if self.pre is None:
            # pre = z @ W1 + b1
            self.pre = (self.enc.ref / m.x_scale) @ m.w1 + m.b1
            self.ops.append(28 * 64)
            return self.torque, self.torque.copy()
        
        n = on - off
        fired = np.flatnonzero(n)
        
        if fired.size:
            # Linear layer -- only the changed rows contirbute
            # One event = ±event_size in scaled units

            # pre += (on − off)[fired] × event_size[fired] @ W1[fired]
            self.pre += (n[fired] * self.event_size[fired]) @ m.w1[fired]
            
            # tanh is nonlinear, so the rest is recomputed (same math as mlpmodel.predict)
            
            hidden = np.tanh(self.pre)
            self.torque = np.clip((hidden @ m.w2 - m.rest_output) * m.y_scale, -m.torque_limit, m.torque_limit)
            self.ops.append(64 * fired.size + 64 * 7)
            
        else:
            self.ops.append(0) # no events no work, hold torque
        
        
        
        return self.torque, self.torque.copy()