import numpy as np

from reflex import Reflex
from mlp_reflex import CausalFeatures, MLPModel
from encoder import SendOnDelta

class DeltaMLPReflex(Reflex):
    name = "delta_mlp"
    
    def __init__(self, model, k):
        # store model, build features + encoder (theta = k * x_scale), then reset
        self.model = model
        self.k = k
        model_dt = float(model.metadata["dt"])

        self.features = CausalFeatures(model.metadata["features"], model_dt)
        self.enc = SendOnDelta(k * model.x_scale)
        
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
            # One event = ±k in scaled units
            
            # pre += k × (on − off)[fired] @ W1[fired]
            self.pre += self.k * (n[fired] @ m.w1[fired])
            
            # tanh is nonlinear, so the rest is recomputed (same math as mlpmodel.predict)
            
            hidden = np.tanh(self.pre)
            self.torque = np.clip((hidden @ m.w2 - m.rest_output) * m.y_scale, -m.torque_limit, m.torque_limit)
            self.ops.append(64 * fired.size + 64 * 7)
            
        else:
            self.ops.append(0) # no events no work, hold torque
        
        
        
        return self.torque, self.torque.copy()