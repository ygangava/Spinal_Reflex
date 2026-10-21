# Reflex interface. A reflex sees only what a real robot could sense (no push force) and returns two things:
#   tau_res  (7,) residual joint torque, added to the base OSC torque every tick
#   estimate (7,) slow summary for a higher layer: its current estimate of the external joint torque
# It also accepts a gain in [0, 1] from that higher layer: scales the torque only (0 = disabled).
# The estimate is reported even when disabled, so the higher layer can watch before enabling.

'''
many reflexes: the delayed oracle, the momentum observer, 
periodic and delta MLPs, the SNN, and later the FPGA co-simulation. 

run_episode never changes .. each reflex is own subclass with its own compute(obs)
don't need to touch osc.py
keeps own state between ticks
rules can be here, contorls what reflex sees, easier to test
'''



import numpy as np



class Reflex:
    name = "base"
    privileged = False   # True only for the oracle: it alone gets the sim's true push information

    def reset(self):
        # Clear internal state (membranes, filters, encoder references) at the start of an episode
        pass

    def compute(self, obs, priv=None):
        # Subclasses return (tau_res, estimate), both (7,). priv is None unless privileged
        raise NotImplementedError

    def step(self, obs, gain=1.0, priv=None):
        # Gain/enable applied here so every reflex obeys it the same way.
        # Privileged info is dropped here for ordinary reflexes, so they can't cheat even if handed it
        tau, est = self.compute(obs, priv if self.privileged else None)
        return gain * np.asarray(tau, dtype=float), np.asarray(est, dtype=float)


class NoReflex(Reflex):
    name = "none"

    def compute(self, obs, priv=None):
        return np.zeros(7), np.zeros(7)


class OracleReflex(Reflex):
    # Cheating upper bound: applies the exact torque that cancels the push (-J^T F, from the sim).
    # "Oracle with delay D" = this reflex + run_episode(reflex_delay=D); the delay is the hardware path, not the reflex
    name = "oracle"
    privileged = True

    def compute(self, obs, priv=None):
        tau = priv["tau_teacher"]
        return tau, tau
