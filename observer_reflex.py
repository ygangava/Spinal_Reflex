import mujoco
import numpy as np

from osc import MODEL, DT
from reflex import Reflex


class MomentumObserver(Reflex):
    name = "momentum_observer"
    # privileged stays False (inherited): uses only the robot model + sensed state, never the push

    def __init__(self, k_o, xml=MODEL, damping=True, mdot=True):
        # Own model + data, separate from the sim's osc.m / osc.d:
        #   - reading/writing them here can't disturb the simulation
        #   - model mismatch later = pass a different xml (wrong payload / link masses)
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        # Start from the same keyframe as the sim so the finger joints (not in obs) match
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("home").id)
        self.M_full = np.zeros((self.m.nv, self.m.nv))  # scratch buffer for mj_fullM (all 9 dofs incl. fingers)
        self.k_o = k_o   # observer gain (1/s): estimate lags the true push with time constant 1/k_o
        self.dt = DT
        # Ablation switches (step 7): drop a known-torque term to see what it costs. Both on = the real observer
        self.damping, self.mdot = damping, mdot
        self.reset()

    def reset(self):
        # Everything carried between ticks. p0 = None marks "first tick of the episode"
        self.p0 = None     # momentum at the start (the integral is measured from here)
        self.I = None      # running integral of the known torques (+ r), i.e. the predicted momentum change
        self.r = None      # current estimate of the external torque
        self.prev = None   # (M, b, passive, dq) from the previous tick

    def compute(self, obs, priv=None):
        # every tick we need compute M, b, and passive
        q, dq = np.asarray(obs["q"]), np.asarray(obs["dq"])
        M, b, passive = self.dynamics(q, dq)
        p = M @ dq    # measured generalized momentum

        if self.p0 is None:
            # First tick: no previous tick to integrate yet. Assume no push so far (r = 0)
            self.p0, self.I, self.r = p, np.zeros(7), np.zeros(7)
        else:
            # Torques that acted DURING the previous tick, i.e. what moved dq[k-1] -> dq[k]:
            #   tau_prev            motor torque actually applied (after limits)
            #   - b, + passive      gravity/Coriolis and damping, at the previous state
            #   + Mdot @ dq         d(M dq)/dt = M ddq + Mdot dq; M changes slowly, so differencing it is safe
            #   + r                 the observer's own estimate, fed back (gives dr/dt = k_o (tau_ext - r))
            M_p, b_p, passive_p, dq_p = self.prev
            Mdot_dq = (M - M_p) / self.dt @ dq_p if self.mdot else 0.0
            passive_p = passive_p if self.damping else 0.0
            self.I = self.I + self.dt * (obs["tau_prev"] - b_p + passive_p + Mdot_dq + self.r)
            # Measured momentum change minus predicted change = what the known torques can't explain = the push
            self.r = self.k_o * (p - self.p0 - self.I)

        self.prev = (M, b, passive, dq.copy())
        # r estimates the external torque; cancelling it means applying -r (same sign as the oracle's -J^T F)
        return -self.r, -self.r

    def dynamics(self, q, dq):
        # The robot's model at the measured state -> M (7x7), b (7,), passive (7,)
        self.d.qpos[:7], self.d.qvel[:7] = q, dq
        mujoco.mj_forward(self.m, self.d)   # recompute everything that depends on qpos/qvel

        # Dense inertia matrix; mj_fullM's signature changed across MuJoCo versions (same as osc.full_inertia)
        if hasattr(self.d, "qM"):
            mujoco.mj_fullM(self.m, self.M_full, self.d.qM)
        else:
            mujoco.mj_fullM(self.m, self.d, self.M_full)

        # .copy(): MuJoCo reuses these buffers, so the next call would overwrite the values we stored
        M = self.M_full[:7, :7].copy()        # includes armature (0.1 per joint)
        b = self.d.qfrc_bias[:7].copy()       # gravity + Coriolis/centrifugal
        passive = self.d.qfrc_passive[:7].copy()  # joint damping (-1.0 * dq here); a known force, not a push
        return M, b, passive


'''
The momentum observer works like this.
The robot knows its own physics: how heavy each link is (M),
what gravity and motion pull on it (b), and what torque its motors applied (τ).
From those, it can predict how much momentum the arm should have if nobody touched it.
It also measures how much momentum the arm actually has, from its joint speeds.
If someone pushes the arm, those two numbers drift apart,
and the size of the gap tells you how hard the push was.
Instead of reacting to the raw gap all at once,
the observer keeps a running guess r of the push
and updates it a little each tick toward whatever the gap says,
so the guess rises smoothly to the true push force over about 1/K_O seconds.
The reflex then applies −r to cancel the push. It never has to compute acceleration,
which is noisy, and it needs no training, because it's just physics bookkeeping.
'''
