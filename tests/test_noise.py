# Sensor-noise wrapper: zero noise changes nothing, seeds reproduce, the inner reflex sees the right noise
import numpy as np

from disturbance import sustained_pushes
from osc import run_episode, DT
from reflex import Reflex, NoisyObs, OracleReflex

P = sustained_pushes(0, DT)[0]
T_END = (P.end_tick + 100) * DT


# Test-only reflex: records what it was shown, outputs a small torque from dq so noise reaches the arm
class Recorder(Reflex):
    name = "recorder"

    def reset(self):
        self.seen = []

    def compute(self, obs, priv=None):
        self.seen.append({k: np.array(obs[k]) for k in ("q", "dq", "tau_prev")})
        return -0.5 * np.asarray(obs["dq"]), np.zeros(7)


# Zero std -> bit-identical to the unwrapped reflex
def test_zero_noise_is_identity():
    a = run_episode(P, reflex=Recorder(), t_end=T_END)
    b = run_episode(P, reflex=NoisyObs(Recorder()), t_end=T_END)
    assert all(np.array_equal(a[k], b[k]) for k in a)


# Same seed -> same run; different seed -> different run
def test_seed_reproducible():
    runs = [run_episode(P, reflex=NoisyObs(Recorder(), dq_std=1e-3, seed=s), t_end=T_END)
            for s in (1, 1, 2)]
    assert np.array_equal(runs[0]["q"], runs[1]["q"])
    assert not np.array_equal(runs[0]["q"], runs[2]["q"])


# The inner reflex's dq minus the true dq has the requested std; tau_prev is untouched when tau_std = 0
def test_inner_sees_requested_noise():
    inner = Recorder()
    log = run_episode(P, reflex=NoisyObs(inner, dq_std=1e-2, seed=3), t_end=T_END)
    noise = np.array([s["dq"] for s in inner.seen]) - log["dq"]
    assert abs(noise.std() - 1e-2) < 5e-4 and abs(noise.mean()) < 5e-4
    tau_seen = np.array([s["tau_prev"] for s in inner.seen])
    assert np.array_equal(tau_seen[1:], log["tau_applied"][:-1])


# Privilege passes through: a wrapped oracle still gets the teacher, so it still cancels the push
def test_privilege_passes_through():
    clean = run_episode(P, reflex=OracleReflex(), t_end=T_END)
    noisy = run_episode(P, reflex=NoisyObs(OracleReflex(), dq_std=1e-2), t_end=T_END)
    assert np.array_equal(clean["e_p"], noisy["e_p"])
