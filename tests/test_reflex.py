# Reflex interface: gain scales only the torque, the estimate is always reported, the residual reaches the sim
import mujoco
import numpy as np
import pytest

from disturbance import Push
from osc import run_episode, DT, m, d
from reflex import Reflex, NoReflex, OracleReflex

TAU = np.array([0.5, -0.5, 0.2, 0.0, 0.1, 0.0, 0.0])
EST = np.arange(7, dtype=float)


# Test-only reflex: fixed torque and fixed estimate, counts resets
class ConstantReflex(Reflex):
    name = "constant"

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def compute(self, obs, priv=None):
        return TAU, EST


def quiet_push():
    return Push(push_id=0, body="hand", onset_tick=round(0.5 / DT), dur_ticks=round(0.02 / DT),
                impulse=2.0, direction=(1.0, 0.0, 0.0), dt=DT)


# Gain scales the torque; the estimate ignores the gain
def test_gain_scales_torque_only():
    r = ConstantReflex()
    for g in [0.0, 0.5, 1.0]:
        tau, est = r.step({}, g)
        assert np.allclose(tau, g * TAU)
        assert np.array_equal(est, EST)


# NoReflex outputs zeros for both
def test_noreflex_zero():
    tau, est = NoReflex().step({}, 1.0)
    assert not tau.any() and not est.any()


# In the sim: tau_cmd = tau_base + gain * torque every tick, estimate logged, reset called once
def test_residual_reaches_sim():
    r = ConstantReflex()
    log = run_episode(quiet_push(), reflex=r, reflex_gain=0.5, t_end=1.0)
    assert r.resets == 1
    assert np.allclose(log["tau_res"], 0.5 * TAU)
    assert np.allclose(log["tau_cmd"], log["tau_base"] + 0.5 * TAU)
    assert np.allclose(log["reflex_est"], EST)


# Gain 0 = disabled: identical physics to no reflex, but the estimate is still logged
def test_gain_zero_is_disabled():
    a = run_episode(quiet_push(), t_end=1.0)
    b = run_episode(quiet_push(), reflex=ConstantReflex(), reflex_gain=0.0, t_end=1.0)
    assert np.array_equal(a["q"], b["q"])
    assert np.allclose(b["reflex_est"], EST)


def sustained_push(dur_ms=80, impulse=4.0):
    return Push(push_id=0, body="hand", onset_tick=round(0.5 / DT), dur_ticks=round(dur_ms / 1000 / DT),
                impulse=impulse, direction=(1.0, 0.0, 0.0), dt=DT)


# Teacher = -J^T F, checked against an independent Jacobian at the logged pose
def test_teacher_matches_jacobian():
    p = sustained_push()
    log = run_episode(p, t_end=0.7)
    b = m.body(p.body).id
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for k in range(p.onset_tick, p.onset_tick + p.dur_ticks, 7):
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        d.qpos[:7] = log["q"][k]
        mujoco.mj_forward(m, d)
        mujoco.mj_jac(m, d, jp, jr, d.xipos[b], b)
        assert np.allclose(log["tau_teacher"][k], -(jp[:, :7].T @ log["f_push"][k]), atol=1e-9)
    assert not log["tau_teacher"][:p.onset_tick].any()


# Ordinary reflexes never see the force or the teacher, even if run_episode hands it over
class SpyReflex(Reflex):
    def __init__(self):
        self.seen = []

    def compute(self, obs, priv=None):
        self.seen.append((set(obs), priv))
        return np.zeros(7), np.zeros(7)


def test_no_cheating():
    r = SpyReflex()
    run_episode(sustained_push(), reflex=r, t_end=0.7)
    assert all(keys == {"k", "q", "dq", "e_p", "e_o", "tau_base"} and priv is None for keys, priv in r.seen)
    assert SpyReflex().step({}, 1.0, priv={"tau_teacher": np.ones(7)})[0].sum() == 0


# Oracle with no delay cancels a sustained push almost exactly
def test_oracle_cancels():
    log = run_episode(sustained_push(), reflex=OracleReflex(), t_end=1.0)
    assert 1000 * np.linalg.norm(log["e_p"], axis=1).max() < 0.05
    assert np.allclose(log["tau_res"], log["tau_teacher"])


# Latency queue: output computed at k is applied at k + D; D = 0 means no delay
@pytest.mark.parametrize("D", [0, 5])
def test_delay_queue(D):
    log = run_episode(quiet_push(), reflex=ConstantReflex(), reflex_delay=D, t_end=0.2)
    assert not log["tau_res"][:D].any() and not log["reflex_est"][:D].any()
    assert np.allclose(log["tau_res"][D:], TAU)
    assert np.allclose(log["tau_res_new"], TAU)


# Delayed oracle regression (FRAMING.md §3b): 4 N*s, 80 ms, hand +x, D = 10 ticks (10 ms) -> ~2.9 mm
def test_delayed_oracle_regression():
    log = run_episode(sustained_push(), reflex=OracleReflex(), reflex_delay=round(0.010 / DT), t_end=1.5)
    assert 1000 * np.linalg.norm(log["e_p"], axis=1).max() == pytest.approx(2.9, rel=0.05)
