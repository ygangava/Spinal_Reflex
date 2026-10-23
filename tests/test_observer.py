import numpy as np
import pytest

from disturbance import Push, sustained_pushes
from observer_reflex import MomentumObserver
from osc import DT, run_episode


QUIET = Push(0, "hand", 200, 20, 0.0, (1., 0., 0.), DT)
# Constant 30 N from tick 100: a clean step, so the observer's first-order response is easy to check
STEP = Push(0, "hand", 100, 300, None, (1., 0., 0.), DT, shape="rect", peak_force=30.0)


def test_silent_without_push():
    # Known torques fully explain the motion -> estimate stays at numerical zero
    log = run_episode(QUIET, reflex=MomentumObserver(100), reflex_gain=0.0, t_end=0.5)
    assert np.abs(log["reflex_est"]).max() < 1e-9


def test_not_privileged():
    assert MomentumObserver.privileged is False


def test_tracks_teacher_on_sustained_push():
    # Gain 0: observe only. Mid-push the estimate should equal the cancelling torque -J^T F
    p = sustained_pushes(0, DT)[13]
    log = run_episode(p, reflex=MomentumObserver(100), reflex_gain=0.0, t_end=(p.end_tick + 50) * DT)
    mid = p.onset_tick + p.dur_ticks // 2
    est, teacher = log["reflex_est"][mid], log["tau_teacher"][mid - 1]
    # Within 1%: the arm still drifts mid-push, so the teacher changes slightly and the 10 ms lag trails it
    assert np.linalg.norm(est - teacher) < 0.01 * np.linalg.norm(teacher)


@pytest.mark.parametrize("k_o", [50, 100, 200])
def test_first_order_lag(k_o):
    # Discrete first-order filter: n ticks after the push first shows up (onset + 1),
    # the estimate has covered 1 - (1 - k_o*DT)^(n+1) of the step
    log = run_episode(STEP, reflex=MomentumObserver(k_o), reflex_gain=0.0, t_end=0.4)
    est, teacher = log["reflex_est"], log["tau_teacher"]
    j = np.argmax(np.abs(teacher[STEP.onset_tick]))      # joint with the largest push torque
    assert abs(est[STEP.onset_tick, j]) < 1e-12          # causal: the push hasn't acted yet (float noise only)
    for n in (0, 5, 20):
        k = STEP.onset_tick + 1 + n
        expected = 1 - (1 - k_o * DT) ** (n + 1)
        assert est[k, j] / teacher[k - 1, j] == pytest.approx(expected, abs=0.01)


def test_gain_zero_is_baseline():
    p = sustained_pushes(0, DT)[13]
    t_end = (p.end_tick + 50) * DT
    base = run_episode(p, t_end=t_end)
    disabled = run_episode(p, reflex=MomentumObserver(100), reflex_gain=0.0, t_end=t_end)
    assert np.array_equal(base["q"], disabled["q"])


def test_reset_between_episodes():
    # Same observer object reused: second episode must match a fresh observer exactly
    r = MomentumObserver(100)
    run_episode(STEP, reflex=r, t_end=0.3)
    reused = run_episode(QUIET, reflex=r, t_end=0.3)
    fresh = run_episode(QUIET, reflex=MomentumObserver(100), t_end=0.3)
    assert np.array_equal(reused["reflex_est"], fresh["reflex_est"])


def test_dropping_damping_leaves_a_false_estimate():
    # Damping is a known torque; leaving it out reads arm motion as a push (step 7 ablation)
    p = sustained_pushes(0, DT)[13]
    t_end = (p.end_tick + 200) * DT
    full = run_episode(p, reflex=MomentumObserver(100), reflex_gain=0.0, t_end=t_end)
    no_damp = run_episode(p, reflex=MomentumObserver(100, damping=False), reflex_gain=0.0, t_end=t_end)
    after = slice(p.end_tick + 100, None)    # push is over: the true external torque is 0
    assert np.abs(full["reflex_est"][after]).max() < 0.02
    assert np.abs(no_damp["reflex_est"][after]).max() > 0.1
