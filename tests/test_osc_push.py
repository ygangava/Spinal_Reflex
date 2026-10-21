# Sim-level checks: is the push applied the way disturbance.py says? (~0.5 s each)
import numpy as np
import pytest

from disturbance import Push
from osc import run_episode, TAU_MAX, DT, DEFAULT_GAINS, HIGH_KP_GAINS

ONSET = 500   # 1.0 s


# Fixed push: no jitter, known onset, on the hand, 20 ms half-sine
def hand_push(impulse, direction):
    return Push(push_id=0, body="hand", onset_tick=ONSET, dur_ticks=10,
                impulse=impulse, direction=direction, dt=DT)


# Peak hand error vs numbers from the earlier sweep (regression values, not ground truth)
@pytest.mark.parametrize("impulse, direction, expected_mm", [
    (2.0, (1.0, 0.0, 0.0), 9.0),
    (2.0, (0.0, 1.0, 0.0), 15.5),
    (7.5, (-1.0, 0.0, 0.0), 42.6),
])
def test_peak_error(impulse, direction, expected_mm):
    log = run_episode(hand_push(impulse, direction), t_end=2.5)
    peak = 1000 * np.linalg.norm(log["e_p"], axis=1).max()
    assert peak == pytest.approx(expected_mm, rel=0.05)


# Logged force integrates to the impulse, and nothing moves before the push
def test_logged_push():
    p = hand_push(2.0, (1.0, 0.0, 0.0))
    log = run_episode(p, t_end=2.5)
    assert np.isclose(np.linalg.norm(log["f_push"].sum(axis=0)) * DT, p.impulse)
    assert np.linalg.norm(log["e_p"][:ONSET], axis=1).max() < 1e-6


# Strong push: commanded torque exceeds limits, applied torque never does
def test_saturation_is_clipped():
    log = run_episode(hand_push(7.5, (-1.0, 0.0, 0.0)), t_end=2.5)
    assert (np.abs(log["tau_cmd"]) > TAU_MAX).any()
    assert (np.abs(log["tau_applied"]) <= TAU_MAX + 1e-9).all()


# Reset works: the same push twice in a row gives identical logs
def test_episodes_independent():
    p = hand_push(2.0, (0.0, 1.0, 0.0))
    a, b = run_episode(p, t_end=2.5), run_episode(p, t_end=2.5)
    assert all(np.array_equal(a[k], b[k]) for k in a)


# Higher KP -> smaller peak error on the same push
def test_high_kp_stiffer():
    p = hand_push(4.0, (1.0, 0.0, 0.0))
    peak = lambda g: np.linalg.norm(run_episode(p, g, t_end=2.5)["e_p"], axis=1).max()
    assert peak(HIGH_KP_GAINS) < peak(DEFAULT_GAINS)
