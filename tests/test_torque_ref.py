import numpy as np
import pytest

from mlp_reflex import CausalFeatures, features_from_log

# The tau_prev channel's reference (SNN_LEARN.md Q7): "hold" = OSC torque at tick 0, "ff" = last tick's tau_ff.


def episode():
    from disturbance import Push
    from osc import DT, run_episode
    # 30 N sustained push, 150 ms, 10 ms ramps (same shape as the sustained set)
    p = Push(0, "hand", 100, 150, 0.0, (1., 0., 0.), DT, shape="trap", rise_ticks=10, peak_force=30.0)
    return p, run_episode(p, t_end=0.4)


# At tick 0 the arm is still and on target, so every feedback term is 0: tau_ff = tau_base exactly
# (up to rounding). Before the push the arm never moves, so they stay equal.
def test_tau_ff_equals_tau_base_while_holding():
    p, log = episode()
    assert np.allclose(log["tau_ff"][:p.onset_tick], log["tau_base"][:p.onset_tick], atol=1e-9)
    # During the push the OSC adds feedback, so tau_base pulls away from the feedforward
    assert np.abs(log["tau_base"] - log["tau_ff"])[p.onset_tick:].max() > 1.0


# Protect the training/inference boundary for the new mode, same as test_mlp's check for "hold"
def test_ff_online_equals_offline():
    rng = np.random.default_rng(1)
    log = {key: rng.normal(size=(20, 7)) for key in ("q", "dq", "tau_base", "tau_applied", "tau_ff")}
    f = CausalFeatures("history", torque_ref="ff")
    online = []
    for k in range(20):
        obs = {key: log[key][k] for key in ("q", "dq", "tau_base", "tau_ff")}
        obs["tau_prev"] = log["tau_base"][0] if k == 0 else log["tau_applied"][k - 1]
        online.append(f.step(obs))
    assert np.array_equal(online, features_from_log(log, "history", 0.001, torque_ref="ff"))


# On a real hold episode, "ff" changes only the tau channel (cols 21-27), and only a little:
# gravity changes as the arm is pushed away from q0. Before the push the two modes are identical.
def test_ff_vs_hold_on_a_hold_episode():
    p, log = episode()
    hold = features_from_log(log, "history", 0.001)
    ff = features_from_log(log, "history", 0.001, torque_ref="ff")
    assert np.array_equal(hold[:, :21], ff[:, :21])
    assert np.allclose(hold[:p.onset_tick + 1], ff[:p.onset_tick + 1], atol=1e-9)
    diff = np.abs(hold[:, 21:] - ff[:, 21:]).max()
    assert 0 < diff < 0.1 * np.abs(hold[:, 21:]).max()


# Default stays "hold", so existing models and data are unchanged; typos fail loudly
def test_default_and_bad_ref():
    assert CausalFeatures().torque_ref == "hold"
    with pytest.raises(ValueError):
        CausalFeatures(torque_ref="gravity")
    with pytest.raises(ValueError):
        features_from_log({}, "history", 0.001, torque_ref="gravity")


# Reflexes read the choice from the model's metadata; models saved before this change have no key -> "hold"
@pytest.mark.parametrize("meta, expected", [({}, "hold"), ({"torque_ref": "ff"}, "ff")])
def test_reflex_reads_torque_ref_from_metadata(meta, expected):
    from snn import SNNModel, SNNReflex
    rng = np.random.default_rng(0)
    m = SNNModel(rng.normal(0, 0.3, (56, 64)), rng.normal(0, 0.05, (64, 7)), 0.9375, 0.1,
                 np.ones(28), np.ones(7), np.full(7, 12.0), {"dt": 0.001, "features": "history", **meta})
    assert SNNReflex(m).features.torque_ref == expected
