import numpy as np
import pytest

from mlp_reflex import CausalFeatures, MLPModel, MLPReflex, features_from_log


def model(mode="history"):
    rng = np.random.default_rng(5)
    width = 28 if mode == "history" else 14
    return MLPModel(rng.normal(0, 0.1, (width, 8)), rng.normal(0, 0.1, 8),
                    rng.normal(0, 0.1, (8, 7)), np.ones(width), np.ones(7),
                    np.full(7, 12.0), {"features": mode, "dt": 0.001})


@pytest.mark.parametrize("mode", ["state", "history"])
def test_online_and_offline_features_are_identical(mode):
    # Protect the training/inference boundary: logged applied torque belongs to
    # the previous row, including after clipping by the simulator's limits.
    rng = np.random.default_rng(1)
    log = {key: rng.normal(size=(20, 7)) for key in ("q", "dq", "tau_base", "tau_applied")}
    encoder = CausalFeatures(mode)
    online = []
    for k in range(20):
        obs = {key: log[key][k] for key in ("q", "dq", "tau_base")}
        obs["tau_prev"] = log["tau_base"][0] if k == 0 else log["tau_applied"][k-1]
        online.append(encoder.step(obs))
    assert np.array_equal(online, features_from_log(log, mode, 0.001))


def test_saved_model_preserves_predictions_and_rejects_wrong_timestep(tmp_path):
    original = model()
    path = tmp_path / "model.npz"
    original.save(path)
    restored = MLPModel.load(path)
    x = np.random.default_rng(2).normal(size=(100, 28))
    assert np.array_equal(original.predict(x), restored.predict(x))
    assert np.array_equal(restored.predict(np.zeros(28)), np.zeros(7))
    with pytest.raises(ValueError, match="timestep"):
        MLPReflex(restored, dt=0.002)


def test_mlp_ignores_tick_and_privileged_teacher_and_resets_history():
    reflex = MLPReflex(model())
    rest = {"q": np.zeros(7), "dq": np.zeros(7), "tau_base": np.ones(7),
            "tau_prev": np.ones(7), "k": 0}
    changed = {**rest, "q": np.full(7, 0.01), "dq": np.full(7, 0.02)}
    outputs = []
    for k, teacher in ((10, np.ones(7)), (10000, np.full(7, 99))):
        reflex.reset()
        reflex.step({**rest, "k": k}, priv={"tau_teacher": teacher})
        outputs.append(reflex.step({**changed, "k": k+1}, priv={"tau_teacher": teacher})[0])
    assert np.array_equal(*outputs)
    assert np.linalg.norm(outputs[0]) > 0
    reflex.reset()
    assert np.array_equal(reflex.step(rest)[0], np.zeros(7))


def test_previous_torque_is_applied_torque_not_unlimited_command():
    from disturbance import Push
    from osc import DT, run_episode
    from reflex import Reflex

    class TorqueRecorder(Reflex):
        def __init__(self):
            self.previous = []

        def compute(self, obs, priv=None):
            self.previous.append(obs["tau_prev"].copy())
            return np.full(7, 100.), np.zeros(7)

    reflex = TorqueRecorder()
    p = Push(0, "hand", 50, 20, 2.0, (1., 0., 0.), DT)
    log = run_episode(p, reflex=reflex, t_end=0.1)
    assert np.array_equal(np.array(reflex.previous)[1:], log["tau_applied"][:-1])
    assert not np.array_equal(log["tau_cmd"], log["tau_applied"])


def test_mlp_is_causal_and_gain_zero_preserves_baseline():
    from disturbance import Push
    from osc import DT, run_episode
    p = Push(0, "hand", 100, 20, 2.0, (1., 0., 0.), DT)
    r = MLPReflex(model())
    log = run_episode(p, reflex=r, t_end=0.25)
    # At onset no sensor has seen this tick's newly applied external force.
    assert np.linalg.norm(log["tau_res_new"][:p.onset_tick+1]) < 1e-7
    assert np.linalg.norm(log["tau_res_new"][p.onset_tick+1]) > 0
    base = run_episode(p, t_end=0.25)
    disabled = run_episode(p, reflex=r, reflex_gain=0.0, t_end=0.25)
    assert np.array_equal(base["q"], disabled["q"])


@pytest.mark.parametrize("mode, width, hidden, macs, params", [
    ("state", 14, 2, 42, 44),
    ("history", 28, 3, 105, 108),
])
def test_cost_counts_follow_actual_layer_shapes(mode, width, hidden, macs, params):
    from scripts.mlp_cost import operation_counts
    m = MLPModel(np.ones((width, hidden), dtype=np.float32),
                 np.zeros(hidden, dtype=np.float32),
                 np.ones((hidden, 7), dtype=np.float32),
                 np.ones(width, dtype=np.float32), np.ones(7, dtype=np.float32),
                 np.full(7, 12.), {"features": mode, "dt": 0.001})
    counts = operation_counts(m)
    assert counts["dense_macs_per_tick"] == macs
    assert counts["dense_macs_per_s"] == macs * 1000
    assert counts["parameters"] == params
    assert counts["parameter_bytes"] == params * 4
    assert counts["calm_dense_macs_per_s"] == counts["pushed_dense_macs_per_s"]
    assert counts["extra_operations_per_tick"]["tanh_evaluations"] == hidden
    assert counts["full_step_scalar_operations_per_tick"]["divisions"] == (35 if mode == "history" else 14)
