import numpy as np
import pytest

from snn import lif_step, layer_step, output_step, snn_step, SNNModel, SNNReflex

# Torch is only needed for training. If it's missing, the NumPy tests still run and the torch ones skip.
try:
    import torch
except ImportError:
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed")

BETA = 0.9375   # n = 4
H = 64          # hidden neurons


def make_model(w_in_std=0.3, w_out_std=0.05, seed=0, torque_limit=None):
    rng = np.random.default_rng(seed)
    limit = np.array([87., 87, 87, 87, 12, 12, 12]) if torque_limit is None else torque_limit
    return SNNModel(rng.normal(0, w_in_std, (56, H)), rng.normal(0, w_out_std, (H, 7)), BETA, 0.1,
                    np.ones(28), np.ones(7), limit, {"dt": 0.001, "features": "history"})


def random_log(T=60, seed=1):
    # Same shape as an osc.run_episode log, just random numbers (enough motion to make events).
    rng = np.random.default_rng(seed)
    return {key: rng.normal(size=(T, 7)) for key in ("q", "dq", "tau_base", "tau_applied")}


def obs_at(log, k):
    # What the reflex sees on tick k (tau_prev = last tick's applied torque, as in test_mlp).
    obs = {key: log[key][k] for key in ("q", "dq", "tau_base")}
    obs["tau_prev"] = log["tau_base"][0] if k == 0 else log["tau_applied"][k - 1]
    return obs


# ================================================================ Lesson 1: lif_step

# P1: constant 0.3 -> spikes on ticks 3, 7, 11, 14, 18; v = 0.092 right after the first spike
def test_lif_p1_constant_input():
    v, ticks = 0.0, []
    for t in range(20):
        v, s = lif_step(v, 0.3, BETA)
        if s:
            ticks.append(t)
        if t == 3:
            assert v == pytest.approx(0.0921, abs=1e-3)
    assert ticks == [3, 7, 11, 14, 18]


# P3: 2.5 once, then nothing -> spike on tick 0 (v = 1.5) and tick 1, then quiet
def test_lif_p3_big_input_spreads_over_ticks():
    v, s = lif_step(0.0, 2.5, BETA)
    assert s and v == pytest.approx(1.5)
    ticks = [0]
    for t in range(1, 10):
        v, s = lif_step(v, 0.0, BETA)
        if s:
            ticks.append(t)
    assert ticks == [0, 1]


# At most one spike per tick: only ONE threshold is subtracted, the rest stays in v
def test_lif_one_spike_per_tick():
    v, s = lif_step(0.0, 5.0, BETA)
    assert bool(s) and v == pytest.approx(4.0)


# Exactly at threshold spikes (>=, same in torch, fixed point and RTL)
def test_lif_spikes_at_exact_threshold():
    v, s = lif_step(0.0, 1.0, BETA)
    assert bool(s) and v == 0.0


# Array check from 1.9: each neuron decides for itself
def test_lif_array_elementwise():
    v, s = lif_step(np.array([0.5, 1.2, 2.5]), 0.0, BETA)
    assert np.array_equal(s, [False, True, True])
    assert np.allclose(v, [0.46875, 0.125, 1.34375])


# No input -> pure geometric leak, never spikes
def test_lif_leak_only():
    v = np.array([0.5, 0.9])
    for _ in range(10):
        v, s = lif_step(v, 0.0, BETA)
        assert not s.any()
    assert np.allclose(v, np.array([0.5, 0.9]) * BETA ** 10)


# The caller's v must not change (returns a new array)
def test_lif_not_in_place():
    v = np.array([0.5, 1.2])
    before = v.copy()
    lif_step(v, 0.3, BETA)
    assert np.array_equal(v, before)


# ================================================================ Lesson 2: layer_step

# Quiet tick: no events -> 0 ops, no spikes, v only leaks
def test_layer_quiet_tick():
    w_in = make_model().w_in
    v = np.full(H, 0.5)
    v_new, s, ops = layer_step(v, np.zeros(28, int), np.zeros(28, int), w_in, BETA)
    assert ops == 0 and not s.any()
    assert np.allclose(v_new, 0.5 * BETA)


# P4: on[3] = 2, off[12] = 1 -> current = 2·row 3 + row 40 (OFF rows start at 28); ops = 64 per count
def test_layer_p4_rows_and_ops():
    w_in = make_model().w_in
    on, off = np.zeros(28, int), np.zeros(28, int)
    on[3], off[12] = 2, 1
    v_new, s, ops = layer_step(np.zeros(H), on, off, w_in, BETA)
    expected_v, expected_s = lif_step(np.zeros(H), 2 * w_in[3] + w_in[40], BETA)
    assert np.allclose(v_new, expected_v)
    assert np.array_equal(s, expected_s)
    assert ops == 64 * 3


# ================================================================ Lesson 3: output_step (bucket with no hole)

# No spikes -> acc holds exactly (non-leaky), 0 ops
def test_output_holds_without_spikes():
    w_out = make_model().w_out
    acc = np.arange(7.0)
    for _ in range(5):
        acc, ops = output_step(acc, np.zeros(H, bool), w_out)
        assert ops == 0
    assert np.array_equal(acc, np.arange(7.0))


# Spikes add their W_out rows; 7 adds per spike
def test_output_adds_rows_of_spiked_neurons():
    w_out = make_model().w_out
    spike = np.zeros(H, bool)
    spike[[2, 10, 63]] = True
    acc, ops = output_step(np.zeros(7), spike, w_out)
    assert np.allclose(acc, w_out[2] + w_out[10] + w_out[63])
    assert ops == 7 * 3


# ================================================================ Phase B: snn_step

# snn_step = layer_step then output_step on the SAME tick's spikes, ops summed
def test_snn_step_is_layer_then_output():
    m = make_model()
    rng = np.random.default_rng(3)
    on, off = rng.integers(0, 3, 28), rng.integers(0, 3, 28)
    v0, acc0 = rng.uniform(0, 1, H), rng.normal(size=7)
    v, acc, s, ops = snn_step(v0, acc0, on, off, m.w_in, m.w_out, BETA)
    v_ref, s_ref, ops1 = layer_step(v0, on, off, m.w_in, BETA)
    acc_ref, ops2 = output_step(acc0, s_ref, m.w_out)
    assert np.array_equal(v, v_ref) and np.array_equal(s, s_ref) and np.array_equal(acc, acc_ref)
    assert ops == ops1 + ops2


# ================================================================ Phase B: SNNModel

# Save -> load gives back identical arrays and metadata
def test_model_save_load_roundtrip(tmp_path):
    m = make_model()
    path = tmp_path / "snn.npz"
    m.save(path)
    r = SNNModel.load(path)
    for name in ("w_in", "w_out", "x_scale", "y_scale", "torque_limit"):
        assert np.array_equal(getattr(m, name), getattr(r, name))
    assert (r.beta, r.k, r.metadata) == (m.beta, m.k, m.metadata)


# Wrong shapes / bad beta / bad k fail at construction, not mid-episode
@pytest.mark.parametrize("change", [
    {"w_in": np.zeros((28, H))},         # forgot the OFF half
    {"w_out": np.zeros((H, 6))},
    {"x_scale": np.ones(14)},
    {"beta": 1.0},                       # no leak = not LIF
    {"beta": 0.0},
    {"k": 0.0},
])
def test_model_rejects_bad_inputs(change):
    m = make_model()
    fields = {**m.__dict__, **change}
    with pytest.raises(ValueError):
        SNNModel(**fields)


# ================================================================ Phase B: SNNReflex

# A held arm makes no events: zero torque, 0 lazy ops, 64 eager ops (leak) every tick
def test_reflex_calm_is_silent():
    r = SNNReflex(make_model())
    calm = {"q": np.zeros(7), "dq": np.zeros(7), "tau_base": np.ones(7), "tau_prev": np.ones(7)}
    for _ in range(20):
        tau, _ = r.step(calm)
        assert np.array_equal(tau, np.zeros(7))
    assert r.ops_lazy == [0] * 20
    assert r.ops == [64] * 20


# Online reflex == offline path used for training (features_from_log -> encode -> snn_step -> clip).
# This is the training/inference boundary: if it breaks, the trained weights see different events online.
def test_reflex_matches_offline_training_path():
    from mlp_reflex import features_from_log
    from scripts.snn_data import encode
    m, log = make_model(), random_log()
    T = len(log["q"])
    r = SNNReflex(m)
    online = np.array([r.step(obs_at(log, k))[0] for k in range(T)])

    x = features_from_log(log, "history", 0.001)
    ev = encode(x[0], x[1:], m.k * m.x_scale)    # tick 0 only sets the encoder reference
    v, acc, lim, offline = np.zeros(H), np.zeros(7), m.torque_limit / m.y_scale, []
    for t in range(len(ev)):
        v, acc, _, _ = snn_step(v, acc, ev[t, :28], ev[t, 28:], m.w_in, m.w_out, m.beta)
        acc = np.clip(acc, -lim, lim)
        offline.append(acc * m.y_scale)

    assert np.array_equal(online[0], np.zeros(7))
    assert np.array_equal(online[1:], np.array(offline))
    assert np.abs(online).sum() > 0             # the test is only meaningful if something fired


# Saturating acc: torque never exceeds the limit, and it does reach it with huge weights
def test_reflex_torque_saturates():
    m = make_model(w_in_std=2.0, w_out_std=10.0)
    log = random_log(T=100)
    r = SNNReflex(m)
    taus = np.array([r.step(obs_at(log, k))[0] for k in range(100)])
    assert np.all(np.abs(taus) <= m.torque_limit + 1e-12)
    assert np.isclose(np.abs(taus), m.torque_limit).any()


# reset() clears v, acc, encoder and features: the same episode twice gives the same torques
def test_reflex_reset_repeats_episode():
    log = random_log()
    r = SNNReflex(make_model())
    runs = []
    for _ in range(2):
        r.reset()
        runs.append(np.array([r.step(obs_at(log, k))[0] for k in range(len(log["q"]))]))
    assert np.array_equal(*runs)


# Not privileged: handing it the teacher torque changes nothing (it can't cheat)
def test_reflex_ignores_teacher():
    log = random_log()
    outs = []
    for teacher in (np.zeros(7), np.full(7, 99.0)):
        r = SNNReflex(make_model())
        outs.append(np.array([r.step(obs_at(log, k), priv={"tau_teacher": teacher})[0]
                              for k in range(len(log["q"]))]))
    assert np.array_equal(*outs)


# ================================================================ Lesson 6: snn_torch

# Forward: 0/1 floats, same dtype as the input, spikes at exactly 0 (>=)
@needs_torch
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_spikefn_forward(dtype):
    from snn_torch import SpikeFn
    x = torch.tensor([-0.5, -1e-6, 0.0, 0.3], dtype=getattr(torch, dtype))
    s = SpikeFn.apply(x)
    assert s.dtype == x.dtype
    assert s.tolist() == [0.0, 0.0, 1.0, 1.0]


# Backward: .sum() makes grad_out = 1, so x.grad is the surrogate itself, 1/(1+10|x|)²
@needs_torch
def test_spikefn_backward_is_surrogate():
    from snn_torch import SpikeFn
    x = torch.tensor([-0.1, 0.0, 0.3], dtype=torch.float64, requires_grad=True)
    SpikeFn.apply(x).sum().backward()
    assert torch.allclose(x.grad, torch.tensor([0.25, 1.0, 0.0625], dtype=torch.float64))


# Backward multiplies by grad_out (chain rule), it doesn't replace it
@needs_torch
def test_spikefn_backward_scales_with_grad_out():
    from snn_torch import SpikeFn
    x = torch.tensor([-0.1, 0.0, 0.3], dtype=torch.float64, requires_grad=True)
    (3.0 * SpikeFn.apply(x)).sum().backward()
    assert torch.allclose(x.grad, 3.0 * torch.tensor([0.25, 1.0, 0.0625], dtype=torch.float64))


# P1 in torch: same spikes and same v as NumPy, every tick
@needs_torch
def test_lif_torch_matches_numpy_p1():
    from snn_torch import lif_step_torch
    v_t, v_n = torch.zeros(1, dtype=torch.float64), 0.0
    for _ in range(20):
        v_t, s_t = lif_step_torch(v_t, torch.tensor([0.3], dtype=torch.float64), BETA)
        v_n, s_n = lif_step(v_n, 0.3, BETA)
        assert bool(s_t.item()) == bool(s_n)
        assert v_t.item() == pytest.approx(float(v_n), abs=1e-12)


# Random batch of 64 neurons over 200 ticks: torch == NumPy
@needs_torch
def test_lif_torch_matches_numpy_random():
    from snn_torch import lif_step_torch
    rng = np.random.default_rng(4)
    cur = rng.normal(0.1, 0.5, (200, 3, H))      # (ticks, batch, neurons)
    v_n = np.zeros((3, H))
    v_t = torch.zeros(3, H, dtype=torch.float64)
    for t in range(200):
        v_n, s_n = lif_step(v_n, cur[t], BETA)
        v_t, s_t = lif_step_torch(v_t, torch.from_numpy(cur[t]), BETA)
        assert np.array_equal(s_t.numpy().astype(bool), s_n)
        assert np.allclose(v_t.numpy(), v_n, atol=1e-12)


# Q6: reset is detached, so d v_new / d v_old = beta exactly, even right at the threshold.
# (Not detached it would be beta·(1 - surrogate), which is ~0 for neurons near threshold.)
@needs_torch
def test_lif_torch_reset_is_detached():
    from snn_torch import lif_step_torch
    v = torch.tensor([0.5, 1.0 / BETA, 1.2], dtype=torch.float64, requires_grad=True)   # middle lands exactly on 1
    v_new, _ = lif_step_torch(v, torch.zeros(3, dtype=torch.float64), BETA)
    v_new.sum().backward()
    assert torch.allclose(v.grad, torch.full((3,), BETA, dtype=torch.float64))


# The spike's gradient w.r.t. the input current is the surrogate at (beta·v + current − 1)
@needs_torch
def test_lif_torch_spike_grad_wrt_current():
    from snn_torch import lif_step_torch, SpikeFn
    v = torch.tensor([0.2, 0.9, 1.5], dtype=torch.float64)
    cur = torch.tensor([0.1, 0.2, 0.0], dtype=torch.float64, requires_grad=True)
    _, s = lif_step_torch(v, cur, BETA)
    s.sum().backward()
    x = BETA * v + cur.detach() - 1.0
    assert torch.allclose(cur.grad, 1.0 / (1.0 + SpikeFn.SLOPE * x.abs()) ** 2)


# ================================================================ Lesson 6: training harness (scripts/snn_train.py)

# The torch network (training copy) == your NumPy snn_step + clip (deployed copy), same weights.
# Small acc limit so the clamp is exercised too. Same idea as agreement(), without the data files.
@needs_torch
def test_train_net_matches_numpy():
    from scripts.snn_train import SNN
    m = make_model()
    lim = np.full(7, 0.3)
    rng = np.random.default_rng(6)
    ev = rng.poisson(0.3, (80, 56))
    net = SNN(BETA, lim, 0).double()
    net.w_in.data, net.w_out.data = torch.from_numpy(m.w_in), torch.from_numpy(m.w_out)
    with torch.no_grad():
        out_t, rate = net(torch.from_numpy(ev[None].astype(np.float64)))
    v, acc, out_n = np.zeros(H), np.zeros(7), []
    for t in range(len(ev)):
        v, acc, _, _ = snn_step(v, acc, ev[t, :28], ev[t, 28:], m.w_in, m.w_out, BETA)
        acc = np.clip(acc, -lim, lim)
        out_n.append(acc)
    # atol 1e-6, not 1e-9: SNN stores acc_limit as float32 (0.3 -> 0.30000001), so clamped ticks differ by ~1e-8
    assert np.allclose(out_t[0].numpy(), np.array(out_n), atol=1e-6)
    assert np.isclose(np.abs(out_t.numpy()), 0.3).any()     # clamp was actually hit
    assert 0.0 < rate.item() <= 1.0


# BPTT gives both trained matrices a nonzero gradient (the surrogate actually lets signal through)
@needs_torch
def test_train_net_gradients_reach_both_weights():
    from scripts.snn_train import SNN
    net = SNN(BETA, np.full(7, 100.0), 0)                    # big limit: clamp mustn't block the gradient
    E = torch.from_numpy(np.random.default_rng(7).poisson(0.3, (2, 40, 56)).astype(np.float32))
    pred, rate = net(E)
    (pred.square().mean() + rate).backward()
    assert net.w_in.grad.abs().sum() > 0
    assert net.w_out.grad.abs().sum() > 0


# Padding ticks (mask 0) don't count in the loss, however wrong they are
@needs_torch
def test_masked_mse_ignores_padding():
    from scripts.snn_train import masked_mse
    Y = torch.zeros(1, 10, 7)
    M = torch.zeros(1, 10)
    M[0, :6] = 1.0
    pred = torch.zeros(1, 10, 7)
    pred[0, 6:] = 100.0                                      # garbage only in the padding
    assert masked_mse(pred, Y, M).item() == 0.0
    pred[0, 0] = 1.0                                         # one real tick off by 1 on all 7 joints
    assert masked_mse(pred, Y, M).item() == pytest.approx(1.0 / 6)


# ================================================================ Lesson 7.7: noise-aware training (floor θ, rescaled W_in)

# With a floor, SNNReflex folds θ/θ_k into W_in; training scales the event counts by the same ratio instead.
# Both must give the same torque (same boundary as test_reflex_matches_offline_training_path, now with a floor).
def test_reflex_with_floor_matches_training_path():
    from mlp_reflex import features_from_log
    from scripts.snn_data import encode
    m, log = make_model(), random_log()
    theta_k = m.k * m.x_scale
    floor = np.r_[np.zeros(14), np.full(7, 0.5), np.zeros(7)]   # raise only the acc channels (0.1 -> 0.5)
    r = SNNReflex(m, floor)
    T = len(log["q"])
    online = np.array([r.step(obs_at(log, k))[0] for k in range(T)])

    x = features_from_log(log, "history", 0.001)
    theta = np.maximum(theta_k, floor)
    ratio = theta / theta_k
    ev = encode(x[0], x[1:], theta) * np.r_[ratio, ratio]
    v, acc, lim, offline = np.zeros(H), np.zeros(7), m.torque_limit / m.y_scale, []
    for t in range(len(ev)):
        v, acc, _, _ = snn_step(v, acc, ev[t, :28], ev[t, 28:], m.w_in, m.w_out, m.beta)
        acc = np.clip(acc, -lim, lim)
        offline.append(acc * m.y_scale)
    assert np.allclose(online[1:], np.array(offline), atol=1e-9)
    assert np.abs(online).sum() > 0


# No floor -> the rescale is exactly 1: W_in is untouched (old results stay bit-identical)
def test_reflex_without_floor_keeps_w_in():
    m = make_model()
    assert np.array_equal(SNNReflex(m).w_in, m.w_in)


# add_noise: zero noise changes nothing; with noise, each feature group gets the noise CausalFeatures would make
def test_add_noise_shapes_and_sizes():
    from scripts.snn_train import add_noise, DT
    rng = np.random.default_rng(0)
    e = {"x0": np.zeros(28), "x": np.zeros((5000, 28)), "kind": "sustained", "end": 10}
    assert np.array_equal(add_noise(e, 0.0, rng)["x"], e["x"])
    dq_std = 1e-3
    x = add_noise(e, dq_std, rng)["x"]
    q_std = dq_std * DT / np.sqrt(2)
    # nq[t] − nq[0]: over time, nq[0] is a constant offset per joint, so each joint's spread is just q_std
    assert np.allclose(x[:, 0:7].std(axis=0), q_std, rtol=0.05)
    assert x[:, 7:14].std() == pytest.approx(dq_std, rel=0.05)
    assert x[:, 14:21].std() == pytest.approx(np.sqrt(2) * dq_std / DT, rel=0.05)  # the finite-difference acc
    assert np.array_equal(x[:, 21:], np.zeros((5000, 7)))                         # tau_prev: no noise
    assert not np.shares_memory(x, e["x"])
