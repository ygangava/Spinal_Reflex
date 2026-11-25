# Send-on-delta encoder: silent when steady, counts whole thresholds, ref tracks the input
# Output hold: moves by delta per event, holds between events, inverts the encoder
import numpy as np

from encoder import SendOnDelta, OutputHold

# Powers of two, so threshold sums are exact in floating point
THETA = np.array([0.5, 0.25, 1.0])


# Steady input -> no events, including the first tick
def test_constant_input_is_silent():
    enc = SendOnDelta(THETA)
    for _ in range(100):
        on, off = enc.step([1.0, -2.0, 3.0])
        assert not on.any() and not off.any()


# Rising by exactly one threshold per tick -> one ON per tick
def test_ramp_one_event_per_tick():
    enc = SendOnDelta(THETA)
    enc.step(np.zeros(3))
    for k in range(1, 20):
        on, off = enc.step(k * THETA)
        assert np.array_equal(on, [1, 1, 1]) and not off.any()


# A 3.5θ jump -> 3 events; the leftover 0.5θ stays in ref until it grows to θ
def test_jump_counts_whole_thresholds():
    for sign in (1, -1):
        enc = SendOnDelta(THETA)
        enc.step(np.zeros(3))
        on, off = enc.step(sign * 3.5 * THETA)
        up, down = (on, off) if sign > 0 else (off, on)
        assert np.array_equal(up, [3, 3, 3]) and not down.any()
        assert np.allclose(enc.ref, sign * 3 * THETA)


# Random signal -> ref stays within one threshold, so events alone rebuild the input
def test_ref_tracks_input():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 0.3, (1000, 3)), axis=0)
    enc = SendOnDelta(THETA)
    rebuilt = None
    for xk in x:
        on, off = enc.step(xk)
        rebuilt = xk.copy() if rebuilt is None else rebuilt + (on - off) * THETA
        assert (np.abs(xk - enc.ref) < THETA).all()
        assert np.allclose(rebuilt, enc.ref)


# reset() forgets ref, so the next sample is a fresh start
def test_reset():
    enc = SendOnDelta(THETA)
    enc.step(np.zeros(3))
    enc.reset()
    on, off = enc.step(10 * THETA)
    assert not on.any() and not off.any()


# No events -> the output holds its value (a steady push stays cancelled)
def test_hold_without_events():
    hold = OutputHold(THETA)
    hold.step([2, 0, 1], [0, 3, 0])
    expected = np.array([1.0, -0.75, 1.0])
    for _ in range(1000):
        assert np.allclose(hold.step(np.zeros(3), np.zeros(3)), expected)


# Encoder -> hold with delta = theta rebuilds the input's change since the first tick
def test_hold_inverts_encoder():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 0.3, (1000, 3)), axis=0)
    enc, hold = SendOnDelta(THETA), OutputHold(THETA)
    for xk in x:
        out = hold.step(*enc.step(xk))
        assert np.allclose(out, enc.ref - x[0])


# Risk 17: tau_prev contains the reflex's own output, so an output change becomes
# an input event next tick. With no network reacting to it, that is one burst, then
# silence. The chatter test (network reacting to its own echo) needs the SNN.
def test_output_echo_is_one_burst():
    hold, tau_enc = OutputHold([1.0]), SendOnDelta([0.5])
    tau_enc.step(hold.value)
    out = hold.step([3], [0])
    counts = [tau_enc.step(out)[0][0]]
    for _ in range(50):
        counts.append(tau_enc.step(hold.step([0], [0]))[0][0])
    assert counts[0] == 6 and sum(counts[1:]) == 0
