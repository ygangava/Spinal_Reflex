import numpy as np
import pytest

# file needs to be named test_*.py for pytest to find it

from disturbance import (Push, profile, push, make_set, IMPULSES, DIRECTIONS_DEG, DURATIONS_S, ONSET_S,
                         grid_pushes, heldout_pushes, HELDOUT_IMPULSES, HELDOUT_DIRECTIONS_DEG)

DT = 0.002 # 2 ms
SEED = 0

def make_push(shape="sine", dur_ticks = 10, impulse = 2.0):
    return Push(push_id=0, body="hand", onset_tick=100, dur_ticks=dur_ticks,
                impulse=impulse, direction=(1.0, 0.0, 0.0), dt=DT, shape=shape)
    
# Push window -- off just before onset, on at first and last tick, off right after
def test_push_window():
    p = make_push()
    last = p.onset_tick + p.dur_ticks -1
    assert np.all(push(p.onset_tick - 1, p) == 0)
    assert np.linalg.norm(push(p.onset_tick, p)) > 0
    assert np.linalg.norm(push(last, p)) > 0
    assert np.all(push(last + 1, p) == 0)
    
# Impulse: sum(force * dt) over the whole run equals p.impulse, 
# for each shape and duration
@pytest.mark.parametrize("shape", ["sine", "rect"]) # shapes
@pytest.mark.parametrize("dur_ticks", [10, 40])
def test_impulse(shape, dur_ticks):
    p = make_push(shape, dur_ticks)
    total = sum(push(k, p) for k in range(p.onset_tick + p.dur_ticks + 10)) * DT
    assert np.isclose(np.linalg.norm(total), p.impulse)
    
# Direction: unit length, horizontal, and the force points along it
def test_direction():
    for p in grid_pushes(SEED, DT):
        assert np.isclose(np.linalg.norm(p.direction), 1.0)
        assert p.direction[2] == 0.0
        f = push(p.onset_tick, p)
        assert np.allclose(f / np.linalg.norm(f), p.direction)

# Sizes: grid is 3x4x2, held-out is 2x4x2
def test_set_sizes():
    assert len(grid_pushes(SEED, DT)) == 24
    assert len(heldout_pushes(SEED, DT)) == 16


# Seed: same seed gives an identical list, a different seed does not
def test_seed_repeatable():
    assert grid_pushes(SEED, DT) == grid_pushes(SEED, DT)
    assert grid_pushes(SEED, DT) != grid_pushes(SEED + 1, DT)

# Held-out shares no impulse or base direction with the grid
def test_heldout_disjoint():
    assert not set(IMPULSES) & set(HELDOUT_IMPULSES)
    assert not set(DIRECTIONS_DEG) & set(HELDOUT_DIRECTIONS_DEG)
    grid_J = {p.impulse for p in grid_pushes(SEED, DT)}
    held_J = {p.impulse for p in heldout_pushes(SEED, DT)}
    assert not grid_J & held_J


# Onsets: every onset tick lies inside ONSET_S converted to ticks
def test_onset_range():
    lo, hi = round(ONSET_S[0] / DT), round(ONSET_S[1] / DT)
    for p in grid_pushes(SEED, DT) + heldout_pushes(SEED, DT):
        assert lo <= p.onset_tick <= hi
        



# ---- Sustained pushes and slow presses (trapezoid, scaled to peak force) ----
from disturbance import sustained_pushes, press_pushes, PRESS_RISES_S, PRESS_FORCE, PRESS_HOLD_S


# Trapezoid: ramps up over `rise` ticks, holds at 1, ramps down symmetrically
def test_trap_profile():
    prof = profile(100, "trap", rise=10)
    assert np.all(np.diff(prof[:10]) > 0) and np.all(prof[10:90] == 1.0)
    assert np.allclose(prof, prof[::-1])
    assert 0 < prof[0] < 0.1


# Peak force overrides impulse: max |force| equals peak_force exactly, onset/end window unchanged
def test_peak_force():
    p = Push(0, "hand", 100, 60, None, (0.0, 1.0, 0.0), DT, shape="trap", rise_ticks=10, peak_force=30.0)
    F = np.array([push(k, p) for k in range(p.end_tick + 5)])
    assert np.isclose(np.linalg.norm(F, axis=1).max(), 30.0)
    assert not F[:p.onset_tick].any() and not F[p.end_tick:].any()
    assert np.allclose(F[p.onset_tick:p.end_tick, 0], 0.0)


# Set sizes, repeatable seeds, and the press sweep's rise times / hold
def test_trap_sets():
    s, pr = sustained_pushes(SEED, DT), press_pushes(SEED, DT)
    assert len(s) == 36 and len(pr) == 28
    assert sustained_pushes(SEED, DT) == s and press_pushes(SEED, DT) == pr
    assert {p.rise_ticks for p in pr} == {round(r / DT) for r in PRESS_RISES_S}
    for p in pr:
        assert p.peak_force == PRESS_FORCE
        assert p.dur_ticks == 2 * p.rise_ticks + round(PRESS_HOLD_S / DT)
