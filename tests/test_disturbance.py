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
        

