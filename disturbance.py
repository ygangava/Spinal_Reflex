# Push disturbances. A push is data; push(k, p) turns it into a world-frame force (N) at tick k.
# No MuJoCo here: the sim loop writes the force into d.xfrc_applied[body, :3].
from dataclasses import dataclass
from itertools import product
import numpy as np

# For every push force x dt - sums to exactlyt he impulse
# force on only from onset_tick up
#push(k, p) takes a tick, not a tim


# Training/eval grid: 3 impulses x 4 directions x 2 durations = 24 pushes
IMPULSES = (2.0, 4.0, 7.5)            # N*s
DIRECTIONS_DEG = (0, 90, 180, 270)    # +x, +y, -x, -y, angle about world z
DURATIONS_S = (0.02, 0.08)
ONSET_S = (0.8, 1.5)
JITTER_DEG = 10.0

# Held-out set: impulses and directions the grid never uses
HELDOUT_IMPULSES = (3.0, 6.0)
HELDOUT_DIRECTIONS_DEG = (45, 135, 225, 315)


@dataclass(frozen=True)
class Push:
    push_id: int
    body: str
    onset_tick: int
    dur_ticks: int
    impulse: float       # N*s
    direction: tuple     # unit vector, world frame
    dt: float
    shape: str = "sine"  # "sine" or "rect"


def profile(n, shape):
    # Per-tick shape in [0, 1]; half-sine sampled at tick midpoints
    if shape == "rect":
        return np.ones(n)
    return np.sin(np.pi * (np.arange(n) + 0.5) / n)


# One push: returns a 3-vector force in world frame, or zero if not active
def push(k, p):
    # Active only for onset_tick <= k < onset_tick + dur_ticks
    j = k - p.onset_tick
    if not 0 <= j < p.dur_ticks:
        return np.zeros(3)

    # Scale so sum(force * dt) equals the impulse exactly
    prof = profile(p.dur_ticks, p.shape)
    f_peak = p.impulse / (p.dt * prof.sum())
    return f_peak * prof[j] * np.asarray(p.direction)


def make_set(impulses, directions_deg, seed, dt, body, shape):
    rng = np.random.default_rng(seed)
    lo, hi = round(ONSET_S[0] / dt), round(ONSET_S[1] / dt)
    pushes = []
    for i, (J, deg, dur) in enumerate(product(impulses, directions_deg, DURATIONS_S)):
        # Random onset tick and direction jitter about world z
        onset = int(rng.integers(lo, hi + 1))
        a = np.radians(deg + rng.uniform(-JITTER_DEG, JITTER_DEG))
        direction = (float(np.cos(a)), float(np.sin(a)), 0.0)
        pushes.append(Push(i, body, onset, round(dur / dt), J, direction, dt, shape))
    return pushes


def grid_pushes(seed, dt, body="hand", shape="sine"):
    return make_set(IMPULSES, DIRECTIONS_DEG, seed, dt, body, shape)


def heldout_pushes(seed, dt, body="hand", shape="sine"):
    return make_set(HELDOUT_IMPULSES, HELDOUT_DIRECTIONS_DEG, seed, dt, body, shape)


if __name__ == "__main__":
    for p in grid_pushes(seed=0, dt=0.002):
        print(p.push_id, p.onset_tick, p.dur_ticks, p.impulse, np.round(p.direction, 3))
