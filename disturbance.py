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


# Sustained pushes: fast rise, hold, fast fall. 3 forces x 4 directions x 3 durations = 36
SUSTAINED_FORCES = (15.0, 30.0, 45.0)     # N, peak
SUSTAINED_DURATIONS_S = (0.1, 0.2, 0.3)   # total, including rise and fall
SUSTAINED_RISE_S = 0.01

# Slow presses: same trapezoid, but the rise time is swept (impact-like -> deliberate). 7 rises x 4 directions = 28
PRESS_FORCE = 30.0                         # N, peak
PRESS_RISES_S = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
PRESS_HOLD_S = 2.0   # long enough to settle: after a fast rise the arm needs ~0.5-1 s to reach F/K


@dataclass(frozen=True)
class Push:
    push_id: int
    body: str
    onset_tick: int
    dur_ticks: int
    impulse: float       # N*s; ignored if peak_force is set
    direction: tuple     # unit vector, world frame
    dt: float
    shape: str = "sine"  # "sine", "rect" or "trap"
    rise_ticks: int = 0  # "trap" only: ramp-up length (ramp-down is the same)
    peak_force: float = None  # N; if set, scale to this peak instead of to the impulse

    @property
    def end_tick(self):
        # First tick after the push
        return self.onset_tick + self.dur_ticks


def profile(n, shape, rise=0):
    # Per-tick shape in [0, 1], sampled at tick midpoints
    if shape == "rect":
        return np.ones(n)
    if shape == "trap":
        # Linear ramp up over `rise` ticks, hold at 1, mirror-image ramp down
        assert 0 < rise and 2 * rise <= n, "trap needs 0 < rise <= dur/2"
        j = np.arange(n) + 0.5
        return np.minimum(1.0, np.minimum(j, n - j) / rise)
    return np.sin(np.pi * (np.arange(n) + 0.5) / n)


# One push: returns a 3-vector force in world frame, or zero if not active
def push(k, p):
    # Active only for onset_tick <= k < onset_tick + dur_ticks
    j = k - p.onset_tick
    if not 0 <= j < p.dur_ticks:
        return np.zeros(3)

    # Scale to the peak force if given, else so sum(force * dt) equals the impulse exactly
    prof = profile(p.dur_ticks, p.shape, p.rise_ticks)
    f_peak = p.peak_force if p.peak_force is not None else p.impulse / (p.dt * prof.sum())
    return f_peak * prof[j] * np.asarray(p.direction)


def make_set(impulses, directions_deg, seed, dt, body, shape):
    rng = np.random.default_rng(seed)
    lo, hi = round(ONSET_S[0] / dt), round(ONSET_S[1] / dt)
    pushes = []
    for i, (J, deg, dur) in enumerate(product(impulses, directions_deg, DURATIONS_S)):
        # Random onset tick and direction jitter about world z
        
        # could just call jittered() here, but this is a bit faster for now and avoids creating a new rng for each push
        onset = int(rng.integers(lo, hi + 1))
        a = np.radians(deg + rng.uniform(-JITTER_DEG, JITTER_DEG))
        direction = (float(np.cos(a)), float(np.sin(a)), 0.0)
        pushes.append(Push(i, body, onset, round(dur / dt), J, direction, dt, shape))
    return pushes


def jittered(rng, deg, dt):
    # Random onset tick and direction jitter about world z (same draws as make_set)
    # Every push 
    lo, hi = round(ONSET_S[0] / dt), round(ONSET_S[1] / dt)
    onset = int(rng.integers(lo, hi + 1))
    a = np.radians(deg + rng.uniform(-JITTER_DEG, JITTER_DEG))
    return onset, (float(np.cos(a)), float(np.sin(a)), 0.0)


def sustained_pushes(seed, dt, body="hand"):
    rng = np.random.default_rng(seed)
    pushes = []
    for i, (F, deg, dur) in enumerate(product(SUSTAINED_FORCES, DIRECTIONS_DEG, SUSTAINED_DURATIONS_S)):
        onset, direction = jittered(rng, deg, dt)
        pushes.append(Push(i, body, onset, round(dur / dt), None, direction, dt,
                           shape="trap", rise_ticks=round(SUSTAINED_RISE_S / dt), peak_force=F))
    return pushes


def press_pushes(seed, dt, body="hand"):
    rng = np.random.default_rng(seed)
    pushes = []
    for i, (rise, deg) in enumerate(product(PRESS_RISES_S, DIRECTIONS_DEG)):
        onset, direction = jittered(rng, deg, dt)
        pushes.append(Push(i, body, onset, round((2 * rise + PRESS_HOLD_S) / dt), None, direction, dt,
                           shape="trap", rise_ticks=round(rise / dt), peak_force=PRESS_FORCE))
    return pushes


def grid_pushes(seed, dt, body="hand", shape="sine"):
    return make_set(IMPULSES, DIRECTIONS_DEG, seed, dt, body, shape)


def heldout_pushes(seed, dt, body="hand", shape="sine"):
    return make_set(HELDOUT_IMPULSES, HELDOUT_DIRECTIONS_DEG, seed, dt, body, shape)


if __name__ == "__main__":
    for p in grid_pushes(seed=0, dt=0.002):
        print(p.push_id, p.onset_tick, p.dur_ticks, p.impulse, np.round(p.direction, 3))
