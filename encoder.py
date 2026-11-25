"""Send-on-delta event encoder, shared by the delta MLP and the SNN.

Each channel remembers the last value it sent (ref). When the input moves a full
threshold away from ref, it emits events and moves ref by that many thresholds.
A steady input emits nothing, so downstream compute only runs when something changes.
"""
import numpy as np


class SendOnDelta:
    def __init__(self, theta):
        self.theta = np.asarray(theta, dtype=float)
        if (self.theta <= 0).any():
            raise ValueError("Thresholds must be positive")
        self.reset()

    def reset(self):
        self.ref = None

    def step(self, x):
        """Return (on, off) event counts per channel for this tick."""
        x = np.asarray(x, dtype=float)
        if self.ref is None:
            # First sample sets the reference; nothing has changed yet.
            self.ref = x.copy()
            return np.zeros(x.shape, int), np.zeros(x.shape, int)
        # Whole thresholds crossed, truncated toward zero (+3.5θ -> 3, -3.5θ -> -3).
        # Counting instead of capping at 1 keeps fast jumps from losing information.
        n = np.fix((x - self.ref) / self.theta).astype(int)
        self.ref += n * self.theta
        return np.maximum(n, 0), np.maximum(-n, 0)


class OutputHold:
    """Per-joint torque that moves by delta per output event and holds between events.

    During a steady push nothing changes, so no events arrive. A leaky output would
    let the push back in; this one holds until told otherwise.
    """

    def __init__(self, delta):
        self.delta = np.asarray(delta, dtype=float)
        if (self.delta <= 0).any():
            raise ValueError("Step sizes must be positive")
        self.reset()

    def reset(self):
        self.value = np.zeros_like(self.delta)

    def step(self, on, off):
        self.value = self.value + (np.asarray(on) - np.asarray(off)) * self.delta
        return self.value.copy()
