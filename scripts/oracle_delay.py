# Delayed-oracle table (FRAMING.md §3b): how good could a perfect reflex be if it reacted D ms late?
# Run from the project root: python -m scripts.oracle_delay
import numpy as np

from disturbance import Push
from osc import run_episode, DT
from reflex import OracleReflex

IMPULSE = 4.0                    # N*s
DURATIONS_MS = (20, 80, 150, 300)
DELAYS_MS = (0, 1, 2, 5, 10, 25, 50)


def peak_mm(log):
    return 1000 * np.linalg.norm(log["e_p"], axis=1).max()


if __name__ == "__main__":
    print(f"{IMPULSE} N*s, hand +x, DT = {DT * 1000:.0f} ms, rate limit on. Peak hand error (mm)")
    print("push ms  OSC only | oracle delay ms: " + " ".join(f"{x:>5d}" for x in DELAYS_MS))
    for dur in DURATIONS_MS:
        p = Push(0, "hand", round(1.0 / DT), round(dur / 1000 / DT), IMPULSE, (1.0, 0.0, 0.0), DT)
        base = peak_mm(run_episode(p, t_end=2.5))
        row = [peak_mm(run_episode(p, reflex=OracleReflex(), reflex_delay=round(x / 1000 / DT), t_end=2.5))
               for x in DELAYS_MS]
        print(f"{dur:6d}   {base:7.1f} |                 " + " ".join(f"{x:5.1f}" for x in row))

'''
push ms  OSC only | oracle delay ms:     0     1     2     5    10    25    50
    20      18.3 |                  14.3  14.7  15.1  15.4  14.5  13.0  16.7
    80      17.3 |                   0.0   0.3   0.6   1.4   2.9   6.9  12.3
   150      16.4 |                   0.0   0.2   0.4   0.9   1.8   4.4   8.3
   300      13.5 |                   0.0   0.1   0.2   0.4   0.9   2.1   4.2


useful data to include!!!

80 ms and longer: a reflex that reacts within about 10 ms cuts the error to about 3 mm or less.
20 ms: even a perfect, instant reflex barely helps, because the 1000 N·m/s rate limit means the torque can't ramp up in time.
Use this as the yardstick: your teammate's reflexes get scored by how close they come to the row at their measured latency.

'''
