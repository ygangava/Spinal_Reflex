# Model mismatch: only the true physics changes, it's always restored, and an unknown payload makes OSC sag
import numpy as np
import pytest

import osc
from disturbance import sustained_pushes
from observer_reflex import MomentumObserver
from osc import run_episode, plant_mismatch, DT

P = sustained_pushes(0, DT)[0]
T_END = P.onset_tick * DT   # calm part only: the sag is what we're testing


# The plant changes inside the block; the controller's model never does; both are restored after
def test_only_plant_changes_and_restores():
    plant, belief = osc.m.body_mass.copy(), osc.m_ctrl.body_mass.copy()
    with plant_mismatch(payload_kg=1.0, link_mass_scale=1.2):
        assert osc.m.body_mass[osc.hand] == pytest.approx(plant[osc.hand] + 1.0)
        assert np.array_equal(osc.m_ctrl.body_mass, belief)
    assert np.array_equal(osc.m.body_mass, plant)


# Restored even if the episode crashes
def test_restores_on_error():
    plant = osc.m.body_mass.copy()
    with pytest.raises(RuntimeError):
        with plant_mismatch(payload_kg=2.0):
            raise RuntimeError
    assert np.array_equal(osc.m.body_mass, plant)


# Unknown payload: OSC under-compensates gravity and sags; the observer reads the extra weight as an
# external force and cancels most of it
def test_payload_sag_and_observer():
    def calm_error(**kw):
        with plant_mismatch(payload_kg=1.0):
            log = run_episode(P, t_end=T_END, **kw)
        return 1000 * np.linalg.norm(log["e_p"][-1])
    osc_sag = calm_error()
    obs_sag = calm_error(reflex=MomentumObserver(100), reflex_delay=1)
    assert osc_sag > 5.0
    assert obs_sag < 0.2 * osc_sag
