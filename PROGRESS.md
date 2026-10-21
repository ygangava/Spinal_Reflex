# Progress

Current stage: 0 — map the loop from memory

## Stage status
- [ ] 0. Map the loop
- [ ] 1. Sim baseline
- [ ] 2. Reflex as control (PD, MLP)
- [ ] 3. SNN in float
- [ ] 4. Fixed-point golden model
- [ ] 5. RTL
- [ ] 6. Integration + comparison

## Log
- 2026-10-16: Env set up (venv, requirements.txt: MuJoCo, robosuite, mink, numpy/scipy, opencv, pytest). Repo on branch `mujocoTest`.
- 2026-10-19: `osc.py`: own OSC in plain MuJoCo (Menagerie Panda, actuators → torque motors, null-space posture, gravity comp via `qfrc_bias`). Holds pose to 0.000 mm; 150 N × 50 ms hand push → 32 mm peak, full recovery.
- 2026-10-20: `disturbance.py`: `Push` dataclass, pure `push(k, p)`, 24-push grid + 16 held-out. `osc.py` refactored to `run_episode(p)` → `.npz` per episode (per-tick log + JSON meta incl. git hash). Tests: `tests/test_disturbance.py`, `tests/test_osc_push.py` — 15/16 pass (`test_heldout_disjoint`: missing import).
- 2026-10-20: Push sweeps (hand, home pose): peak error ∝ impulse; duration/shape change it <10% but change torque demand/saturation a lot. y moves hand ~1.7× more than x (m_eff ≈ 3.9 vs 6.7 kg). Body matters most: link1–2 ≈ no effect, effect grows toward hand. Recovery to <1 mm: 450–800 ms.
- 2026-10-20: OSC alone recovers all 24 grid pushes (final err < 0.005 mm) → "does it recover" is the wrong question. Checked `reflex_sim`: 41/41 tests pass, harness + teacher + encoder only; no trained reflex/SNN/RTL; same finding (only the cheating oracle helps).
- 2026-10-20: Delayed-oracle runs (perfect −JᵀF, D ms late; DT 1 ms, 87/12 N·m, 1000 N·m/s rate limit): sustained pushes (≥80 ms) 17 → ~3 mm at 10 ms delay; 20 ms impacts barely helpable (rate limit). New framing written up in `FRAMING.md`.
- 2026-10-21: KP sweep (4 N·s, 80 ms): peak ∝ ~1/√KP; KP 150 → 17.4 mm, 400 → 10.7, 1000 → 6.4 (vs ~3 mm for a 10 ms oracle).
- 2026-10-21: Fixed viewer import + held-out test import. Gains now parameters: `make_gains()`, `DEFAULT_GAINS`, `HIGH_KP_GAINS`, passed `run_episode(p, gains)` → `osc(g)` → meta. 17/17 tests pass. DT still 2 ms (1 ms switch not done yet).


## Decisions
<!-- e.g. control rate, push magnitude/duration, recovery metric, Q formats -->
- In robosuite the physics steps every 2 ms, and the base controller recomputes torques at every physics step, so 500 Hz. - BUT no longer since own OSC
  - error space
  - End effector error should go into event encode
  - joint error - log and see if it is useful or a diagnostic for end effector error later
  - MuJoCo is deterministic for the same initial state on the same machine. robosuite, however, adds initialization noise to the starting joint positions by default, so runs differ unless you pass initialization_noise=None or seed it. Turn it off for the baseline. Your test is to run no-push twice and diff the joint logs. They should match exactly, and the joints should stay nearly constant during a hold.
  - time window - half open window [t, t+dt) for pushes. avoids double counting at end
  - half sine push profile - Impulse = F_peak * duration * 2/pi
  - parameterize by impulse, can ge tpeak froce from that ... duration can vary independlty then and can test short and sharp angainst long and soft pushes with the same impulse. For now, we can just use a short hard push.
  - initial direction x and y 
  - seeds - default_rng(seed) - using seperate seeds for training and testing

The direction jitter needs an axis. I assume ±10° is a rotation about world z, so pushes stay horizontal.

- 2026-10-20: Sim: plain MuJoCo + Menagerie Panda + own OSC (not robosuite). Full control of per-tick order: `mj_step1` → OSC + residual → push → `mj_step2`. 500 Hz, DT = 2 ms.
- 2026-10-20: OSC gains KP 150 (pos, ori), null 10; KD = 2√KP (critical).
- 2026-10-20: Log row = state at START of tick k (before `mj_step2`). Log both `tau_cmd` and `tau_applied` (`actuator_force`, after forcerange clip).
- 2026-10-20: Time in integer ticks; seconds only for printing. Tick counts from `round(x / DT)`.
- 2026-10-20: Push grid: impulse 2 / 4 / 7.5 N·s (clean / at limit / saturates), duration 20 / 80 ms, ±x ±y + ±10° jitter about z, onset uniform 0.8–1.5 s. Half-sine default, rect optional. Impulse exact per tick.
- 2026-10-20: Held-out: 3 / 6 N·s, diagonals. Never used for training or tuning.
- 2026-10-20: One push per episode for Stages 1–2. Push body = hand for now; bodies later (skip link1–2).
- 2026-10-20: Framing (draft, pending teammate): reflex = fast push *cancellation* (label −JᵀF), ignores slow presses; computation/latency is the contribution, not beating the momentum observer. See `FRAMING.md`.
- 2026-10-21: Gains as a dict; KD always = 2√KP. HIGH_KP = pos 400 / ori 300 / null 20 (pos ≈ 2900 N/m in x, top of Franka's 10–3000 N/m). Changes 3 gains at once — deliberate.
- 2026-10-21: Planned: DT 1 ms (match Franka 1 kHz), torque-rate limit 1000 N·m/s on total torque, latency queue on reflex output.




## Open questions
<!-- things I don't understand yet -->
- Teacher? how to train the SNN? How to implement the event encoder? How to implement the LIF neurons? How to integrate with the OSC controller? How to measure recovery quality and compute/energy cost?

- Encoder input details to decide:
  - Orientation error: rotation-vector (axis-angle) error vs euler angles have discontinuities
  - Velocity: decide whether to include end-effector velocity. Position plus orientation is 6 channels; adding velocity makes 12. With ON/OFF events per channel, 12 channels means 24 input neurons.
  - Output: the residual torque is still 7 joint torques. Input is 6 or 12 channels, output is 7.

- Where to put the event encoder and SNN in the loop? 
  - After the OSC controller, or before it? 
  - The residual torque is added to the OSC output, so it seems like the SNN should be after the OSC controller. BUT then the SNN sees the error after the OSC controller has already acted. 
  - if before -- SNN sees error before OSC acs, and output is added to OSC which may not be what we want. 

- Recovery metric: threshold + hold time? Needs a joint-space term too (link pushes can move joints more than the hand).
- Push strength across bodies: same impulse ≠ same disturbance. Scale per body, or define strength by resulting peak hand displacement?
- Reflex latency (FPGA delay): where to model it in the loop?
- Agree held-out rule with teammate before training starts.
- For teammate (`FRAMING.md` §8): adopt framing? reflex inputs (error only vs + torque-side; error-only may cancel its own input)? label raw vs delay-shifted −JᵀF? core controller list?
- Reflex interface: `reflex(k, obs) -> tau_res`, with privileged force access for the oracle only — design before coding.

LLM says:
- Where to insert the residual. Check how your robosuite version structures env.step. In recent versions it loops over sim substeps, around 500 Hz. Every substep it recomputes the OSC torques and writes them to data.ctrl, but it only updates the goal at control_freq, which defaults to 20 Hz. That gives you two options:

- Between env.step calls, via data.qfrc_applied. This is the easy option: MuJoCo adds those joint forces to the dynamics. The catch is that your reflex then only updates at 20 Hz, which is far too slow to be a reflex. Raising control_freq helps but changes the OSC behaviour too.
Inside the substep loop. Subclass the OSC controller and override run_controller to return its torques plus your residual. The reflex then runs at sim rate, and it literally matches the CLAUDE.md description: “added to OSC output”. OSC’s torque clipping also applies to the sum.

- I’d lean toward the second option. Whichever you choose, grep the robosuite source to confirm where torques get written, and make sure nothing else writes to qfrc_applied. Keep the push on xfrc_applied, which is a separate array, so the two don’t collide.

