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
- 2026-10-21: DT → 1 ms; torque-rate limit (`limit_torque`, no limit on first tick so the arm doesn't sag); regression peaks re-measured (8.96 / 16.47 / 45.66 mm, was 9.0 / 15.5 / 42.6 at 2 ms, no rate limit). Reflex interface `reflex.py`: `step(obs, gain) -> (tau_res, estimate)`.
- 2026-10-21: Teacher `tau_teacher = −JᵀF` (`mj_applyFT` at body COM `xipos`) logged every tick. `OracleReflex` (privileged, only one that gets `priv`). Latency queue `reflex_delay` in `run_episode`. `scripts/oracle_delay.py` reproduces `FRAMING.md` §3b exactly (e.g. 80 ms push, 10 ms delay → 2.9 mm). 28/28 tests pass. Committed as "Oracle reflex".
- 2026-10-21: New disturbances in `disturbance.py`: `"trap"` shape (ramp up / hold / ramp down) + `peak_force` option + `p.end_tick`. `sustained_pushes` (15/30/45 N × 4 dirs × 100/200/300 ms, 10 ms rise; 36) and `press_pushes` (30 N, rise 20 ms → 2 s, 2 s hold; 28). Shared `jittered()` for onset + direction. 31/31 tests pass.
- 2026-10-21: Press hold was 0.5 s → too short: after a fast rise the arm needs ~0.5–1 s to settle (+x: 9 mm at 0.1 s, 30.3 mm at ≥1 s; +y 53.5 mm). Made OSC "yield" look rise-dependent. Fixed: hold 2 s, measure last quarter.
- 2026-10-21: First baseline plot `renders/baseline.png` via `python -m scripts.baseline` (440 episodes, ~40 s; CSV in `runs/baseline/`). Means — impacts peak: OSC 32.1, high-KP 25.4, oracle 0/5/25 ms 20.7/22.9/25.5 mm. Sustained peak: 26.9 / 12.9 / 0.5 / 1.6 / 5.1 mm. Press hold deflection: 43.1 / 16.3 / 0 / 0 / 0 mm. → impacts barely helpable; sustained = where a reflex wins; oracle is rigid to slow presses (the failure the event-driven reflex must avoid).
- 2026-10-21: Pool workers pinned to 1 BLAS thread (1.8× faster per episode, no oversubscription). matplotlib installed in venv (not yet in requirements.txt).


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
- 2026-10-21: Done: DT 1 ms, rate limit on by default (`rate_limit=False` to compare). Limits apply to total (base + reflex) torque.
- 2026-10-21: Reflex returns torque + slow estimate (external joint torque); gain in [0, 1] scales torque only, estimate always reported. Gain applied in `Reflex.step` so every reflex obeys it.
- 2026-10-21: Privileged info (`tau_teacher`) never in `obs`; only `privileged = True` reflexes get `priv`, double-guarded (run_episode + `Reflex.step`).
- 2026-10-21: Latency lives in `run_episode`, not in reflexes: one output delay lumps sensing → FPGA → comms → motor; same for every reflex (fair). Torque and estimate both delayed. Log `tau_res_new` (computed) and `tau_res` (applied). Equivalent to input delay for time-invariant reflexes.
- 2026-10-21: Sustained push and slow press are the same trapezoid; only rise time differs → press sweep = onset-rate sweep. Both scaled by peak force (same steady force across rise times), impacts still by impulse.
- 2026-10-21: Keep random onsets (onset time doesn't change physics at home pose — checked, 3e-13 mm — but stops a learned reflex using `k` as a timing cue; matters once the arm moves). Align analysis on `onset_tick`. Compare controllers on the same seed/set.
- 2026-10-21: Target reflex behavior: like the oracle on fast-onset pushes, like plain OSC (yields) on slow presses. Onset speed is a proxy for intent, not intent: slow-unwanted → falls back to OSC (no worse than baseline); fast-wanted → reflex fights it (worse; limitation). Higher layer covers this via the gain/enable input and the disturbance estimate.
- 2026-10-21: Metrics so far: peak hand error (from onset), AUC over push + 1 s, press = mean deflection over last quarter of hold, saturation fraction.




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
  - 2026-10-21: resolved (see Decisions). Reflex latency → `reflex_delay` queue.
- Gain is constant per episode; make it a function of k if a higher layer should enable the reflex mid-episode?
- Where's the resist/yield boundary on the press sweep (e.g. rise < 50 ms resist, > 200 ms yield)? This is the spec for training.
- Remove `k` from `obs`? A reflex shouldn't need a clock.
- Baseline plot averages over directions (hides x vs y ~1.7×) and over sustained forces — add per-direction breakdown?
- Add matplotlib to requirements.txt for teammate.
