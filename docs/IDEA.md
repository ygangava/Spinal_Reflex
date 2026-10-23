# Framing: an event-driven reflex that cancels pushes

Draft, 2026-10-20; updated 2026-10-21 (sim steps 1–4 done, baseline matrix run, decisions in §9, open options in §10); 2026-10-23 (periodic MLP result, §3e).
Citations live in `docs/Research_clauded.md` (referred to below as **[EB §n]**).
All numbers below come from our own sim (`osc.py`, `disturbance.py`). The run conditions are given with each table.

---

## 1. The claim

> On a Franka arm under a realistic fast low-level controller, an event-driven SNN reflex gets close to
> observer-quality cancellation of sustained pushes, ignores slow deliberate presses, and does it with far
> fewer operations than periodic or delta-triggered MLP reflexes, at low latency on an FPGA.

- **Push rejection is the setting.**
- **Computation and latency are the contribution.**
- We do **not** claim the SNN rejects pushes better than a model-based observer.

---

## 2. Why the framing changed

| | Original (implicit) | Now |
|---|---|---|
| Why a reflex | "The brain is slow; the reflex fills the gap" | The fast controller is error-driven, so it can only push back *after* the arm moves. A reflex can infer the push and cancel it before error builds. |
| Baseline | A slow controller | A fast (1 kHz-class) stiff OSC, which is what real robots run under a slow policy [EB §1] |
| What the reflex learns | "Correct the error" (≈ more gain) | "Cancel the push": the label is −JᵀF |
| Key axis | Brain rate | **Reflex latency** |
| Classical rival | None | De Luca momentum observer, which the Franka already exposes [EB §3] |

**Framings considered and dropped:**
- **Slow down or delay the OSC.** This is a strawman. Real arms have a 1 kHz impedance loop under the policy, and the delay belongs on policy targets, not torques.
- **Soft arm plus reflex as the only story.** In a hold task with no contact, stiffness costs nothing, so stiff OSC wins outright. It survives only as the **slow-press test**: the reflex must leave deliberate presses alone.
- **PD reflex as the main result.** A PD reflex running every tick is just extra gain.

The loop in CLAUDE.md doesn't change: push → error → encoder → SNN → residual added to OSC → arm.

---

## 3. Evidence from our sim

### 3a. OSC alone always recovers
Conditions: KP = 150, critically damped, DT = 2 ms, home pose, hand push, 24-push grid (seed 0).
(Measured before the 1 ms / rate-limit switch; at 1 ms with the rate limit, peaks change by < 1% from DT and up to ~7% from the rate limit.)

| Impulse | Peak hand error | Time to < 1 mm | Saturated ticks |
|---|---|---|---|
| 2 N·s | 9–15 mm | 0.46–0.52 s | 0 |
| 4 N·s | 18–31 mm | 0.56–0.60 s | 0–5 |
| 7.5 N·s | 33–61 mm | 0.66–0.71 s | 0–31 |

Final error < 0.005 mm in every case.

**So:** "does it recover" is the wrong question. The metric is how much error builds up and how fast it falls.

### 3b. Room for a reflex: delayed oracle
The oracle applies the true −JᵀF, D ms late. Conditions: DT = 1 ms, 87/12 N·m limits, torque-rate limit 1000 N·m/s on the **total** torque on or off, hand push in +x. Values are peak hand error in mm.

**4 N·s**

| Push length | Rate limit | OSC only | Oracle 0 ms | 1 ms | 2 ms | 5 ms | 10 ms | 25 ms | 50 ms |
|---|---|---|---|---|---|---|---|---|---|
| 20 ms | off | 17.8 | 7.1 | 7.5 | 7.8 | 8.7 | 9.6 | 11.5 | 15.8 |
| 20 ms | on | 18.3 | 14.3 | 14.7 | 15.1 | 15.4 | 14.5 | 13.0 | 16.7 |
| 80 ms | on | 17.3 | 0.0 | 0.3 | 0.6 | 1.4 | 2.9 | 6.9 | 12.3 |
| 150 ms | on | 16.4 | 0.0 | 0.2 | 0.4 | 0.9 | 1.8 | 4.4 | 8.3 |
| 300 ms | on | 13.5 | 0.0 | 0.1 | 0.2 | 0.4 | 0.9 | 2.1 | 4.2 |

**7.5 N·s, rate limit on**

| Push length | OSC only | Oracle 0 ms | 5 ms | 10 ms | 25 ms | 50 ms |
|---|---|---|---|---|---|---|
| 20 ms | 38.5 | 34.9 | 36.7 | 35.8 | 34.4 | 35.7 |
| 80 ms | 31.9 | 5.0 | 8.2 | 11.4 | 18.7 | 25.0 |
| 150 ms | 30.1 | 0.0 | 1.7 | 3.3 | 8.1 | 15.4 |
| 300 ms | 24.9 | 0.0 | 0.8 | 1.6 | 3.9 | 7.7 |

**Takeaways**
- **Sustained pushes (≥ 80 ms) leave a lot of room.** A 10 ms reflex cuts the error 5–15×, and the rate limit barely matters here.
- **Short impacts (20 ms) can't be helped.** Even a perfect, instant reflex gains little once the rate limit is on. Report this as the physical limit; it isn't a failure.
- **Error grows steadily with delay.** That's the quantitative case for a fast, event-driven, FPGA reflex.

### 3c. Other sim facts
- **Peak error scales with impulse.** At equal impulse, duration and shape change it by < 10%, but they change torque demand and saturation a lot.
- **Direction matters.** Effective mass at the hand is about 7.2 / 4.0 / 4.6 kg in x / y / z, so y pushes move the hand about 1.7× more than x pushes.
- **Static stiffness in our OSC form** is KP · m_eff: about 1076 / 597 / 689 N/m. In franka_ros impedance form it's K directly, and Λ only shapes the transient [EB §2].
- **The pushed body matters most.** The same push on link1–2 does almost nothing; on the hand it does the most.
- **Raising gains helps only like 1/√KP** (4 N·s, 80 ms): KP 150 → 17.4 mm, 400 → 10.7, 1000 → 6.4. Even KP 1000 (beyond Franka's 3000 N/m) loses to a 10 ms oracle (2.9 mm).
- **Onset time doesn't change the physics at home pose:** the same push at 0.8 s vs 1.5 s gives identical error curves (3e-13 mm) once aligned on onset.
- **Settling after a press takes ~0.5–1 s** (+x: 9 mm at 0.1 s, 30.3 mm at ≥ 1 s; +y 53.5 mm). Short holds under-measure yielding.

### 3d. Baseline matrix (`python -m scripts.baseline` → `renders/baseline.png`)
Conditions: DT 1 ms, rate limit on, hand pushes, seed 0, mean over 4 directions (± 10° jitter) and each set's other factors. 440 episodes.

| | OSC | High-KP | Oracle 0 ms | 5 ms | 25 ms |
|---|---|---|---|---|---|
| Impacts (2/4/7.5 N·s, 20/80 ms): peak error | 32.1 mm | 25.4 | 20.7 | 22.9 | 25.5 |
| Sustained (15/30/45 N, 100–300 ms, 10 ms rise): peak error | 26.9 mm | 12.9 | 0.5 | 1.6 | 5.1 |
| Slow press (30 N, rise 20 ms → 2 s, 2 s hold): deflection | 43.1 mm | 16.3 | 0.0 | 0.0 | 0.0 |

- **Impacts:** even a perfect reflex helps modestly; at 25 ms it's no better than raising gains.
- **Sustained:** where a reflex wins — even 25 ms late, 2.5× better than high-KP, 5× better than OSC.
- **Press:** OSC and high-KP yield like fixed springs at every rise time; the oracle is **rigid at every rise time**. Nothing yet has the target behavior (rigid to fast onsets, OSC-like to slow ones) — that gap is the reflex's job.

### 3e. Periodic MLP (teammate, branch `codex/periodic-mlp-reflex`; details in `docs/MLP.md`, `docs/MLP_COST.md`)
Conditions: home pose, hand pushes, noise-free, held-out seeds 200/201; 28 → 64 tanh → 7; inputs q − q0, dq, finite-difference acc, previous applied torque (`tau_prev`) − hold torque; label −JᵀF shifted 1 tick; 2 data rounds (DAgger). MLP at +1 tick output delay, oracle at 2 (same force → torque latency).

| | OSC | High-KP | MLP | Oracle 2 ms |
|---|---|---|---|---|
| Sustained: peak error | 28.9 mm | 14.1 | 0.75 | 0.69 |
| Impacts: peak error | 30.9 mm | 22.3 | 17.6 | 17.7 |
| Press: hold deflection | 43.1 mm | 16.3 | 0.33 | 0.0 |

- **Near-oracle on sustained pushes, but rigid to presses** (label cancels every force).
- **It is a learned observer-like estimator.** Dynamics give −τ_ext = τ_applied − M·q̈ − bias, so at a fixed pose the label ≈ (`tau_prev` − τ_ref) − M·acc, linear in two of its inputs. That formula alone is within 0.5% of the label on near-cancelled data (8.8% on OSC-only). Closed-loop ablation: zeroing `tau_prev` → 101% error, acc → 5.8%, q / dq → ~1%.
- While cancelling, acc ≈ 0, so output ≈ `tau_prev` − τ_ref: it **holds its own last output**. This is what fixes risk 1.
- **Cost:** 2,240 dense MACs every tick, calm or pushed; 2,304 params (9 KiB FP32). Host latency isn't relevant to the FPGA comparison.
- **So:** the inputs decide the result more than the network does. The SNN can't win on rejection; it has to win on cost (and ideally on yielding to presses).

### 3f. Momentum observer (`observer_reflex.py`, `python -m scripts.observer ablate|sweep|compare` → `results/observer/`)
Same held-out cases and metrics as §3e (sustained = 20 / 40 N, 150 / 250 ms, diagonals). Observer at +1 tick output delay (2-tick total, same as MLP).

| | OSC | High-KP | MLP | Observer K_O=100 | Observer K_O=1000 | Oracle 2 ms |
|---|---|---|---|---|---|---|
| Sustained: peak error | 28.9 mm | 14.1 | 0.755 | 2.11 | 0.687 | 0.687 |
| Impacts: peak error | 30.9 mm | 22.3 | 17.6 | 17.6 | 17.7 | 17.7 |
| Press: hold deflection | 43.1 mm | 16.3 | 0.33 | ≈0 | ≈0 | ≈0 |
| Own ops / tick | – | – | 2,240 MACs | ~112 MACs | ~112 MACs | – |

- **Lag = 1/K_O.** Discrete: K_O·DT = 1 is exact after one tick (= oracle); > 1 overshoots; > 2 diverges. Validation sweep (sustained peak): K_O 25 / 50 / 100 / 200 / 500 / 1000 → 6.49 / 3.76 / 2.24 / 1.53 / 1.12 / 0.98 mm.
- **Ablation:** dropping joint damping leaves a 0.37 N·m false estimate after release; dropping Ṁq̇ only 0.013.
- **Latency:** every reflex loses ~0.2 mm per extra ms (sustained, held-out).
- **Op count excludes M(q) and b, which the OSC already computes every tick.**
- **So:** with a perfect model and no noise, a training-free observer ties the oracle and beats the MLP with ~20× fewer of its own ops. K_O = 1000 is only realistic noise-free; real observers use ~50–200, where the MLP wins (0.755 vs 2.11 mm). Noise + model mismatch (§8 step 8) decide whether learning earns its place.

---

## 4. Approach

### 4a. Sim changes (in order)
1. ✅ **Switch to DT = 1 ms.** This matches the Franka's 1 kHz loop and makes a 1-tick sensing delay a realistic 1 ms.
2. ✅ **Enforce limits on the total torque:** 87/12 N·m and 1000 N·m/s (`limit_torque`; no rate limit on the first tick, so the arm doesn't sag). `rate_limit=False` to compare.
3. ✅ **Latency queue on the reflex output** (`run_episode(reflex_delay=D)`). One output delay lumps sensing → FPGA → comms → motor; equivalent to input delay for a time-invariant reflex. A separate sensing delay is *not* modeled yet.
4. ✅ **Log the teacher torque every tick:** −JᵀF at the pushed body's COM (`xipos`) via `mj_applyFT`.
5. ✅ **New disturbances in `disturbance.py`:**
   - sustained pushes, 100–300 ms (`"trap"` shape, 10 ms rise, scaled by peak force)
   - short impacts, kept as the limit case (half-sine, scaled by impulse)
   - slow-press ramps, rise swept 20 ms → 2 s, 2 s hold — same trapezoid as sustained, only the rise differs
6. ✅ **Gains as parameters** (`make_gains`, KD = 2√KP always). Kept the Λ-scaled OSC form; stiffness reported as KP·m_eff in N/m. `HIGH_KP_GAINS` = pos 400 / ori 300 / null 20 (≈ 2900 N/m in x, top of Franka's range).

### 4b. Controllers

| Controller | Role | Priority |
|---|---|---|
| OSC alone | Floor | core |
| High-KP OSC | Answers "just raise the gains" | core |
| OSC + momentum observer feedforward | Classical rival; more important now that the MLP ≈ learned observer (§3e). Not covered by the MLP | core |
| Same, high-pass filtered | Fair rival for the slow-press test | stretch |
| Oracle at 0 / 5 / 25 ms delay | Upper bound | core |
| Periodic MLP | Dense learned baseline. Done at home pose (§3e); retrain on the moving arm | core |
| Delta MLP | "Spikes aren't necessary" ablation [EB §7, Neil 2017] | core |
| SNN | The contribution | core |

### 4c. Reflex design notes (for training)
- **Label:** −JᵀF, the teacher, not "teacher − base". Expect the student to be bounded by an observer-like teacher [EB §8].
- **Inputs must include more than joint error.** See risk 1 below: encode changes in `tau_base`, or in the reflex's own output. The MLP uses `tau_prev` (previous applied torque, now in `obs`) + acc + q − q0 + dq (§3e). **The SNN and delta MLP need the same inputs**, or they lose for reasons unrelated to spiking.
- **Share one send-on-delta encoder** across the delta MLP and the SNN, so the comparison is fair. Its channels now include `tau_prev` and acc/dq, not just error. The encoder mechanics are unchanged.
- **SNN must hold its output during a steady push.** Nothing changes, so there are no events; if the output leaks, the push returns. Option: output spikes encode torque *changes*, and a per-joint accumulator holds the value.
- **RTL impact:** same core (LIF, time-multiplexed layer, event queue). Adds motor torque as an FPGA input, a per-joint output hold register, and torque (±87 N·m) / acc ranges for the Q formats. Freeze inputs before the golden model.
- **Interface (`reflex.py`, done):** `step(obs, gain) -> (torque, estimate)`. `obs` = only what a robot can sense (k, q, dq, e_p, e_o, tau_base). The estimate is a slow summary (external joint torque) for a higher layer. `gain` in [0, 1] scales the torque only; 0 = disabled, estimate still reported. Only `privileged = True` reflexes (the oracle) get the teacher, double-guarded.
- **Raw vs delay-shifted label:** raw = "cancel the push happening now" (impossible at onset — no evidence yet); shifted by L ticks = "cancel what you could have detected by now" (achievable; the first L ticks go uncancelled, i.e. the delayed oracle at L). Option: train raw, score against the delayed oracle.

---

## 5. Evaluation

**Matrix:** controllers × disturbances (sustained, short impact, slow-press sweep) × held-out set.

| Metric group | Metrics |
|---|---|
| Rejection | Peak error (from onset); area under the error curve over push + 1 s; time to < 1 mm (not yet computed) |
| Yielding (slow press) | Mean deflection over the last quarter of a 2 s hold (implemented); force per unit displacement; should match plain OSC |
| Cost | Saturation %; events/s and operations/s (vs 1000 dense evaluations/s); FPGA latency, resources and power from synthesis |

- **Headline figure:** rejection vs. computation, one point per reflex, with the oracle-delay curve as context.
- **Second figure:** yield on slow presses vs. ramp time, one curve per controller.
- **Protocol:** grid pushes for training and tuning; held-out pushes (3 and 6 N·s, diagonal directions) for final numbers only. Agree this with the teammate before training.
- **Comparisons:** every controller runs on the same seed/set; analysis aligned on `onset_tick`. Keep random onsets (stops a learned reflex using `k` as a timing cue; matters once the arm moves). Held-out versions of the sustained/press sets don't exist yet.

---

## 6. Things to consider (risks)

1. **The reflex cancels its own input.** Good cancellation drives the error toward 0, so events stop, the membranes leak, the output decays and the push comes back. With error-only inputs, expect release mid-push or oscillation. **Biggest design risk.** Give the reflex torque-side inputs (changes in `tau_base` or in its own output), as the observer does. Solved in the MLP by `tau_prev` (§3e).
2. **Sustained push vs. slow press is about onset rate, not duration.** A pure high-pass reflex also releases a 300 ms sustained push after its time constant. Desired behavior: triggered by a fast onset, held while the force persists. Define the boundary as ramp rate and test it with the slow-press sweep.
3. **The observer is cheap and already on the robot.** It's a few matrix operations per tick and needs no training, so the SNN likely loses to it on compute as well as rejection. Where a learned reflex can honestly win is **model mismatch**: an unknown payload or wrong link masses, which break a model-based observer. Our sim's model is perfect, so add a mismatch test.
4. **Label timing.** At push onset the label is nonzero but there's no evidence yet. Shift the labels by the expected detection latency, or score against the delayed oracle.
5. **Distribution shift.** Labels collected without the reflex come from large errors. A working reflex keeps errors small, so the network sees inputs it never trained on. Plan closed-loop data collection (DAgger-style rounds).
6. **Observer bandwidth.** The momentum observer lags with time constant 1/K_O: 5–20 ms at typical gains [EB §3]. "Faster than feedback" ≠ instant. Tune K_O fairly.
7. **Energy claims.** Loihi energy numbers don't transfer to an FPGA [EB §7]. Report our own synthesis (resources, latency, power) and per-operation counts; cite neuromorphic chip numbers only as context.
8. **Wrist torque limits.** Joints 5–7 are capped at 12 N·m, so pushes on the hand saturate first. Report the saturation % for every controller.
9. **Single pose.** All numbers so far are at home pose. Effective mass changes with pose, so add a moving trajectory (scripted end-effector path) or several hold poses before the final runs.
10. **Push vs. human.** Error alone can't tell an object hit from a person bumping the arm. State this as a limitation; the slow-press test only covers deliberate, slow contact.
11. **Scope.** About six weeks to Dec 2, including RTL. Run the "core" rows first; add "stretch" rows only if time allows.
12. **Damping in impedance form.** If we switch to franka_ros-style F = K·e − D·v, then D = 2√K is critically damped only for a 1 kg apparent mass, so it's underdamped in heavy directions [EB §2]. Either keep the Λ-scaled OSC or design D with Λ. (Decided for now: keep the Λ-scaled OSC.)
13. **Onset speed is a proxy for intent, not intent.** Slow-but-unwanted (a heavy object leaning) → reflex yields, falls back to plain OSC: no worse than baseline. Fast-but-wanted (a person tapping to stop or quickly guiding the arm) → reflex fights it: *worse* than baseline, the safety-relevant failure. Mitigation: the higher layer uses the gain/enable input and watches the estimate.
14. **Hand-only so far.** All results push the hand. Other links should show the same pattern but different magnitudes; the hand-error metric can miss joint motion from link pushes (needs a joint-space term).
15. **`k` is in `obs`.** A learned reflex could use the tick count as a timing cue. Random onsets make that useless; removing `k` closes it completely.
16. **Measurement windows.** Results depend on windows long enough to settle (press hold was 0.5 s and gave a false rise-time trend — fixed at 2 s). Check settling for any new metric.
17. **Event feedback.** `tau_prev` contains the reflex's own output, so an output change triggers an input event, then a recompute, then maybe another output change. This can chatter. Test it once the encoder exists.
18. **The MLP learned M at one pose.** `q − q0` is relative to the episode start. Expect it to degrade on a moving arm. Finite-difference acc multiplies velocity noise by 1/DT (sim is noise-free).

---

## 7. Decisions needed (with teammate)

- [ ] Adopt this framing (push cancellation + slow-press yielding, computation as the contribution)
- [ ] Reflex inputs: error only, or error + torque-side signals — MLP used torque-side (`tau_prev` + acc); confirm
- [ ] Label: raw −JᵀF, or delay-shifted — MLP used a 1-tick shift; confirm
- [ ] Core controller list (§4b)
- [ ] Held-out protocol
- [x] DT = 1 ms switch (done on the sim side; tell teammate — all training data should be generated at 1 ms)
- [ ] Resist/yield boundary on the press sweep (e.g. rise < 50 ms resist, > 200 ms yield) — this is the training spec
- [ ] Remove `k` from `obs`?

## 8. Next steps (sim side)

1. ✅ DT = 1 ms, torque and torque-rate limits, latency queue (§4a 1–3)
2. ✅ Teacher torque in the log, plus oracle-with-delay as a condition (`scripts/oracle_delay.py` reproduces §3b)
3. ✅ New disturbance types (sustained, slow-press sweep)
4. ✅ Baseline matrix without learning (§3d, `renders/baseline.png`)
5. ✅ Periodic MLP at home pose (teammate, done early, §3e). Inputs and label still to confirm (§7). Merge `codex/periodic-mlp-reflex` → `mujocoTest`.
Order rule: build what doesn't depend on motion first; tune or train anything that depends on the input signals only after the arm moves, so nothing is tuned on conditions we'll throw away.

6. **Momentum observer (+ high-pass version)** at the fixed pose. Independent of motion (dynamics model + joint momentum). **Next:** the MLP (§3e) behaves like one but isn't one, so this is the honest reference for "why learn at all".
7. **Encoder mechanics only:** send-on-delta logic, ON/OFF channels, tests. A pure function of its input, so safe to write now. Do *not* pick thresholds yet — at a fixed pose the error is exactly 0 before the push, so any threshold looks silent. Channels: `tau_prev`, acc/dq, error (§4c). Include the output-hold accumulator and a test for event feedback (risk 17).
8. **Sensor noise + model-mismatch options in `run_episode`** (velocity/torque noise; wrong payload or link mass in the controller's model). Independent of motion. Needed for risk 18 (acc noise) and risk 3 (where learning can beat the observer). Rerun the MLP and the observer with these on.
9. **Moving arm (important):** a slow scripted policy (5–10 Hz) reaching toward a target, on top of the 1 kHz OSC — the realistic hierarchy. Covers risk 9 and makes random onsets meaningful (different pose at each push). Rerun the baseline matrix on it.
   - Policy sends waypoints at 5–10 Hz; a 1 kHz interpolator (linear or min-jerk) gives the OSC a smooth `x_d(t)`, `v_d(t)`. Avoid hold-and-jump: each jump is a "fake push" the encoder/reflex would react to.
   - OSC gets feedforward: a = KP·e + KD·(v_d − v) + a_d (today it assumes v_d = 0).
   - Replace global `x_d` with a target function `target(k) -> (x_d, v_d, a_d)` passed to `run_episode`; a held pose is a constant target, so existing tests stay valid.
   - Teacher (−JᵀF) is unaffected by motion — labels stay clean.
   - Task metrics: final position error, time to reach, deviation from the planned path during/after the push.
10. **Encoder thresholds** tuned on the moving arm (measure events during normal motion with no push), then **toy reflexes:** leaky event-driven PD (no learning) and a tiny periodic MLP trained on the teacher (teammate's home-pose MLP done early, §3e; retrain here). Teammate's training-data generation starts here, on the moving arm.
11. **Yield spec before SNN training:** pick the resist/yield boundary on the press sweep (§7), and a label or gating that achieves it. Today every learned reflex is rigid to presses (§3e).
12. **Delta MLP, then SNN** on the shared encoder and the same inputs. Report MACs vs synaptic ops, calm and pushed (`docs/MLP_COST.md`).
13. Multi-body check (links 4–7), then `run_episode` taking a list of pushes (overlapping / sequential).

---

## 9. Decisions made so far (2026-10-20 → 21)

- Plain MuJoCo + Menagerie Panda + own Λ-scaled OSC; KP 150 default, KD = 2√KP; high-KP = 400/300/20.
- DT 1 ms; 87/12 N·m and 1000 N·m/s enforced on the **total** torque.
- Log row = state at the start of tick k; log commanded and applied torque, base / reflex / teacher separately.
- Time in integer ticks; tick counts from `round(x / DT)`.
- One push per episode for now, on the hand; random onset 0.8–1.5 s + ±10° jitter about z; same seed → identical set.
- Impacts scaled by impulse; sustained and presses scaled by peak force (same trapezoid, rise time differs).
- Teacher = −JᵀF at the pushed body's COM; only privileged reflexes see it.
- Latency = one output queue in `run_episode`, same for every reflex; torque and estimate both delayed.
- Reflex outputs torque + slow estimate; accepts gain/enable from a higher layer.
- Target behavior: oracle-like on fast-onset pushes, OSC-like (yield) on slow presses.

## 10. Open options (choose later)

| Area | Options | Leaning |
|---|---|---|
| Framing | Pure cancellation · cancellation + slow-press yield (hybrid) · soft-arm compliance only | Hybrid (pending teammate) |
| Reflex inputs | Joint error only · + `tau_base` changes · + own output | More than error (risk 1); MLP: `tau_prev` + acc + q − q0 + dq |
| Label | Raw −JᵀF · delay-shifted · train raw, score vs delayed oracle | 1-tick shift (MLP); score vs oracle at the same total latency |
| Error space for encoder | End-effector (6 or 12 ch) · joint (14 ch: 7 position + 7 velocity errors) | Open |
| Classical rival | Momentum observer · + high-pass version · model-mismatch test | Observer core; mismatch test is where learning can win |
| Pushes | Hand only · links 4–7 · multiple bodies · overlapping / sequential pushes | Hand until the reflex works, then generalize |
| Strength across bodies | Same force · scale per body · define by resulting peak displacement | Open |
| Motion | Hold at home · several hold poses · slow scripted policy (5–10 Hz) reaching a target | Scripted policy; after first reflexes work, before final runs |
| Policy → OSC targets | Hold-and-jump every 100–200 ms · interpolate waypoints at 1 kHz (linear / min-jerk) | Interpolate (jumps look like pushes to the reflex) |
| Reflex error signal (moving arm) | Raw tracking error (realistic, includes normal lag) · deviation from a no-push reference run (clean, sim-only) | Open — decide with teammate; teacher is the same either way |
| Moving-arm task metrics | Final error · time to reach · path deviation during/after push | Path deviation primary for push effects |
| Gain schedule | Constant per episode · function of time / estimate | Constant for now |
| Metrics | Peak · AUC · time to 1 mm · joint-space term | Peak + AUC primary; add joint term for link pushes |
