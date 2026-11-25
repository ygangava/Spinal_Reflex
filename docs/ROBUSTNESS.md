# Robustness: sensor noise and model mismatch (step 9, 2026-11-24)

Part 1 is sensor noise (9a–9c). Part 2 is model mismatch (9d–9e). Part 3 puts both together.

# Part 1: Sensor noise

## The gist
- **Noise kills the event savings.** The encoder can't tell noise from a real change, so even tiny
  velocity noise (1e-4 rad/s) makes it fire almost every tick. Calm cost goes from 0 to ~1,100 MACs/tick.
- **Noise hurts the MLP badly.** Its acceleration input is `Δdq / DT`, which multiplies velocity noise by 1,000.
  At 1e-2 rad/s the arm jitters ~10 mm while nothing is happening.
- **The observer (K=100) barely cares.** It integrates over ~10 ms, which averages noise out. The price is
  speed: ~2× worse peak error than the MLP when there's no noise.
- **Fix: set each event threshold above that channel's noise floor (6σ).** Then calm noise almost never fires
  (calm cost ≈ 0 again), and noise smaller than the threshold never reaches the network at all.
  The encoder becomes a noise filter.
- **Result:** the delta MLP with noise-floor thresholds is the most noise-robust reflex on sustained pushes,
  and cheapest when calm. It's worse on impacts at high noise, because big thresholds hide fast onsets.
- **Not free:** a bigger threshold also hides small *real* changes and reacts later, so accuracy still slips
  as noise grows (1.03 → 2.29 mm). Noise still costs something; the threshold just keeps that cost small.
- **For the SNN:** it uses the same encoder, so it gets this filtering for free. Its leaky membrane is a second
  filter on top, which helps accuracy but not compute.

---

## In plain terms

**What the encoder does.** For each signal it remembers the last value it reported. Every tick it asks: has the
signal moved at least θ away from that? Yes → send an event, the network updates. No → stay quiet, no work.

**The problem.** Real sensors jitter: even with the arm perfectly still, the velocity reading wobbles a little
every tick. σ (sigma) is the typical size of that wobble. If θ is smaller than the wobble, jitter alone crosses
the threshold almost every tick. The encoder can't tell "the arm moved" from "the sensor jittered", so it fires
constantly: no cheaper than the periodic MLP, and the network reacts to the jitter, so the arm shakes.

**The fix.** Set θ to 6σ. Jitter almost never gets that big (beyond 6σ is ~1 in a billion for Gaussian noise), so
while the arm is still, the encoder stays quiet. And since it only reports moves bigger than θ, small jitter never
reaches the network. Like "tell me only if the temperature changes by more than 2°" when the thermometer flickers
by ±0.1°: you never hear about the flicker.

**Why it's best on sustained pushes.** A sustained push moves the arm far more than 6σ, so the real change gets
through and the jitter doesn't. The periodic MLP sees every jittery reading, so it shakes.

**Why it's worse on impacts.** In the first few ms of a short hit, the clearest sign is acceleration, which is
also the noisiest signal (computed from velocity differences, ~1,000× the jitter). So its 6σ threshold is huge,
and a short hit often doesn't cross it. The reflex only notices once position and velocity have moved: a bit late.

## Raising the threshold is not the same as having no noise

The threshold doesn't remove noise. It ignores *everything* smaller than θ, real changes included. So noise
still has a role: it decides how big θ must be, and a bigger θ costs two things.

1. **Coarser information.** With no noise, θ can be tiny and the network sees the arm finely. With θ = 6σ, any
   real movement smaller than θ is invisible: the network works from a rougher picture.
2. **Slower reaction.** A real change has to grow past θ before it's reported. Bigger θ → longer wait.
   This is why impacts got worse.

| dq noise (rad/s) | 0 | 1e-3 | 3e-3 | 1e-2 |
|---|---|---|---|---|
| Delta + floor, sustained peak mm | **1.03** | 1.21 | 1.63 | 2.29 |
| MLP (no filtering), for contrast | 1.01 | 1.38 | 3.54 | 16.77 |

More noise → bigger θ → accuracy slips. Far better than not filtering, but not free.

## Can the SNN filter noise?

Partly, in a different way, with its own cost.

**How.** The membrane adds up incoming events and slowly leaks the total away. Noise events point in random
directions (an ON, then an OFF), so they mostly cancel in the running total. A real push sends events in one
direction, so the total builds up and the neuron fires. That's averaging over time, the same trick the observer uses.

**Costs.**
1. **Averaging takes time.** The membrane collects events over its time constant τ. Longer τ = better filtering
   but later reaction. Same speed-vs-noise trade-off as θ and the observer gain; no filter avoids it.
2. **It doesn't save compute.** The membrane filters *after* events arrive, and every arriving event still costs
   synaptic adds. Lower θ and let the membrane filter → noise makes events again → calm cost goes back up.

| Filter | Blocks noise for… | Cost |
|---|---|---|
| Encoder threshold θ | compute **and** accuracy | coarser, slower |
| SNN membrane τ | accuracy only | slower |

**Where the SNN could win.** A moderate θ (cheap enough when calm) plus the membrane cleaning up the rest could
give finer resolution than the delta MLP's big 6σ thresholds with similar noise rejection, and so maybe better
impacts. θ and τ are chosen together; training can tune τ. A testable claim, not a guaranteed result.

---

## Setup
- **Noise model:** `NoisyObs(reflex, q_std, dq_std, seed)` in `reflex.py` adds Gaussian noise to the `q` / `dq`
  the reflex sees. The sim and the OSC still see the true state (limitation: noise only in the reflex's sensing).
  `tau_prev` is clean: it's the torque the controller commanded.
- **Knob:** dq noise std. q noise is tied to it as if dq were a finite difference of q:
  `q_std = dq_std · DT / √2`.
- **Feature noise** (`feature_noise` in `delta_mlp_reflex.py`): q − q0 → q_std, dq → dq_std,
  acc → `√2 · dq_std / DT` (the 1,000× blow-up), tau_prev → 0.
- **Noise floor:** `θ = max(k · x_scale, 6 · feature_noise)` per channel. A robot can measure its sensor noise at
  rest, so this isn't privileged, but it does assume σ is known.
- **Cases:** validation seed 100 (24 impacts, 36 sustained), +1 tick output delay for every reflex. Each
  episode's noise seed is shared across controllers (paired comparison).
- **Reproduce:** `python -m scripts.noise_sweep` (~3 min) → `results/noise/noise_sweep.csv`.

## Results

Cell = mean **peak mm / calm jitter mm**; delta rows add **calm / push MACs per tick**.
Calm jitter = mean hand error before the push (skipping the first 200 ms). OSC alone isn't affected
(noise only reaches the reflex). Observer cost is ~112 MACs/tick at all times; periodic MLP 2,240.

### Sustained pushes
| dq noise (rad/s) | 0 | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 |
|---|---|---|---|---|---|---|
| OSC | 26.88 / 0 | 26.88 / 0 | 26.88 / 0 | 26.88 / 0 | 26.88 / 0 | 26.88 / 0 |
| Observer K=100 | 2.24 / 0 | 2.24 / 0 | 2.24 / 0 | 2.26 / 0.01 | 2.34 / 0.05 | 2.65 / 0.25 |
| Observer K=1000 | 0.98 / 0 | 1.00 / 0.01 | 1.10 / 0.08 | 1.53 / 0.39 | 3.23 / 1.25 | 10.40 / 4.70 |
| MLP | 1.01 / 0 | 1.03 / 0 | 1.08 / 0.04 | 1.38 / 0.26 | 3.54 / 1.46 | 16.77 / 10.49 |
| Delta k=0.01 | 1.03 / 0 · 0/451 | 1.03 / 0.01 · 1095/1193 | 1.07 / 0.04 · 1303/1371 | 1.38 / 0.25 · 1535/1566 | 3.53 / 1.44 · 1708/1726 | 16.99 / 10.27 · 1864/1894 |
| Delta k=0.1 | 1.52 / 0 · 0/158 | 1.38 / 0.01 · 225/366 | 1.18 / 0.12 · 676/721 | 1.43 / 0.27 · 1019/1044 | 3.48 / 1.36 · 1210/1226 | 16.99 / 10.19 · 1423/1433 |
| **Delta k=0.01 + floor** | **1.03 / 0 · 0/451** | **1.16 / 0 · 0/593** | **1.19 / 0 · 0/709** | **1.21 / 0 · 0/853** | **1.63 / 0 · 0/931** | **2.29 / 0.02 · 15/936** |

### Impacts
| dq noise (rad/s) | 0 | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 |
|---|---|---|---|---|---|---|
| OSC | 32.01 / 0 | 32.01 / 0 | 32.01 / 0 | 32.01 / 0 | 32.01 / 0 | 32.01 / 0 |
| Observer K=100 | 21.95 / 0 | 21.95 / 0 | 21.96 / 0 | 22.00 / 0.01 | 22.15 / 0.05 | 22.71 / 0.24 |
| Observer K=1000 | 21.72 / 0 | 21.70 / 0.01 | 21.75 / 0.09 | 21.85 / 0.38 | 23.13 / 1.20 | 28.74 / 4.60 |
| MLP | 21.76 / 0 | 21.73 / 0 | 21.78 / 0.04 | 21.85 / 0.24 | 22.66 / 1.46 | 30.71 / 10.15 |
| Delta k=0.01 | 21.77 / 0 · 0/1544 | 21.75 / 0.01 · 1095/1688 | 21.77 / 0.04 · 1302/1744 | 21.85 / 0.24 · 1533/1812 | 22.63 / 1.44 · 1708/1897 | 30.78 / 10.14 · 1865/1997 |
| Delta k=0.1 | 21.95 / 0 · 0/1062 | 21.88 / 0.01 · 222/1099 | 21.83 / 0.11 · 675/1187 | 21.90 / 0.26 · 1017/1334 | 22.66 / 1.35 · 1207/1444 | 30.49 / 9.91 · 1420/1562 |
| **Delta k=0.01 + floor** | 21.77 / 0 · 0/1544 | 21.98 / 0 · 0/1437 | 22.11 / 0 · 0/1378 | 22.59 / 0 · 0/1288 | 23.34 / 0.01 · 1/1200 | 27.84 / 0.02 · 18/1105 |

## What got worse, and why
- **Event savings vanish without a floor.** Send-on-delta fires whenever `|x − ref| ≥ θ`. With θ = k · x_scale,
  θ on the acc channels is far below the acc noise (e.g. 1.4 rad/s² of noise vs θ ≈ 0.04–0.17 rad/s² at
  k = 0.01, dq noise 1e-3). Every tick looks like a change.
- **MLP and delta MLP (no floor) degrade together.** Events change *when* the network runs, not *what* it sees:
  `ref` follows the noise to within θ, so noise passes straight through to the torque.
- **Deadbeat observer (K=1000) degrades like the MLP.** A one-tick estimator does no averaging. The MLP is worse
  because of the 1,000× acc amplification.
- **Floor version on impacts at high noise (27.8 vs 22.7 mm for the K=100 observer).** A 6σ threshold on acc
  at 1e-2 rad/s noise is ~85 rad/s², so the onset of a short hit is mostly invisible until q / dq move.
  Same speed-vs-filtering trade-off as the observer gain.

## What got better, and why
- **Calm cost ≈ 0 at every noise level** with the floor (≤ 18 MACs/tick at 1e-2), vs a constant ~112 for the
  observer. That's the event-driven argument surviving noise.
- **Accuracy better than the periodic MLP under noise.** Noise smaller than θ never moves `ref`, so the network
  sees a clean, quantized signal: send-on-delta has a built-in deadband.
- **Best of all reflexes on sustained pushes at high noise:** 2.29 mm vs 2.65 (observer K=100) and 16.8 (MLP),
  with ~10× less calm jitter than the observer.
- **Quirk: dithering.** Delta k=0.1 without a floor gets slightly *better* with a little noise (1.52 → 1.18 mm at
  3e-4): noise triggers extra updates, so the held output goes stale less often.

## Caveats
- σ is assumed known (measured at rest). A wrong σ either floods events (too low) or hides real signal (too high).
- Noise reaches only the reflex, not the OSC. Real sensor noise would also move the base controller.
- White Gaussian noise only; no bias, quantization, or filtered (colored) dq as on a real Franka.
- One pose (home), noise-free training data. Retraining the MLP with noise (or filtered acc) is untested.
- Validation seed only; held-out test seeds (200/201) not run yet.

---

# Part 2: Model mismatch

## The gist
- **Model mismatch = the robot's idea of its own body is wrong.** E.g. it's holding a 1 kg object it wasn't told
  about, or its links are 20% heavier than the datasheet says.
- **Reviewers suggested this is where a learned reflex could beat the observer,** because the observer depends on
  the model and a learned reflex doesn't (directly).
- **At a held pose, the opposite happened.** The observer was completely unaffected: zero sag, same push
  recovery. The MLP got *worse* than having no reflex at all on sag, and its push recovery got up to 15× worse.
- **Why:** a wrong mass at a held pose mostly means wrong gravity, a constant extra force. The observer can't tell
  that from a push, so it cancels it, which is exactly right. The MLP was only trained on the correct robot, so
  a heavier one is input it has never seen.
- **Conclusion:** mismatch (mass/payload, held pose) is not where learning wins. Noise is (Part 1).

## In plain terms

**What "the model" is.** To hold the arm still, the controller (OSC) has to know how heavy each part is: it
pushes up exactly as hard as gravity pulls down, plus whatever's needed to correct position errors. The momentum
observer uses the same knowledge differently: "given the torque I applied and the masses I know, the arm should
move like *this*; if it moves differently, something external is pushing it."

**What mismatch does.** If the real arm is heavier than the model says:
- The OSC holds it up with too little force, so the arm **sags** until the position error makes up the
  difference. Plain OSC has no integral term, so the sag stays.
- The observer sees "the arm is being pulled down more than my model explains" and reports it as an external
  force. That's actually correct: an unknown payload *is* an external load. Cancelling it removes the sag.

**How we simulate it.** The sim now has two copies of the robot:
- **The plant** (`osc.m`): the true physics MuJoCo runs.
- **The belief** (`osc.m_ctrl`): what the controller thinks. The observer loads its own copy of the same nominal file.

`plant_mismatch(payload_kg, link_mass_scale)` changes only the plant, inside a `with` block, and always restores it:
- `payload_kg`: extra mass at the hand's center of mass (an unknown object in the gripper). Franka's rated payload is 3 kg.
- `link_mass_scale`: links 1–7 mass and inertia scaled together (wrong datasheet values).

With no mismatch, everything is bit-identical to before the split (tested).

**Why two numbers per episode.** A wrong mass does two separate things, so we measure both:
- **Calm sag (mm):** steady hand error before the push, from 0.5 s (sag has settled) to push onset.
- **Push peak (mm):** the biggest *push-induced* deviation, `‖e(t) − e(onset)‖`. Recovery quality on top of
  whatever sag was already there. Without this, a sagging arm would look "worse at pushes" just from the offset.

## Setup
- Validation seed 100 (24 impacts, 36 sustained), no noise. Every controller keeps the nominal model.
- Output delay +1 tick for learned reflexes and the observer, +2 for the oracle (as in `scripts/mlp.py`).
- **Reproduce:** `python -m scripts.mismatch_sweep` (~3 min) → `results/mismatch/mismatch_sweep.csv`.
- Code: `osc.py` (`m_ctrl`, `d_ctrl`, `plant_mismatch`), `tests/test_mismatch.py`.

## Results

Cell = mean **calm sag mm / push peak mm**; delta row adds **calm / push MACs per tick**.

### Sustained pushes
| | nominal | +0.5 kg | +1 kg | +2 kg | links ×0.8 | links ×1.2 |
|---|---|---|---|---|---|---|
| OSC | 0 / 26.88 | 7.15 / 26.85 | 14.20 / 26.80 | 28.07 / 26.63 | 10.74 / 26.75 | 10.89 / 26.97 |
| Observer K=100 | 0 / 2.24 | **0 / 2.20** | **0.01 / 2.18** | **0.01 / 2.16** | **0 / 2.32** | **0 / 2.19** |
| Observer K=1000 | 0 / 0.98 | **0 / 0.91** | **0 / 0.86** | **0 / 0.77** | **0 / 1.12** | **0 / 0.88** |
| MLP | 0 / 1.01 | 9.08 / 1.87 | 16.05 / 3.73 | 29.66 / 15.12 | 13.18 / 3.77 | 12.95 / 2.68 |
| Delta k=0.01 | 0 / 1.03 · 0/451 | 9.09 / 1.89 · 44/586 | 16.09 / 3.74 · 186/688 | 29.67 / 15.07 · 526/993 | 13.31 / 3.84 · 194/718 | 12.96 / 2.68 · 98/664 |
| Oracle | 0 / 0.98 | 7.15 / 0.93 | 14.20 / 0.89 | 28.07 / 0.81 | 10.74 / 1.06 | 10.89 / 0.92 |

### Impacts
| | nominal | +0.5 kg | +1 kg | +2 kg | links ×0.8 | links ×1.2 |
|---|---|---|---|---|---|---|
| OSC | 0 / 32.01 | 7.15 / 30.46 | 14.20 / 29.24 | 28.07 / 27.45 | 10.74 / 34.18 | 10.89 / 30.46 |
| Observer K=100 | 0 / 21.95 | 0 / 19.96 | 0.01 / 18.38 | 0.01 / 16.04 | 0 / 24.73 | 0 / 19.89 |
| Observer K=1000 | 0 / 21.72 | 0 / 19.58 | 0 / 17.88 | 0 / 15.36 | 0 / 24.75 | 0 / 19.49 |
| MLP | 0 / 21.76 | 9.08 / 19.71 | 16.06 / 18.44 | 29.65 / 20.23 | 13.18 / 25.58 | 12.95 / 19.81 |
| Delta k=0.01 | 0 / 21.77 · 0/1544 | 9.09 / 19.72 · 43/1580 | 16.09 / 18.44 · 183/1630 | 29.66 / 20.24 · 525/1739 | 13.31 / 25.61 · 193/1630 | 12.96 / 19.82 · 97/1622 |
| Oracle | 0 / 21.73 | 7.15 / 20.06 | 14.20 / 18.75 | 28.07 / 16.81 | 10.74 / 24.22 | 10.89 / 20.02 |

## What it means, and why

- **The observer is unaffected: zero sag, same or slightly better push recovery.** A wrong mass at a held pose is
  mostly a wrong gravity term, which is constant. The observer lumps every unexplained torque into "external",
  so the payload's weight gets cancelled along with the push. Inertia errors only matter while the arm
  accelerates, which at a hold is only the brief push transient. (Push peaks shrink with more payload because a
  heavier arm moves less under the same force.)
- **The MLP sags *more* than no reflex** (9.08 vs 7.15 mm at +0.5 kg), and push recovery degrades badly
  (15.1 mm at +2 kg vs 1.01 nominal). It learned "this pattern of q, dq, acc, tau_prev means this push torque"
  on the correct robot only. A heavier robot produces patterns it never saw, so its output is wrong, and nothing
  in it says "ignore what you don't understand." This is the out-of-distribution problem.
- **Lighter links also hurt** (×0.8: sag 13.2 mm for the MLP, 10.7 for OSC). Any mass error, either direction,
  is a gravity error.
- **The delta MLP's calm cost is no longer 0** (up to 526 MACs/tick at +2 kg). The sagging arm keeps its inputs
  drifting, and the MLP's changing output feeds back into its own `tau_prev` input (risk 17: event echo).
  "Calm" isn't calm when the robot is fighting its own model error.
- **The oracle sags exactly like OSC.** It only cancels the push (it knows the true push force), not the payload.
  That's a reminder that the oracle is "perfect at its label", not "perfect at holding still".

## Caveats
- **Held pose only.** On a moving arm, a wrong inertia makes the observer report fake forces every time the arm
  accelerates. That's where it would likely degrade; not tested (moving arm was cut for time).
- **Only mass/inertia mismatch.** Unmodeled joint friction (stiction, Coulomb) is the error most likely to fool the
  observer during a push; not tested.
- **The MLP was trained on the nominal robot only.** Training with randomized payloads/masses (domain
  randomization) would likely fix most of its sag, but then it's learning to do what the observer does by design.
- Payload is a point mass at the hand's COM, not offset in the gripper. Validation seed only.

---

# Part 3: Putting it together

| Condition | Best reflex | Why |
|---|---|---|
| No noise, nominal model | MLP / delta MLP ≈ observer K=1000 ≈ oracle | Everything works when the world matches training |
| **Sensor noise** | **Delta MLP + noise-floor thresholds** (sustained); observer K=100 (impacts at high noise) | Send-on-delta ignores changes below θ: a deadband that keeps noise out of the network and keeps calm cost ≈ 0 |
| **Mass / payload mismatch (held pose)** | **Observer** | It treats unexplained torque as external, so it cancels the payload too. The MLP sees inputs it was never trained on |

**What this means for the project's claim.**
- Event-driven compute **can** beat the observer, but on noise and cost, not on model mismatch. The encoder's
  threshold is doing double duty: it saves compute *and* filters noise.
- The observer is the stronger rival under mismatch. The honest framing: the reflex is cheaper when calm and more
  noise-robust on sustained pushes, while the observer is more robust to wrong masses and needs no training.
- The SNN shares the encoder, so it inherits the noise result. It will also inherit the MLP's mismatch weakness
  unless it's trained with randomized masses.

**Cost summary (MACs per tick).**

| | calm, no noise | calm, noise 1e-2 | calm, +2 kg payload | during sustained push (nominal) |
|---|---|---|---|---|
| Periodic MLP | 2,240 | 2,240 | 2,240 | 2,240 |
| Observer | ~112 | ~112 | ~112 | ~112 |
| Delta MLP k=0.01 | 0 | 1,864 | 526 | 451 |
| Delta MLP + floor | 0 | 15 | — | 451 (936 at noise 1e-2) |

## Next
- Optional: domain-randomized MLP retraining, or a friction-mismatch test (where the observer might actually break).
- Run the SNN through both sweeps once it exists; its membrane time constant τ is the extra filter to tune.
