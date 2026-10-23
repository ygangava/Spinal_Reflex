# Momentum observer reflex

A classical, non-learned baseline for the reflex slot. It estimates the push torque from the robot's own
physics model and cancels it. No training. Used as the "why learn at all?" reference for the MLP and SNN.

## 1. Idea

- The robot knows its model: inertia `M(q)`, gravity + Coriolis `b`, joint damping, and the motor torque it applied.
- From those it predicts how the arm's momentum `p = M·q̇` *should* change.
- It measures how the momentum *did* change.
- The gap is the push. The observer keeps a running estimate `r` of it and applies `−r`.

## 2. Math

Dynamics: `M·q̈ + b = τ + τ_damp + τ_ext`, so the momentum obeys

```
ṗ = τ + τ_damp − b + Ṁ·q̇ + τ_ext
```

Observer:

```
r = K_O · [ p − p₀ − ∫ (τ + τ_damp − b + Ṁ·q̇ + r) dt ]
```

Differentiating gives `ṙ = K_O·(τ_ext − r)`: `r` follows the true external torque through a first-order
low-pass filter with time constant **1/K_O**. No acceleration is needed: velocity is integrated, never differentiated.

**Discrete version** (one update per 1 ms tick k):

```
I[k] = I[k−1] + DT · ( τ_prev − b[k−1] + τ_damp[k−1] + (M[k] − M[k−1])/DT · q̇[k−1] + r[k−1] )
r[k] = K_O · ( M[k]·q̇[k] − p₀ − I[k] )
output = −r[k]
```

The sum uses tick k−1 values because `τ_prev` and the state at k−1 are what moved `q̇[k−1] → q̇[k]`.

**Stability.** The error obeys `e[k] = (1 − K_O·DT)·e[k−1]`:

| K_O·DT | Behavior |
|---|---|
| < 1 | smooth first-order lag |
| = 1 (K_O = 1000) | exact after one tick ("deadbeat") |
| 1–2 | overshoots, oscillates |
| > 2 | diverges |

## 3. Code

| File | What |
|---|---|
| [`observer_reflex.py`](../observer_reflex.py) | `MomentumObserver(Reflex)` |
| [`tests/test_observer.py`](../tests/test_observer.py) | 9 tests |
| [`scripts/observer.py`](../scripts/observer.py) | ablation, K_O sweep, held-out comparison, figures |
| `results/observer/` | CSVs, `summary.json`, figures |

`MomentumObserver(k_o, xml=MODEL, damping=True, mdot=True)`:

- **`__init__`** loads its **own** MuJoCo model/data copy. It can't disturb the sim, and model mismatch later = pass a
  different `xml`. Not privileged: it never sees the push.
- **`dynamics(q, dq)`** writes the sensed state into its copy, runs `mj_forward`, and returns `M` (7×7, includes
  armature), `b = qfrc_bias`, `τ_damp = qfrc_passive` (−1.0·q̇). Values are `.copy()`'d because MuJoCo reuses buffers.
- **`reset()`** clears `p0`, `I`, `r`, `prev`. `p0 = None` marks the first tick.
- **`compute(obs)`** — first tick: store `p₀`, set `I = r = 0`. Later ticks: the discrete update above.
  Returns `(−r, −r)` (torque, estimate), the same sign as the oracle's `−JᵀF`.
- **`damping`, `mdot`** are ablation switches only. Both on = the real observer.

Inputs used from `obs`: `q`, `dq`, `tau_prev` (the motor torque actually applied last tick, after limits).

## 4. Tests

`python -m pytest -q tests/test_observer.py` → 9 pass (48 total).

| Test | Checks |
|---|---|
| silent without push | estimate < 1e-9 on a quiet episode (actual ~1e-16) |
| not privileged | never gets the teacher |
| tracks teacher | mid sustained push, estimate within 1% of `−JᵀF` |
| first-order lag (K_O 50/100/200) | zero before the push acts; n ticks later = `1 − (1 − K_O·DT)^(n+1)` of a step |
| gain 0 = baseline | identical trajectory to no reflex |
| reset | reused object = fresh object |
| damping ablation | dropping damping leaves a false estimate after the push ends |

## 5. Results

All runs: home pose, hand pushes, noise-free, DT 1 ms, torque + rate limits on.
Observer and MLP get +1 tick output delay; the oracle gets +2 (it skips the 1-tick sensing lag). So every reflex has a
**2 ms total latency** from push to cancelling torque.

### Ablation (`python -m scripts.observer ablate`)
K_O = 100, one 30 N sustained push.

| Variant | False estimate after push ends | Peak error |
|---|---|---|
| Full observer | 0.001 N·m | 1.49 mm |
| No damping term | 0.368 N·m | 1.64 mm |
| No Ṁ·q̇ term | 0.013 N·m | 1.49 mm |

Damping matters most. Leaving it out reads ordinary arm motion as a push.

### K_O sweep (`python -m scripts.observer sweep`, validation seed 100)
Mean peak hand error.

| K_O | Lag | Sustained | Impacts |
|---|---|---|---|
| 25 | 40 ms | 6.49 mm | 23.91 mm |
| 50 | 20 ms | 3.76 | 22.47 |
| 100 | 10 ms | 2.24 | 21.95 |
| 200 | 5 ms | 1.53 | 21.90 |
| 500 | 2 ms | 1.12 | 21.78 |
| 1000 | 1 ms | 0.98 | 21.72 |
| MLP | | 1.01 | 21.76 |
| Oracle | | 0.98 | 21.73 |

Figure: `results/observer/ko_sweep.png`.

### Held-out comparison (`python -m scripts.observer compare`, seeds 200/201)
Same cases and metrics as `docs/MLP.md`. Sustained = 20 / 40 N, 150 / 250 ms, 10 ms ramps, diagonal directions.
Impacts = 3 / 6 N·s. Presses = 30 N, rise 20 ms → 2 s, 2 s hold.

| Metric | OSC | High-KP | MLP | Obs K_O=100 | Obs K_O=1000 | Oracle |
|---|---|---|---|---|---|---|
| Sustained peak (mm) | 28.9 | 14.1 | 0.755 | 2.11 | **0.687** | 0.687 |
| Sustained AUC (mm·s) | 8.46 | 3.18 | 0.221 | 0.643 | **0.208** | 0.208 |
| Impact peak (mm) | 30.9 | 22.3 | 17.6 | 17.6 | 17.7 | 17.7 |
| Impact AUC (mm·s) | 7.29 | **2.97** | 3.90 | 3.91 | 3.95 | 3.96 |
| Press hold deflection (mm) | 43.1 | 16.3 | 0.33 | ≈0 | ≈0 | ≈0 |
| Quiet max error (mm) | ≈0 | ≈0 | 3e-5 | ≈0 | ≈0 | ≈0 |

OSC / high-KP / MLP / oracle columns reproduce `results/mlp/summary.json` exactly.
Figures: `comparison.png` (bars), `trace.png` (one push: hand error, and estimate vs teacher at onset).

### Latency sweep
Held-out sustained pushes, total latency 1 / 2 / 3 / 6 / 11 ms. Every reflex loses about **0.2 mm per extra ms**.
Observer K_O = 1000 lies exactly on the oracle line; the MLP sits just above it. Figure: `latency.png`.

### Presses
Every reflex is rigid at every rise time (all ≈0–0.3 mm vs OSC 43 mm). None yields to slow presses.
Figure: `press.png`.

### Cost per tick

| | Own arithmetic | Plus |
|---|---|---|
| MLP | 2,240 MACs | feature extraction |
| Observer | ~112 MACs (2 matrix-vector products) + ~150 add/sub/div | `M(q)`, `b`, damping — **already computed by the OSC every tick** |

Both run every tick, calm or pushed.

## 6. What it means

- **Perfect model, no noise:** a training-free observer at K_O = 1000 *is* the oracle at its latency. It beats the
  MLP with ~20× fewer of its own ops. Learning adds nothing to rejection here.
- **Realistic gains:** real observers use K_O ≈ 50–200, because high gain passes sensor noise straight through. At
  K_O = 100 the MLP is ~3× better (0.755 vs 2.11 mm). But the MLP's finite-difference acceleration is also noise-sensitive,
  so this is only a fair comparison once both see noise.
- **Impacts:** all reflexes ≈ 17.6 mm, limited by the 1000 N·m/s torque-rate limit, not by the estimator.
- **Presses:** observer, MLP and oracle all cancel every external force, so none yields. That is still unsolved.
- **For the SNN:** it can't beat this on rejection. Its case is (1) far fewer ops via event-driven compute, especially
  when calm, (2) possible robustness to noise / model mismatch, (3) yielding to slow presses.

## 7. Limits and next

- Home pose, hand pushes, no sensor noise, perfect model (the observer uses the same model as the sim).
- Next: add sensor noise and model mismatch (wrong payload/link masses via `xml=`), rerun MLP and observer.
  Then the high-pass observer variant for the press test.

## Reproduce

```sh
source venv/bin/activate
python -m pytest -q tests/test_observer.py
python -m scripts.observer ablate
python -m scripts.observer sweep      # ~50 s on 8 workers
python -m scripts.observer compare    # ~2 min on 8 workers; needs models/mlp_history.npz
```
