# Periodic MLP cost and host latency

This measures the delivered `models/mlp_history.npz`, without retraining or
changing its control behavior. The scope is the additional reflex; base OSC,
physics, sensor acquisition, communication, and actuator limiting are excluded.
No power, energy, FPGA resource, or FPGA latency measurement is available.

## Algorithmic work

The dense network is **28 -> 64 tanh -> 7**, with no output bias. Its rest
output is precomputed once at load time, then subtracted each tick.

| Quantity | Per 1 ms tick | Per second at 1 kHz |
| --- | ---: | ---: |
| Input-to-hidden dense MACs | 1,792 | 1,792,000 |
| Hidden-to-output dense MACs | 448 | 448,000 |
| **Total dense MACs** | **2,240** | **2,240,000** |
| Dense FLOPs, conventional 2 per MAC | 4,480 | 4,480,000 |
| tanh evaluations, listed separately | 64 | 64,000 |

The MAC count is identical when calm, being pushed, and recovering: this is a
periodic dense baseline, with no zero-skipping or event-triggered computation.
All 2,240 matrix weights are nonzero. The conventional FLOP number includes
only the dense MACs and must not be presented as total controller FLOPs.

The current full `MLPReflex.step`, including causal features, normalization,
output scaling, gain, and clipping, has the following algorithmic scalar work:

| Operation | Count per tick |
| --- | ---: |
| Multiplication | 2,254 |
| Addition | 2,233 |
| Subtraction | 28 |
| Division | 35 |
| Negation | 7 |
| Clipping comparison | 14 |
| tanh evaluation | 64 |

For these scalar counts, a length-n dot has n multiplies and n-1 additions.
This differs from counting every MAC as two FLOPs; do not add the two tables
together. tanh has no assigned FLOP or energy equivalent. The counts are
algorithmic operations on finite inputs, not CPU instructions, SIMD/FMA
instructions, memory transactions, or measured joules. Copies, allocation,
dtype conversions, and Python overhead are excluded from these arithmetic
counts but included when they occur inside the timed reflex call.

## Model storage

There are **2,304 FP32 trainable parameters**: 2,240 matrix weights and 64 hidden
biases. Their raw storage is **9,216 bytes = 9 KiB**. Including normalization
scales, torque limits, and the cached rest output, numeric model arrays occupy
9,440 bytes. This excludes Python objects, metadata, feature history and
temporary arrays. The three seven-element float64 history/reference arrays
occupy another 168 bytes after initialization. The compressed checkpoint is
11,164 bytes, including its metadata and archive overhead.

These are actual floating-point storage values, not an int8/fixed-point
hardware footprint. Inputs from the simulator are float64, so the current
NumPy inference uses float64 intermediate arithmetic with FP32 stored weights.

## Single-tick latency on the current host

Host: **Apple M5 Pro**, macOS 26.7.1 arm64, Python 3.12.14, NumPy 1.26.4.
BLAS-related thread environment variables were set to 1. There is no CPU
affinity, real-time scheduling, or fixed-frequency guarantee.

| Timed scope | Median (us) | P95 (us) | P99 (us) | Observed max (us) |
| --- | ---: | ---: | ---: | ---: |
| Network + normalization/scaling/clipping, precomputed features | 3.834 | 4.167 | 5.125 | 107.834 |
| **Full reflex step**, including feature extraction and gain | **6.333** | **6.919** | **8.250** | **3,973.875** |

The full-reflex median is about 0.63% of a 1 ms tick, but the observed maximum
exceeds a tick. The outlier is retained; this benchmark did not identify its
cause. An observed maximum is not a worst-case execution-time bound, and the
median/P99 cannot establish a hard real-time deadline. These host results also
do not establish the kickoff's FPGA timing target or end-to-end reaction time.

Method: replay the first held-out sustained episode from seed 200, with frozen
weights. Observations are constructed before timing. After three complete
warmup passes, ten measured passes supply **19,760 single-call samples per
scope**, without batching. Each replay resets causal history at its start.
`perf_counter_ns` surrounds each call. The first tick's initialization is
included; model loading, reset calls, replay bookkeeping, simulation and CSV
writing are not. Timer overhead is recorded separately and not subtracted.
Raw samples retain repeat, tick, phase, and wall-clock latency.

Full-reflex medians are 6.375 us in calm segments and 6.292 us during the push;
the similar numbers do not indicate skipped work. Both execute the same dense
network every tick. The measured replay contains one disturbance trajectory;
it is a host microbenchmark, not an exhaustive timing stress test.

## Reproduce and compare fairly

```sh
source .venv/bin/activate
python -m scripts.mlp_cost --repeats 10 --warmup 3
```

`results/mlp/cost/summary.json` records counts, latency distributions, timing
scope, environment and hashes. `latency_samples.csv` contains individual
measurements. The CPU model label was verified separately with
`sysctl -n machdep.cpu.brand_string`; arithmetic counts are stable across hosts,
while measured latency will vary. Parameterized tests also check that counts
follow actual matrix dimensions rather than assuming this network size.

For future SNN/delta-MLP comparisons, use the same inputs and task conditions.
Report dense MACs, synaptic accumulations, neuron/leak updates, and encoder work
as separate counts; a synaptic addition and a MAC are not interchangeable energy
measurements. Include work in quiet periods as well as pushed periods. Actual
power/energy or FPGA resources need their own hardware or synthesis evidence.
