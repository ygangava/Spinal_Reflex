# Periodic MLP reflex baseline

This is a supervised ANN baseline for the midterm. It runs every 1 ms in the
existing `Reflex` slot. It is not an SNN, an event-triggered controller, or an
FPGA implementation. The experiment holds the Panda's home pose and pushes the
hand in a noise-free simulator.

## Model and causal inputs

The delivered model is **28 -> 64 tanh -> 7**, with 2,304 trainable parameters
and 2,240 dense MACs per tick (2.24 million MAC/s at 1 kHz). That count excludes
feature extraction, normalization, activations, biases, and output clipping.
PyTorch trains the model; NumPy loads the exported `.npz` for inference, without
pickle or a PyTorch dependency in the controller.

See [MLP_COST.md](MLP_COST.md) for the operation breakdown, memory accounting,
and measured single-tick host latency. Regenerate those measurements with
`python -m scripts.mlp_cost`.

The 28 inputs are four seven-dimensional vectors:

1. Joint displacement from the episode's initial posture.
2. Current joint velocity.
3. Backward finite-difference acceleration, `(dq[k] - dq[k-1]) / DT`.
4. Previous **applied** motor torque minus the initial hold torque.

The last vector is exposed as `obs['tau_prev']` in `osc.py`. It includes the
effects of the existing magnitude and slew-rate limits. At reset, it is the
initial hold torque. No force, teacher label, push timing, episode identifier,
or tick counter is an input. Offline and online feature definitions are tested
for exact agreement. `--features state` also supports the kickoff's 14-input
joint displacement/velocity baseline; use a separate output filename for it.

History matters because a successfully cancelled sustained force can coexist
with almost zero tracking error. Acceleration plus the previous applied torque
helps the network distinguish that condition from a force-free arm. This is a
learned dynamics-residual estimator; it is not an error-only MLP. Future SNN and
delta-MLP comparisons need the same information, or an explicit input ablation.

## Training and latency

The target at tick `k` is the previous tick's cancellation torque,
`tau_teacher[k-1] = -J[k-1]^T F[k-1]`. Measurements at the start of tick `k`
can reflect force applied in tick `k-1`; they cannot reveal a new force in tick
`k` before it acts. The label is therefore shifted by one tick rather than
requiring the network to anticipate onset. No extra subtraction of OSC torque
is applied to this label.

Training seeds are **0, 1, 2**; validation seed is **100**. Each seed has the
existing 24 impact and 36 sustained-push conditions. Entire episodes, not
individual adjacent time samples, are assigned to splits. Scales are fitted
only on training data. Initial demonstrations use OSC alone and an oracle with
5 ms output delay; this supplies both large-error and nearly-cancelled states.
A second data-aggregation round adds the student's own closed-loop trajectories
with offline teacher labels. The learned reflex never receives those labels.

The default fit uses CPU PyTorch, seed 17, Adam at 0.002, batch size 1,024, up
to 80 epochs per round, and the best validation checkpoint within each round.
The loss is MSE after per-joint torque scaling. Input RMS normalization has
physical floors for low-variance channels. The network's rest prediction is
subtracted, so zero features produce zero residual torque. Residuals are
clipped to the joint magnitude limits; base + residual still goes through the
existing total-torque magnitude and slew-rate limits. Gain/enable is inherited
from `Reflex.step`.

Evaluation defaults to another **1 ms output delay**, giving the MLP a nominal
two-tick sensing-plus-output path. Its oracle comparator is delayed **2 ms**
for the same reason. This is a simulated timing convention, not a measured
hardware latency.

## Reproduce

From the repository root, using Python 3.12:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mlp.txt
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
git -C mujoco_menagerie sparse-checkout set franka_emika_panda
python -m pytest -q
python -m scripts.mlp train
python -m scripts.mlp validate
python -m scripts.mlp evaluate --out results/mlp
```

The recorded Menagerie revision is in the model metadata. Reproducing the
reported numbers requires that revision; if the clone has advanced, fetch and
checkout the recorded commit. The separate requirements file is the minimal
MLP environment, leaving the team's original full environment intact.

For an existing model, skip training and evaluate directly. To insert it in
another script:

```python
from osc import DT, run_episode
from mlp_reflex import MLPModel, MLPReflex

model = MLPModel.load('models/mlp_history.npz')
log = run_episode(push, reflex=MLPReflex(model, DT), reflex_delay=1)
```

`train` saves model weights to `models/mlp_history.npz` and a training record to
`runs/mlp/training.json`. `validate` defaults to `runs/mlp/validation`.
Evaluation writes per-episode metrics, a summary with configuration/source
hashes, and PNG/PDF comparison plots. `example_trace.npz` stores the plotted
episode, selected in advance as the first sustained episode of the first seed.

## Held-out results

The frozen two-round model was evaluated in **488 episodes** (122 conditions
times four controllers). Means over the 32 paired episodes in each disturbance
group, with 1 ms control ticks and magnitude/slew-rate limits enabled:

| Disturbance / metric | OSC | High-KP OSC | MLP | Oracle (2 ms) |
| --- | ---: | ---: | ---: | ---: |
| Sustained: peak error (mm) | 28.902 | 14.079 | **0.755** | 0.687 |
| Sustained: error AUC (mm s) | 8.462 | 3.177 | **0.221** | 0.208 |
| Impact: peak error (mm) | 30.851 | 22.306 | **17.569** | 17.720 |
| Impact: error AUC (mm s) | 7.290 | **2.968** | 3.898 | 3.958 |

The MLP reduces sustained-push peak error by 97.4% relative to OSC and 94.6%
relative to high-KP OSC. Impact peak error improves 43.1% relative to OSC, but
impact AUC is worse than high-KP: this is not a win on every recovery metric.
Delayed exact force cancellation is a reference, not a proof of optimality
under actuator limits; the slightly lower MLP impact peak does not establish
that it outperforms an optimal controller.

The two quiet runs have maximum post-onset error about **0.000033 mm**. In
56 press diagnostics, mean late-hold deflection is **0.333 mm** for the MLP,
versus 43.149 mm for OSC and 16.253 mm for high-KP. The model cancels deliberate
slow force too: the yielding requirement remains unsolved.

Artifacts are in `results/mlp/`: `results.csv` has one row per episode;
`summary.json` records the split and means; `comparison.png` / `comparison.pdf`
show the comparison and a preselected trajectory; `training.json` records the
two fits; `environment.txt` records installed packages. The tests passed
**39/39**, including causal input timing, training/inference feature parity,
checkpoint round-trip, reset/gain behavior, operation counts, and the existing simulator checks.
The actual training run also checked PyTorch/NumPy prediction agreement.

## Held-out protocol and limits

Test seeds **200, 201** are used only after freezing the model. Impact tests use
the existing held-out impulses (3 and 6 N s) and diagonal directions. Sustained
tests use unseen forces (20 and 40 N), durations (150 and 250 ms), and diagonal
directions. Each controller receives identical paired pushes. There are 32
impact and 32 sustained episodes per controller, plus 56 press diagnostics and
two quiet episodes. The comparisons are OSC, high-KP OSC, MLP, and delayed oracle.

Peak hand error and error AUC cover onset through one second after force release.
For presses, mean deflection is also measured over the final quarter of the
constant-force hold. Saturation fraction counts requested torques exceeding
magnitude limits. Torque-limited fraction counts any discrepancy between
requested and applied torque, including slew-rate limiting. These are fractions
of the complete episode. Results are descriptive paired means from two seeds,
not confidence bounds or evidence of real-world robustness.

Slow presses are a diagnostic, not part of the training objective: the label
asks for cancellation of every external force, so yielding to deliberate
presses is not expected. The model has not been validated on moving trajectories,
other initial poses, link pushes, sensor noise, model mismatch, or real hardware.
Finite-difference acceleration will be sensitive to velocity noise. No speed,
energy, sparsity, or FPGA-resource advantage is claimed by this baseline.
