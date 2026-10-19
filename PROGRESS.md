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


## Decisions
<!-- e.g. control rate, push magnitude/duration, recovery metric, Q formats -->
- In robosuite the physics steps every 2 ms, and the base controller recomputes torques at every physics step, so 500 Hz.
  - error space
  - End effector error should go into event encode
  - joint error - log and see if it is useful or a diagnostic for end effector error later
  - MuJoCo is deterministic for the same initial state on the same machine. robosuite, however, adds initialization noise to the starting joint positions by default, so runs differ unless you pass initialization_noise=None or seed it. Turn it off for the baseline. Your test is to run no-push twice and diff the joint logs. They should match exactly, and the joints should stay nearly constant during a hold.


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

LLM says:
- Where to insert the residual. Check how your robosuite version structures env.step. In recent versions it loops over sim substeps, around 500 Hz. Every substep it recomputes the OSC torques and writes them to data.ctrl, but it only updates the goal at control_freq, which defaults to 20 Hz. That gives you two options:

- Between env.step calls, via data.qfrc_applied. This is the easy option: MuJoCo adds those joint forces to the dynamics. The catch is that your reflex then only updates at 20 Hz, which is far too slow to be a reflex. Raising control_freq helps but changes the OSC behaviour too.
Inside the substep loop. Subclass the OSC controller and override run_controller to return its torques plus your residual. The reflex then runs at sim rate, and it literally matches the CLAUDE.md description: “added to OSC output”. OSC’s torque clipping also applies to the sum.

- I’d lean toward the second option. Whichever you choose, grep the robosuite source to confirm where torques get written, and make sure nothing else writes to qfrc_applied. Keep the push on xfrc_applied, which is a separate array, so the two don’t collide.

