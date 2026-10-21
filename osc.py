"""Stage 1 baseline: Panda + hand-written OSC in plain MuJoCo, one push, logged.

Model: mujoco_menagerie/franka_emika_panda/scene.xml
  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
  
mjpython osc.py
"""
import mujoco
import mujoco.viewer  # plain `import mujoco` doesn't load the viewer submodule
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import contextlib
import subprocess # for the git
import json
import os
from dataclasses import asdict
from collections import deque

from disturbance import push, grid_pushes
from reflex import NoReflex

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "mujoco_menagerie/franka_emika_panda/scene.xml")
# MODEL = "mujoco_menagerie/franka_emika_panda/scene.xml"
m = mujoco.MjModel.from_xml_path(MODEL) # robot desc that doesn’t change: links, masses, joints, actuators
d = mujoco.MjData(m) # state that changes every step: joint angles, velocities, forces, positions.

# 1 ms tick to match the Franka's 1 kHz control loop (Menagerie default is 2 ms)
m.opt.timestep = 0.001
DT = m.opt.timestep

# Menagerie's arm actuators are position servos (force = kp*(ctrl - q) - kv*qdot).
'''
Make the motors take torque commands. 
Menagerie’s motors are position servos (you send an angle). 
Need to send torques, so the script rewrites them so that motor force = ctrl.
'''

# Turn the 7 arm actuators into pure torque motors: force = 1*ctrl + 0.
TAU_MAX = np.array([87, 87, 87, 87, 12, 12, 12.0])   # Panda joint torque limits (Nm)
m.actuator_gainprm[:7, :] = 0
m.actuator_gainprm[:7, 0] = 1
m.actuator_biasprm[:7, :] = 0
m.actuator_ctrlrange[:7] = np.c_[-TAU_MAX, TAU_MAX]

# Franka torque-rate limit, every joint (Nm/s). MuJoCo doesn't enforce this, so limit_torque() does
TAU_RATE_MAX = 1000.0


def limit_torque(cmd, prev):
    # Rate limit against last tick's applied torque, then magnitude limit.
    # prev is None on the first tick: the arm was already holding this torque before t = 0,
    # so ramping up from 0 would make it sag under gravity (~25 ticks to reach joint 2's hold torque)
    if prev is not None:
        step = TAU_RATE_MAX * DT
        cmd = prev + np.clip(cmd - prev, -step, step)
    return np.clip(cmd, -TAU_MAX, TAU_MAX)

# Put Arm in Start pose
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
mujoco.mj_forward(m, d)
hand = m.body("hand").id

x_d = d.xpos[hand].copy()                  # target position = start pose
R_d = d.xmat[hand].reshape(3, 3).copy()    # target orientation = start pose
q0 = d.qpos[:7].copy()                     # null-space posture target

# Gains. spring and damper strengths
# Because F = Lambda * a, the task behaves like a unit mass: kd = 2*sqrt(kp) is critical damping.
# PID -- P - proportional to error --- hard it pulls back toward the target. Bigger means stiffer.
# D - derivative of error --- damperit resists velocity so the arm doesn’t oscillate. 2*sqrt(KP) is critical damping, the fastest return with no overshoot
# KP_POS, KP_ORI, KP_NULL = 150.0, 150.0, 10.0
# KD_POS, KD_ORI, KD_NULL = (2 * np.sqrt(k) for k in (KP_POS, KP_ORI, KP_NULL))

# make gains as parameters
def make_gains(kp_pos=150.0, kp_ori=150.0, kp_null=10.0):
    # KD = 2 * sqrt(KP) # critical damping, fastest return with no overshoot
    return {"kp_pos": kp_pos, "kp_ori": kp_ori, "kp_null": kp_null,
            "kd_pos": 2 * np.sqrt(kp_pos), "kd_ori": 2 * np.sqrt(kp_ori), "kd_null": 2 * np.sqrt(kp_null)}
    
DEFAULT_GAINS = make_gains()
HIGH_KP_GAINS = make_gains(kp_pos=400.0, kp_ori=300.0, kp_null=20.0)


jp = np.zeros((3, m.nv))
jr = np.zeros((3, m.nv))
M_full = np.zeros((m.nv, m.nv))


def full_inertia(dst):
    """Dense joint-space inertia matrix. mj_fullM's signature changed across MuJoCo versions."""
    if hasattr(d, "qM"):
        mujoco.mj_fullM(m, dst, d.qM)    # older MuJoCo
    else:
        mujoco.mj_fullM(m, d, dst)       # newer MuJoCo (3.16+)


def osc(g):
    """Return OSC joint torques (7,) plus position and orientation error."""
    # Jacobian: maps joint velocities -> hand linear (jp) and angular (jr) velocity. Arm = first 7 dofs.
    mujoco.mj_jacBody(m, d, jp, jr, hand)
    J = np.vstack([jp, jr])[:, :7]                       # 6x7
    full_inertia(M_full)
    M = M_full[:7, :7]                                   # joint-space inertia
    Minv = np.linalg.inv(M)
    Lam = np.linalg.pinv(J @ Minv @ J.T)                 # task-space inertia; pinv survives singularities

    dq = d.qvel[:7]
    v = J @ dq                                           # hand twist [lin, ang]
    e_p = x_d - d.xpos[hand]
    R_cur = R.from_matrix(d.xmat[hand].reshape(3, 3))
    e_o = (R.from_matrix(R_d) * R_cur.inv()).as_rotvec()  # world-frame axis*angle from current to target

    
    a = np.r_[g["kp_pos"] * e_p - g["kd_pos"] * v[:3],
              g["kp_ori"] * e_o - g["kd_ori"] * v[3:]]

    F = Lam @ a                                          # task wrench

    # Null space: 7 joints vs 6 task dofs leaves 1 dof (elbow swivel). Pull it to q0 without disturbing the hand.
    N = np.eye(7) - J.T @ Lam @ J @ Minv
    tau_null = M @ (g["kp_null"] * (q0 - d.qpos[:7]) - g["kd_null"] * dq)

    tau = J.T @ F + N @ tau_null + d.qfrc_bias[:7]       # qfrc_bias = gravity + Coriolis compensation
    return tau, e_p, e_o

# RUN ORiginally
# created the ppush time, force, steps based off wanted dur, viewer, for range 
# if push, apply push force on hand body, else 0.0, stored info taken afer mj_step1 except for applied force, which is after mj_step2.


# Episode settings
T_END = 3.0          # latest onset ~1.5 s + ~0.8 s recovery + margin

# Per-tick log: name -> width. Every row is the state at the START of tick k (before mj_step2)
LOG_FIELDS = {
    "e_p": 3, "e_o": 3,              # hand position / orientation error
    "ee_pos": 3,                     # hand position (target is x_d, constant for now)
    "q": 7, "dq": 7,                 # joint state
    "tau_base": 7,                   # OSC torque
    "tau_teacher": 7,                # -J^T F: torque that exactly cancels the push (training label, oracle)
    "tau_res_new": 7,                # reflex torque computed this tick (after gain), before the latency queue
    "tau_res": 7,                    # reflex torque applied this tick (computed reflex_delay ticks ago)
    "reflex_est": 7,                 # reflex's slow disturbance estimate, applied this tick (same delay)
    "tau_cmd": 7,                    # torque we asked for (base + residual), before any limit
    "tau_applied": 7,                # torque MuJoCo actually applied (after rate + magnitude limits)
    "f_push": 3,                     # push force this tick, world frame
}

qfrc_push = np.zeros(m.nv)


def teacher_torque(f, body):
    # joint torque that cancels the push force f,
    # applied where xfrc_applied acts: the body's COM (xipos, not xpos).
    # Needs current kinematics, so call after mj_step1
    # JᵀF depends on the arm's current pose: the Jacobian and xipos are both computed in mj_step1

    qfrc_push[:] = 0 # push stored in joint coordinates, qfrc - generalized force
    mujoco.mj_applyFT(m, d, f, np.zeros(3), d.xipos[body], body, qfrc_push)
    return -qfrc_push[:7].copy()


def run_episode(p, gains=DEFAULT_GAINS, reflex=None, reflex_gain=1.0, reflex_delay=0, rate_limit=True,
                t_end=T_END, viewer=False):
    # Fresh start every episode: same pose, zero velocity, zero applied forces
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    body = m.body(p.body).id
    n = round(t_end / DT)
    log = {name: np.zeros((n, w)) for name, w in LOG_FIELDS.items()}
    reflex = reflex if reflex is not None else NoReflex()
    reflex.reset()
    prev = None

    # Latency queue: reflex output computed at tick k is applied at k + reflex_delay. Starts full of zeros
    queue = deque([(np.zeros(7), np.zeros(7))] * reflex_delay)

    # One loop for both modes: nullcontext() stands in for the viewer and yields None
    ctx = mujoco.viewer.launch_passive(m, d) if viewer else contextlib.nullcontext()
    with ctx as v:
        for k in range(n):
            if v is not None and not v.is_running():
                n = k
                break
            start = time.time()

            # Controller sees the current state
            mujoco.mj_step1(m, d)
            tau, e_p, e_o = osc(gains)

            # Push this tick and the torque that cancels it (before the reflex, so the oracle can use it this tick)
            f = push(k, p)
            tau_teacher = teacher_torque(f, body)

            # Reflex sees only what a real robot could sense; only a privileged reflex also gets the teacher
            obs = {"k": k, "q": d.qpos[:7].copy(), "dq": d.qvel[:7].copy(),
                   "e_p": e_p, "e_o": e_o, "tau_base": tau,
                   # Previous applied motor torque is observable, unlike the push force.
                   # At reset the arm is assumed to be holding its initial bias torque.
                   "tau_prev": tau.copy() if prev is None else prev.copy()}
            priv = {"tau_teacher": tau_teacher} if reflex.privileged else None
            tau_res_new, est_new = reflex.step(obs, reflex_gain, priv)
            

            # Latency: everything leaving the reflex is delayed the same
            # Lumps potnetch sensing -> FPGA -> communication -> motor into one delay on the reflex output
            # The reflex computes from fresh inputs at tick k; its output is applied at k + D.
            # For a time-invariant reflex this equals seeing the push D ticks late (only total delay matters).

            queue.append((tau_res_new, est_new))
            tau_res, est = queue.popleft()

            # Total torque through the Franka's rate and magnitude limits
            cmd = tau + tau_res
            applied = limit_torque(cmd, prev if rate_limit else None)
            prev = applied
            d.ctrl[:7] = applied
            d.xfrc_applied[body, :3] = f

            # Log the start-of-tick state before step2 changes it
            log["e_p"][k], log["e_o"][k] = e_p, e_o
            log["ee_pos"][k] = d.xpos[hand]
            log["q"][k], log["dq"][k] = d.qpos[:7], d.qvel[:7]
            log["tau_base"][k], log["tau_teacher"][k] = tau, tau_teacher
            log["tau_res_new"][k], log["tau_res"][k], log["reflex_est"][k] = tau_res_new, tau_res, est
            log["tau_cmd"][k] = cmd
            log["f_push"][k] = f

            # Physics; actuator_force is the torque actually used during this tick
            mujoco.mj_step2(m, d)
            log["tau_applied"][k] = d.actuator_force[:7]

            if v is not None:
                v.sync()
                time.sleep(max(0, DT - (time.time() - start)))

    # Trim if the viewer was closed early
    return {name: a[:n] for name, a in log.items()}


def git_commit():
    # Short hash + "-dirty" if there are uncommitted changes, so a run can be traced to code
    run = lambda *a: subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "--short", "HEAD") + ("-dirty" if run("status", "--porcelain") else "")


def episode_meta(p, seed, gains, reflex_name="none", reflex_gain=1.0, reflex_delay=0, rate_limit=True):
    return {
        "seed": seed,
        "push": asdict(p),
        "dt": DT,
        "gains": gains,
        "reflex": reflex_name,
        "reflex_gain": reflex_gain,
        "reflex_delay_ticks": reflex_delay,
        "tau_max": TAU_MAX.tolist(),
        "tau_rate_max": TAU_RATE_MAX if rate_limit else None,
        "model": os.path.relpath(MODEL, HERE),
        "mujoco": mujoco.__version__,
        "git": git_commit(),
    }


def save_episode(path, log, meta):
    # One .npz per episode: per-tick arrays plus metadata as a JSON string
    np.savez(path, meta=json.dumps(meta), **log)


if __name__ == "__main__":
    SEED = 0
    out = os.path.join(HERE, "runs", f"grid_seed{SEED}")
    os.makedirs(out, exist_ok=True)

    for p in grid_pushes(SEED, DT):
        log = run_episode(p)
        save_episode(os.path.join(out, f"push_{p.push_id:02d}.npz"), log, episode_meta(p, SEED, DEFAULT_GAINS))

        peak = 1000 * np.linalg.norm(log["e_p"], axis=1).max()
        sat = (np.abs(log["tau_cmd"]) > TAU_MAX).any(axis=1).sum()
        print(f"push {p.push_id:2d}  J={p.impulse:3.1f}  dur={p.dur_ticks:2d}  "
              f"dir=({p.direction[0]:+.2f},{p.direction[1]:+.2f})  peak={peak:5.1f} mm  sat_ticks={sat}")

