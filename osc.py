"""Stage 1 baseline: Panda + hand-written OSC in plain MuJoCo, one push, logged.

Model: mujoco_menagerie/franka_emika_panda/scene.xml
  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
  
mjpython osc.py
"""
import mujoco
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

MODEL = "mujoco_menagerie/franka_emika_panda/scene.xml"
m = mujoco.MjModel.from_xml_path(MODEL) # robot desc that doesn’t change: links, masses, joints, actuators
d = mujoco.MjData(m) # state that changes every step: joint angles, velocities, forces, positions.

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
KP_POS, KP_ORI, KP_NULL = 150.0, 150.0, 10.0
KD_POS, KD_ORI, KD_NULL = (2 * np.sqrt(k) for k in (KP_POS, KP_ORI, KP_NULL))

jp = np.zeros((3, m.nv))
jr = np.zeros((3, m.nv))
M_full = np.zeros((m.nv, m.nv))


def full_inertia(dst):
    """Dense joint-space inertia matrix. mj_fullM's signature changed across MuJoCo versions."""
    if hasattr(d, "qM"):
        mujoco.mj_fullM(m, dst, d.qM)    # older MuJoCo
    else:
        mujoco.mj_fullM(m, d, dst)       # newer MuJoCo (3.16+)


def osc():
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

    a = np.r_[KP_POS * e_p - KD_POS * v[:3],             # desired task acceleration (PD)
              KP_ORI * e_o - KD_ORI * v[3:]]
    F = Lam @ a                                          # task wrench

    # Null space: 7 joints vs 6 task dofs leaves 1 dof (elbow swivel). Pull it to q0 without disturbing the hand.
    N = np.eye(7) - J.T @ Lam @ J @ Minv
    tau_null = M @ (KP_NULL * (q0 - d.qpos[:7]) - KD_NULL * dq)

    tau = J.T @ F + N @ tau_null + d.qfrc_bias[:7]       # qfrc_bias = gravity + Coriolis compensation
    return tau, e_p, e_o


# Run
DT = m.opt.timestep
T_END = 6.0
PUSH_T, PUSH_DUR = 1.0, 0.05  # starts at t=1 s, lasts 50 ms
PUSH_F = np.array([150.0, 0, 0])                          # N, world frame, on the hand body

steps = int(T_END / DT)
log = np.zeros((steps, 1 + 3 + 3 + 7 + 7))               # t, e_p, e_o, q, tau

yes_viewer = True
if(yes_viewer):
    with mujoco.viewer.launch_passive(m, d) as viewer:
        for k in range(steps):
            if not viewer.is_running():
                break
            start = time.time()
            t = k * DT
            mujoco.mj_step1(m, d)        # kinematics + inertia + bias for the CURRENT state
            q = d.qpos[:7].copy()      # .copy() matters: d.qpos changes in place during step2
            dq = d.qvel[:7].copy()
            tau, e_p, e_o = osc() # get the torques
            tau_res = np.zeros(7)        # <-- reflex residual goes here
            d.ctrl[:7] = tau + tau_res # send torques to motors
            
            # Apply push if it's time
            d.xfrc_applied[hand, :3] = PUSH_F if PUSH_T <= t < PUSH_T + PUSH_DUR else 0.0
            
            mujoco.mj_step2(m, d)
            tau_applied = d.actuator_force[:7].copy()
            log[k] = np.r_[t, e_p, e_o, q, dq, tau + tau_res, tau_applied]
            
            viewer.sync()                                          # redraw
            
            time.sleep(max(0, DT - (time.time() - start)))         # run at real time
        
else:
    for k in range(steps):
        t = k * DT
        mujoco.mj_step1(m, d)        # kinematics + inertia + bias for the CURRENT state
        tau, e_p, e_o = osc() # get the torques
        tau_res = np.zeros(7)        # <-- reflex residual goes here
        d.ctrl[:7] = tau + tau_res # send torques to motors
        
        # Apply push on EE hand if time
        d.xfrc_applied[hand, :3] = PUSH_F if PUSH_T <= t < PUSH_T + PUSH_DUR else 0.0
        mujoco.mj_step2(m, d)        # dynamics + integrate one step
        log[k] = np.r_[t, e_p, e_o, d.qpos[:7], d.ctrl[:7]]

np.save("run.npy", log)
pos_mm = 1000 * np.linalg.norm(log[:, 1:4], axis=1)
ori_deg = np.degrees(np.linalg.norm(log[:, 4:7], axis=1))
pre = log[:, 0] < PUSH_T
print(f"dt={DT*1000:.1f} ms, {steps} steps")
print(f"pre-push  max pos err {pos_mm[pre].max():.3f} mm, ori {ori_deg[pre].max():.3f} deg")
print(f"peak      pos err {pos_mm.max():.1f} mm at t={log[pos_mm.argmax(), 0]:.3f} s, ori {ori_deg.max():.2f} deg")
print(f"final     pos err {pos_mm[-1]:.3f} mm, ori {ori_deg[-1]:.3f} deg")
print(f"final joint drift from q0 {np.abs(log[-1, 7:14] - q0).max():.4f} rad")


'''
Impulse, force × duration: 40 N × 0.05 s = 2 N·s. 
A short hard push and a longer soft one with the same impulse disturb the arm similarly. 
For experiments will want a set of pushes w diff strengths and directions
Can move this into a new disturbance.py file
'''
