"""
for test
- claude did 

found 

impulse	peak err (x / y)	worst torque/limit at 20 ms	at 80 ms
2 N·s	9 / 15 mm	0.65	0.47
4 N·s	18 / 31 mm	1.11 (−x only)	0.76
7.5 N·s	34–43 / 60 mm	2.04	1.37


"""
import sys, numpy as np, mujoco
src = open("osc.py").read().split("# Run")[0]
g = {}
exec(compile(src, "osc.py", "exec"), g)
m, d, osc, hand, q0 = g["m"], g["d"], g["osc"], g["hand"], g["q0"]
LIM = np.array([87, 87, 87, 87, 12, 12, 12.0])
DT = m.opt.timestep
ONSET = 500

def run(J, dur_ticks, direction, shape, body=hand, T=3.0):
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    dur = dur_ticks * DT
    F = J / (dur * 2 / np.pi) if shape == "sine" else J / dur
    n = int(T / DT)
    err = np.zeros(n); sat = np.zeros(n, bool); ratio = 0.0; req = 0.0
    for k in range(n):
        mujoco.mj_step1(m, d)
        tau, e_p, e_o = osc()
        d.ctrl[:7] = tau
        j = k - ONSET
        if 0 <= j < dur_ticks:
            s = np.sin(np.pi * (j + 0.5) / dur_ticks) if shape == "sine" else 1.0
            d.xfrc_applied[body, :3] = F * s * direction
        else:
            d.xfrc_applied[body, :3] = 0
        mujoco.mj_step2(m, d)
        err[k] = np.linalg.norm(e_p)
        r = np.abs(tau) / LIM
        req = max(req, r.max()); sat[k] = (r > 1).any()
    post = err[ONSET:]
    peak = post.max()
    # recovery: first tick after peak where err stays below 1 mm for good
    above = np.where(post > 1e-3)[0]
    rec = (above[-1] + 1) * DT if len(above) else 0.0
    return F, peak * 1000, req, sat.sum(), rec * 1000

dirs = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0]}
print("shape dur  J    dir  Fpeak  peak_mm  max|tau|/lim  sat_ticks  recover_ms(<1mm)")
for shape in ["sine", "rect"]:
    for dur in [10, 25, 40]:
        for J in [2, 4, 7.5, 10]:
            for dn, dv in dirs.items():
                F, pk, req, ns, rec = run(J, dur, np.array(dv, float), shape)
                print(f"{shape:4s} {dur*2:3d}ms {J:4.1f} {dn}  {F:6.0f}  {pk:6.1f}   {req:5.2f}   {ns:4d}   {rec:6.0f}")
