# Spinal Reflexes for Robots: Annotated Evidence Base for an Event-Driven SNN Residual Reflex on a Franka Panda

Most of the project's framing is well supported: the slow-policy / 1 kHz-controller hierarchy, the OSC and impedance foundations, the Franka torque limits and the reflex-latency numbers all have primary sources. Three claims need rewording before a reviewer sees them. First, "end-effector stiffness depends on (Λ⁻¹)ᵢᵢ" mixes up stiffness and inertia. Second, "feedback can only act after the robot has moved" is true but incomplete, because momentum observers also have finite bandwidth. Third, neuromorphic energy savings are mostly measured on Loihi, not on FPGA. Even the strongest real-robot result, an RL-trained SNN on Loihi doing force control with a real KUKA arm (arXiv:2403.08928), reports "one order of magnitude less energy usage in comparison to state-of-the-art low-energy edge-hardware," which it says needs "hundreds of µJ per inference." The strongest rival a reviewer will raise is the De Luca momentum observer. It is a model-based, 1 kHz, one-parameter disturbance estimator, and the Franka already exposes its output through FCI. Your SNN must be benchmarked against it, not only against plain OSC.

## TL;DR
- **Well supported (cite directly):** FCI/libfranka runs at 1 kHz with joint torque limits of 87 Nm (A1–A4) and 12 Nm (A5–A7) and a 1000 Nm/s torque-rate limit. VLA and visuomotor policies run at roughly 3–30 Hz with roughly 30–140 ms latency (π0: 73 ms on an RTX 4090; OpenVLA: about 6 Hz; Diffusion Policy: 0.1 s). Human short- and long-latency reflexes take 20–45 ms and 50–100 ms, and voluntary responses take more than 100 ms. Khatib 1987, Hogan 1985 and Albu-Schäffer et al. 2003 are the canonical control foundations.
- **Needs rewording:** Λ⁻¹ governs the *effective mass*, i.e. the initial acceleration under a push, not the static stiffness, which is set by K. Momentum observers have a first-order lag set by the gain K_O, so they are faster than error-driven feedback but not instantaneous. The franka_ros damping rule D = 2√K is "critical damping" only for unit apparent mass. The 1000 Nm/s torque-rate limit means even a perfect reflex needs at least about 87 ms to ramp a proximal joint to full torque on real hardware.
- **Weakest / most contestable:** SNN energy advantages come from neuromorphic chips, not from an FPGA running a tiny network. Loihi measures 23.6 pJ per synaptic op. DeWolf et al. 2023 (Neuromorph. Comput. Eng. 3:014007) report that "the neuromorphic implementation on Loihi reduces the dynamic energy cost per inference by two orders of magnitude," but that is dynamic energy only, with background hardware energy removed, and it compares first-generation Loihi against CPU/GPU. A well-tuned momentum observer or DOB plus feedforward cancellation may match or beat a learned SNN reflex on rejection. Frame the contribution as latency and event-sparsity under a compliance constraint, not as better rejection than a DOB.

## Key Findings by Claim

### Claim 1 — Hierarchical control: slow policy (≈1–30 Hz) over a fast 1 kHz low-level loop
**Strength: strong.**

- **Franka Robotics, "Franka Control Interface Overview," FCI documentation (official docs), https://frankarobotics.github.io/docs/overview.html.** Supports the 1 kHz low-level loop. Quote: "Using libfranka, our open-source C++ interface, you can send real-time control values at 1 kHz," including "gravity and friction compensated joint-level torque commands." The same page says FCI also returns "estimated externally applied torques and forces" and "collision and contact detection information" at 1 kHz.\[1\] This is relevant to Claim 3: the robot already ships a disturbance estimate.
- **Black, K., Brown, N., Driess, D., et al., "π0: A Vision-Language-Action Flow Model for General Robot Control," arXiv:2410.24164, 2024, https://arxiv.org/abs/2410.24164.** Supports policy latency in the tens of ms and action chunking. Table I (RTX 4090) gives image encoders 14 ms, observation forward pass 32 ms, 10 flow steps 27 ms, "total on-board inference 73 ms" and "total off-board inference 86 ms." The action chunk is H = 50, enabling control "at frequencies of up to 50 Hz."\[2\]\[3\] Inference itself runs at roughly 10–14 Hz, and the 50 Hz comes from chunking.
- **Kim, M. J., Pertsch, K., Karamcheti, S., et al., "OpenVLA: An Open-Source Vision-Language-Action Model," arXiv:2406.09246 (CoRL 2024), https://arxiv.org/abs/2406.09246.** Supports the low end of policy rates. OpenVLA "requires 15GB of GPU memory when loaded in bfloat16 precision … and runs at approximately 6Hz on one NVIDIA RTX 4090 GPU."\[4\]\[5\]
- **Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., Song, S., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," RSS 2023; extended in IJRR 2025, https://arxiv.org/abs/2303.04137.** Supports the ~100 ms visuomotor latency. "Using DDIM with 100 training iterations and 10 inference iterations enables 0.1 s inference latency on an Nvidia 3080 GPU." The policy is "able to maintain peak performance with latency up to 4 steps."\[6\]\[7\]
- **Jiang, W., Clemons, J., Sankaralingam, K., Kozyrakis, C. (NVIDIA Research), "How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf," arXiv:2602.18397, 2026, https://arxiv.org/abs/2602.18397.** Supports the systematic latency landscape. The authors "define a 10 Hz inference frequency as acceptable … and 100 Hz as high-performance." Modeled π0 end-to-end latency is 52.57 ms (19.0 Hz) on Jetson Thor, 31.06 ms (32.2 Hz) on an RTX 4090 and 3.18 ms on a B100. Caveat: these are roofline (upper-bound) estimates, and the authors report that real systems achieve 73–83% of roofline.\[8\] Present them as best-case figures.
- **Black, K., Galliker, M. Y., Levine, S., "Real-Time Execution of Action Chunking Flow Policies," arXiv:2506.07339, 2025, https://arxiv.org/abs/2506.07339.** Gives measured end-to-end latency for π0.5 with 5 denoising steps on an RTX 4090 over a LAN: 108.76–138.98 ms total, of which about 97 ms is the model itself.\[9\] This is useful as a measured counterpart to VLA-Perf's estimates.
- **Supporting example of the full hierarchy on a Franka:** "Compliance for Free: Learning Identifiable Impedance via Bilateral Teleoperation," arXiv:2609.19976, 2026. Here the FR3 "is driven at 1 kHz by a Cartesian impedance controller … while the policy replans at ∼10 Hz."\[10\] This is a recent preprint, so cite it as an illustration, not as a foundation.

**Interpretation:** policy and reflex time-scales differ by 1–2 orders of magnitude. A push lasting 50–100 ms is entirely invisible to a 6–10 Hz policy. That is the core motivation for a reflex layer, and it is well grounded.

### Claim 2 — OSC and impedance foundations; effective mass; damping design; franka_ros defaults
**Strength: strong for the foundations; one technical claim needs correction.**

- **Khatib, O., "A unified approach for motion and force control of robot manipulators: The operational space formulation," IEEE Journal of Robotics and Automation, 3(1):43–53, 1987, DOI 10.1109/JRA.1987.1087068.** Supports task-space dynamics with Λ = (J M⁻¹ Jᵀ)⁻¹, the unified motion/force control approach, and the extension to redundant manipulators.\[11\] The redundant-manipulator extension underlies the dynamically consistent null-space projection.
- **Hogan, N., "Impedance Control: An Approach to Manipulation: Part I—Theory," ASME J. Dynamic Systems, Measurement, and Control, 107(1):1–7, 1985, DOI 10.1115/1.3140702.** Supports impedance control as the regulation of dynamic behavior. "Control of position or force alone is inadequate; control of dynamic behavior is also required."\[12\]\[13\] Parts II and III (same issue) cover implementation and applications.\[14\]
- **Albu-Schäffer, A., Ott, C., Frese, U., Hirzinger, G., "Cartesian impedance control of redundant robots: recent results with the DLR-light-weight-arms," Proc. IEEE ICRA 2003, vol. 3, pp. 3704–3709, DOI 10.1109/ROBOT.2003.1242165.** Supports damping design with non-diagonal inertia. The paper compares a "factorization design" with a "double diagonalization design," which simultaneously diagonalizes Λ and K, and shows step responses with chosen damping ratios (ξ = 0.3, 1.0).\[15\]\[16\]
- **franka_ros cartesian_impedance_example_controller source, https://github.com/frankaemika/franka_ros/blob/develop/franka_example_controllers/src/cartesian_impedance_example_controller.cpp (mirror: docs.ros.org melodic API).** Supports the damping rule. The source contains the comment "// Damping ratio = 1" followed by `2.0 * sqrt(config.translational_stiffness)` and `2.0 * sqrt(config.rotational_stiffness)`. The torque law is `Jᵀ(−K·error − D·(J·dq))` with no inertia shaping. The libfranka 0.15.0 documented example `cartesian_impedance_control.cpp`, described as "a simple cartesian impedance controller without inertia shaping," uses `const double translational_stiffness{150.0}; const double rotational_stiffness{10.0};` with damping `2.0 * sqrt(translational_stiffness)`, the same 2√K rule.
- **Default stiffness values: partially verified.** The 2√K rule is confirmed in source. I could not open the dynamic-reconfigure file `compliance_param.cfg` in this session. My recollection is a default translational stiffness of 200 N/m (range 0–400), rotational 10 Nm/rad and nullspace 0.5; check these against the .cfg before citing. A related, independent ROS 2 controller (Mayr et al., Cartesian-Impedance-Controller, GitHub) does ship a 200 N/m translational and 20 Nm/rad rotational default with the "2*sqrt(stiffness)" damping rule.\[17\]
- **Franka datasheet (Franka Research 3, R02212 v2.4), https://franka.de/hubfs/Datasheet_FrankaResearch3_R02212_2.4.pdf.** Gives the hardware range: "Adjustable translational stiffness 10 − 3000 N/m; adjustable rotational stiffness 1 − 300 Nm/rad."\[18\] A 200 N/m example is therefore near the compliant end of the range.

**Correction a reviewer will make:** with the franka_ros law (no inertia shaping), static stiffness at the end effector is exactly K = 200 N/m in every direction, independent of Λ. What Λ⁻¹ controls is the *transient*. Immediately after a push f, before feedback has built up, the end-effector acceleration is ẍ ≈ Λ⁻¹f, so early displacement is x(t) ≈ ½(Λ⁻¹f)t². The *effective mass* along a unit direction u is m_u = 1/(uᵀΛ⁻¹u); this is Khatib's later "inertial properties" concept, which you should check and cite separately. For axis i, "(Λ⁻¹)ᵢᵢ" is therefore the right quantity for *how fast the arm yields in the first milliseconds*, not for stiffness.

The same point means D = 2√K gives damping ratio ζ = 1 only if the apparent mass is 1 kg. For a real Λ with entries of several kg, ζ = 1/√m_u < 1, so the default is underdamped in heavy directions. This is a good, defensible argument for why a reflex helps exactly in the early-transient window, and it is a reason to cite Albu-Schäffer 2003.

### Claim 3 — Error-driven feedback acts only after displacement; observers reject faster; delay and gain limits
**Strength: moderate. The core rival is well sourced; the "delay degrades rejection" evidence is generic.**

- **De Luca, A., Mattone, R., "Sensorless robot collision detection and hybrid force/motion control," Proc. IEEE ICRA 2005, pp. 999–1004.** This is the original generalized-momentum residual for sensorless collision detection.\[19\] The residual r obeys ṙ = K_O(τ_ext − r), i.e. it is a first-order low-pass estimate of the external joint torque. (Standard result; verify the equation form in the paper.)
- **De Luca, A., Albu-Schäffer, A., Haddadin, S., Hirzinger, G., "Collision detection and safe reaction with the DLR-III lightweight manipulator arm," Proc. IEEE/RSJ IROS 2006, pp. 1623–1630, DOI 10.1109/IROS.2006.282053.** The abstract presents a collision detection method "based on physical quantities such as total energy and generalized momentum … that uses only proprioceptive robot sensors and provides also directional information for a safe robot reaction after collision."\[19\]\[20\]\[21\]
- **Haddadin, S., De Luca, A., Albu-Schäffer, A., "Robot Collisions: A Survey on Detection, Isolation, and Identification," IEEE Transactions on Robotics, 33(6):1292–1312, 2017, DOI 10.1109/TRO.2017.2723903.** The canonical survey of the collision-event pipeline (detection, isolation, identification, reaction).\[19\]\[22\] It argues that physical HRI needs "fast and reliable handling of possible collisions on the entire robot structure … along with control strategies for safe robot reaction."\[23\]\[24\]
- **Chen, W.-H., Yang, J., Guo, L., Li, S., "Disturbance-Observer-Based Control and Related Methods—An Overview," IEEE Transactions on Industrial Electronics, 63(2):1083–1095, 2016.** The standard DOB/ADRC survey, covering four decades of industrial use.\[25\]
- **Ohishi, K., Nakao, M., Ohnishi, K., Miyachi, K., "Microprocessor-controlled DC motor for load-insensitive position servo system," IEEE Trans. Industrial Electronics, IE-34(1):44–49, 1987.** The classic origin of the DOB in motion control.\[26\]\[27\]

**Why "faster" needs nuance:** the momentum observer's estimate lags the true τ_ext with time constant 1/K_O. At K_O = 50–200 s⁻¹ that is 5–20 ms, comparable to your reflex budget. The rigorous claim is that observer-based cancellation does not need a position error to accumulate, while impedance feedback does. Both are bandwidth-limited.

**Gap:** this session did not retrieve a quantitative source showing that delay in disturbance compensation degrades rejection, or that gain increases are limited by sampling, delay and torque saturation. These are textbook results: the Q-filter bandwidth trade-off in the Chen 2016 survey, and the delay–phase-margin limits in standard sampled-data control texts. Pull a specific figure from Chen et al. 2016 or from Ohnishi's stability analyses of DOB-based motion control rather than asserting it.

### Claim 4 — Compliance/safety trade-off; distinguishing collisions from intended contact
**Strength: moderate–strong.**

- **ISO/TS 15066:2016, "Robots and robotic devices — Collaborative robots," ISO.** It supplements ISO 10218-1/-2 and defines power-and-force limiting (PFL). Annex A gives pain-onset force and pressure limits for 29 body areas in 12 regions, separately for quasi-static (clamping) and transient (impact) contact. It also gives effective spring constants and masses used to derive permissible energy transfer, E = F²/2k, and hence speed. Examples from a secondary reproduction: chest quasi-static 140 N, with transient limits 2× the quasi-static values. Caveats: reviewers note that many limits are "preliminary," and the content has been folded into ISO 10218-2:2025 (Annex M).\[28\]\[29\]\[30\]\[31\]\[32\]\[33\] Check which version your course expects.
- **"Safe Physical Human-Robot Interaction through Variable Impedance Control based on ISO/TS 15066," arXiv:2311.13814, 2023.** Supports stiffening or softening only as needed. The proposed controller "significantly increases the maximum permissible velocity … while limiting contact forces to below safety thresholds" by adjusting effective mass, and it enforces the Panda limits of 87/12 Nm and 1000 Nm/s in simulation.\[34\]
- **Haddadin, S., Albu-Schäffer, A., Hirzinger, G., "Requirements for Safe Robots: Measurements, Analysis and New Insights," International Journal of Robotics Research, 28(11–12), 2009.** Supports the claim that impact severity scales with robot mass and velocity. It reports ADAC automobile crash-test-facility impacts with the DLR LWR-III, evaluating "the influence of the robot mass and velocity."\[35\]
- **Haddadin et al. 2017 TRO survey (above), plus Golz, S., Osendorfer, C., Haddadin, S., "Using tactile sensation for learning contact knowledge: Discriminate collision from physical interaction," ICRA 2015.** Supports separating accidental collisions from intended contact.\[22\] The survey's framing of collisions as fast, high-frequency events versus slow intentional interaction is a reasonable reading. However, I did not retrieve a verbatim high-pass-filter quote in this session, so treat the frequency-separation claim as **moderately supported** until you pull the relevant section.

### Claim 5 — Biological reflex latencies and bio-inspired robot reflexes
**Strength: strong for physiology; weaker for robot reflex layers.**

- **Kurtzer, I. L., "Long-latency reflexes account for limb biomechanics through several supraspinal pathways," Frontiers in Integrative Neuroscience, 8:99, 2014, https://www.frontiersin.org/journals/integrative-neuroscience/articles/10.3389/fnint.2014.00099/pdf.** The cleanest single source for the three latency bands: "the short-latency reflex (SLR 20–45 ms) and the more delayed voluntary reaction (100 ms)," with the LLR "around 50–100 ms."\[36\]
- **Pruszynski, J. A., Scott, S. H., "Optimal feedback control and the long-latency stretch response," Experimental Brain Research, 218(3):341–359, 2012, DOI 10.1007/s00221-012-3041-8.** Supports the claim that LLRs (50–100 ms in the upper limb) are task-dependent and consistent with optimal feedback control. Secondary citers summarize: "long-latency responses (LLRs) refer to the burst of muscular activity observed 50–100 ms … postperturbation (Pruszynski and Scott, 2012)."\[37\]\[38\]
- **Scott, S. H., "A functional taxonomy of bottom-up sensory feedback processing for motor actions," Trends in Neurosciences, 2016 (as cited in PMC11209647).** Cited for upper-limb spinal delays of "20–30 ms" and transcortical delays of "50–60 ms."\[37\] The spread of numbers across sources (20–30 vs 20–45 vs 20–50 ms) reflects different epoch definitions.\[38\]\[39\]\[40\] State the R1 window you use, for example 20–45 ms as in Kurtzer 2014.
- **Robot reflex layers.** The most direct robotic analogue is Vorndamme, J., Melone, A., Kirschner, R., Figueredo, L., Haddadin, S., "Safe robot reflexes: A taxonomy-based decision and modulation framework," IEEE Transactions on Robotics, vol. 41, pp. 982–1001, 2025, DOI 10.1109/TRO.2024.3519421. It introduces a "reflex context taxonomy" and a "reflex base elements taxonomy" that "serves as a succinct yet comprehensive summary of validated robot reflexes," together with a Robot Safety Assessment Pipeline (RSAP). I did not retrieve a legged-robot spinal-CPG or push-recovery reflex paper in this session. The thin spot is partly filled by Škerlj, J., Haddadin, S., "Sub-Millisecond Reflexes for Continuous Physical Human-Robot Interaction," arXiv:2610.21913 (15 Oct 2026; preprint). On a Franka arm it reports "switching delays below 0.1 ms, more than four orders of magnitude faster than conventional ros2_control," and its code is open source (github.com/RoboRef/franka_reflex_engine, ROS 2, Franka Panda).

**Mapping point:** at 1 kHz a 20 ms spinal-reflex budget is 20 control ticks, so the robot "spinal" layer is not limited by compute. It is limited by sensing, observer bandwidth and torque-rate limits (see Claim 9).

### Claim 6 — Event-driven / send-on-delta sampling
**Strength: strong for the concepts; the performance-parity claim is system-dependent.**

- **Åström, K. J., Bernhardsson, B. M., "Comparison of Riemann and Lebesgue sampling for first order stochastic systems," Proc. 41st IEEE CDC, vol. 2, pp. 2011–2016, 2002, https://lup.lub.lu.se/search/publication/537765.** Supports event-based sampling as competitive with periodic sampling. "Lebesgue sampling or event based sampling … means that signals are sampled only when measurements pass certain limits. In this paper it is shown that Lebesgue sampling gives better performance for some simple systems."\[41\] Note the scope: first-order stochastic systems only.
- **Miśkowicz, M., "Send-On-Delta Concept: An Event-Based Data Reporting Strategy," Sensors, 6(1):49–63, 2006, DOI 10.3390/s6010049.** Supports send-on-delta as the canonical level-crossing scheme. A sample is reported only when the signal changes by more than δ since the last report.\[42\] The paper derives a lower bound on send-on-delta effectiveness versus periodic sampling and calls it "attractive … due to effective energy consumption."\[43\]\[44\] By construction, a constant (steady-state) error never re-triggers. That is the "change detector" property your reflex relies on, and the reason it ignores slow intentional loads.
- **Tabuada, P., "Event-triggered real-time scheduling of stabilizing control tasks," IEEE Transactions on Automatic Control, 52(9):1680–1685, 2007.** Supports state-dependent triggering with stability guarantees.\[45\]
- **Heemels, W. P. M. H., Johansson, K. H., Tabuada, P., "An introduction to event-triggered and self-triggered control," Proc. 51st IEEE CDC, pp. 3270–3285, 2012, DOI 10.1109/CDC.2012.6425820.** The standard tutorial: sensing and actuation are "performed when needed," motivated by resource-constrained embedded and wireless control.\[46\]\[47\]

**Caveat:** "comparable performance with fewer samples" is shown for specific system classes and trigger designs, and Zeno behavior and minimum inter-event times must be handled. Claim it for your plant only with your own data, for example events per second versus 1 kHz and rejection RMS.

### Claim 7 — SNNs for robot control; training; energy; FPGA; delta networks
**Strength: moderate. Energy claims transfer poorly to FPGA.**

- **DeWolf, T., Patel, K., Jaworski, P., Leontie, R., Hays, J., Eliasmith, C., "Neuromorphic control of a simulated 7-DOF arm using Loihi," Neuromorphic Computing and Engineering, 3(1):014007, 2023, DOI 10.1088/2634-4386/acb286.** The closest precedent: a fully spiking *operational space controller* for a simulated 7-DOF Kinova arm in MuJoCo. Its RMSE is "only slightly worse than the analytic solution, with 4.13% more deviation from the ideal trajectory." Per the paper, "the neuromorphic implementation on Loihi reduces the dynamic energy cost per inference by two orders of magnitude," measured with "the background energy cost of running the hardware" removed and comparing first-generation Loihi against CPU/GPU.
- **DeWolf, T., Stewart, T. C., Slotine, J.-J., Eliasmith, C., "A spiking neural model of adaptive arm control," Proceedings of the Royal Society B, 283(1843):20162134, 2016, DOI 10.1098/rspb.2016.2134.** The REACH model: spiking-neuron OSC plus adaptive control that learns unknown dynamics online.\[48\]\[49\]\[50\]\[51\]
- **Stagsted, R. K., Vitale, A., Binz, J., Renner, A., Larsen, L. B., Sandamirskaya, Y., "Towards neuromorphic control: A spiking neural network based PID controller for UAV," RSS XVI, 2020, DOI 10.15607/RSS.2020.XVI.074, https://www.roboticsproceedings.org/rss16/p074.pdf.** Described as "the first implementation of a fully neuromorphic PID controller," run on Loihi for a 1-DoF UAV with performance "comparable … to a conventional digital controller." It used 27 ms time steps and 54 of 128 cores.\[52\] Note the 27 ms step, which is far slower than your 1 kHz target.
- **Bing, Z., Meschede, C., Röhrbein, F., Huang, K., Knoll, A. C., "A Survey of Robotics Control Based on Learning-Inspired Spiking Neural Networks," Frontiers in Neurorobotics, 12:35, 2018, DOI 10.3389/fnbot.2018.00035.** A qualitative survey: SNNs "have showed great potential for achieving advanced robotic intelligence in terms of speed, energy efficiency, and computation capabilities."\[53\]\[54\]
- **Neftci, E. O., Mostafa, H., Zenke, F., "Surrogate Gradient Learning in Spiking Neural Networks," IEEE Signal Processing Magazine, 36(6):51–63, 2019, arXiv:1901.09948.** Surrogate gradients are "a particularly flexible and efficient method" for overcoming SNNs' "binary and dynamical nature."\[55\]
- **Eshraghian, J. K., Ward, M., Neftci, E. O., Wang, X., Lenz, G., Dwivedi, G., Bennamoun, M., Jeong, D. S., Lu, W. D., "Training Spiking Neural Networks Using Lessons From Deep Learning," Proceedings of the IEEE, 111(9):1016–1054, 2023, DOI 10.1109/JPROC.2023.3308088.** The snnTorch tutorial and reference paper.\[53\]\[56\]\[57\]
- **Davies, M., Srinivasa, N., Lin, T.-H., et al., "Loihi: A Neuromorphic Manycore Processor with On-Chip Learning," IEEE Micro, 38(1):82–99, 2018, DOI 10.1109/MM.2018.112130359.** Gives measured costs on a 14 nm chip: 23.6 pJ minimum per synaptic spike op, 3.5 ns maximum per synaptic op, and 81/52 pJ per active/inactive neuron update. It also reports ">3 orders of magnitude" better energy-delay product on LASSO than a CPU.\[58\]\[59\]
- **Horowitz, M., "Computing's energy problem (and what we can do about it)," IEEE ISSCC 2014, pp. 10–14, DOI 10.1109/ISSCC.2014.6757323.** Gives 45 nm per-op energies: 32-bit float add 0.9 pJ, float multiply 3.7 pJ, int add 0.1 pJ, and 8 KB/32 KB/1 MB cache access 10/20/100 pJ. The paper's text puts DRAM access at "1 to 2 nJ." The widely quoted "640 pJ per 32-bit DRAM read" comes from the Han et al. EIE paper (arXiv:1602.01528) and talk slides, not from this ISSCC paper, so cite accordingly.\[60\]\[61\]\[62\] Use these numbers to argue that a synaptic op (accumulate only, no multiply) costs roughly an int add, while a MAC costs roughly a multiply plus an add. That gives about an order of magnitude per op before sparsity. The real savings come from activity sparsity and from avoiding memory traffic.
- **Neil, D., Lee, J. H., Delbruck, T., Liu, S.-C., "Delta Networks for Optimized Recurrent Network Computation," ICML 2017 (PMLR 70), arXiv:1612.05571.** The best evidence for event-driven versus dense compute in ANNs. It reports a "9X reduction in cost with negligible loss of accuracy" on TIDIGITS, "5.7x" on WSJ speech, and RNN cost "reduced by a substantial 100X" on a driving dataset.\[63\] This is the fair non-spiking baseline: a delta-RNN on an FPGA gets most of the sparsity benefit without spikes.
- **Carpegna, A., Savino, A., Di Carlo, S., "Spiker: an FPGA-optimized Hardware acceleration for Spiking Neural Networks," arXiv:2201.06993, 2022 (ISVLSI 2022; venue not confirmed).** An FPGA latency datapoint: 400 LIF neurons on an Artix-7 at 215 µs and 13 mJ per MNIST image.\[64\]\[65\] Sub-millisecond inference fits inside a 1 ms control tick, but 13 mJ per inference is not obviously better than a microcontroller running a tiny MLP.

### Claim 8 — Residual learning on top of a nominal controller
**Strength: strong.**

- **Silver, T., Allen, K., Tenenbaum, J., Kaelbling, L., "Residual Policy Learning," arXiv:1812.06298, 2018.** On six MuJoCo tasks, "RPL can perform long-horizon, sparse-reward tasks for which reinforcement learning alone fails" and "consistently and substantially improves on the initial controllers."\[66\]
- **Johannink, T., Bahl, S., Nair, A., Luo, J., Kumar, A., Loskyll, M., Ojea, J. A., Solowjow, E., Levine, S., "Residual Reinforcement Learning for Robot Control," IEEE ICRA 2019, arXiv:1812.03201.** "The final control policy is a superposition of both control signals" (conventional feedback plus learned residual), demonstrated on real-robot block assembly with contacts.\[67\]

**Gap:** this session did not retrieve a specific "learn the disturbance-compensation torque from a teacher signal" paper. The closest framing is supervised imitation of an observer: train the SNN to regress −τ̂_ext from the momentum observer as the teacher. Say explicitly that this is your design choice. It also makes the DOB comparison unavoidable, because the student is bounded by its teacher unless it exploits event timing.

### Claim 9 — Physics limit: actuator saturation bounds impact rejection
**Strength: strong, and stronger than the user framed it.**

- **Franka Robotics, "Control Interface Specification and Robot Limits," FCI documentation, https://frankarobotics.github.io/docs/robot_specifications.html.** For both FER (Panda) and FR3, τ_max = 87, 87, 87, 87, 12, 12, 12 Nm for joints 1–7, and the torque-rate limit is **1000 Nm/s on every joint**.\[68\]
- **Franka Research 3 datasheet (R02212 v2.4), https://franka.de/hubfs/Datasheet_FrankaResearch3_R02212_2.4.pdf.** Gives "Joint torque limits A1-A4: ±87 Nm, A5-A7: ±12 Nm." The **Franka Emika Robot Instruction Handbook** (generationrobots.com mirror) says: "Axes 1 & 2 … Axes 3 & 4: allowed, repeatable peak torque <= 87 Nm; Axes 5, 6, 7 … <= 12 Nm."\[18\]\[69\]

**Interpretation:** two separate hardware limits bound your reflex.
- **Magnitude.** A force F at a lever arm of about 0.5–0.8 m needs roughly F·r of joint torque. 87 Nm at about 0.8 m resists only about 110 N at full extension, and the 12 Nm wrist joints saturate far earlier for off-axis wrenches. Any push above a few hundred newtons cannot be cancelled by torque control.
- **Slew rate.** At 1000 Nm/s, ramping from 0 to 87 Nm takes at least 87 ms, longer than the human long-latency reflex window. On real hardware the reflex can therefore add at most about 20 Nm within its first 20 ms. Report this as a hard constraint and enforce both limits in MuJoCo, where torque-rate limits are not enforced by default and must be implemented. Otherwise simulated results will overstate what the reflex can do.

## Strongest vs. Weakest Claims

| Claim | Support | Main risk |
|---|---|---|
| 1. Policy/controller rate hierarchy | Strong (π0, OpenVLA, Diffusion Policy, VLA-Perf, FCI docs) | VLA-Perf figures are roofline upper bounds |
| 2. OSC/impedance foundations | Strong | "Stiffness ∝ (Λ⁻¹)ᵢᵢ" is wrong; it is effective mass/transient. franka_ros 200 N/m default not verified from .cfg |
| 3. Observers beat error-driven feedback | Moderate | Observers have a 1/K_O lag; delay-degradation evidence not yet quantified |
| 4. Compliance/safety, collision vs. intended contact | Moderate–strong | Frequency-separation claim needs a verbatim source; ISO/TS 15066 now superseded by ISO 10218-2:2025 |
| 5. Reflex latencies | Strong (physiology); thin (robot reflex layers) | Latency bands vary by source definition |
| 6. Event-triggered sampling | Strong (concept) | Performance parity is system-specific |
| 7. SNN efficiency | Moderate | Loihi energy numbers don't transfer to FPGA; delta-ANNs get similar sparsity |
| 8. Residual learning | Strong | Teacher-signal framing is your own design |
| 9. Torque saturation | Strong | Rate limit (1000 Nm/s) is the bigger and under-appreciated constraint |

## Counter-Evidence a Reviewer Will Raise

1. **"Why not just use the momentum observer?"** It is model-based, has a single gain K_O, runs at 1 kHz, is already exposed by FCI ("estimated externally applied torques"), and has been the standard since De Luca 2005/2006. Feeding −τ̂_ext forward is the obvious baseline. Expect it to match a learned SNN on rejection RMS. Your defensible advantages are (a) event sparsity, measured as ops per second relative to 1 kHz dense compute, (b) a learned nonlinear trigger that separates impacts from intentional slow loads, and (c) a hardware latency figure on an FPGA. Do not claim better rejection unless your own data show it.
2. **"Just raise K."** Answer with ISO/TS 15066 transient limits, the Haddadin 2009 crash data and the variable-impedance literature (arXiv:2311.13814). Add that the default 2√K damping is already underdamped in heavy directions.
3. **"Spikes aren't necessary."** Delta networks (Neil 2017) reach 5.7–100× compute reduction without spikes. Include a delta-MLP ablation.
4. **"Energy claims are borrowed from Loihi."** Report FPGA resource use, latency and power from your own synthesis. Treat the Davies 2018 and DeWolf 2023 numbers as context only.
5. **"Simulation ignores rate limits and sensing delay."** Enforce 87/12 Nm and 1000 Nm/s in MuJoCo, and add about 1 ms sensing plus actuation delay.

## Recommendations

- Reword Claim 2 to: "The early-transient yield to a push is governed by Λ⁻¹; steady-state stiffness is K." Cite Khatib 1987 and Albu-Schäffer 2003.
- Use three baselines: OSC alone, OSC + momentum-observer feedforward, and OSC + stiffened K. The SNN reflex should be the fourth row.
- Verify before submission: the `compliance_param.cfg` defaults, the Neftci 2019 DOI, the Johannink ICRA pages, and a verbatim frequency-separation quote from the Haddadin 2017 survey.

## Caveats

- Several 2026 arXiv items (VLA-Perf, Compliance for Free) are preprints, not peer-reviewed.
- ISO standard contents were confirmed through secondary sources, because the standard itself is paywalled.
- The Spiker venue and the provenance of the Horowitz DRAM figure are flagged above.

## Sources

1. [Franka Control Interface Overview — Franka Control Interface (FCI) documentation](https://frankarobotics.github.io/docs/overview.html)
2. [𝜋\_0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/html/2410.24164v1)
3. [\$\\pi\_0\$: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/pdf/2410.24164)
4. [OpenVLA: An Open-Source Vision-Language-Action Model Moo Jin Kim∗,1](https://arxiv.org/pdf/2406.09246)
5. [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/html/2406.09246v3)
6. [Diffusion policy: Visuomotor policy learning via action diffusion - Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song, 2025](https://journals.sagepub.com/doi/10.1177/02783649241273668)
7. [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/pdf/2303.04137)
8. <https://arxiv.org/pdf/2602.18397>
9. [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/pdf/2506.07339)
10. [Compliance for Free: Learning Identifiable Impedance via Bilateral Teleoperation](https://arxiv.org/pdf/2609.19976)
11. [A unified approach for motion and force control of robot manipulators: The operational space formulation - ADS](https://ui.adsabs.harvard.edu/abs/1987IJRA....3...43K/abstract)
12. [\[PDF\] Impedance Control: An Approach to Manipulation: Part I—Theory](https://www.semanticscholar.org/paper/Impedance-Control:-An-Approach-to-Manipulation:-Hogan/2ec225fe0745aeba026002a98cea18cd57c58b06)
13. [Impedance Control: An Approach to Manipulation: Part I—Theory](https://asmedigitalcollection.asme.org/dynamicsystems/article/107/1/1/400604/Impedance-Control-An-Approach-to-Manipulation-Part)
14. [Impedance control - An approach to manipulation. I - Theory. II - Implementation. III - Applications - ADS](https://ui.adsabs.harvard.edu/abs/1985ATJDS.107....1H/abstract)
15. [Cartesian Impedance Control of Redundant Robots: Recent](https://www.informatik.uni-bremen.de/agebv2/downloads/published/albuschaeffericra03.pdf)
16. [Cartesian impedance control of redundant robots: Recent results with the DLR-Light-Weight-Arms - Technical University of Munich](https://portal.fis.tum.de/en/publications/cartesian-impedance-control-of-redundant-robots-recent-results-wi/)
17. [GitHub - matthias-mayr/Cartesian-Impedance-Controller: A C++ implementation of Cartesian impedance control for torque-controlled manipulators with ROS 2 & 1 bindings. · GitHub](https://github.com/matthias-mayr/Cartesian-Impedance-Controller)
18. [Copyright © Franka Robotics The Reference Platform for AI & Robotics FRANKA](https://franka.de/hubfs/Datasheet_FrankaResearch3_R02212_2.4.pdf?hsLang=en)
19. [JRM Vol.38 p.589 (2026)](https://www.fujipress.jp/jrm/rb/robot003800020589/)
20. [Collision detection and safe reaction with the DLR-III lightweight manipulator arm - Technical University of Munich](https://portal.fis.tum.de/en/publications/collision-detection-and-safe-reaction-with-the-dlr-iii-lightweigh/)
21. [\[PDF\] Collision Detection and Safe Reaction with the DLR-III Lightweight Manipulator Arm](https://www.semanticscholar.org/paper/Collision-Detection-and-Safe-Reaction-with-the-Arm-Luca-Albu-Sch%C3%A4ffer/95ac2dee3497073f6f139ba0f16cf64c5e9d327c)
22. [Safe Collision and Clamping Reaction for Parallel Robots During Human-Robot Collaboration](https://arxiv.org/pdf/2308.09656)
23. [electronic library - Robot Collisions: A Survey on Detection, Isolation, and Identification](https://elib.dlr.de/117080/)
24. [Robot Collisions: A Survey on Detection, Isolation, and Identification](https://cir.nii.ac.jp/crid/1363951794738948608)
25. [Disturbance-Observer-Based Control and Related Methods - An Overview - PolyU Scholars Hub](https://research.polyu.edu.hk/en/publications/disturbance-observer-based-control-and-related-methods-an-overvie/)
26. [Reduced-order disturbance observer design for discrete-time linear stochastic systems - Jinya Su, Baibing Li, Wen-Hua Chen, Jun Yang, 2016](https://doi.org/10.1177/0142331216634425)
27. [Disturbance Observer Based Control Design via Active Disturbance Rejection Control: A PMSM Example - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2405896320324861)
28. [Energy Tank-based Control Framework for Satisfying the ISO/TS 15066 Constraint](https://arxiv.org/pdf/2304.14059)
29. [Evolution of Safety Requirements in Industrial Robotics: Comparative Analysis of ISO 10218-1/2 (2011 vs. 2025) and Integration of ISO/TS 15066](https://arxiv.org/pdf/2602.17822)
30. [A Statistical Model to Determine Biomechanical Limits for Physically Safe Interactions With Collaborative Robots](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8850785/)
31. [ISO/TS 15066 in Plain English: Force Limits, Tooling, and Real-World Examples](https://automationinside.com/article/iso-ts-15066-in-plain-english-force-limits-tooling-and-real-world-examples)
32. [Biomechanical Limits](http://eidtech.dyndns-at-work.com/support/cobotta_manual/007496.html)
33. [ISO/TS 15066 Body Region Force & Pressure Limits Table](https://www.inmotion.global/resources/cobot-safety/iso-ts-15066-body-region-force-limits/)
34. [Safe Physical Human-Robot Interaction through Variable Impedance Control based on ISO/TS 15066](https://arxiv.org/pdf/2311.13814)
35. [Requirements for Safe Robots: Measurements, Analysis and New Insights: International Journal of Robotics Research: Vol 28, No 11-12](https://dl.acm.org/doi/abs/10.1177/0278364909343970)
36. [www.frontiersin.org](https://www.frontiersin.org/journals/integrative-neuroscience/articles/10.3389/fnint.2014.00099/pdf)
37. [Erroneous Compensation for Long-Latency Feedback Delays as Origin of Essential Tremor - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11209647/)
38. [Feedback responses rapidly scale with the urgency to correct for external perturbations](https://www.researchgate.net/publication/245029712_Feedback_responses_rapidly_scale_with_the_urgency_to_correct_for_external_perturbations)
39. [Assessing Neural Connectivity and Associated Time Delays of Muscle Responses to Continuous Position Perturbations](https://link.springer.com/article/10.1007/s10439-020-02573-2)
40. [Task-dependent modulation of spinal and transcortical stretch reflexes linked to motor learning rate](https://www.biorxiv.org/content/10.1101/223842v1.full)
41. [Comparison of Riemann and Lebesgue sampling for first order stochastic systems](https://lup.lub.lu.se/search/publication/816e63c3-2672-4bef-90a5-b882b74d550b)
42. [Output feedback MPC with send-on-delta measurements for uncertain systems - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2405896316319747)
43. [\[PDF\] Send-On-Delta Concept: An Event-Based Data Reporting Strategy](https://www.semanticscholar.org/paper/Send-On-Delta-Concept:-An-Event-Based-Data-Strategy-Mi%C5%9Bkowicz/e16d4fa920e6886be986d915b768dc166e7761e0)
44. [Send-On-Delta Concept: An Event-Based Data Reporting Strategy](https://www.proquest.com/scholarly-journals/send-on-delta-concept-event-based-data-reporting/docview/1537498974/se-2)
45. [Adaptive Predictive Sampling and Communication for Real-time Monitoring](https://arxiv.org/pdf/2609.26493)
46. [An introduction to event-triggered and self-triggered control - Research portal Eindhoven University of Technology](https://research.tue.nl/en/publications/an-introduction-to-event-triggered-and-self-triggered-control/)
47. [(PDF) An introduction to event-triggered and self-triggered control](https://www.researchgate.net/publication/261490436_An_introduction_to_event-triggered_and_self-triggered_control)
48. [A spiking neural model of adaptive arm control](https://royalsocietypublishing.org/doi/10.1098/rspb.2016.2134)
49. [A neural model of the motor control system](https://www.semanticscholar.org/paper/A-neural-model-of-the-motor-control-system-DeWolf/603065cfadc9795867c641c822c575cb89dce35f)
50. [A Spiking Neural Architecture for Coordinating Arm and Locomotor Control](https://arxiv.org/html/2606.11034)
51. [A Spiking Neural Architecture for Coordinating Arm and Locomotor Control](https://arxiv.org/pdf/2606.11034)
52. <https://www.roboticsproceedings.org/rss16/p074.pdf>
53. [Spiking Neural Networks for Continuous Control via End-to-End Model-Based Learning](https://arxiv.org/pdf/2509.05356)
54. [Frontiers](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2018.00035/full)
55. [Surrogate Gradient Learning in Spiking Neural Networks](https://arxiv.org/abs/1901.09948)
56. [Training Spiking Neural Networks Using Lessons From Deep Learning - the UWA Profiles and Research Repository](https://research-repository.uwa.edu.au/en/publications/training-spiking-neural-networks-using-lessons-from-deep-learning-2/)
57. [GitHub - jeshraghian/snntorch: Deep and online learning with spiking neural networks in Python · GitHub](https://github.com/jeshraghian/snntorch)
58. [FEATURE ARTICLE: Neuromorphic Computing Loihi: A Neuromorphic](https://redwood.berkeley.edu/wp-content/uploads/2021/08/Davies2018.pdf)
59. [Loihi: A Neuromorphic Manycore Processor with On-Chip Learning](https://www.semanticscholar.org/paper/Loihi:-A-Neuromorphic-Manycore-Processor-with-Davies-Srinivasa/ccc560f932e36b8561a847e02fcf20522bc71c5f)
60. [arXiv:1602.01528v2 \[cs.CV\] 3 May 2016](https://www.cse.iitd.ac.in/~rijurekha/course/eie.pdf)
61. [Exploration of Efficient Computation for Trajectory Planning ...](https://dl.gi.de/bitstreams/62353637-dac7-43a0-8f34-7a8a481166bd/download)
62. [Computing's Energy Problem (and what we can do about it)](https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf)
63. [Delta Networks for Optimized Recurrent Network Computation](https://arxiv.org/abs/1612.05571)
64. [Spiker: an FPGA-optimized Hardware acceleration for Spiking Neural Networks](https://www.alphaxiv.org/abs/2201.06993)
65. [(PDF) Spiker: an FPGA-optimized Hardware acceleration for Spiking Neural Networks](https://www.researchgate.net/publication/357953311_Spiker_an_FPGA-optimized_Hardware_acceleration_for_Spiking_Neural_Networks)
66. [Residual Policy Learning](https://arxiv.org/abs/1812.06298)
67. [Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201)
68. [Control Interface Specification and Robot Limits — Franka Control Interface (FCI) documentation](https://frankarobotics.github.io/docs/robot_specifications.html)
69. [FRANKA EMIKA ROBOT’S INSTRUCTION HANDBOOK SECTION 1 General information](https://www.generationrobots.com/media/franka-emika-robot-handbook.pdf)
