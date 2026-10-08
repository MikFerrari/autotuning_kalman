# Paper → code map

M. Menner, K. Berntorp, S. Di Cairano, *Automated Controller Calibration by Kalman Filtering*,
IEEE TCST 31(6), 2023, doi:10.1109/TCST.2023.3254213.

## Method

| Paper | Code |
|---|---|
| Random-walk parameter model, training objective as measurement (Eq. 4, 8, 9) | `calibrators.py` module docstring |
| Training objective ‖y − r(X, U)‖²_{C_v⁻¹} (Eq. 5) | `metrics.training_cost` |
| EKF update (Eq. 11a–e) | `EKFCalibrator.update`. H_k by finite differences (`ask`/`tell`) or analytic (`update`) |
| UKF update (Eq. 12a–i), sigma points and weights (Sec. III-B) | `UKFCalibrator` |
| Sliding window replay h(θ) = r′(θ, x_{k−N}, z_{k−N}, W_{k−N\|k−1}) (Sec. III) | `online.SlidingWindow` + `disturbances()` / replay functions per system |
| Safety check, Theorem 1, condition (16) | `safety.LyapunovSafetyCheck`, `safety.quadratic_lyapunov` |
| Decay factor (Eq. 20) | `metrics.decay_factor` |

## Experiments

| Paper | Code |
|---|---|
| Double integrator (18), Ts = 0.1 s | `systems/double_integrator.py` |
| Seven controllers (Sec. IV-A1): state feedback (2), LQR/DARE (4), incremental PID (3), H∞ loop shaping (8), sliding mode (2), dynamic output feedback (4), NN 2-10-10-1 leaky ReLU (151) | same file, `CONTROLLERS` |
| Objectives (19a), (19b) with overshoot penalty | `tracking_spec`, `episodic_objective` |
| Fig. 2, Fig. 3, decay factors | `experiments/di_episodic.py` |
| Table I (online, dv ~ N(0,1) and dv = 1) | `experiments/di_online.py` |
| Kinematic single-track model (21), LQ optimal controller, h₁, h₂ | `systems/vehicle.py` (`LQCostCalibrationTask`) |
| BO baseline: SE kernel l = 1, σ² = 1, UCB | `bayesopt.GPUCB` |
| Figs. 4, 5 (UKF vs EKF vs BO; non-differentiable objective) | `experiments/vehicle_compare.py` |
| Lane-change controllers (23), (24), objective (22), Figs. 8, 9 | `systems/vehicle.py`, `experiments/lane_change_online.py` |

## Assumptions where the paper is silent

| Item | Choice made here |
|---|---|
| Initial covariance P₀ | P₀ = C_θ |
| UKF weight w₀ | 1/3 (the paper only requires −1 < w₀ < 1). Results are sensitive to it, see README |
| Sigma-point covariance | Eq. (12) literally: sigma points from P_{k−1\|k−1}, C_θ added afterwards (12h). `sigma_from="predicted"` gives the textbook ordering (equal to the EKF for linear h) |
| Initial controller parameters (Sec. IV-A) | θ₀ = 0, except LQR (Q = I, R = 1), H∞ (W₁ = W₂ = 1, i.e. θ = 1), NN (N(0, 0.1²), seed 0) and dynamic output feedback (observer gains −1, −2: in the noise-free episodic task p̂ = p, so the objective carries no information about them) |
| LQR / LQ with non-PSD sigma points | Q, R projected onto the PSD/PD cone (eigenvalue clipping, floor 1e-6 on R) |
| H∞ loop shaping | Continuous Glover–McFarlane design on G = 1/s² (Skogestad & Postlethwaite Table 9.2, D = 0, γ = 1.1 γ_min), K = W₁ K_s W₂ in positive feedback u = K e, Tustin discretisation. Synthesis failures (e.g. θ₃ = 0) get a large finite penalty |
| Diverging rollouts | States and inputs clipped at ±1e3 so that unstable sigma points stay finite |
| Episode indexing in (19a) | h = [p₁…p₁₅₀; u₀…u₁₄₉] (inputs that produce the positions) |
| Table I | 600 steps (60 s), window N = 50, cost = mean(p² + u²), start from the episodic parameters (C_θ = 0.1 I). Online: C_θ = 0.01 I, C_v = 10⁴ I, because the paper's I/I destabilises per-step updates (see README) |
| Vehicle | l_f = 1.2 m, l_r = 1.6 m. Euler with Ts = 0.25 s. The "optimal controller" (horizon 20) is unconstrained receding-horizon LQ on the linearisation at x_ref (first-stage gain of a 20-step Riccati recursion), applied to the nonlinear model. Random PD initial Q = GGᵀ/4 + 0.05 I, R = GGᵀ/2 + 0.05 I |
| θ for Q, R | Upper-triangular entries: 10 for Q, 3 for R (13 parameters) |
| BO acquisition optimisation | Cost standardised. LCB μ − 2σ minimised over 1000 random PD candidates plus 1000 Gaussian perturbations (σ = 0.3) of the incumbent |
| Lane change C_θ | 0.01 I (C_θ = I goes unstable after the first manoeuvre) |
| CarSim (Sec. V) | Not available. Replaced by a dynamic single-track model with linear, saturated tyres (m = 1500 kg, I_z = 2500 kg m², C_f = C_r = 8e4 N/rad), simulated at 1 ms. Steering limits |δ| ≤ 0.5 rad, |δ̇| ≤ 1 rad/s. Speed 70 → 50 km/h at t = 50 s. LQR (23) is linearised at the speed at the start of the replay window |
