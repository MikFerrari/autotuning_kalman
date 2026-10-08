# autotuning_kalman

Python implementation of **M. Menner, K. Berntorp, S. Di Cairano, "Automated Controller Calibration by
Kalman Filtering", IEEE TCST 31(6), 2023** ([doi](https://doi.org/10.1109/TCST.2023.3254213)).

A Kalman filter estimates **controller parameters** θ instead of the system state. The "measurement" is a
training objective that scores closed-loop behaviour: data from the last N steps are replayed through a
model with candidate parameters, and the filter moves θ towards the parameters that would have met the
specification. It works for any parametrised controller (PID gains, MPC weights, NN weights, …).

## Quick start

```bash
cd pixi_ws/src/autotuning_kalman
pixi install
pixi run test              # 31 unit tests, ~2 s
pixi run di-episodic       # Fig. 2/3: 7 controllers on a double integrator   (~1 min)
pixi run di-online         # Table I: online calibration under disturbances  (needs di-episodic first)
pixi run vehicle-compare   # Fig. 4/5: UKF vs EKF vs Bayesian optimisation    (--trials 500 as in the paper)
pixi run lane-change       # Fig. 8/9: online lane-change calibration with model mismatch
```

Figures and parameter files go to `outputs/`. Each script takes `--help`. The main knob is
`--c-theta` (C_θ = c·I).

## Using the calibrator

Both filters have an ask/tell interface, so h(θ) can come from any simulator: a vectorised numpy
model, MuJoCo, JAX rollouts, and so on.

```python
from autotuning_kalman.calibrators import UKFCalibrator

cal = UKFCalibrator(theta0, C_theta=0.1, C_v=1.0)   # Eq. (12)
for k in ...:                                        # once per episode or per time step
    thetas = cal.ask()                               # (2L+1, L) sigma points
    H = np.stack([h(th) for th in thetas])           # specification r(X, U) for each candidate
    theta = cal.tell(y_desired, H)                   # new parameters for the controller
```

`EKFCalibrator` (Eq. 11) has the same interface: `ask` returns the finite-difference points. You can
also call `update(y, h(θ), H)` with an analytic Jacobian. For online use, `online.SlidingWindow` stores
(x, u, z, ref). The replay recovers the disturbances W_i = x_{i+1} − f(x_i, u_i) from the data, so model
mismatch is part of the objective (Section III). `safety.LyapunovSafetyCheck` implements the Theorem 1
gate between the filter and the controller.

## Layout

```
autotuning_kalman/
  calibrators.py          UKFCalibrator (Eq. 12), EKFCalibrator (Eq. 11)
  online.py               sliding window for receding-horizon calibration
  safety.py               Theorem 1 safety check
  metrics.py              training objective (5), decay factor (20)
  bayesopt.py             GP-UCB baseline (Sec. IV-B1)
  systems/double_integrator.py   model (18) + the 7 controllers of Sec. IV-A, batched over θ
  systems/vehicle.py      kinematic model (21), LQ cost calibration, lane change, dynamic plant
experiments/              one script per figure/table
docs/PAPER_NOTES.md       equation → code map and every assumption made where the paper is silent
```

## Results

All runs below use the defaults in this repo. Paper numbers are given only where the paper states them
in the text (its tables and figures are images).

**Fig. 4/5, vehicle: UKF vs EKF vs BO** (100 trials, median). This matches the paper qualitatively.

| | cost at iteration 0 → 10 → 100 | mean cost over the run |
|---|---|---|
| UKF, h₁ | 37.8 → 31.8 → 31.7 | 31.9 |
| EKF, h₁ | 37.8 → 31.6 → 31.6 | 31.7 |
| BO, h₁ | 37.8 → 34.4 → 33.3 | 36.7 (sometimes explores very costly parameters) |
| UKF, h₂ (sign-change penalty) | 69.0 → 46.7 → **37.8** | 43.8 |
| EKF, h₂ | 69.0 → 56.6 → 47.6 (stalls on plateaus) | 50.9 |

**Fig. 2/3, double integrator, episodic** (100 UKF iterations, C_θ = 0.1 I). Cost of objective (19a); the
LQR optimum for this objective is about 14.2.

| controller | start → end | | controller | start → end |
|---|---|---|---|---|
| state feedback | 150 → 14.2 | | sliding mode | 150 → 33.8 |
| optimal control (LQR) | 14.5 → 14.2 | | dynamic output feedback | 150 → 14.4 |
| PID | 150 → 45.9 | | neural network (151 params) | 1237 → 21.1 |
| H∞ loop shaping | 27.3 → 16.8 | | | |

With the overshoot penalty (19b), H∞ overshoot drops from 1.20 to 1.08; state feedback and sliding mode
never overshoot. **Differences from the paper:** PID settles on a poorly damped response (overshoot 1.7)
that the penalty cannot fix. With the paper's C_θ = I, PID, sliding mode and the NN oscillate or diverge,
and the outcome is very sensitive to the unreported sigma-point weight w₀. The decay factor (Eq. 20) is
reported by the script but is ill-conditioned for non-monotone curves, so I don't compare it with the paper.

**Table I, online with disturbances** (60 s, window N = 50, C_θ = 0.01 I, C_v = 10⁴ I; cost = mean p² + u²).

| controller | dv ~ N(0,1): fixed → online | dv = 1: fixed → online |
|---|---|---|
| state feedback | 13.5 → 13.8 (−3%) | 185 → 116 (+37%) |
| optimal control | 12.9 → 13.0 (−0.5%) | 217 → 142 (+35%) |
| PID | 48 → 129 (**worse**) | 160 → diverges |
| H∞ | 68.8 → 27.1 (+61%) | 9939 → 438 (+96%) |
| sliding mode | 93.5 → 24.2 (+74%) | 191199 → 2478 (+99%) |
| dynamic output feedback | 18.4 → 19.4 (−5%) | 375 → 145 (+61%) |
| neural network | 129 → 27.6 (+79%) | 234 → 106 (+55%) |

Compensating a systematic disturbance works for every controller except PID. **Important for later use:**
with the paper's C_θ = C_v = I, per-step updates on overlapping noisy windows destabilise the loop. When
H P Hᵀ ≫ C_v the Kalman gain becomes a full Gauss–Newton step to the optimum of one noise realisation,
whatever size C_θ has. A large C_v is what damps the step.

**Fig. 8/9, lane change with model mismatch** (dynamic bicycle as the plant, kinematic model in the
calibrator, C_θ = 0.01 I). Starting from θ = 0, both controllers learn to change lanes within the first
manoeuvre. State feedback learns the negative gains described in the paper (θ ≈ [−0.1, −1.8, −6.1] at
50 s) and re-adapts after the speed drop. For LQR the learned yaw weight is higher at the *lower* speed,
the opposite of the paper's trend. With C_θ = I both controllers go unstable after the first lane change.
