"""Section IV-B (Figs. 4-5): UKF vs. EKF vs. Bayesian optimisation on the kinematic vehicle.

Episodic calibration of the cost matrices Q (4x4) and R (2x2) of an optimal
controller, with random positive definite initialisations per trial.
Fig. 4: objective h_1. Fig. 5: h_2 = [h_1; #sign changes of delta_dot]
(non-differentiable), UKF vs. EKF only.

Compute placement: the UKF/EKF trials advance in lock-step and the sigma points
of *all* trials are stacked into one batched rollout (GPU if available). BO is
sequential within a trial, so its trials run on the CPU process pool, in
parallel with the GPU work.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from autotuning_kalman import backend
from autotuning_kalman.bayesopt import GPUCB
from autotuning_kalman.calibrators import EKFCalibrator, UKFCalibrator
from autotuning_kalman.metrics import training_cost
from autotuning_kalman.plotting import METHOD_COLORS, add_band, save, style
from autotuning_kalman.systems.vehicle import LQCostCalibrationTask

OUT = Path(__file__).resolve().parents[1] / "outputs"


def run_kalman_batched(cls, task, theta0s, iterations):
    """Run one filter per trial; every iteration evaluates all trials' points in one batch."""
    cals = [cls(th, C_theta=1.0, C_v=1.0) for th in theta0s]
    costs = [training_cost(task.y, task.h(np.array([c.theta for c in cals])))]
    for _ in range(iterations):
        pts = [c.ask() for c in cals]
        H = task.h(np.vstack(pts))
        for c, Hc in zip(cals, backend.chunks(H, [len(p) for p in pts])):
            c.tell(task.y, Hc)
        costs.append(training_cost(task.y, task.h(np.array([c.theta for c in cals]))))
    return np.array(costs).T  # (trials, iterations + 1)


def _bo_trial(job):
    theta0, iterations, seed, penalty = job
    task = LQCostCalibrationTask(sign_change_penalty=penalty, dev=torch.device("cpu"))
    bo = GPUCB(lambda rng, n: np.array([task.random_theta(rng) for _ in range(n)]), seed=seed)
    costs, theta = [], theta0
    for _ in range(iterations + 1):
        c = float(training_cost(task.y, task.h(theta[None]))[0])
        bo.tell(theta, c)
        costs.append(c)
        theta = bo.ask()
    return np.array(costs)


def figure(curves: dict, title: str, stem: Path):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["cost of operation at each time step",
                                        "average cost of operation so far"])
    for name, C in curves.items():
        x = np.arange(C.shape[1])
        for row, data in [(1, C), (2, np.cumsum(C, axis=1) / np.arange(1, C.shape[1] + 1))]:
            q25, q50, q75 = np.percentile(data, [25, 50, 75], axis=0)
            add_band(fig, x, q25, q50, q75, name, METHOD_COLORS[name], row=row, col=1, showlegend=row == 1)
    fig.update_yaxes(type="log", title_text="cost", row=1, col=1)
    fig.update_yaxes(type="log", title_text="average cost", row=2, col=1)
    fig.update_xaxes(title_text="time step (calibration iteration)", row=2, col=1)
    style(fig, title, 850, 650)
    print("wrote", *save(fig, stem))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=500, help="the paper uses 500")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--no-bo", action="store_true")
    args = ap.parse_args()
    print(backend.describe())

    for penalty, stem in [(False, "fig4_vehicle"), (True, "fig5_vehicle")]:
        task = LQCostCalibrationTask(sign_change_penalty=penalty)
        rng = np.random.default_rng(0)
        theta0s = [task.random_theta(rng) for _ in range(args.trials)]
        use_bo = not penalty and not args.no_bo
        with ThreadPoolExecutor(1) as tp:
            bo_future = tp.submit(backend.parallel_map, _bo_trial,
                                  [(th, args.iterations, i, penalty) for i, th in enumerate(theta0s)]) if use_bo else None
            t0 = time.perf_counter()
            curves = {"UKF": run_kalman_batched(UKFCalibrator, task, theta0s, args.iterations),
                      "EKF": run_kalman_batched(EKFCalibrator, task, theta0s, args.iterations)}
            print(f"  UKF+EKF ({args.trials} trials batched on {task.dev}): {time.perf_counter() - t0:.1f} s")
            if use_bo:
                curves["BO"] = np.array(bo_future.result())
                print(f"  BO done after {time.perf_counter() - t0:.1f} s (CPU pool)")

        print(f"{stem} ({'h2, sign-change penalty' if penalty else 'h1'}), {args.trials} trials")
        for k, C in curves.items():
            med = np.median(C, axis=0)
            print(f"  {k:4s} median cost: it0 {med[0]:9.2f}  it10 {med[10]:9.2f}  "
                  f"it{args.iterations} {med[-1]:9.2f} | median avg cost {np.median(C.mean(axis=1)):9.2f}")
        title = ("Fig. 5 — partially non-differentiable objective h<sub>2</sub>" if penalty
                 else "Fig. 4 — UKF vs. EKF vs. Bayesian optimisation")
        figure(curves, f"{title} ({args.trials} trials, median and 25–75 %)", OUT / stem)


if __name__ == "__main__":
    main()
