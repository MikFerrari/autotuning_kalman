"""Section IV-A2 (Fig. 2): episodic calibration of seven controllers on the double integrator.

Each controller runs in its own process (multicore CPU); the neural network's
303 sigma-point rollouts run on the GPU when one is available. Writes Fig. 2,
the decay factors of Eq. (20) and the calibrated parameters used by
di_online.py. Fig. 3 has its own script (di_fig3.py).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from autotuning_kalman import backend
from autotuning_kalman.calibrators import UKFCalibrator
from autotuning_kalman.metrics import decay_factor, training_cost
from autotuning_kalman.plotting import CONTROLLER_COLORS, LABELS, save, style
from autotuning_kalman.systems.double_integrator import CONTROLLERS, episodic_objective

OUT = Path(__file__).resolve().parents[1] / "outputs"
PAPER_DECAY = {"state_feedback": 16.4, "optimal_control": 79.2, "pid": 6.55, "h_infinity": 7.82,
               "sliding_mode": 37.6, "dynamic_output_feedback": 15.3, "neural_network": 6.39}


def calibrate_episodic(ctrl, iterations=100, overshoot=False, C_theta=1.0, theta0=None, decay=1.0):
    """UKF between repetitions of the tracking task; returns (thetas, costs).

    ``decay`` < 1 shrinks C_theta geometrically, i.e. the parameter freeze of Sec. III-D.
    """
    y, h = episodic_objective(ctrl, overshoot=overshoot)
    C0 = np.asarray(C_theta, dtype=float)
    cal = UKFCalibrator(ctrl.theta0() if theta0 is None else theta0, C_theta=C0)
    thetas = [cal.theta.copy()]
    for k in range(iterations):
        cal.C_theta = np.diag(np.broadcast_to(C0 * decay**k, (cal.n,)))
        thetas.append(cal.step(y, h).copy())
    thetas = np.array(thetas)
    return thetas, training_cost(y, h(thetas))


def _job(job):
    name, iterations, c_theta = job
    t0 = time.perf_counter()
    thetas, costs = calibrate_episodic(CONTROLLERS[name], iterations, C_theta=c_theta)
    return name, thetas, costs, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--controllers", nargs="*", default=list(CONTROLLERS))
    ap.add_argument("--c-theta", type=float, default=1.0, help="C_theta = c * I (paper: 1)")
    args = ap.parse_args()
    print(backend.describe())

    results = {r[0]: r[1:] for r in backend.parallel_map(
        _job, [(n, args.iterations, args.c_theta) for n in args.controllers])}
    print(f"{'controller':26s} {'cost_0':>10s} {'cost_end':>10s} {'decay %':>9s} {'paper %':>8s} {'time s':>7s}")
    for name, (thetas, costs, dt) in results.items():
        with np.errstate(all="ignore"):
            df = 100 * decay_factor(costs)
        print(f"{name:26s} {costs[0]:10.2f} {costs[-1]:10.2f} {df:9.2f} {PAPER_DECAY[name]:8.2f} {dt:7.1f}")
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "di_episodic_theta.npz", **{k: v[0][-1] for k, v in results.items()})

    fig = go.Figure()
    for name, (_, costs, _) in results.items():
        fig.add_trace(go.Scatter(y=costs, name=LABELS[name], line=dict(color=CONTROLLER_COLORS[name], width=2)))
    fig.update_yaxes(type="log", title="closed-loop cost (19a)")
    fig.update_xaxes(title="Kalman-filter iteration")
    style(fig, f"Fig. 2 — convergence of the closed-loop cost (C<sub>θ</sub> = {args.c_theta:g} I)", 850, 480)
    print("wrote", *save(fig, OUT / "fig2_di_convergence"))


if __name__ == "__main__":
    main()
