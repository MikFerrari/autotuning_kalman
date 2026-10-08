"""Section IV-A3 (Table I): online calibration under noise / constant disturbance.

Regulation at p_ref = p0 = 0 with dv_k ~ N(0, 1) or dv_k = 1. Both runs start
from the parameters of 100 episodic iterations (Section IV-A2); the online
run executes the UKF at every time step on a sliding window of N steps,
replaying the window with the disturbances recovered from the data.

Per-step updates on overlapping, noisy windows need C_theta = 0.01 I and
C_v = 1e4 I (the paper uses I for both in the episodic setting): when
H P H^T >> C_v the UKF takes a full Gauss-Newton step to the optimum of a
single noise realisation, which destabilises the loop.

All (controller, case, fixed/online) runs are independent and run in parallel
processes; the neural network's window replays run on the GPU if available.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from autotuning_kalman import backend
from autotuning_kalman.calibrators import UKFCalibrator
from autotuning_kalman.online import SlidingWindow
from autotuning_kalman.plotting import CONTROLLER_COLORS, LABELS, save, style
from autotuning_kalman.systems.double_integrator import (CONTROLLERS, STATE_CLIP, disturbances,
                                                         simulate, step, tracking_spec)

OUT = Path(__file__).resolve().parents[1] / "outputs"
CPU = torch.device("cpu")


def run_closed_loop(ctrl, theta0, dv, window=None, C_theta=0.01, C_v=1e4):
    """Closed loop for len(dv) steps; online UKF calibration if ``window`` is given.

    The plant/controller step is a single sample (CPU); the window replays are
    batched over sigma points on the controller's preferred device.
    """
    t = lambda a: backend.to_tensor(a, CPU)  # noqa: E731
    x = np.zeros(2)
    theta = np.asarray(theta0, dtype=float)
    prm = ctrl.prepare(t(theta[None]))
    z = ctrl.init_state(prm, t(x))[0].numpy()
    cal = UKFCalibrator(theta, C_theta=C_theta, C_v=C_v) if window else None
    buf = SlidingWindow(window) if window else None
    X, U = [x], []
    for k in range(len(dv)):
        if buf is not None and buf.full:
            w = buf.window(x)
            W = disturbances(w["X"], w["U"])

            def h(thetas):
                Xr, Ur, valid = simulate(ctrl, thetas, w["X"][0], window, 0.0, W=W, z0=w["z0"])
                hv = tracking_spec(Xr, Ur)
                hv[~valid] = STATE_CLIP
                return hv

            theta = cal.step(np.zeros(2 * window), h)
            prm = ctrl.prepare(t(theta[None]))
        u, zn = ctrl.act(prm, t(x[None]), 0.0, t(z[None]))
        u = u.clamp(-STATE_CLIP, STATE_CLIP)
        if buf is not None:
            buf.push(x, float(u[0]), z)
        x = step(t(x[None]), u, t([0.0, dv[k]]))[0].numpy()
        z = zn[0].numpy()
        X.append(x)
        U.append(float(u[0]))
    return np.array(X), np.array(U), theta


def cost(X, U):
    """Mean per-step training objective (19a) with y = 0: p^2 + u^2."""
    return float(np.mean(X[1:, 0] ** 2 + U**2))


def _job(job):
    name, theta0, case, dv, online, window, c_theta, c_v = job
    t0 = time.perf_counter()
    X, U, _ = run_closed_loop(CONTROLLERS[name], theta0, dv, window if online else None, c_theta, c_v)
    return (name, case, online), cost(X, U), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=600, help="closed-loop steps (Ts = 0.1 s)")
    ap.add_argument("--window", type=int, default=50, help="sliding window N")
    ap.add_argument("--c-theta", type=float, default=0.01,
                    help="C_theta = c * I; per-step updates need a much smaller c than episodic ones")
    ap.add_argument("--c-v", type=float, default=1e4,
                    help="C_v = c * I; large values damp the near Gauss-Newton step under noise")
    ap.add_argument("--controllers", nargs="*", default=list(CONTROLLERS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(backend.describe())

    path = OUT / "di_episodic_theta.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found: run experiments/di_episodic.py first")
    theta_ep = np.load(path)
    cases = {"dv ~ N(0,1)": np.random.default_rng(args.seed).standard_normal(args.steps),
             "dv = 1": np.ones(args.steps)}
    jobs = [(n, theta_ep[n], case, dv, online, args.window, args.c_theta, args.c_v)
            for n in args.controllers for case, dv in cases.items() for online in (False, True)]
    # longest jobs first so the pool stays busy
    jobs.sort(key=lambda j: -(CONTROLLERS[j[0]].n_params * (10 if j[4] else 1)))
    res = {key: (c, dt) for key, c, dt in backend.parallel_map(_job, jobs)}

    print(f"{'controller':26s} {'case':12s} {'fixed':>11s} {'online':>11s} {'improv.':>8s} {'time s':>7s}")
    for n in args.controllers:
        for case in cases:
            c_fix, c_onl = res[(n, case, False)][0], res[(n, case, True)][0]
            print(f"{n:26s} {case:12s} {c_fix:11.3f} {c_onl:11.3f} {100 * (1 - c_onl / c_fix):7.1f}% "
                  f"{res[(n, case, True)][1]:7.1f}")

    fig = make_subplots(rows=1, cols=2, subplot_titles=list(cases), horizontal_spacing=0.08)
    names = [LABELS[n] for n in args.controllers]
    for col, case in enumerate(cases, start=1):
        for online, pattern in [(False, ""), (True, "/")]:
            fig.add_trace(go.Bar(
                x=names, y=[res[(n, case, online)][0] for n in args.controllers],
                name="online calibration" if online else "fixed (episodic) parameters",
                marker=dict(color=[CONTROLLER_COLORS[n] for n in args.controllers],
                            opacity=1.0 if online else 0.45, pattern_shape=pattern),
                showlegend=col == 1), row=1, col=col)
    fig.update_yaxes(type="log", title_text="mean p² + u²", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    style(fig, "Table I — fixed vs. online-calibrated parameters under disturbances", 1100, 520)
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"))
    print("wrote", *save(fig, OUT / "table1_di_online"))


if __name__ == "__main__":
    main()
