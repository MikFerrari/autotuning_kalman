"""Section V (Figs. 8-9): online calibration of lane-change controllers under model mismatch.

The "real" vehicle is a dynamic single-track model with tyre forces (stand-in
for CarSim, which is proprietary); the controller and the UKF only know the
kinematic model (21). The lateral reference switches between +-1.5 m every
7.5 s, the speed drops from 70 to 50 km/h at t = 50 s, and theta_0 = 0.
The UKF runs every 0.1 s on a 5 s window with training objective (22).

The two controllers run in parallel processes; each run is a sequential,
single-thread simulation (the rollouts have only 5-7 sigma points).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from autotuning_kalman import backend
from autotuning_kalman.calibrators import UKFCalibrator
from autotuning_kalman.online import SlidingWindow
from autotuning_kalman.plotting import save, style
from autotuning_kalman.systems.vehicle import (DELTA_MAX, DynamicBicycle, LQRLaneChange,
                                               StateFeedbackLaneChange, kinematic_step, lateral_input)

OUT = Path(__file__).resolve().parents[1] / "outputs"
TS = 0.1
CONTROLLERS = {"optimal_control": (LQRLaneChange(), ["th_y", "th_psi"]),
               "state_feedback": (StateFeedbackLaneChange(), ["th_y", "th_psi", "th_delta"])}
THETA_LABELS = {"th_y": "θ<sub>y</sub>", "th_psi": "θ<sub>ψ</sub>", "th_delta": "θ<sub>δ</sub>"}
PARAM_COLORS = ["#d62728", "#2ca02c", "#9467bd"]


def window_objective(ctrl, w):
    """y and h(theta) of Eq. (22) for the data window w (kinematic replay)."""
    Xw, Uw, refs = w["X"], w["U"], w["refs"]
    W = Xw[1:] - kinematic_step(Xw[:-1], Uw, TS)
    N = len(Uw)
    y = np.concatenate([refs, np.zeros(2 * N)])

    def h(thetas):
        K = ctrl.gains(thetas, Xw[0, 3])
        x = np.tile(Xw[0], (len(K), 1))
        pY, psi, dd = [], [], []
        for i in range(N):
            ddot = lateral_input(K, x, refs[i])
            u = np.stack([np.full(len(K), Uw[i, 0]), ddot], axis=1)
            x = kinematic_step(x, u, TS) + W[i]
            x[:, 4] = np.clip(x[:, 4], -DELTA_MAX, DELTA_MAX)
            pY.append(x[:, 1]), psi.append(x[:, 2]), dd.append(ddot)
        return np.concatenate([np.array(pY).T, 0.1 * np.array(psi).T, 10 * np.array(dd).T], axis=1)

    return y, h


def run(ctrl, duration=120.0, window=50, C_theta=0.01, v_hi=70 / 3.6, v_lo=50 / 3.6, t_drop=50.0):
    plant = DynamicBicycle(vx0=v_hi)
    cal = UKFCalibrator(np.zeros(ctrl.n_params), C_theta=C_theta, C_v=1.0)
    buf = SlidingWindow(window)
    log = {"t": [], "x": [], "ref": [], "theta": []}
    for k in range(int(round(duration / TS))):
        t = k * TS
        x = plant.kinematic_state()
        ref = 1.5 if int(t // 7.5) % 2 == 0 else -1.5
        if buf.full:
            cal.step(*window_objective(ctrl, buf.window(x)))
        ddot = lateral_input(ctrl.gains(cal.theta[None], x[3]), x[None], ref)[0]
        a = np.clip((v_hi if t < t_drop else v_lo) - x[3], -3.0, 3.0)  # simple speed controller
        buf.push(x, [a, ddot], None, ref)
        plant.advance(a, ddot, TS)
        for key, val in zip(log, [t, x, ref, cal.theta.copy()]):
            log[key].append(val)
    return {k: np.array(v) for k, v in log.items()}


def _job(job):
    name, duration, c_theta = job
    return name, run(CONTROLLERS[name][0], duration, C_theta=c_theta)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--c-theta", type=float, default=0.01, help="C_theta = c * I")
    args = ap.parse_args()
    print(backend.describe())
    logs = dict(backend.parallel_map(_job, [(n, args.duration, args.c_theta) for n in CONTROLLERS]))

    for name, log in logs.items():
        labels = CONTROLLERS[name][1]
        err = log["x"][:, 1] - log["ref"]
        segs = [(0, 15), (15, 50), (50, 65), (65, args.duration)]
        rms = "  ".join(f"[{a:g},{b:g}) s: {np.sqrt(np.mean(err[(log['t'] >= a) & (log['t'] < b)] ** 2)):.3f}"
                        for a, b in segs)
        print(f"{name}: RMS lateral error  {rms}")
        print(f"  theta at 50 s: {np.round(log['theta'][int(50 / TS) - 1], 3)}  "
              f"end: {np.round(log['theta'][-1], 3)}")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            row_heights=[0.4, 0.4, 0.2])
        fig.add_trace(go.Scatter(x=log["t"], y=log["ref"], name="reference", line=dict(color="#444", dash="dash",
                                 width=1.2, shape="hv")), row=1, col=1)
        fig.add_trace(go.Scatter(x=log["t"], y=log["x"][:, 1], name="lateral position p<sub>Y</sub>",
                                 line=dict(color="#1f77b4", width=2)), row=1, col=1)
        for i, lab in enumerate(labels):
            fig.add_trace(go.Scatter(x=log["t"], y=log["theta"][:, i], name=THETA_LABELS[lab],
                                     line=dict(color=PARAM_COLORS[i], width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=log["t"], y=3.6 * log["x"][:, 3], name="speed", showlegend=False,
                                 line=dict(color="#555", width=2)), row=3, col=1)
        fig.update_yaxes(title_text="p<sub>Y</sub> [m]", row=1, col=1)
        fig.update_yaxes(title_text="parameters", row=2, col=1)
        fig.update_yaxes(title_text="speed [km/h]", row=3, col=1)
        fig.update_xaxes(title_text="time [s]", row=3, col=1)
        num = 8 if name == "optimal_control" else 9
        kind = "optimal (LQR) controller" if name == "optimal_control" else "state-feedback controller"
        style(fig, f"Fig. {num} — online calibration of the {kind}", 950, 700)
        print("wrote", *save(fig, OUT / f"fig{num}_lane_change_{name}"))


if __name__ == "__main__":
    main()
