"""Section IV-A2 (Fig. 3): state trajectories after calibration with (19a) and (19b).

Protocol (the paper gives no settings for this figure):
  1. calibrate with the tracking objective (19a) from theta_0;
  2. add the overshoot specification (19b) and keep calibrating from there.
Both phases run 100 UKF iterations with C_theta shrinking by 0.95 per
iteration, i.e. the parameter freeze of Section III-D. Without it the filter
keeps oscillating across the discontinuous 1.1 threshold.

C_theta = 0.1 I, except for PID: its good integral gain (~4e-4) is ~5000x
smaller than the other gains, so the per-parameter prior step size (Sec. III-C)
is scaled accordingly, C_theta = diag(1e-3, 1e-8, 1e-3).

The four controllers run in parallel processes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from autotuning_kalman import backend
from autotuning_kalman.metrics import training_cost
from autotuning_kalman.plotting import CONTROLLER_COLORS, LABELS, save, style
from autotuning_kalman.systems.double_integrator import CONTROLLERS, episodic_objective, simulate
from di_episodic import calibrate_episodic

OUT = Path(__file__).resolve().parents[1] / "outputs"
SELECTION = ["state_feedback", "pid", "h_infinity", "sliding_mode"]
C_THETA = {"pid": [1e-3, 1e-8, 1e-3]}
DECAY = 0.95
STEPS = 150


def _job(job):
    name, iterations = job
    ctrl = CONTROLLERS[name]
    c = C_THETA.get(name, 0.1)
    th_a = calibrate_episodic(ctrl, iterations, C_theta=c, decay=DECAY)[0][-1]
    th_b = calibrate_episodic(ctrl, iterations, overshoot=True, C_theta=c, theta0=th_a, decay=DECAY)[0][-1]
    out = {}
    for key, th in [("19a", th_a), ("19b", th_b)]:
        X, U, _ = simulate(ctrl, th, [0.0, 0.0], STEPS, 1.0)
        y, h = episodic_objective(ctrl, overshoot=key == "19b")
        out[key] = dict(theta=th, X=X[0], U=U[0], cost=float(training_cost(y, h(th[None]))[0]))
    return name, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, default=100)
    args = ap.parse_args()
    print(backend.describe())
    results = dict(backend.parallel_map(_job, [(n, args.iterations) for n in SELECTION]))

    print(f"{'controller':16s} {'max p (19a)':>12s} {'max p (19b)':>12s} {'cost (19a)':>11s} {'cost (19b)':>11s}")
    for name, r in results.items():
        print(f"{name:16s} {r['19a']['X'][:, 0].max():12.3f} {r['19b']['X'][:, 0].max():12.3f} "
              f"{r['19a']['cost']:11.2f} {r['19b']['cost']:11.2f}")

    t = 0.1 * np.arange(STEPS + 1)
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes="rows",
                        horizontal_spacing=0.04, vertical_spacing=0.07,
                        subplot_titles=["calibrated with (19a)", "calibrated with (19b): + overshoot cost"])
    for col, key in enumerate(["19a", "19b"], start=1):
        for name, r in results.items():
            for row, idx in [(1, 0), (2, 1)]:
                fig.add_trace(go.Scatter(x=t, y=r[key]["X"][:, idx], name=LABELS[name], legendgroup=name,
                                         showlegend=row == 1 and col == 1,
                                         line=dict(color=CONTROLLER_COLORS[name], width=2)), row=row, col=col)
        fig.add_hline(y=1.1, line=dict(color="#444", dash="dot", width=1), row=1, col=col,
                      annotation_text="1.1" if col == 2 else None, annotation_position="top right")
    fig.update_yaxes(title_text="position p", row=1, col=1)
    fig.update_yaxes(title_text="velocity v", row=2, col=1)
    fig.update_xaxes(title_text="time [s]", row=2)
    style(fig, "Fig. 3 — state trajectories after calibration", 1000, 620)
    fig.update_layout(legend=dict(orientation="h", y=-0.13, x=0.5, xanchor="center"))
    print("wrote", *save(fig, OUT / "fig3_di_trajectories"))


if __name__ == "__main__":
    main()
