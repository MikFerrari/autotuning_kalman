import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve_discrete_are

from autotuning_kalman.backend import to_tensor

from autotuning_kalman.online import SlidingWindow
from autotuning_kalman.systems import vehicle as vh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from lane_change_online import window_objective  # noqa: E402


def test_linearisation_matches_finite_differences():
    ts, v = 0.25, 10.0
    x0, eps = np.array([0.0, 0.0, 0.0, v, 0.0]), 1e-6
    J = np.zeros((4, 4))
    for j, idx in enumerate([1, 2, 3, 4]):
        dx = np.zeros(5)
        dx[idx] = eps
        J[:, j] = (vh.kinematic_step(x0 + dx, np.zeros(2), ts) - vh.kinematic_step(x0 - dx, np.zeros(2), ts))[1:] / (2 * eps)
    Ad, _ = vh.lateral_linearisation(v, ts)
    np.testing.assert_allclose(J, Ad, atol=1e-6)


def test_finite_horizon_gains_approach_dare():
    task = vh.LQCostCalibrationTask(dev=torch.device("cpu"))
    task.horizon = 400
    theta = task.random_theta(np.random.default_rng(3))
    Q, R = vh.sym_from_vec(theta[:10], 4), vh.sym_from_vec(theta[10:], 2)
    Ad, Bd = task.Ad.numpy(), task.Bd.numpy()
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    np.testing.assert_allclose(task.gains(to_tensor(theta[None], task.dev))[0].numpy(), K, atol=1e-6)


def test_objective_shapes_and_sign_changes():
    assert vh.sign_changes(np.array([[1.0, -1.0, 0.0, -2.0, 3.0]]))[0] == 2
    assert vh.sign_changes(np.array([[0.0, 1.0, 0.0, 0.0, -1.0, -1.0, 2.0]]))[0] == 2
    for pen, n in [(False, 124), (True, 125)]:
        task = vh.LQCostCalibrationTask(sign_change_penalty=pen)
        th = np.stack([task.random_theta(np.random.default_rng(i)) for i in range(3)])
        assert task.h(th).shape == (3, n) and task.y.shape == (n,)


def test_lane_change_replay_reproduces_dynamic_plant():
    """With the applied parameters, the kinematic replay + recovered W matches the data."""
    ctrl, th = vh.StateFeedbackLaneChange(), np.array([-0.3, -1.5, -2.0])
    plant, buf = vh.DynamicBicycle(vx0=19.0), SlidingWindow(20)
    for _ in range(20):
        x = plant.kinematic_state()
        ddot = vh.lateral_input(ctrl.gains(th[None], x[3]), x[None], 1.5)[0]
        buf.push(x, [0.5, ddot], None, 1.5)
        plant.advance(0.5, ddot, 0.1)
    w = buf.window(plant.kinematic_state())
    _, h = window_objective(ctrl, w)
    np.testing.assert_allclose(h(th[None])[0, :20], w["X"][1:, 1], atol=1e-9)


def test_stacked_trials_equal_individual_evaluation():
    """Batching the sigma points of several trials must not change any result."""
    task = vh.LQCostCalibrationTask(sign_change_penalty=True)
    th = np.stack([task.random_theta(np.random.default_rng(i)) for i in range(6)])
    stacked = task.h(th)
    for i in range(6):
        np.testing.assert_allclose(task.h(th[i:i + 1])[0], stacked[i], atol=1e-10)
