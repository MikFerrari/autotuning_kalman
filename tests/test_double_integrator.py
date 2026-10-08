import numpy as np
import pytest
import torch
from scipy.linalg import solve_discrete_are

from autotuning_kalman.backend import to_tensor

from autotuning_kalman.calibrators import UKFCalibrator
from autotuning_kalman.metrics import training_cost
from autotuning_kalman.safety import LyapunovSafetyCheck, quadratic_lyapunov
from autotuning_kalman.systems import double_integrator as di


def test_parameter_counts_match_paper():
    counts = {n: c.n_params for n, c in di.CONTROLLERS.items()}
    assert counts == {"state_feedback": 2, "optimal_control": 4, "pid": 3, "h_infinity": 8,
                      "sliding_mode": 2, "dynamic_output_feedback": 4, "neural_network": 151}
    for c in di.CONTROLLERS.values():
        assert c.theta0().shape == (c.n_params,)


def test_lqr_gain_matches_dare():
    Q, R = np.array([[2.0, 0.5], [0.5, 1.0]]), np.array([[0.3]])
    P = solve_discrete_are(di.A, di.B[:, None], Q, R)
    K = np.linalg.solve(R + di.B[None] @ P @ di.B[:, None], di.B[None] @ P @ di.A)[0]
    np.testing.assert_allclose(di.LQR.gain(np.array([2.0, 0.5, 1.0, 0.3])), K)


def test_pid_incremental_form_equals_positional_form():
    """u_k = P e_k + I sum_j (e_j + e_{j-1}) + D (e_k - e_{k-1}) with e_{-1} = e_{-2} = 0."""
    P, I, D = 0.7, 0.2, 0.4
    e = np.random.default_rng(1).standard_normal(20)
    pid = di.CONTROLLERS["pid"]
    cpu = torch.device("cpu")
    prm = pid.prepare(to_tensor([[P, I, D]], cpu))
    z = torch.zeros(1, 3, dtype=torch.float64)
    e_prev, integ = 0.0, 0.0
    for ek in e:
        u, z = pid.act(prm, to_tensor([[ek, 0.0]], cpu), 0.0, z)
        integ += ek + e_prev
        assert float(u[0]) == pytest.approx(P * ek + I * integ + D * (ek - e_prev))
        e_prev = ek


@pytest.mark.parametrize("theta", [np.ones(8), np.array([2.0, 1, 1, 3, 1, 5, 1, 2])])
def test_hinf_loopshaping_stabilises(theta):
    K = di.hinf_loopshaping(theta)
    A_cl = np.block([[di.A + np.outer(di.B, [K.D[0, 0], 0.0]), np.outer(di.B, K.C[0])],
                     [np.outer(K.B[:, 0], [1.0, 0.0]), K.A]])
    assert K.A.shape == (6, 6)
    assert np.max(np.abs(np.linalg.eigvals(A_cl))) < 1.0


def test_hinf_rejects_improper_compensator():
    assert di.hinf_loopshaping(np.array([1.0, 1, 0, 1, 1, 1, 1, 1])) is None


@pytest.mark.parametrize("name", list(di.CONTROLLERS))
def test_window_replay_reproduces_closed_loop(name):
    """h(theta) at the applied theta must reproduce the data (Section III, W from (2))."""
    ctrl = di.CONTROLLERS[name]
    th = ctrl.theta0()
    if name in ("state_feedback", "pid", "sliding_mode", "dynamic_output_feedback"):
        th = {"state_feedback": [-1.0, -1.5], "pid": [-1.0, -0.05, -2.0],
              "sliding_mode": [1.0, 0.5], "dynamic_output_feedback": [-1.0, -1.5, -0.5, -0.5]}[name]
    th = np.asarray(th, dtype=float)
    dv = np.random.default_rng(2).standard_normal(40) * 0.1
    cpu = torch.device("cpu")
    prm = ctrl.prepare(to_tensor(th[None], cpu))
    x = to_tensor([[0.3, 0.0]], cpu)
    z = ctrl.init_state(prm, x[0])
    X, U, Z = [x[0].numpy()], [], [z[0].numpy()]
    for k in range(40):
        u, z = ctrl.act(prm, x, 0.0, z)
        x = di.step(x, u, to_tensor([0.0, dv[k]], cpu))
        X.append(x[0].numpy()), U.append(float(u[0])), Z.append(z[0].numpy())
    X, U = np.array(X), np.array(U)
    s = 15  # window starts at k = 15
    W = di.disturbances(X[s:], U[s:])
    Xr, Ur, valid = di.simulate(ctrl, th, X[s], len(U) - s, 0.0, W=W, z0=Z[s])
    assert valid[0]
    np.testing.assert_allclose(Xr[0], X[s:], atol=1e-9)
    np.testing.assert_allclose(Ur[0], U[s:], atol=1e-9)


@pytest.mark.parametrize("name, C_theta, ratio", [
    ("state_feedback", 1.0, 0.2), ("dynamic_output_feedback", 1.0, 0.2),
    ("optimal_control", 1.0, 1.0),  # theta0 (Q = I, R = 1) is already near-optimal
    ("pid", 0.1, 0.5), ("neural_network", 0.1, 0.1)])
def test_episodic_ukf_reduces_cost(name, C_theta, ratio):
    ctrl = di.CONTROLLERS[name]
    y, h = di.episodic_objective(ctrl)
    cal = UKFCalibrator(ctrl.theta0(), C_theta=C_theta)
    c0 = training_cost(y, h(cal.theta[None]))[0]
    for _ in range(30):
        cal.step(y, h)
    assert training_cost(y, h(cal.theta[None]))[0] < ratio * c0


def test_safety_check_theorem1():
    def V(theta, x):
        return quadratic_lyapunov(di.A + np.outer(di.B, theta), x)

    gate = LyapunovSafetyCheck(V)
    x = np.array([1.0, 0.0])
    th_old = np.array([-1.0, -1.5])
    assert np.array_equal(gate(th_old, np.array([1.0, -1.5]), x), th_old)  # destabilising
    assert gate.n_rejected == 1
    th_new = np.array([-1.2, -1.6])
    expected = th_new if V(th_new, x) <= V(th_old, x) else th_old
    np.testing.assert_array_equal(gate(th_old, th_new, x), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
@pytest.mark.parametrize("name", ["neural_network", "pid", "h_infinity"])
def test_gpu_rollouts_match_cpu(name):
    ctrl = di.CONTROLLERS[name]
    th = ctrl.theta0() + 0.01 * np.random.default_rng(4).standard_normal((9, ctrl.n_params))
    Xc, Uc, _ = di.simulate(ctrl, th, [0.0, 0.0], 150, 1.0, dev=torch.device("cpu"))
    Xg, Ug, _ = di.simulate(ctrl, th, [0.0, 0.0], 150, 1.0, dev=torch.device("cuda"))
    np.testing.assert_allclose(Xg, Xc, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(Ug, Uc, rtol=1e-9, atol=1e-9)
