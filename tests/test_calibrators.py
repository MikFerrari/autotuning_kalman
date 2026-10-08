import numpy as np
import pytest

from autotuning_kalman.calibrators import EKFCalibrator, UKFCalibrator
from autotuning_kalman.metrics import decay_factor

RNG = np.random.default_rng(0)
H_LIN = RNG.standard_normal((6, 3))
THETA_TRUE = np.array([1.0, -2.0, 0.5])


def h_lin(thetas):
    return thetas @ H_LIN.T


def test_sigma_points_reproduce_mean_and_covariance():
    P = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 0.5]])
    cal = UKFCalibrator(THETA_TRUE, P0=P, w0=0.2)
    sp, w = cal.ask(), cal.weights
    assert sp.shape == (7, 3)
    np.testing.assert_allclose(w @ sp, THETA_TRUE)
    d = sp - THETA_TRUE
    np.testing.assert_allclose((w[:, None] * d).T @ d, P, atol=1e-12)


def test_ukf_posterior_ordering_matches_eq12_for_linear_h():
    cal = UKFCalibrator(np.zeros(3), C_theta=0.5, C_v=2.0, P0=np.eye(3))
    y = h_lin(THETA_TRUE[None])[0]
    P, Ct, Cv = np.eye(3), 0.5 * np.eye(3), 2.0 * np.eye(6)
    S = H_LIN @ P @ H_LIN.T + Cv
    K = P @ H_LIN.T @ np.linalg.inv(S)
    cal.step(y, h_lin)
    np.testing.assert_allclose(cal.theta, K @ y, atol=1e-10)
    np.testing.assert_allclose(cal.P, P + Ct - K @ S @ K.T, atol=1e-10)


def test_ukf_predicted_ordering_equals_ekf_for_linear_h():
    ukf = UKFCalibrator(np.zeros(3), C_theta=0.3, C_v=0.5, sigma_from="predicted")
    ekf = EKFCalibrator(np.zeros(3), C_theta=0.3, C_v=0.5)
    y = h_lin(THETA_TRUE[None])[0]
    for _ in range(5):
        ukf.step(y, h_lin)
        ekf.step(y, h_lin)
    np.testing.assert_allclose(ukf.theta, ekf.theta, atol=1e-7)
    np.testing.assert_allclose(ukf.P, ekf.P, atol=1e-7)


@pytest.mark.parametrize("cls", [UKFCalibrator, EKFCalibrator])
def test_converges_on_static_least_squares(cls):
    cal = cls(np.zeros(3), C_theta=1.0, C_v=1.0)
    y = h_lin(THETA_TRUE[None])[0]
    for _ in range(60):
        cal.step(y, h_lin)
    np.testing.assert_allclose(cal.theta, THETA_TRUE, atol=1e-6)


@pytest.mark.parametrize("central", [False, True])
def test_ekf_finite_difference_jacobian(central):
    def h(thetas):
        return np.stack([np.sin(thetas[:, 0]) * thetas[:, 1], thetas[:, 0] ** 2], axis=1)

    th = np.array([0.3, 2.0])
    cal = EKFCalibrator(th, fd_step=1e-6, central=central)
    H_ref = np.array([[np.cos(0.3) * 2.0, np.sin(0.3)], [0.6, 0.0]])
    ref = EKFCalibrator(th)
    ref.update(np.zeros(2), h(th[None])[0], H_ref)
    cal.step(np.zeros(2), h)
    np.testing.assert_allclose(cal.theta, ref.theta, atol=1e-5)


def test_decay_factor_eq20():
    # c_bar = [1, .5, .25, 0] -> every ratio (c_i - c_{i+1}) / c_{i+1} equals 1
    assert decay_factor([7.0, 5.0, 4.0, 3.0]) == pytest.approx(1.0)
