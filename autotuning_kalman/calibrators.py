"""Kalman filters that estimate controller parameters (Section III of the paper).

The parameters follow a random walk and the training objective is the
"measurement" (Eq. 8):

    theta_{k+1} = theta_k + dtheta_k,     dtheta_k ~ N(0, C_theta)
    y_k         = h(theta_k) + v_k,       v_k      ~ N(0, C_v)

``y_k`` holds the desired values of the specification function and
``h(theta)`` is the specification function evaluated on closed-loop data
produced with parameters ``theta`` (Eq. 5-6).

Both filters use an ask/tell interface so that h can be evaluated by any
means (vectorised model, MuJoCo, JAX, ...):

    thetas = cal.ask()            # (M, n_theta) points at which h is needed
    cal.tell(y, h(thetas))        # (M, n_r) evaluations -> new cal.theta

``step(y, h_batch)`` does both in one call.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

HBatch = Callable[[np.ndarray], np.ndarray]


def _as_cov(c, n: int) -> np.ndarray:
    """Scalar, diagonal vector or full matrix -> (n, n) covariance."""
    c = np.asarray(c, dtype=float)
    if c.ndim == 0:
        return float(c) * np.eye(n)
    if c.ndim == 1:
        return np.diag(c)
    return c


def _chol(P: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor, falling back to an eigen-clipped matrix."""
    P = 0.5 * (P + P.T)
    try:
        return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(P)
        return np.linalg.cholesky((V * np.maximum(w, 1e-12)) @ V.T)


class _Calibrator:
    def __init__(self, theta0, C_theta=1.0, C_v=1.0, P0=None):
        self.theta = np.asarray(theta0, dtype=float).copy()
        self.n = self.theta.size
        self.C_theta = _as_cov(C_theta, self.n)
        self._C_v = C_v  # n_r is only known once h has been evaluated
        # The paper does not state P_0; the prior step covariance is used.
        self.P = self.C_theta.copy() if P0 is None else _as_cov(P0, self.n)

    def C_v(self, n_r: int) -> np.ndarray:
        return _as_cov(self._C_v, n_r)

    def step(self, y, h_batch: HBatch) -> np.ndarray:
        """One recursion: evaluate h at the requested points and update."""
        thetas = self.ask()
        self.tell(y, np.asarray(h_batch(thetas), dtype=float))
        return self.theta

    def ask(self) -> np.ndarray:
        raise NotImplementedError

    def tell(self, y, h_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class UKFCalibrator(_Calibrator):
    """Unscented Kalman filter calibration, Eq. (12) and Section III-B.

    Sigma points: theta, theta +/- sqrt(L / (1 - w0)) [A]_i with
    P_{k-1|k-1} = A A^T, weights w_{a,0} = w_{c,0} = w0 and
    w_{a,i} = w_{c,i} = (1 - w0) / (2L).

    ``sigma_from="posterior"`` follows Eq. (12) literally: sigma points are
    drawn from P_{k-1|k-1} and C_theta is added afterwards (12h).
    ``"predicted"`` is the textbook UKF ordering (sigma points drawn from
    P_{k-1|k-1} + C_theta); for a linear h it coincides with the EKF.
    """

    def __init__(self, theta0, C_theta=1.0, C_v=1.0, P0=None, w0: float = 1.0 / 3.0,
                 sigma_from: str = "posterior"):
        super().__init__(theta0, C_theta, C_v, P0)
        if not -1.0 < w0 < 1.0:
            raise ValueError("w0 must lie in (-1, 1)")
        if sigma_from not in ("posterior", "predicted"):
            raise ValueError("sigma_from must be 'posterior' or 'predicted'")
        self.w0 = w0
        self.sigma_from = sigma_from
        L = self.n
        self.weights = np.full(2 * L + 1, (1.0 - w0) / (2 * L))
        self.weights[0] = w0
        self._scale = np.sqrt(L / (1.0 - w0))

    def _sigma_cov(self) -> np.ndarray:
        return self.P + self.C_theta if self.sigma_from == "predicted" else self.P

    def ask(self) -> np.ndarray:
        A = self._scale * _chol(self._sigma_cov())
        return np.vstack([self.theta, self.theta + A.T, self.theta - A.T])

    def tell(self, y, h_values: np.ndarray) -> np.ndarray:
        thetas = self.ask()
        Ysp = np.atleast_2d(h_values)
        w = self.weights
        theta_hat = w @ thetas                                    # (12b)
        y_hat = w @ Ysp                                           # (12d)
        dT, dY = thetas - theta_hat, Ysp - y_hat
        S = self.C_v(Ysp.shape[1]) + (w[:, None] * dY).T @ dY     # (12f)
        C_sz = (w[:, None] * dT).T @ dY                           # (12g)
        P_sig = (w[:, None] * dT).T @ dT
        P_pred = P_sig + self.C_theta if self.sigma_from == "posterior" else P_sig  # (12h)
        K = np.linalg.solve(S, C_sz.T).T                          # (12e)
        self.theta = self.theta + K @ (np.asarray(y, dtype=float) - y_hat)  # (12a), (4)
        P = P_pred - K @ S @ K.T                                  # (12i)
        self.P = 0.5 * (P + P.T)
        return self.theta


class EKFCalibrator(_Calibrator):
    """Extended Kalman filter calibration, Eq. (11) and Section III-A.

    H_k = dh/dtheta is computed by finite differences on the points returned
    by ``ask`` (forward differences: n+1 points, central: 2n+1 points). An
    analytic Jacobian can be passed to ``update`` instead.
    """

    def __init__(self, theta0, C_theta=1.0, C_v=1.0, P0=None, fd_step: float = 1e-4,
                 central: bool = False):
        super().__init__(theta0, C_theta, C_v, P0)
        self.fd_step = fd_step
        self.central = central

    def _steps(self) -> np.ndarray:
        return self.fd_step * np.maximum(1.0, np.abs(self.theta))

    def ask(self) -> np.ndarray:
        E = np.diag(self._steps())
        pts = [self.theta, self.theta + E]
        if self.central:
            pts.append(self.theta - E)
        return np.vstack(pts)

    def tell(self, y, h_values: np.ndarray) -> np.ndarray:
        Y = np.atleast_2d(h_values)
        n, eps = self.n, self._steps()
        if self.central:
            H = ((Y[1:n + 1] - Y[n + 1:]) / (2 * eps[:, None])).T
        else:
            H = ((Y[1:n + 1] - Y[0]) / eps[:, None]).T
        return self.update(y, Y[0], H)

    def update(self, y, h_theta: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Eq. (11) given h(theta_k) and its Jacobian H_k (n_r, n_theta)."""
        P_pred = self.P + self.C_theta                            # (11d)
        S = H @ P_pred @ H.T + self.C_v(H.shape[0])               # (11c)
        K = np.linalg.solve(S, H @ P_pred).T                      # (11b)
        self.theta = self.theta + K @ (np.asarray(y, dtype=float) - h_theta)  # (11a)
        P = (np.eye(self.n) - K @ H) @ P_pred                     # (11e)
        self.P = 0.5 * (P + P.T)
        return self.theta
