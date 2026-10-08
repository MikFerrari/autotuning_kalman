"""Safety check of Section III-D (Fig. 1, Theorem 1).

The check sits between the Kalman filter and the controller: it decides
whether the filter's current estimate may be applied. Rejections do not
touch the filter state, so the full data history still shapes later updates.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

# V(theta, x) -> Lyapunov value, or None if theta is not stabilising (condition 1).
LyapunovFn = Callable[[np.ndarray, np.ndarray], Optional[float]]


class LyapunovSafetyCheck:
    """Accept theta_new at state x only if V_new(x) <= V_old(x), Eq. (16).

    Condition 1 (theta_new stabilising when held constant) is enforced by
    ``lyapunov`` returning None for non-stabilising parameters.
    """

    def __init__(self, lyapunov: LyapunovFn):
        self.lyapunov = lyapunov
        self.n_rejected = 0

    def __call__(self, theta_old: np.ndarray, theta_new: np.ndarray, x: np.ndarray) -> np.ndarray:
        v_new = self.lyapunov(theta_new, x)
        v_old = self.lyapunov(theta_old, x)
        if v_new is not None and (v_old is None or v_new <= v_old):
            return np.asarray(theta_new, dtype=float).copy()
        self.n_rejected += 1
        return np.asarray(theta_old, dtype=float).copy()


def quadratic_lyapunov(A_cl: np.ndarray, x: np.ndarray, Q=None) -> Optional[float]:
    """x^T P x with A_cl^T P A_cl - P = -Q, or None if A_cl is not Schur."""
    if np.max(np.abs(np.linalg.eigvals(A_cl))) >= 1.0:
        return None
    Q = np.eye(A_cl.shape[0]) if Q is None else Q
    P = solve_discrete_lyapunov(A_cl.T, Q)
    return float(x @ P @ x)
