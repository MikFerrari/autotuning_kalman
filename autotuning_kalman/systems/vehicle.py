"""Vehicle models and controllers of Sections IV-B and V.

Kinematic single-track model, Eq. (21), state x = [pX, pY, psi, v, delta],
input u = [v_dot, delta_dot], discretised with forward Euler:

    pX'    = v cos(psi + beta) / cos(beta)
    pY'    = v sin(psi + beta) / cos(beta)
    psi'   = v tan(delta) / L
    v'     = v_dot,     delta' = delta_dot,     beta = arctan(lr tan(delta) / L)

CarSim (Section V) is replaced by a dynamic single-track model with linear
tyres (``DynamicBicycle``) so that the calibrator, which only knows the
kinematic model, still faces a model mismatch.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import solve_discrete_are

from autotuning_kalman import backend

LF, LR = 1.2, 1.6  # not given in the paper (CarSim data in Table II); typical sedan
L = LF + LR


def kinematic_f(x, u):
    """Continuous-time right-hand side of Eq. (21b) for batched x (..., 5), u (..., 2).

    Works on numpy arrays and torch tensors alike.
    """
    xp = torch if isinstance(x, torch.Tensor) else np
    psi, v, delta = x[..., 2], x[..., 3], x[..., 4]
    beta = xp.arctan(LR * xp.tan(delta) / L)
    return xp.stack([v * xp.cos(psi + beta) / xp.cos(beta),
                     v * xp.sin(psi + beta) / xp.cos(beta),
                     v * xp.tan(delta) / L,
                     u[..., 0], u[..., 1]], -1)


def kinematic_step(x, u, ts):
    return x + ts * kinematic_f(x, u)


def lateral_linearisation(v: float, ts: float):
    """Euler-discretised linearisation of [pY, psi, v, delta] at psi = delta = 0."""
    Ac = np.zeros((4, 4))
    Ac[0, 1], Ac[0, 3], Ac[1, 3] = v, v * LR / L, v / L
    Bc = np.zeros((4, 2))
    Bc[2, 0] = Bc[3, 1] = 1.0
    return np.eye(4) + ts * Ac, ts * Bc


# ---------------------------------------------------- Section IV-B: LQ optimal control
def sym_from_vec(vec, n: int):
    """Upper-triangular entries (row-major) -> symmetric matrices, batched (numpy or torch)."""
    iu = np.triu_indices(n)
    if isinstance(vec, torch.Tensor):
        M = vec.new_zeros(vec.shape[:-1] + (n, n))
    else:
        M = np.zeros(vec.shape[:-1] + (n, n))
    M[..., iu[0], iu[1]] = vec
    M[..., iu[1], iu[0]] = vec
    return M


def vec_from_sym(M: np.ndarray) -> np.ndarray:
    return M[..., np.triu_indices(M.shape[-1])[0], np.triu_indices(M.shape[-1])[1]]


def project_psd(M: torch.Tensor, floor: float) -> torch.Tensor:
    w, V = torch.linalg.eigh(M)
    return (V * w.clamp_min(floor)[..., None, :]) @ V.transpose(-1, -2)


class LQCostCalibrationTask:
    """Reference tracking with an optimal controller, Section IV-B.

    theta = [triu(Q) (10), triu(R) (3)]; cost sum ||M(x - x_ref)||_Q + ||u||_R
    with M = [0 I4]. The optimal controller is unconstrained receding-horizon
    LQ (horizon N = 20) on the linearised model -- the paper does not specify
    the solver -- and is applied to the nonlinear model (21).

    ``h`` is batched over parameter vectors and runs in PyTorch, on the GPU if
    available: stack the sigma points of many trials to fill it.
    """

    n_params = 13
    ts, horizon, steps = 0.25, 20, 20
    x0 = np.array([0.0, 2.0, 0.0, 12.0, 0.0])
    x_ref = np.array([0.0, 0.0, 0.0, 10.0, 0.0])

    def __init__(self, sign_change_penalty: bool = False, dev: torch.device | None = None):
        self.sign_change_penalty = sign_change_penalty
        self.dev = backend.device(large_batch=True) if dev is None else dev
        Ad, Bd = lateral_linearisation(self.x_ref[3], self.ts)
        self.Ad, self.Bd = backend.to_tensor(Ad, self.dev), backend.to_tensor(Bd, self.dev)

    def random_theta(self, rng: np.random.Generator) -> np.ndarray:
        """Random positive definite Q and R, as used to initialise every trial."""
        Gq, Gr = rng.standard_normal((4, 4)), rng.standard_normal((2, 2))
        return np.concatenate([vec_from_sym(Gq @ Gq.T / 4 + 0.05 * np.eye(4)),
                               vec_from_sym(Gr @ Gr.T / 2 + 0.05 * np.eye(2))])

    def gains(self, th: torch.Tensor) -> torch.Tensor:
        """Batched finite-horizon Riccati recursion; returns the first-stage gains (B, 2, 4)."""
        Q = project_psd(sym_from_vec(th[:, :10], 4), 0.0)
        R = project_psd(sym_from_vec(th[:, 10:], 2), 1e-6)
        A_, B_ = self.Ad, self.Bd
        P = Q.clone()
        for _ in range(self.horizon):
            BtP = B_.T @ P
            K = torch.linalg.solve(R + BtP @ B_, BtP @ A_)
            P = Q + A_.T @ P @ (A_ - B_ @ K)
            P = 0.5 * (P + P.transpose(-1, -2))
        return K

    def rollout(self, thetas):
        th = backend.to_tensor(np.atleast_2d(thetas), self.dev)
        K = self.gains(th)
        x_ref = backend.to_tensor(self.x_ref, self.dev)
        x = backend.to_tensor(self.x0, self.dev).expand(len(th), 5).clone()
        X, U = [x], []
        for _ in range(self.steps):
            u = -(K @ (x[:, 1:] - x_ref[1:])[:, :, None])[:, :, 0]
            x = kinematic_step(x, u, self.ts)
            X.append(x)
            U.append(u)
        return torch.stack(X, 1), torch.stack(U, 1)

    def h(self, thetas) -> np.ndarray:
        """h_1 (and h_2 = [h_1; #sign changes of delta_dot]) with y = 0."""
        X, U = self.rollout(thetas)
        x_ref = backend.to_tensor(self.x_ref, self.dev)
        parts = [(X[:, :, 1:] - x_ref[1:]).reshape(len(X), -1), U.reshape(len(U), -1)]
        if self.sign_change_penalty:
            parts.append(sign_changes(U[:, :, 1])[:, None])
        return torch.cat(parts, 1).cpu().numpy()

    @property
    def y(self) -> np.ndarray:
        return np.zeros((self.steps + 1) * 4 + self.steps * 2 + int(self.sign_change_penalty))


def sign_changes(u):
    """Number of sign changes along the last axis, zeros skipped (numpy or torch, batched)."""
    u = torch.as_tensor(u)
    s = torch.sign(u)
    last = torch.zeros_like(s[..., 0])  # sign of the last non-zero entry so far
    count = torch.zeros_like(last)
    for k in range(s.shape[-1]):
        sk = s[..., k]
        count += ((sk != 0) & (last != 0) & (sk != last)).to(count.dtype)
        last = torch.where(sk != 0, sk, last)
    return count


# ------------------------------------------------------- Section V: lane change
DELTA_MAX, DELTA_DOT_MAX = 0.5, 1.0  # actuator limits (rad, rad/s); not given in the paper


class StateFeedbackLaneChange:
    """delta_dot = [th_y th_psi th_delta] [pY - pY_ref; psi; delta], Eq. (24)."""
    n_params = 3

    def gains(self, thetas, v):
        return np.atleast_2d(thetas)


class LQRLaneChange:
    """LQR for J = sum th_y pY^2 + th_psi psi^2 + delta_dot^2, Eq. (23), via DARE.

    Linearised kinematic model at [0 0 0 v 0]; returns the equivalent gains
    on [pY - pY_ref; psi; delta]. Negative weights are clipped to ~0.
    """
    n_params = 2

    def gains(self, thetas, v):
        Ad, Bd = lateral_linearisation(v, 0.1)
        A3, B3 = Ad[np.ix_([0, 1, 3], [0, 1, 3])], Bd[[0, 1, 3], 1:]
        out = []
        for th in np.atleast_2d(thetas):
            Q = np.diag([max(th[0], 1e-6), max(th[1], 1e-6), 1e-6])
            try:
                P = solve_discrete_are(A3, B3, Q, np.eye(1))
                out.append(-np.linalg.solve(np.eye(1) + B3.T @ P @ B3, B3.T @ P @ A3)[0])
            except (np.linalg.LinAlgError, ValueError):
                out.append(np.zeros(3))
        return np.array(out)


def lateral_input(K: np.ndarray, x: np.ndarray, y_ref) -> np.ndarray:
    """delta_dot for batched gains K (B, 3) and states x (B, 5), with rate limit."""
    e = np.stack([x[:, 1] - y_ref, x[:, 2], x[:, 4]], axis=1)
    return np.clip(np.sum(K * e, axis=1), -DELTA_DOT_MAX, DELTA_DOT_MAX)


class DynamicBicycle:
    """Dynamic single-track model with linear, saturated tyres (stand-in for CarSim).

    State [X, Y, psi, vx, vy, r, delta]; inputs a (longitudinal acceleration)
    and delta_dot. Integrated with explicit Euler at ``dt``.
    """

    m, Iz, Cf, Cr, mu, g = 1500.0, 2500.0, 8.0e4, 8.0e4, 1.0, 9.81

    def __init__(self, vx0: float, dt: float = 1e-3):
        self.s = np.array([0.0, 0.0, 0.0, vx0, 0.0, 0.0, 0.0])
        self.dt = dt

    def kinematic_state(self) -> np.ndarray:
        X, Y, psi, vx, _, _, delta = self.s
        return np.array([X, Y, psi, vx, delta])

    def advance(self, a: float, delta_dot: float, duration: float) -> None:
        Fz_f = self.m * self.g * LR / L
        Fz_r = self.m * self.g * LF / L
        for _ in range(int(round(duration / self.dt))):
            X, Y, psi, vx, vy, r, delta = self.s
            af = delta - np.arctan2(vy + LF * r, vx)
            ar = -np.arctan2(vy - LR * r, vx)
            Fyf = np.clip(self.Cf * af, -self.mu * Fz_f, self.mu * Fz_f)
            Fyr = np.clip(self.Cr * ar, -self.mu * Fz_r, self.mu * Fz_r)
            ds = np.array([vx * np.cos(psi) - vy * np.sin(psi),
                           vx * np.sin(psi) + vy * np.cos(psi),
                           r,
                           a + vy * r - Fyf * np.sin(delta) / self.m,
                           (Fyf * np.cos(delta) + Fyr) / self.m - vx * r,
                           (LF * Fyf * np.cos(delta) - LR * Fyr) / self.Iz,
                           delta_dot])
            self.s = self.s + self.dt * ds
            self.s[6] = np.clip(self.s[6], -DELTA_MAX, DELTA_MAX)
