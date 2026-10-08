"""Double integrator and the seven controllers of Section IV-A.

    [p; v]_{k+1} = [[1, Ts], [0, 1]] [p; v]_k + Ts [0; 1] u_k + [0; 1] dv_k   (18a)
    e_k = p_k - p_ref                                                          (18b)

Rollouts are batched over parameter vectors (all UKF sigma points at once) and
run in PyTorch. Controllers with a large parameter count (``large_batch``,
i.e. the neural network) run on the GPU when one is available; the others run
on the CPU, where small batches are faster. Gain synthesis that needs scipy
(DARE, H-infinity Riccati equations) is done on the CPU and moved to the
rollout device.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.linalg import LinAlgWarning, solve_continuous_are, solve_discrete_are
from scipy.signal import cont2discrete

from autotuning_kalman.backend import DTYPE, device as pick_device, to_tensor

TS = 0.1
A = np.array([[1.0, TS], [0.0, 1.0]])
B = np.array([0.0, TS])
STATE_CLIP = 1e3  # keeps diverging sigma-point rollouts finite


def step(x: torch.Tensor, u: torch.Tensor, w=0.0) -> torch.Tensor:
    """Eq. (18a) for batched x (B, 2), u (B,); w is an additive state disturbance."""
    p, v = x[:, 0], x[:, 1]
    xn = torch.stack([p + TS * v, v + TS * u], dim=1) + w
    return xn.clamp(-STATE_CLIP, STATE_CLIP)


def disturbances(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """W_i = x_{i+1} - f(x_i, u_i) from measured data (Section III)."""
    return X[1:] - (X[:-1] @ A.T + U[:, None] * B)


# ---------------------------------------------------------------- controllers
class Controller:
    name: str
    n_params: int
    nz: int = 0               # dimension of the controller's internal state z_k
    large_batch: bool = False  # worth running on the GPU

    def theta0(self) -> np.ndarray:
        return np.zeros(self.n_params)

    def prepare(self, th: torch.Tensor) -> dict:
        """Parameter-dependent precomputation (gains, synthesis). Must set 'valid'."""
        return {"theta": th, "valid": torch.ones(len(th), dtype=torch.bool, device=th.device)}

    def init_state(self, prm: dict, x0: torch.Tensor) -> torch.Tensor:
        th = prm["theta"]
        return torch.zeros(len(th), self.nz, dtype=th.dtype, device=th.device)

    def act(self, prm: dict, x: torch.Tensor, p_ref: float, z: torch.Tensor):
        """Return (u (B,), z_next (B, nz))."""
        raise NotImplementedError


class StateFeedback(Controller):
    """u = [th1 th2] [e; v]."""
    name, n_params = "state_feedback", 2

    def act(self, prm, x, p_ref, z):
        th = prm["theta"]
        return th[:, 0] * (x[:, 0] - p_ref) + th[:, 1] * x[:, 1], z


class LQR(Controller):
    """Infinite-horizon LQR with Q = [[th1, th2], [th2, th3]], R = th4 (via DARE).

    Q and R must be PSD/PD for the DARE; parameters outside that set are
    projected onto it (eigenvalue clipping), which the paper does not detail.
    """
    name, n_params = "optimal_control", 4

    def theta0(self):
        return np.array([1.0, 0.0, 1.0, 1.0])

    @staticmethod
    def gain(th: np.ndarray) -> Optional[np.ndarray]:
        w, V = np.linalg.eigh(np.array([[th[0], th[1]], [th[1], th[2]]]))
        Q = (V * np.maximum(w, 1e-6)) @ V.T
        r = max(th[3], 1e-6)
        try:
            P = solve_discrete_are(A, B[:, None], Q, np.array([[r]]))
        except (np.linalg.LinAlgError, ValueError):
            return None
        return (B @ P @ A) / (r + B @ P @ B)

    def prepare(self, th):
        gains = [self.gain(t) for t in th.cpu().numpy()]
        valid = [g is not None for g in gains]
        K = np.array([g if g is not None else np.zeros(2) for g in gains])
        return {"theta": th, "K": to_tensor(K, th.device),
                "valid": torch.tensor(valid, device=th.device)}

    def act(self, prm, x, p_ref, z):
        K = prm["K"]
        return -(K[:, 0] * (x[:, 0] - p_ref) + K[:, 1] * x[:, 1]), z


class PID(Controller):
    """Incremental PID: u_k = u_{k-1} + (P+I+D) e_k + (I-P-2D) e_{k-1} + D e_{k-2}.

    z = (u_{k-1}, e_{k-1}, e_{k-2}).
    """
    name, n_params, nz = "pid", 3, 3

    def act(self, prm, x, p_ref, z):
        P, I, D = prm["theta"].T
        e = x[:, 0] - p_ref
        u = z[:, 0] + (P + I + D) * e + (I - P - 2 * D) * z[:, 1] + D * z[:, 2]
        u = u.clamp(-STATE_CLIP, STATE_CLIP)
        return u, torch.stack([u, e, z[:, 1]], dim=1)


@dataclass
class _SS:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray


def _first_order(a, b, c, d) -> _SS:
    """(a s + b) / (c s + d) as a one-state realisation (requires c != 0)."""
    p = d / c
    return _SS(np.array([[-p]]), np.array([[1.0]]), np.array([[(b - a * p) / c]]), np.array([[a / c]]))


def _series(s1: _SS, s2: _SS) -> _SS:
    """y = s2(s1(u))."""
    n1, n2 = s1.A.shape[0], s2.A.shape[0]
    A_ = np.block([[s1.A, np.zeros((n1, n2))], [s2.B @ s1.C, s2.A]])
    return _SS(A_, np.vstack([s1.B, s2.B @ s1.D]), np.hstack([s2.D @ s1.C, s2.C]), s2.D @ s1.D)


G_DI = _SS(np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0], [1.0]]),
           np.array([[1.0, 0.0]]), np.zeros((1, 1)))


def hinf_loopshaping(theta: np.ndarray, gamma_rel: float = 1.1) -> Optional[_SS]:
    """Glover-McFarlane H-infinity loop shaping (Skogestad & Postlethwaite, Table 9.2).

    Shaped plant G_s = W2 G W1 with W1 = (th1 s + th2)/(th3 s + th4) and
    W2 = (th5 s + th6)/(th7 s + th8); the final controller K = W1 K_s W2 maps
    e = p - p_ref to u (positive-feedback convention u = K e) and is
    discretised with the Tustin transform. Returns None if synthesis fails.
    """
    if abs(theta[2]) < 1e-6 or abs(theta[6]) < 1e-6:
        return None
    W1, W2 = _first_order(*theta[:4]), _first_order(*theta[4:])
    Gs = _series(_series(W1, G_DI), W2)  # strictly proper: D_s = 0
    a, b, c = Gs.A, Gs.B, Gs.C
    try:
        X = solve_continuous_are(a, b, c.T @ c, np.eye(1))
        Z = solve_continuous_are(a.T, c.T, b @ b.T, np.eye(1))
    except (np.linalg.LinAlgError, ValueError):
        return None
    gamma = gamma_rel * np.sqrt(1.0 + np.max(np.real(np.linalg.eigvals(X @ Z))))
    Lm = (1.0 - gamma**2) * np.eye(a.shape[0]) + X @ Z
    LZ = gamma**2 * np.linalg.solve(Lm.T, Z @ c.T)
    Ks = _SS(a - b @ b.T @ X + LZ @ c, LZ, b.T @ X, np.zeros((1, 1)))
    K = _series(_series(W2, Ks), W1)
    with warnings.catch_warnings():  # sigma points may put a pole near 2/Ts
        warnings.simplefilter("ignore", LinAlgWarning)
        Ad, Bd, Cd, Dd, _ = cont2discrete((K.A, K.B, K.C, K.D), TS, method="bilinear")
    if not np.all(np.isfinite(Ad)):
        return None
    return _SS(Ad, Bd, Cd, Dd)


class HInfinity(Controller):
    """Output feedback (position only): z+ = A z + b e, u = c z + d e."""
    name, n_params, nz = "h_infinity", 8, 6

    def theta0(self):
        return np.ones(8)  # W1 = W2 = 1: plain loop-shaping design, stabilising

    def prepare(self, th):
        n = len(th)
        Ad, Bd, Cd, Dd = np.zeros((n, 6, 6)), np.zeros((n, 6)), np.zeros((n, 6)), np.zeros(n)
        valid = np.zeros(n, dtype=bool)
        for i, t in enumerate(th.cpu().numpy()):
            K = hinf_loopshaping(t)
            if K is not None:
                Ad[i], Bd[i], Cd[i], Dd[i] = K.A, K.B[:, 0], K.C[0], K.D[0, 0]
                valid[i] = True
        dev = th.device
        return {"theta": th, "A": to_tensor(Ad, dev), "B": to_tensor(Bd, dev), "C": to_tensor(Cd, dev),
                "D": to_tensor(Dd, dev), "valid": torch.tensor(valid, device=dev)}

    def act(self, prm, x, p_ref, z):
        e = x[:, 0] - p_ref
        u = (prm["C"] * z).sum(1) + prm["D"] * e
        zn = torch.einsum("bij,bj->bi", prm["A"], z) + prm["B"] * e[:, None]
        return u, zn.clamp(-1e6, 1e6)


class SlidingMode(Controller):
    """u = -th1 v - th2 sign(e + th1 v); sliding surface e + th1 v = 0."""
    name, n_params = "sliding_mode", 2

    def act(self, prm, x, p_ref, z):
        th = prm["theta"]
        s = x[:, 0] - p_ref + th[:, 0] * x[:, 1]
        return -th[:, 0] * x[:, 1] - th[:, 1] * torch.sign(s), z


class DynamicOutputFeedback(Controller):
    """Observer-based output feedback; only p is measured. z = (p_hat, v_hat).

    u = [th1 th2] [p_hat - p_ref; v_hat]
    z+ = A z + [0; Ts] u + [th3; th4] (p_hat - p)
    """
    name, n_params, nz = "dynamic_output_feedback", 4, 2

    def theta0(self):
        # Stable observer (error poles ~0.72, 0.28): in the noise-free episodic task
        # p_hat = p, so the objective carries no information about th3, th4.
        return np.array([0.0, 0.0, -1.0, -2.0])

    def init_state(self, prm, x0):
        z = super().init_state(prm, x0)
        z[:, 0] = x0[0]
        return z

    def act(self, prm, x, p_ref, z):
        th = prm["theta"]
        u = th[:, 0] * (z[:, 0] - p_ref) + th[:, 1] * z[:, 1]
        zn = step(z, u) + th[:, 2:4] * (z[:, :1] - x[:, :1])
        return u, zn.clamp(-STATE_CLIP, STATE_CLIP)


class NeuralNetwork(Controller):
    """2-10-10-1 MLP with leaky ReLU (slope 0.1): 30 + 110 + 11 = 151 parameters."""
    name, n_params, large_batch = "neural_network", 151, True
    SHAPES = [(10, 2), (10,), (10, 10), (10,), (1, 10), (1,)]

    def theta0(self):
        return 0.1 * np.random.default_rng(0).standard_normal(self.n_params)

    def prepare(self, th):
        ws, i = [], 0
        for shape in self.SHAPES:
            n = int(np.prod(shape))
            ws.append(th[:, i:i + n].reshape(len(th), *shape))
            i += n
        return {"theta": th, "w": ws, "valid": torch.ones(len(th), dtype=torch.bool, device=th.device)}

    def act(self, prm, x, p_ref, z):
        W_in, b_in, W_lay, b_lay, W_out, b_out = prm["w"]
        sig = torch.nn.functional.leaky_relu
        inp = torch.stack([x[:, 0] - p_ref, x[:, 1]], dim=1)
        h1 = sig(torch.bmm(W_in, inp[:, :, None])[:, :, 0] + b_in, 0.1)
        h2 = sig(torch.bmm(W_lay, h1[:, :, None])[:, :, 0] + b_lay, 0.1)
        return torch.bmm(W_out, h2[:, :, None])[:, 0, 0] + b_out[:, 0], z


CONTROLLERS = {c.name: c for c in [StateFeedback(), LQR(), PID(), HInfinity(), SlidingMode(),
                                   DynamicOutputFeedback(), NeuralNetwork()]}


# ----------------------------------------------------------------- simulation
def simulate(ctrl: Controller, thetas, x0, steps: int, p_ref: float = 0.0,
             W: Optional[np.ndarray] = None, z0: Optional[np.ndarray] = None,
             dev: Optional[torch.device] = None):
    """Closed-loop rollouts for a batch of parameter vectors.

    W: (steps, 2) additive disturbances (e.g. replayed from data); z0: initial
    controller state shared by the batch. Returns numpy arrays X (B, steps+1, 2),
    U (B, steps) and the 'valid' mask from ``ctrl.prepare``.
    """
    dev = pick_device(ctrl.large_batch) if dev is None else dev
    th = to_tensor(np.atleast_2d(thetas), dev)
    prm = ctrl.prepare(th)
    x0_t = to_tensor(x0, dev)
    x = x0_t.expand(len(th), 2).clone()
    z = ctrl.init_state(prm, x0_t) if z0 is None else to_tensor(z0, dev).expand(len(th), -1).clone()
    W_t = None if W is None else to_tensor(W, dev)
    X, U = [x], []
    for k in range(steps):
        u, z = ctrl.act(prm, x, p_ref, z)
        u = u.clamp(-STATE_CLIP, STATE_CLIP)
        x = step(x, u, 0.0 if W_t is None else W_t[k])
        X.append(x)
        U.append(u)
    return (torch.stack(X, 1).cpu().numpy(), torch.stack(U, 1).cpu().numpy(),
            prm["valid"].cpu().numpy())


def tracking_spec(X: np.ndarray, U: np.ndarray, overshoot: bool = False) -> np.ndarray:
    """h(theta) of Eq. (19a) / (19b): [p_{1:N}; u_{0:N-1} (; cost_overshoot)]."""
    h = np.concatenate([X[:, 1:, 0], U], axis=1)
    if overshoot:
        h = np.concatenate([h, np.where(X[:, 1:, 0].max(axis=1) > 1.1, 10.0, 0.0)[:, None]], axis=1)
    return h


def episodic_objective(ctrl: Controller, p_ref: float = 1.0, steps: int = 150,
                       overshoot: bool = False, invalid_value: float = STATE_CLIP):
    """Reference tracking task of Section IV-A2: returns (y, h_batch)."""
    y = np.concatenate([np.full(steps, p_ref), np.zeros(steps)] + ([np.zeros(1)] if overshoot else []))

    def h_batch(thetas):
        X, U, valid = simulate(ctrl, thetas, [0.0, 0.0], steps, p_ref)
        h = tracking_spec(X, U, overshoot)
        h[~valid] = invalid_value  # failed synthesis (H-inf / LQR): large finite penalty
        return h

    return y, h_batch
