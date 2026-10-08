"""Bayesian optimisation baseline of Section IV-B1 (comparison only).

Gaussian process with a squared-exponential kernel (length scale 1, output
variance 1) and an upper-confidence-bound acquisition, written as a lower
confidence bound because the cost is minimised. The acquisition is optimised
over random candidates drawn from the search space (e.g. random positive
definite matrices) and around the incumbent; the paper does not detail this.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class GPUCB:
    def __init__(self, sample_space: Callable[[np.random.Generator, int], np.ndarray],
                 length_scale: float = 1.0, variance: float = 1.0, beta: float = 2.0,
                 noise: float = 1e-6, n_candidates: int = 2000, local_std: float = 0.3,
                 seed: int = 0):
        self.sample_space = sample_space
        self.l, self.sf2, self.beta, self.noise = length_scale, variance, beta, noise
        self.n_candidates, self.local_std = n_candidates, local_std
        self.rng = np.random.default_rng(seed)
        self.X: list[np.ndarray] = []
        self.c: list[float] = []

    def _k(self, A, B):
        d2 = np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None] - 2 * A @ B.T
        return self.sf2 * np.exp(-0.5 * np.maximum(d2, 0.0) / self.l**2)

    def ask(self) -> np.ndarray:
        X, c = np.array(self.X), np.array(self.c)
        y = (c - c.mean()) / (c.std() + 1e-12)
        n_loc = self.n_candidates // 2
        best = X[np.argmin(c)]
        cand = np.vstack([self.sample_space(self.rng, self.n_candidates - n_loc),
                          best + self.local_std * self.rng.standard_normal((n_loc, X.shape[1]))])
        Kxx = self._k(X, X) + self.noise * np.eye(len(X))
        Lc = np.linalg.cholesky(Kxx)
        Ks = self._k(cand, X)
        alpha = np.linalg.solve(Lc.T, np.linalg.solve(Lc, y))
        mu = Ks @ alpha
        v = np.linalg.solve(Lc, Ks.T)
        sd = np.sqrt(np.maximum(self.sf2 - np.sum(v**2, 0), 1e-12))
        return cand[np.argmin(mu - self.beta * sd)]

    def tell(self, theta: np.ndarray, cost: float) -> None:
        self.X.append(np.asarray(theta, dtype=float))
        self.c.append(float(cost))
