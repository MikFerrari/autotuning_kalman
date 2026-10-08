"""Evaluation metrics used in Section IV."""

from __future__ import annotations

import numpy as np


def training_cost(y, h, C_v_inv=None) -> np.ndarray:
    """||y - h||^2_{C_v^-1}, Eq. (5); h may carry leading batch dimensions."""
    e = np.asarray(y) - np.asarray(h)
    if C_v_inv is None:
        return np.sum(e * e, axis=-1)
    return np.einsum("...i,ij,...j->...", e, C_v_inv, e)


def decay_factor(costs) -> float:
    """Average per-iteration cost reduction, Eq. (20).

    With costs c_0..c_n and c_bar_i = (c_i - c_n) / (c_0 - c_n):
    dF = 1/(n-1) * sum_{i=0}^{n-2} (c_bar_i - c_bar_{i+1}) / c_bar_{i+1}.
    For n = 100 this is exactly the 1/99 * sum_{i=0}^{98} of the paper.
    """
    c = np.asarray(costs, dtype=float)
    n = c.size - 1
    c_bar = (c - c[n]) / (c[0] - c[n])
    return float(np.mean((c_bar[:n - 1] - c_bar[1:n]) / c_bar[1:n]))
