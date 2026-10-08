"""Sliding window of closed-loop data for online (receding-horizon) calibration.

At time k the objective is h(theta) = r'(theta, x_{k-N}, z_{k-N}, W_{k-N|k-1})
(Section III): the last N steps are replayed with candidate parameters
from the stored initial state x_{k-N}, the controller state z_{k-N} and the
disturbances W recovered from the data, W_i = x_{i+1} - f(x_i, u_i).
"""

from __future__ import annotations

from collections import deque

import numpy as np


class SlidingWindow:
    def __init__(self, N: int):
        self.N = N
        self._buf = deque(maxlen=N)  # (x_i, u_i, z_i, ref_i) for i = k-N .. k-1

    def push(self, x, u, z=None, ref=None) -> None:
        """Record the state, applied input, controller state and reference of one step."""
        self._buf.append((np.array(x, dtype=float), np.array(u, dtype=float),
                          None if z is None else np.array(z, dtype=float), ref))

    @property
    def full(self) -> bool:
        return len(self._buf) == self.N

    def window(self, x_now) -> dict:
        """Data of the last N steps; X includes the current state x_k."""
        xs, us, zs, refs = zip(*self._buf)
        return {"X": np.vstack(list(xs) + [np.asarray(x_now, dtype=float)]),
                "U": np.asarray(us), "z0": zs[0], "refs": np.asarray(refs)}
