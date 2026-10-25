"""Descriptions of what an environment accepts and returns.

* :class:`Discrete` ``(n)``: the actions are the integers ``0 … n−1``.
* :class:`Box` ``(low, high, shape)``: observations are float vectors of a fixed shape with
  every value in ``[low, high]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Discrete:
    def __init__(self, n: int) -> None:
        self.n = n

    def contains(self, x: Any) -> bool:
        return isinstance(x, int | np.integer) and 0 <= int(x) < self.n


class Box:
    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: type = np.float32):
        self.low, self.high, self.shape, self.dtype = low, high, tuple(shape), dtype

    def contains(self, x: Any) -> bool:
        x = np.asarray(x)
        return (
            x.shape == self.shape
            and x.dtype == self.dtype
            and bool(np.all(x >= self.low) and np.all(x <= self.high))
        )
