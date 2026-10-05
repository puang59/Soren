"""Graph-size curriculum: start training on small graphs and widen to the full range."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from soren.data.schema import GraphRecord


@dataclass
class CurriculumConfig:
    enabled: bool = False
    start_cap: int = 20
    """Largest graph, in nodes, sampled at the start."""
    growth: float = 1.5
    """Factor applied to the cap each time it widens."""
    success_threshold: float = 0.8
    """Widen once the rolling success rate reaches this."""
    window: int = 200
    """Episodes in the rolling success window."""
    widen_every: int | None = None
    """If set, widen on a fixed schedule of this many episodes, ignoring success."""

    def __post_init__(self) -> None:
        if self.start_cap < 1:
            raise ValueError("start_cap must be at least 1")
        if self.growth <= 1.0:
            raise ValueError("growth must be greater than 1")
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if self.widen_every is not None and self.widen_every < 1:
            raise ValueError("widen_every must be at least 1")


class CurriculumSampler:
    """Sample uniformly among the graphs no larger than the current cap.

    The environment calls the sampler to pick a graph and reports each finished episode back
    through :meth:`report`. The cap widens when the rolling success rate over a full window
    reaches the threshold, or every ``widen_every`` episodes if a schedule is set. Once the cap
    covers the largest graph, sampling is uniform over the whole set.

    Each environment owns its sampler, so with several environments the caps advance
    independently; they stay close because every environment sees the same policy.
    """

    def __init__(self, graphs: Sequence[GraphRecord], config: CurriculumConfig | None = None):
        if not graphs:
            raise ValueError("CurriculumSampler needs at least one graph")
        self.config = config or CurriculumConfig(enabled=True)
        sizes = np.array([graph.num_nodes for graph in graphs])
        self._order = np.argsort(sizes, kind="stable")
        self._sorted_sizes = sizes[self._order]
        self.max_size = int(sizes.max())
        # Never start below the smallest graph, or there would be nothing to sample.
        self.cap = min(max(self.config.start_cap, int(sizes.min())), self.max_size)
        self._recent: deque[bool] = deque(maxlen=self.config.window)
        self._episodes_at_cap = 0

    @property
    def finished(self) -> bool:
        return self.cap >= self.max_size

    @property
    def eligible(self) -> int:
        """Number of graphs within the current cap."""
        return int(np.searchsorted(self._sorted_sizes, self.cap, side="right"))

    def __call__(self, rng: np.random.Generator) -> int:
        return int(self._order[rng.integers(self.eligible)])

    def report(self, success: bool) -> None:
        """Record the outcome of one finished episode and widen the cap if it is time."""
        if self.finished:
            return
        self._recent.append(bool(success))
        self._episodes_at_cap += 1
        cfg = self.config
        if cfg.widen_every is not None:
            ready = self._episodes_at_cap >= cfg.widen_every
        else:
            full = len(self._recent) == cfg.window
            ready = full and float(np.mean(self._recent)) >= cfg.success_threshold
        if ready:
            self._widen()

    def _widen(self) -> None:
        before = self.eligible
        cap = self.cap
        # Grow until at least one new graph becomes eligible, so a widening is never a no-op.
        while True:
            cap = min(self.max_size, max(cap + 1, math.ceil(cap * self.config.growth)))
            self.cap = cap
            if self.eligible > before or self.finished:
                break
        self._recent.clear()
        self._episodes_at_cap = 0
