"""Sanity-reference baselines that ignore the control flow entirely."""

from __future__ import annotations

import numpy as np

from soren.baselines.base import OrderSearcher
from soren.data.schema import GraphRecord


class RandomOrder(OrderSearcher):
    """Inspect nodes in a uniformly random permutation.

    With a single vulnerable node among ``n`` the expected number of nodes inspected is
    ``(n + 1) / 2``, which makes this a useful calibration point for the other methods.
    """

    name = "random_order"
    stochastic = True

    def order(self, graph: GraphRecord, rng: np.random.Generator) -> list[int]:
        return [int(node) for node in rng.permutation(graph.num_nodes)]


class LineOrder(OrderSearcher):
    """Inspect nodes top to bottom by source line: reading the function as written.

    If vulnerable statements cluster at a typical position, this baseline shows it.
    """

    name = "line_order"

    def order(self, graph: GraphRecord, rng: np.random.Generator) -> list[int]:
        return sorted(range(graph.num_nodes), key=lambda node: (graph.nodes[node].line, node))
