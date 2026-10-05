"""Breadth-first search baseline."""

from __future__ import annotations

from collections import deque

import numpy as np

from soren.baselines.base import OrderSearcher
from soren.data.schema import GraphRecord


class BFS(OrderSearcher):
    """Breadth-first from ENTRY, successors in canonical order."""

    name = "bfs"

    def order(self, graph: GraphRecord, rng: np.random.Generator) -> list[int]:
        order = [graph.entry]
        seen = {graph.entry}
        queue = deque([graph.entry])
        while queue:
            for succ in graph.successors(queue.popleft()):
                if succ not in seen:
                    seen.add(succ)
                    order.append(succ)
                    queue.append(succ)
        return order
