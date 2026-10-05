"""Depth-first search baseline."""

from __future__ import annotations

import numpy as np

from soren.baselines.base import EnvSearcher
from soren.env.cfg_nav_env import CFGNavEnv


class DFS(EnvSearcher):
    """Depth-first from ENTRY, successors in canonical order, backtracking at dead ends.

    Every step is a real environment action, so its action count is directly comparable with
    the learned agent's.
    """

    name = "dfs"

    def choose(self, env: CFGNavEnv, rng: np.random.Generator) -> int | None:
        for slot, succ in enumerate(env.graph.successors(env.node)):
            if env.visits[succ] == 0:
                return slot
        return env.backtrack_action if env.stack else None
