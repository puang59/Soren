"""Random walk baseline."""

from __future__ import annotations

import numpy as np

from soren.baselines.base import EnvSearcher
from soren.env.cfg_nav_env import CFGNavEnv


class RandomWalk(EnvSearcher):
    """Uniform choice among the valid ``MOVE`` and ``BACKTRACK`` actions."""

    name = "random_walk"
    stochastic = True

    def choose(self, env: CFGNavEnv, rng: np.random.Generator) -> int | None:
        moves = [a for a in env.valid_actions() if a != env.declare_action]
        return int(rng.choice(moves)) if moves else None
