"""Scripted reference policies for validating the environment and the reward scale.

These are not baselines for the report (see ``soren.baselines``). They exist to answer
questions about the environment itself: is every graph solvable, are trajectories
deterministic, and does the reward make the intended behaviour the best one?
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from soren.env.cfg_nav_env import CFGNavEnv


class Policy(Protocol):
    def act(self, env: CFGNavEnv, rng: np.random.Generator) -> int:
        """Choose a valid action for the environment's current state."""
        ...


class OraclePolicy:
    """Walk a shortest forward path to the nearest vulnerable node, then declare it."""

    def act(self, env: CFGNavEnv, rng: np.random.Generator) -> int:
        graph, start = env.graph, env.node
        if start in graph.vuln_set:
            return env.declare_action
        # BFS over forward edges, remembering the first move taken from the start node.
        first_move = {start: -1}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for slot, v in enumerate(graph.successors(u)):
                if v in first_move:
                    continue
                first_move[v] = slot if u == start else first_move[u]
                if v in graph.vuln_set:
                    return first_move[v]
                queue.append(v)
        # The target is behind us; this only happens if the oracle is started mid-episode.
        return env.backtrack_action


class RandomPolicy:
    """Uniform over the valid actions."""

    def act(self, env: CFGNavEnv, rng: np.random.Generator) -> int:
        return int(rng.choice(env.valid_actions()))


class DeclareImmediatelyPolicy:
    """Degenerate: declare at the first node where declaring is allowed."""

    def act(self, env: CFGNavEnv, rng: np.random.Generator) -> int:
        valid = env.valid_actions()
        return env.declare_action if env.declare_action in valid else valid[0]


class NeverDeclarePolicy:
    """Degenerate: wander at random and never declare, so every episode times out."""

    def act(self, env: CFGNavEnv, rng: np.random.Generator) -> int:
        moves = [a for a in env.valid_actions() if a != env.declare_action]
        return int(rng.choice(moves))


@dataclass
class Rollout:
    graph_id: str
    actions: list[int]
    rewards: list[float]
    nodes: list[int]
    info: dict[str, Any]

    @property
    def success(self) -> bool:
        return bool(self.info["is_success"])

    @property
    def episode_return(self) -> float:
        """Unshaped return."""
        return float(self.info["episode_return"])


def rollout(
    env: CFGNavEnv,
    policy: Policy,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
) -> Rollout:
    """Run one episode of ``policy`` in ``env`` and return its trajectory."""
    rng = np.random.default_rng(seed)
    _, info = env.reset(seed=seed, options=options)
    actions: list[int] = []
    rewards: list[float] = []
    nodes = [info["current_node"]]
    done = False
    while not done:
        action = policy.act(env, rng)
        _, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        nodes.append(info["current_node"])
        done = terminated or truncated
    return Rollout(info["graph_id"], actions, rewards, nodes, info)
