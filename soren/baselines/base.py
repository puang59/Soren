"""Common interface for traversal baselines.

BFS, DFS and a random walk define an order in which nodes are visited. They have no rule for
declaring a node vulnerable, so here they run under **Protocol A** (oracle stop): the search
succeeds the moment it first visits a vulnerable node. This favours the baselines, since the
learned agent must both reach the node and choose to declare it.

Two step counts are reported:

* ``nodes_inspected``: distinct nodes visited, including ENTRY and the target. Comparable
  across every method, and the primary metric.
* ``actions_taken``: raw actions, including backtracks, revisits and the final declaration.
  Comparable between methods that walk the graph edge by edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from soren.data.schema import GraphRecord
from soren.env.cfg_nav_env import CFGNavEnv, EnvConfig
from soren.env.rewards import RewardConfig


@dataclass
class EpisodeResult:
    method: str
    graph_id: str
    success: bool
    nodes_inspected: int
    actions_taken: int
    cumulative_reward: float
    """Unshaped return under the environment's reward configuration."""
    declared_node: int | None = None
    first_hit_step: int | None = None
    end_reason: str = ""
    visit_order: list[int] = field(default_factory=list)
    """Nodes in the order they were first visited."""


class Searcher(Protocol):
    name: str

    def run(self, graph: GraphRecord, max_steps: int, rng: np.random.Generator) -> EpisodeResult:
        """Search ``graph`` for a vulnerable node within ``max_steps`` actions."""
        ...


class EnvSearcher:
    """A baseline that walks the real environment one action at a time.

    Subclasses choose movement actions; this class applies the oracle stop by issuing
    ``DECLARE`` as soon as the walk stands on a vulnerable node.
    """

    name = "env_searcher"

    def __init__(
        self, env_config: EnvConfig | None = None, reward_config: RewardConfig | None = None
    ) -> None:
        self.env_config = env_config or EnvConfig()
        self.reward_config = reward_config or RewardConfig()

    def choose(self, env: CFGNavEnv, rng: np.random.Generator) -> int | None:
        """Return a movement action, or ``None`` when the search space is exhausted."""
        raise NotImplementedError

    def run(self, graph: GraphRecord, max_steps: int, rng: np.random.Generator) -> EpisodeResult:
        env = CFGNavEnv([graph], self.env_config, self.reward_config)
        _, info = env.reset(options={"graph_index": 0, "max_steps": max_steps})
        visit_order = [info["current_node"]]
        seen = set(visit_order)
        done = False
        exhausted = False
        while not done:
            if env.node in graph.vuln_set:
                action = env.declare_action
            else:
                action = self.choose(env, rng)
                if action is None:
                    exhausted = True
                    break
            *_, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if info["current_node"] not in seen:
                seen.add(info["current_node"])
                visit_order.append(info["current_node"])

        reward = info["episode_return"]
        end_reason = info["end_reason"] or ""
        if exhausted:
            # Nothing left to explore: score it like running out of steps.
            reward += self.reward_config.timeout
            end_reason = "exhausted"
        return EpisodeResult(
            method=self.name,
            graph_id=graph.sample_id,
            success=bool(info["is_success"]),
            nodes_inspected=len(visit_order),
            actions_taken=info["steps"],
            cumulative_reward=float(reward),
            declared_node=info["declared_node"],
            first_hit_step=info["first_hit_step"],
            end_reason=end_reason,
            visit_order=visit_order,
        )
