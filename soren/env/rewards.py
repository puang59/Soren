"""Reward function for the CFG navigation environment.

The base reward charges a small cost per movement, pays a large reward for a correct
declaration, and penalises wrong declarations and running out of steps.

Optional potential-based shaping adds ``gamma * phi(s') - phi(s)`` with
``phi(s) = -beta * d(s)``, where ``d`` is the distance from the current node to the nearest
vulnerable node. Shaping of this form leaves the optimal policy unchanged. It reads the label,
so it is a training-time signal only: reported returns always use the unshaped reward.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from soren.data.schema import GraphRecord


@dataclass
class StepEvent:
    """What happened in one environment step; the input to the reward function."""

    graph: GraphRecord
    action: str  # "move" | "backtrack" | "declare" | "invalid"
    prev_node: int
    node: int
    revisit: bool = False
    correct: bool | None = None  # set for "declare"
    timeout: bool = False
    dead_end: bool = False
    terminal: bool = False


@dataclass
class RewardConfig:
    step: float = -0.01
    """Cost of every move, backtrack or wasted (masked) action."""
    revisit: float = -0.02
    """Extra cost of moving onto an already visited node."""
    correct: float = 1.0
    wrong: float = -0.5
    timeout: float = -0.5
    """Charged when the step budget runs out or the agent is stuck in a dead end."""
    shaping: bool = False
    shaping_beta: float = 0.05
    gamma: float = 0.99
    """Discount used by the shaping term; keep it equal to the agent's discount."""


def base_reward(event: StepEvent, cfg: RewardConfig) -> float:
    """The unshaped reward for one step."""
    reward = 0.0
    if event.action == "declare":
        reward += cfg.correct if event.correct else cfg.wrong
    else:
        reward += cfg.step
        if event.action == "move" and event.revisit:
            reward += cfg.revisit
    if event.timeout or event.dead_end:
        reward += cfg.timeout
    return reward


def distance_to_vulnerable(graph: GraphRecord) -> np.ndarray:
    """Hop distance from every node to the nearest vulnerable node.

    Edges are treated as undirected: the agent moves forward along edges and backward along its
    own path, so the distance stays finite once it has walked past the target.
    """
    dist = np.full(graph.num_nodes, -1, dtype=np.int64)
    queue: deque[int] = deque()
    for v in graph.vuln_nodes:
        dist[v] = 0
        queue.append(v)
    while queue:
        u = queue.popleft()
        for v in (*graph.successors(u), *graph.predecessors(u)):
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                queue.append(v)
    # Nodes disconnected from every vulnerable node get the largest finite distance.
    dist[dist < 0] = graph.num_nodes
    return dist


class RewardModel:
    """Computes unshaped and shaped rewards, caching per-graph distances."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()
        self._distances: dict[str, np.ndarray] = {}

    def potential(self, graph: GraphRecord, node: int) -> float:
        dist = self._distances.get(graph.sample_id)
        if dist is None:
            dist = self._distances[graph.sample_id] = distance_to_vulnerable(graph)
        return -self.config.shaping_beta * float(dist[node])

    def shaping_term(self, event: StepEvent) -> float:
        """``gamma * phi(s') - phi(s)``, with the potential of a terminal state fixed at 0."""
        phi_next = 0.0 if event.terminal else self.potential(event.graph, event.node)
        return self.config.gamma * phi_next - self.potential(event.graph, event.prev_node)

    def __call__(self, event: StepEvent) -> tuple[float, float]:
        """Return ``(unshaped, training)`` rewards; they are equal when shaping is off."""
        unshaped = base_reward(event, self.config)
        if not self.config.shaping:
            return unshaped, unshaped
        return unshaped, unshaped + self.shaping_term(event)
