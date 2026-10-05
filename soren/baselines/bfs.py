"""Breadth-first search baseline."""

from __future__ import annotations

from collections import deque

import numpy as np

from soren.baselines.base import EpisodeResult
from soren.data.schema import GraphRecord
from soren.env.rewards import RewardConfig


class BFS:
    """Breadth-first from ENTRY, successors in canonical order.

    BFS jumps between frontier nodes that are not adjacent, which the environment's actions
    cannot do. It is therefore simulated directly: each newly inspected node costs one step,
    and the final declaration is one more action. Its ``actions_taken`` is a lower bound on
    what an edge-by-edge walk in the same order would need.
    """

    name = "bfs"

    def __init__(self, reward_config: RewardConfig | None = None) -> None:
        self.reward_config = reward_config or RewardConfig()

    def run(self, graph: GraphRecord, max_steps: int, rng: np.random.Generator) -> EpisodeResult:
        cfg = self.reward_config
        visit_order = [graph.entry]
        seen = {graph.entry}
        queue = deque([graph.entry])
        steps = 0
        reward = 0.0
        hit: int | None = None
        end_reason = "exhausted"

        while queue and hit is None and end_reason != "timeout":
            node = queue.popleft()
            for succ in graph.successors(node):
                if succ in seen:
                    continue
                steps += 1
                reward += cfg.step
                seen.add(succ)
                visit_order.append(succ)
                queue.append(succ)
                if steps >= max_steps:
                    end_reason = "timeout"
                    break
                if succ in graph.vuln_set:
                    hit = succ
                    break

        first_hit_step = None
        if hit is not None:
            first_hit_step = steps
            steps += 1  # the declaration
            reward += cfg.correct
            end_reason = "correct"
        else:
            reward += cfg.timeout
        return EpisodeResult(
            method=self.name,
            graph_id=graph.sample_id,
            success=hit is not None,
            nodes_inspected=len(visit_order),
            actions_taken=steps,
            cumulative_reward=reward,
            declared_node=hit,
            first_hit_step=first_hit_step,
            end_reason=end_reason,
            visit_order=visit_order,
        )
