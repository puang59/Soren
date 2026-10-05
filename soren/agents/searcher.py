"""Run a trained policy through the environment and report it like a baseline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from soren.baselines.base import EpisodeResult
from soren.data.schema import GraphRecord
from soren.env.cfg_nav_env import CFGNavEnv, EnvConfig
from soren.env.rewards import RewardConfig


def evaluate_policy(
    model: Any,
    graphs: Sequence[GraphRecord],
    env_config: EnvConfig | None = None,
    reward_config: RewardConfig | None = None,
    deterministic: bool = True,
    name: str = "ppo",
    max_steps: int | None = None,
) -> list[EpisodeResult]:
    """Run one episode per graph and return the results in graph order.

    Unlike the traversal baselines, the policy gets no oracle stop: it succeeds only if it
    reaches a vulnerable node *and* declares it. Shaping is switched off, so
    ``cumulative_reward`` is the unshaped return.
    """
    reward_config = replace(reward_config or RewardConfig(), shaping=False)
    env = CFGNavEnv(graphs, env_config, reward_config)
    results = []
    for index, graph in enumerate(graphs):
        options: dict[str, Any] = {"graph_index": index}
        if max_steps is not None:
            options["max_steps"] = max_steps
        obs, info = env.reset(options=options)
        visit_order = [info["current_node"]]
        seen = set(visit_order)
        done = False
        while not done:
            action, _ = model.predict(
                obs, action_masks=env.action_masks(), deterministic=deterministic
            )
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            if info["current_node"] not in seen:
                seen.add(info["current_node"])
                visit_order.append(info["current_node"])
        results.append(
            EpisodeResult(
                method=name,
                graph_id=graph.sample_id,
                success=bool(info["is_success"]),
                nodes_inspected=len(visit_order),
                actions_taken=info["steps"],
                cumulative_reward=float(info["episode_return"]),
                declared_node=info["declared_node"],
                first_declared_node=info["first_declared_node"],
                first_hit_step=info["first_hit_step"],
                end_reason=info["end_reason"] or "",
                visit_order=visit_order,
            )
        )
    return results


class PolicySearcher:
    """Adapter that makes a trained policy usable wherever a baseline ``Searcher`` is."""

    def __init__(
        self,
        model: Any,
        env_config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
        deterministic: bool = True,
        name: str = "ppo",
    ) -> None:
        self.model = model
        self.env_config = env_config
        self.reward_config = reward_config
        self.deterministic = deterministic
        self.name = name
        self.stochastic = not deterministic

    def run(self, graph: GraphRecord, max_steps: int, rng: np.random.Generator) -> EpisodeResult:
        return evaluate_policy(
            self.model,
            [graph],
            self.env_config,
            self.reward_config,
            self.deterministic,
            self.name,
            max_steps,
        )[0]
