"""Common interface for traversal baselines.

BFS, DFS and a random walk define an order in which nodes are visited. They have no rule for
declaring a node vulnerable, so by default they run under **Protocol A** (oracle stop): the
search succeeds the moment it first visits a vulnerable node. This favours the baselines,
since the learned agent must both reach the node and choose to declare it.

Passing a ``declare_rule`` replaces the oracle stop. **Protocol B** uses a rule that declares
at the first node whose score passes a threshold (see ``soren.baselines.heuristic``), so a
baseline can be wrong, like the agent.

Two step counts are reported:

* ``nodes_inspected``: distinct nodes visited, including ENTRY and the target. Comparable
  across every method, and the primary metric.
* ``actions_taken``: raw actions, including backtracks, revisits and the final declaration.
  Comparable between methods that walk the graph edge by edge.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from soren.data.schema import GraphRecord
from soren.env.cfg_nav_env import CFGNavEnv, EnvConfig
from soren.env.rewards import RewardConfig
from soren.env.wrappers import TraceRecorder
from soren.viz.trace import INSPECT, Trace, TraceStep

DeclareRule = Callable[[GraphRecord, int], bool]
"""Decides, on arriving at a node, whether to declare it vulnerable."""


def oracle_stop(graph: GraphRecord, node: int) -> bool:
    """Protocol A: declare exactly when standing on a vulnerable node."""
    return node in graph.vuln_set


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
    """The last node declared."""
    first_declared_node: int | None = None
    """The first node declared; differs from ``declared_node`` only with a declare budget."""
    first_hit_step: int | None = None
    end_reason: str = ""
    visit_order: list[int] = field(default_factory=list)
    """Nodes in the order they were first visited."""


class Searcher(Protocol):
    name: str
    stochastic: bool
    """Whether results depend on the random generator passed to ``run``."""

    def run(self, graph: GraphRecord, max_steps: int, rng: np.random.Generator) -> EpisodeResult:
        """Search ``graph`` for a vulnerable node within ``max_steps`` actions."""
        ...


class EnvSearcher:
    """A baseline that walks the real environment one action at a time.

    Subclasses choose movement actions; this class applies the oracle stop by issuing
    ``DECLARE`` as soon as the walk stands on a vulnerable node.
    """

    name = "env_searcher"
    stochastic = False

    def __init__(
        self,
        env_config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
        declare_rule: DeclareRule | None = None,
        name: str | None = None,
    ) -> None:
        self.env_config = env_config or EnvConfig()
        self.reward_config = reward_config or RewardConfig()
        self.declare_rule = declare_rule or oracle_stop
        if name is not None:
            self.name = name

    def choose(self, env: CFGNavEnv, rng: np.random.Generator) -> int | None:
        """Return a movement action, or ``None`` when the search space is exhausted."""
        raise NotImplementedError

    def run(self, graph: GraphRecord, max_steps: int, rng: np.random.Generator) -> EpisodeResult:
        return self._episode(graph, max_steps, rng, None)

    def trace(self, graph: GraphRecord, max_steps: int, rng: np.random.Generator) -> Trace:
        """Run one episode and return it as a replayable trace."""
        recorder = TraceRecorder(CFGNavEnv([graph], self.env_config, self.reward_config), self.name)
        result = self._episode(graph, max_steps, rng, recorder)
        trace = recorder.trace
        assert trace is not None
        # Covers the case where the search ran out of nodes before the episode ended; the
        # penalty for that is charged to the last step, as a timeout would be.
        if result.end_reason == "exhausted" and trace.steps:
            trace.steps[-1].reward += self.reward_config.timeout
        trace.outcome = {
            "success": result.success,
            "declared_node": result.declared_node,
            "return": result.cumulative_reward,
            "end_reason": result.end_reason,
            "nodes_inspected": result.nodes_inspected,
        }
        return trace

    def _episode(
        self,
        graph: GraphRecord,
        max_steps: int,
        rng: np.random.Generator,
        recorder: TraceRecorder | None,
    ) -> EpisodeResult:
        env = recorder.core if recorder else CFGNavEnv([graph], self.env_config, self.reward_config)
        stepper = recorder or env
        _, info = stepper.reset(options={"graph_index": 0, "max_steps": max_steps})
        visit_order = [info["current_node"]]
        seen = set(visit_order)
        done = False
        exhausted = False
        while not done:
            if self.declare_rule(graph, env.node) and env.action_masks()[env.declare_action]:
                action = env.declare_action
            else:
                action = self.choose(env, rng)
                if action is None:
                    exhausted = True
                    break
            *_, terminated, truncated, info = stepper.step(action)
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
            first_declared_node=info["first_declared_node"],
            first_hit_step=info["first_hit_step"],
            end_reason=end_reason,
            visit_order=visit_order,
        )


class OrderSearcher:
    """A baseline defined only by the order in which it inspects nodes.

    Such a search may jump between nodes that are not adjacent, which the environment's
    actions cannot do, so it is simulated directly: the first node is free (like standing on
    ENTRY), each further node costs one step, and the final declaration is one more action.
    ``actions_taken`` is therefore a lower bound on what an edge-by-edge walk in the same
    order would need.
    """

    name = "order_searcher"
    stochastic = False

    def __init__(
        self,
        reward_config: RewardConfig | None = None,
        declare_rule: DeclareRule | None = None,
        name: str | None = None,
    ) -> None:
        self.reward_config = reward_config or RewardConfig()
        self.declare_rule = declare_rule or oracle_stop
        if name is not None:
            self.name = name

    def order(self, graph: GraphRecord, rng: np.random.Generator) -> list[int]:
        """Nodes in the order they are inspected."""
        raise NotImplementedError

    def run(self, graph: GraphRecord, max_steps: int, rng: np.random.Generator) -> EpisodeResult:
        cfg = self.reward_config
        visit_order: list[int] = []
        steps = 0
        reward = 0.0
        declared: int | None = None
        first_hit_step: int | None = None
        end_reason = "exhausted"

        for position, node in enumerate(self.order(graph, rng)):
            visit_order.append(node)
            if position > 0:
                steps += 1
                reward += cfg.step
                if steps >= max_steps:
                    end_reason = "timeout"
                    break
            if first_hit_step is None and node in graph.vuln_set:
                first_hit_step = steps
            declarable = graph.nodes[node].kind not in ("ENTRY", "EXIT")
            if declarable and self.declare_rule(graph, node):
                declared = node
                break

        success = declared is not None and declared in graph.vuln_set
        if declared is not None:
            steps += 1  # the declaration
            reward += cfg.correct if success else cfg.wrong
            end_reason = "correct" if success else "wrong_declare"
        else:
            reward += cfg.timeout
        return EpisodeResult(
            method=self.name,
            graph_id=graph.sample_id,
            success=success,
            nodes_inspected=len(visit_order),
            actions_taken=steps,
            cumulative_reward=reward,
            declared_node=declared,
            first_declared_node=declared,
            first_hit_step=first_hit_step,
            end_reason=end_reason,
            visit_order=visit_order,
        )

    def trace(self, graph: GraphRecord, max_steps: int, rng: np.random.Generator) -> Trace:
        """Run one search and return it as a replayable trace of ``INSPECT`` jumps."""
        result = self.run(graph, max_steps, rng)
        cfg = self.reward_config
        order = result.visit_order
        steps = [
            TraceStep(t=i, node=order[i], action=INSPECT, next_node=order[i + 1], reward=cfg.step)
            for i in range(len(order) - 1)
        ]
        if result.declared_node is not None:
            node = order[-1]
            reward = cfg.correct if result.success else cfg.wrong
            steps.append(TraceStep(len(steps), node, "DECLARE", node, reward))
        elif steps:
            steps[-1].reward += cfg.timeout
        return Trace(
            graph_id=graph.sample_id,
            method=self.name,
            start_node=order[0],
            steps=steps,
            outcome={
                "success": result.success,
                "declared_node": result.declared_node,
                "return": result.cumulative_reward,
                "end_reason": result.end_reason,
                "nodes_inspected": result.nodes_inspected,
            },
        )
