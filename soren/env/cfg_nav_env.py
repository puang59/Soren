"""The CFG navigation environment.

Each episode is one vulnerable function. The agent starts at the function's ENTRY node and at
every step either moves to a successor of the current node, backtracks along the path it came
by, or declares the current node vulnerable.

Actions (``Discrete(K + 2)``):

* ``0 .. K-1``  ``MOVE_i``: move to the i-th successor in canonical (line) order
* ``K``         ``BACKTRACK``: pop the path stack and return to the previous node
* ``K + 1``     ``DECLARE``: declare the current node vulnerable

The number of successors varies per node, so :meth:`CFGNavEnv.action_masks` reports which
actions are valid. It is the hook ``sb3_contrib.MaskablePPO`` looks for.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from soren.data.features import feature_dim, featurize
from soren.data.schema import NODE_KINDS, GraphRecord
from soren.env.rewards import RewardConfig, RewardModel, StepEvent

VISIT_CLIP = 4
STACK_CLIP = 32
SIZE_REF = 512  # log(|V|) is scaled by log(SIZE_REF)
CONTEXT_DIM = 5

_KIND_INDEX = {kind: i for i, kind in enumerate(NODE_KINDS)}


class InvalidActionError(ValueError):
    """Raised when a masked action is taken and ``strict_masks`` is on."""


@dataclass
class EnvConfig:
    k: int = 6
    """Number of successor slots. Graphs with a larger out-degree are rejected."""
    max_steps_cap: int = 200
    max_steps_per_node: float = 4.0
    """Step budget is ``min(max_steps_cap, max_steps_per_node * |V|)``."""
    max_declares: int = 1
    """Declarations allowed per episode; the episode ends when a wrong one uses the last."""
    feature_tier: str = "L"
    allow_backtrack: bool = True
    strict_masks: bool = True
    """Raise on masked actions. When off, a masked action is a wasted step instead."""

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be at least 1")
        if self.max_declares < 1:
            raise ValueError("max_declares must be at least 1")

    def max_steps_for(self, num_nodes: int) -> int:
        return max(1, min(self.max_steps_cap, math.ceil(self.max_steps_per_node * num_nodes)))


Sampler = Callable[[np.random.Generator], int]


class CFGNavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        graphs: Sequence[GraphRecord],
        config: EnvConfig | None = None,
        reward: RewardConfig | None = None,
        sampler: Sampler | None = None,
    ) -> None:
        super().__init__()
        if not graphs:
            raise ValueError("CFGNavEnv needs at least one graph")
        self.graphs = list(graphs)
        self.config = config or EnvConfig()
        self.reward_model = RewardModel(reward)
        self.sampler = sampler

        k = self.config.k
        for graph in self.graphs:
            widest = max(len(graph.successors(node.id)) for node in graph.nodes)
            if widest > k:
                raise ValueError(
                    f"{graph.sample_id}: out-degree {widest} exceeds k={k}; "
                    "drop the graph or raise k"
                )

        self._index_by_id = {graph.sample_id: i for i, graph in enumerate(self.graphs)}
        self._feature_cache: dict[int, np.ndarray] = {}
        self._feature_dim = feature_dim(self.config.feature_tier)
        self._slot_dim = self._feature_dim + 3
        self._backtrack_dim = 2 + len(NODE_KINDS)
        obs_dim = self._feature_dim + 2 + k * self._slot_dim + self._backtrack_dim + CONTEXT_DIM

        self.action_space = spaces.Discrete(k + 2)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)

        self._graph_index = 0
        self._started = False
        self._begin_episode(0, None)
        self._started = False

    # ------------------------------------------------------------------ actions

    @property
    def backtrack_action(self) -> int:
        return self.config.k

    @property
    def declare_action(self) -> int:
        return self.config.k + 1

    def action_name(self, action: int) -> str:
        if action == self.backtrack_action:
            return "BACKTRACK"
        if action == self.declare_action:
            return "DECLARE"
        return f"MOVE_{action}"

    def action_masks(self) -> np.ndarray:
        """Boolean mask over actions; ``True`` marks a valid action in the current state."""
        mask = np.zeros(self.config.k + 2, dtype=bool)
        if self._done:
            return mask
        mask[: len(self.graph.successors(self.node))] = True
        mask[self.backtrack_action] = self.config.allow_backtrack and bool(self.stack)
        mask[self.declare_action] = (
            self.graph.nodes[self.node].kind not in ("ENTRY", "EXIT")
            and self.node not in self.declared
            and self.declares_left > 0
        )
        return mask

    def valid_actions(self) -> list[int]:
        return [int(a) for a in np.flatnonzero(self.action_masks())]

    # ----------------------------------------------------------------- episode

    def _begin_episode(self, index: int, max_steps: int | None) -> None:
        self._graph_index = index
        self.graph = self.graphs[index]
        self.node = self.graph.entry
        self.stack: list[int] = []
        self.visits = np.zeros(self.graph.num_nodes, dtype=np.int64)
        self.visits[self.node] = 1
        self.declared: list[int] = []
        self.declares_left = self.config.max_declares
        self.t = 0
        self.max_steps = max_steps or self.config.max_steps_for(self.graph.num_nodes)
        self.first_hit_step: int | None = 0 if self.node in self.graph.vuln_set else None
        self.is_success = False
        self.end_reason: str | None = None
        self.episode_return = 0.0  # unshaped; this is the return to report
        self.shaped_return = 0.0
        self._last_reward = 0.0
        self._done = False
        self._last_action = "RESET"
        self._started = True

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start an episode.

        ``options`` may pin the graph with ``graph_index`` or ``graph_id`` (a ``sample_id``),
        and may override the step budget with ``max_steps``.
        """
        super().reset(seed=seed)
        options = options or {}
        if "graph_index" in options:
            index = int(options["graph_index"])
            if not 0 <= index < len(self.graphs):
                raise ValueError(f"graph_index {index} out of range")
        elif "graph_id" in options:
            try:
                index = self._index_by_id[options["graph_id"]]
            except KeyError:
                raise ValueError(f"unknown graph_id {options['graph_id']!r}") from None
        elif self.sampler is not None:
            index = int(self.sampler(self.np_random))
        else:
            index = int(self.np_random.integers(len(self.graphs)))
        self._begin_episode(index, options.get("max_steps"))
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._started:
            raise RuntimeError("call reset() before step()")
        if self._done:
            raise RuntimeError("episode is over; call reset()")
        action = int(action)
        graph, prev = self.graph, self.node
        terminated = truncated = False

        if not self.action_space.contains(action) or not self.action_masks()[action]:
            if self.config.strict_masks:
                raise InvalidActionError(
                    f"{self.action_name(action)} is not valid at node {prev} of {graph.sample_id}"
                )
            event = StepEvent(graph, "invalid", prev, prev)
        elif action < self.config.k:
            target = graph.successors(prev)[action]
            self.stack.append(prev)
            event = StepEvent(graph, "move", prev, target, revisit=bool(self.visits[target]))
            self._arrive(target)
        elif action == self.backtrack_action:
            target = self.stack.pop()
            event = StepEvent(graph, "backtrack", prev, target)
            self._arrive(target)
        else:
            correct = prev in graph.vuln_set
            self.declared.append(prev)
            self.declares_left -= 1
            event = StepEvent(graph, "declare", prev, prev, correct=correct)
            if correct:
                self.is_success = True
                terminated, self.end_reason = True, "correct"
            elif self.declares_left == 0:
                terminated, self.end_reason = True, "wrong_declare"

        self.t += 1
        self._last_action = self.action_name(action)
        if self.first_hit_step is None and self.node in graph.vuln_set:
            self.first_hit_step = self.t

        if not terminated:
            if self.t >= self.max_steps:
                truncated, self.end_reason = True, "timeout"
                event.timeout = True
            elif not self.action_masks().any():
                # Only reachable with backtracking disabled: nowhere left to go.
                terminated, self.end_reason = True, "dead_end"
                event.dead_end = True
        event.terminal = terminated or truncated
        self._done = event.terminal

        unshaped, reward = self.reward_model(event)
        self._last_reward = unshaped
        self.episode_return += unshaped
        self.shaped_return += reward
        return self._observation(), reward, terminated, truncated, self._info()

    def _arrive(self, node: int) -> None:
        self.node = node
        self.visits[node] += 1

    # ------------------------------------------------------------- observation

    def _features(self) -> np.ndarray:
        feats = self._feature_cache.get(self._graph_index)
        if feats is None:
            feats = featurize(self.graph, self.config.feature_tier)
            self._feature_cache[self._graph_index] = feats
        return feats

    def _observation(self) -> np.ndarray:
        cfg, graph, feats = self.config, self.graph, self._features()
        f = self._feature_dim
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # 1. Current node features.
        obs[:f] = feats[self.node]
        # 2. Current node episode features.
        obs[f] = min(int(self.visits[self.node]), VISIT_CLIP) / VISIT_CLIP
        obs[f + 1] = float(self.node in self.declared)
        pos = f + 2

        # 3. Successor slots: [valid, features, visited_before, is_back_edge].
        for slot, succ in enumerate(graph.successors(self.node)):
            base = pos + slot * self._slot_dim
            obs[base] = 1.0
            obs[base + 1 : base + 1 + f] = feats[succ]
            obs[base + 1 + f] = float(self.visits[succ] > 0)
            obs[base + 2 + f] = float(graph.is_back_edge(self.node, succ))
        pos += cfg.k * self._slot_dim

        # 4. Backtrack target summary: [valid, visit count, kind one-hot].
        if cfg.allow_backtrack and self.stack:
            target = self.stack[-1]
            obs[pos] = 1.0
            obs[pos + 1] = min(int(self.visits[target]), VISIT_CLIP) / VISIT_CLIP
            obs[pos + 2 + _KIND_INDEX[graph.nodes[target].kind]] = 1.0
        pos += self._backtrack_dim

        # 5. Episode context.
        n = graph.num_nodes
        obs[pos] = min(self.t / self.max_steps, 1.0)
        obs[pos + 1] = self.visited_unique / n
        obs[pos + 2] = min(len(self.stack), STACK_CLIP) / STACK_CLIP
        obs[pos + 3] = min(math.log(n) / math.log(SIZE_REF), 1.0)
        obs[pos + 4] = self.declares_left / cfg.max_declares
        return obs

    @property
    def visited_unique(self) -> int:
        return int(np.count_nonzero(self.visits))

    def _info(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph.sample_id,
            "current_node": self.node,
            "action_name": self._last_action,
            "steps": self.t,
            "visited_unique": self.visited_unique,
            "is_success": self.is_success,
            "declared_node": self.declared[-1] if self.declared else None,
            "first_declared_node": self.declared[0] if self.declared else None,
            "first_hit_step": self.first_hit_step,
            "end_reason": self.end_reason,
            "reward_unshaped": self._last_reward,
            "episode_return": self.episode_return,
            "shaped_return": self.shaped_return,
        }
