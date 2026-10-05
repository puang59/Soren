"""A hand-written suspiciousness score, and Protocol B built on top of it.

Under Protocol A the traversal baselines are stopped by an oracle the moment they reach a
vulnerable node. Protocol B makes them decide for themselves: each walks in its own order and
declares at the first node whose score reaches a threshold. With the same per-node information
available to every method, what is left to compare is the navigation.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
import pandas as pd

from soren.baselines.base import EnvSearcher, OrderSearcher, Searcher
from soren.baselines.bfs import BFS
from soren.baselines.dfs import DFS
from soren.baselines.random_walk import RandomWalk
from soren.baselines.reference import LineOrder, RandomOrder
from soren.data.features import LEXICAL_NAMES, TIER_L_NAMES, Lexicon, featurize
from soren.data.schema import GraphRecord
from soren.env.cfg_nav_env import EnvConfig
from soren.env.rewards import RewardConfig

DEFAULT_WEIGHTS: dict[str, float] = {
    "dangerous_call": 3.0,
    "array_subscript": 2.0,
    "pointer_arith": 2.0,
    "pointer_deref": 1.0,
    "alloc_call": 1.0,
    "length_like_ident": 1.0,
    "arith_op": 0.5,
}
"""Weights of the Tier L flags, chosen by hand for memory-safety CWEs and not fitted."""


class Scorer(Protocol):
    name: str

    def scores(self, graph: GraphRecord) -> np.ndarray:
        """A suspiciousness score in ``[0, 1]`` for every node of ``graph``."""
        ...


class HeuristicScorer:
    """Weighted sum of Tier L flags, scaled so that a node with every flag set scores 1."""

    name = "heuristic"

    def __init__(
        self, weights: dict[str, float] | None = None, lexicon: Lexicon | None = None
    ) -> None:
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        unknown = sorted(set(self.weights) - set(LEXICAL_NAMES))
        if unknown:
            raise ValueError(f"unknown lexical features: {', '.join(unknown)}")
        if not self.weights or min(self.weights.values()) < 0 or sum(self.weights.values()) <= 0:
            raise ValueError("weights must be non-negative with a positive sum")
        self.lexicon = lexicon
        self._columns = [TIER_L_NAMES.index(name) for name in self.weights]
        self._weights = np.array(list(self.weights.values()), dtype=np.float64)
        self._cache: dict[str, np.ndarray] = {}

    def scores(self, graph: GraphRecord) -> np.ndarray:
        cached = self._cache.get(graph.sample_id)
        if cached is None:
            features = featurize(graph, "L", self.lexicon)[:, self._columns]
            cached = (features @ self._weights) / self._weights.sum()
            self._cache[graph.sample_id] = cached
        return cached


class ThresholdRule:
    """Declare at a node whose score is at least ``threshold``."""

    def __init__(self, scorer: Scorer, threshold: float) -> None:
        self.scorer = scorer
        self.threshold = threshold

    def __call__(self, graph: GraphRecord, node: int) -> bool:
        return bool(self.scorer.scores(graph)[node] >= self.threshold)


class HeuristicFirst(OrderSearcher):
    """Best-first search: always inspect the highest-scoring node on the frontier.

    The frontier holds the successors of every node inspected so far; ties go to the earlier
    source line. A strong non-learned baseline, since it uses the same lexical flags the agent
    sees.
    """

    name = "heuristic_first"

    def __init__(
        self,
        reward_config: RewardConfig | None = None,
        declare_rule: Callable[[GraphRecord, int], bool] | None = None,
        name: str | None = None,
        scorer: Scorer | None = None,
    ) -> None:
        super().__init__(reward_config, declare_rule, name)
        self.scorer = scorer or HeuristicScorer()

    def order(self, graph: GraphRecord, rng: np.random.Generator) -> list[int]:
        scores = self.scorer.scores(graph)
        order: list[int] = []
        seen = {graph.entry}
        frontier = [(0.0, graph.nodes[graph.entry].line, graph.entry)]
        while frontier:
            _, _, node = heapq.heappop(frontier)
            order.append(node)
            for succ in graph.successors(node):
                if succ not in seen:
                    seen.add(succ)
                    heapq.heappush(frontier, (-float(scores[succ]), graph.nodes[succ].line, succ))
        return order


BASELINE_FACTORIES: dict[str, Callable[..., Searcher]] = {
    "dfs": lambda env, reward, **kw: DFS(env, reward, **kw),
    "random_walk": lambda env, reward, **kw: RandomWalk(env, reward, **kw),
    "bfs": lambda env, reward, **kw: BFS(reward, **kw),
    "random_order": lambda env, reward, **kw: RandomOrder(reward, **kw),
    "line_order": lambda env, reward, **kw: LineOrder(reward, **kw),
    "heuristic_first": lambda env, reward, **kw: HeuristicFirst(reward, **kw),
}


def make_baseline(
    method: str,
    env_config: EnvConfig | None = None,
    reward_config: RewardConfig | None = None,
    scorer: Scorer | None = None,
    threshold: float | None = None,
) -> Searcher:
    """Build a baseline under Protocol A, or under Protocol B when a scorer is given.

    A Protocol B searcher is named ``<method>+<scorer>`` so both protocols can share a table.
    """
    if method not in BASELINE_FACTORIES:
        raise ValueError(f"unknown baseline {method!r}; available: {sorted(BASELINE_FACTORIES)}")
    factory = BASELINE_FACTORIES[method]
    if scorer is None:
        return factory(env_config, reward_config)
    if threshold is None:
        raise ValueError("Protocol B needs a threshold")
    return factory(
        env_config,
        reward_config,
        declare_rule=ThresholdRule(scorer, threshold),
        name=f"{method}+{scorer.name}",
    )


def threshold_curve(
    method: str,
    scorer: Scorer,
    graphs: Sequence[GraphRecord],
    thresholds: Sequence[float],
    env_config: EnvConfig | None = None,
    reward_config: RewardConfig | None = None,
    seeds: Sequence[int] = (0,),
) -> pd.DataFrame:
    """Success rate, mean nodes inspected and mean return of ``method`` at each threshold."""
    env_config = env_config or EnvConfig()
    rows = []
    for threshold in thresholds:
        searcher = make_baseline(method, env_config, reward_config, scorer, threshold)
        results = []
        for seed in seeds if searcher.stochastic else seeds[:1]:
            rng = np.random.default_rng(seed)
            results += [
                searcher.run(graph, env_config.max_steps_for(graph.num_nodes), rng)
                for graph in graphs
            ]
        rows.append(
            {
                "method": method,
                "scorer": scorer.name,
                "threshold": float(threshold),
                "success_rate": float(np.mean([r.success for r in results])),
                "declared_rate": float(np.mean([r.declared_node is not None for r in results])),
                "nodes_inspected": float(np.mean([r.nodes_inspected for r in results])),
                "cumulative_reward": float(np.mean([r.cumulative_reward for r in results])),
            }
        )
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame) -> float:
    """The threshold with the highest success rate; ties go to fewer nodes inspected."""
    ranked = curve.sort_values(
        ["success_rate", "nodes_inspected", "threshold"], ascending=[False, True, True]
    )
    return float(ranked.iloc[0]["threshold"])


__all__ = [
    "BASELINE_FACTORIES",
    "DEFAULT_WEIGHTS",
    "EnvSearcher",
    "HeuristicFirst",
    "HeuristicScorer",
    "Scorer",
    "ThresholdRule",
    "best_threshold",
    "make_baseline",
    "threshold_curve",
]
