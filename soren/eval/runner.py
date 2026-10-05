"""Run any searcher over a set of graphs and collect one row per episode.

Every number in the report is derived from the table this module produces, so a result can
always be traced back to a single file.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from soren.baselines.base import Searcher
from soren.data.schema import GraphRecord
from soren.env.cfg_nav_env import EnvConfig


class TestSplitError(RuntimeError):
    """Raised when the test split is evaluated without explicit permission."""

    __test__ = False  # not a pytest test class, despite the name


def shortest_path_to_vulnerable(graph: GraphRecord) -> int:
    """Forward hops from ENTRY to the nearest vulnerable node; -1 if none is reachable."""
    depth = {graph.entry: 0}
    queue = deque([graph.entry])
    while queue:
        u = queue.popleft()
        if u in graph.vuln_set:
            return depth[u]
        for v in graph.successors(u):
            if v not in depth:
                depth[v] = depth[u] + 1
                queue.append(v)
    return -1


def graph_metadata(graph: GraphRecord) -> dict[str, object]:
    return {
        "graph_id": graph.sample_id,
        "project": graph.project,
        "cwe": graph.cwe,
        "num_nodes": graph.num_nodes,
        "num_vuln": len(graph.vuln_nodes),
        "dist_to_vuln": shortest_path_to_vulnerable(graph),
    }


def run_evaluation(
    searchers: Sequence[Searcher],
    graphs: Sequence[GraphRecord],
    env_config: EnvConfig | None = None,
    seeds: Sequence[int] = (0,),
    split: str = "val",
    allow_test: bool = False,
    seed_overrides: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Evaluate each searcher on each graph; return one row per ``(method, seed, graph)``.

    Deterministic searchers run once; stochastic ones run once per entry of ``seeds``.
    ``seed_overrides`` maps a position in ``searchers`` to the seed to record for it, which is
    how several checkpoints of one method are told apart.

    The test split is refused unless ``allow_test`` is set: it is meant to be evaluated once,
    after every design choice is frozen.
    """
    if split == "test" and not allow_test:
        raise TestSplitError(
            "refusing to evaluate the test split; pass allow_test=True (--allow-test) "
            "once the configuration is frozen"
        )
    env_config = env_config or EnvConfig()
    seed_overrides = seed_overrides or {}
    metadata = [graph_metadata(graph) for graph in graphs]
    budgets = [env_config.max_steps_for(graph.num_nodes) for graph in graphs]

    rows: list[dict[str, object]] = []
    for position, searcher in enumerate(searchers):
        if position in seed_overrides:
            run_seeds: Sequence[int] = [seed_overrides[position]]
        else:
            run_seeds = seeds if searcher.stochastic else seeds[:1]
        for seed in run_seeds:
            rng = np.random.default_rng(seed)
            for graph, meta, budget in zip(graphs, metadata, budgets, strict=True):
                result = searcher.run(graph, budget, rng)
                first = result.first_declared_node
                rows.append(
                    {
                        "method": result.method,
                        "seed": seed,
                        "split": split,
                        **meta,
                        "max_steps": budget,
                        "success": result.success,
                        "first_declare_correct": first is not None and first in graph.vuln_set,
                        "nodes_inspected": result.nodes_inspected,
                        "actions_taken": result.actions_taken,
                        "cumulative_reward": result.cumulative_reward,
                        "first_hit_step": result.first_hit_step,
                        "end_reason": result.end_reason,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["first_hit_step"] = frame["first_hit_step"].astype("Int64")
    return frame


def with_max_declares(env_config: EnvConfig, max_declares: int) -> EnvConfig:
    """Copy of ``env_config`` with a different declare budget, for Top-k evaluation."""
    return replace(env_config, max_declares=max_declares)


def save_results(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def load_results(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
