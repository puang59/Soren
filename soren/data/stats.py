"""Statistics over a set of graphs, used to size the environment."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from soren.data.schema import GraphRecord
from soren.eval.runner import shortest_path_to_vulnerable

PERCENTILES = (5, 25, 50, 75, 95, 99, 100)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    summary = {"mean": float(array.mean())}
    for q, value in zip(PERCENTILES, np.percentile(array, PERCENTILES), strict=True):
        summary[f"p{q}"] = float(value)
    return summary


def out_degree_table(records: Sequence[GraphRecord]) -> list[dict[str, float]]:
    """For each out-degree ``d``: the share of nodes and of graphs that fit within ``d``."""
    per_node = Counter(len(g.successors(n.id)) for g in records for n in g.nodes)
    per_graph = [max(len(g.successors(n.id)) for n in g.nodes) for g in records]
    total_nodes = sum(per_node.values())
    rows = []
    covered = 0
    for degree in range(max(per_node) + 1):
        covered += per_node.get(degree, 0)
        rows.append(
            {
                "out_degree": degree,
                "nodes": per_node.get(degree, 0),
                "nodes_covered": covered / total_nodes,
                "graphs_covered": float(np.mean([widest <= degree for widest in per_graph])),
            }
        )
    return rows


def smallest_k(records: Sequence[GraphRecord], node_coverage: float = 0.99) -> int:
    """The smallest number of successor slots that fits ``node_coverage`` of all nodes."""
    for row in out_degree_table(records):
        if row["nodes_covered"] >= node_coverage:
            return max(int(row["out_degree"]), 1)
    raise AssertionError("unreachable: the last row covers every node")


def graph_statistics(records: Sequence[GraphRecord]) -> dict[str, Any]:
    """Size, branching and label statistics for ``records``."""
    if not records:
        raise ValueError("no graphs to describe")
    num_nodes = [g.num_nodes for g in records]
    num_vuln = [len(g.vuln_nodes) for g in records]
    distances = [shortest_path_to_vulnerable(g) for g in records]
    reachable = [d for d in distances if d >= 0]
    return {
        "graphs": len(records),
        "nodes": _distribution(num_nodes),
        "vulnerable_nodes": _distribution(num_vuln),
        "single_vulnerable_node_share": float(np.mean([v == 1 for v in num_vuln])),
        "vulnerable_node_fraction": _distribution(
            [v / n for v, n in zip(num_vuln, num_nodes, strict=True)]
        ),
        "distance_entry_to_vulnerable": _distribution(reachable),
        "out_degree": out_degree_table(records),
        "smallest_k_for_99_percent_of_nodes": smallest_k(records),
        "kinds": dict(sorted(Counter(n.kind for g in records for n in g.nodes).items())),
        "by_cwe": dict(sorted(Counter(g.cwe for g in records).items())),
    }
