"""Node featurizers.

Each tier maps a :class:`GraphRecord` to a ``[num_nodes, F]`` float32 matrix with every value in
``[0, 1]``. Features never use node ids or absolute line numbers, and nothing here may read the
ground-truth labels.

Tier S (structural) is the state described in the project description: node kind, in/out
degree and position in the control flow.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from soren.data.schema import NODE_KINDS, GraphRecord

DEGREE_CLIP = 4

TIER_S_NAMES: tuple[str, ...] = (
    *(f"kind_{kind.lower()}" for kind in NODE_KINDS),
    "in_degree",
    "out_degree",
    "depth",
    "in_loop",
    "loop_header",
    "rel_line",
)

_KIND_INDEX = {kind: i for i, kind in enumerate(NODE_KINDS)}


def _tier_s(graph: GraphRecord) -> np.ndarray:
    n = graph.num_nodes
    out = np.zeros((n, len(TIER_S_NAMES)), dtype=np.float32)
    col = {name: i for i, name in enumerate(TIER_S_NAMES)}

    max_depth = max(1, max(node.depth for node in graph.nodes))
    num_lines = max(1, len(graph.source_lines), max(node.line for node in graph.nodes))
    loop_headers = {target for _, target in graph.back_edges}

    for node in graph.nodes:
        row = out[node.id]
        row[_KIND_INDEX[node.kind]] = 1.0
        row[col["in_degree"]] = min(len(graph.predecessors(node.id)), DEGREE_CLIP) / DEGREE_CLIP
        row[col["out_degree"]] = min(len(graph.successors(node.id)), DEGREE_CLIP) / DEGREE_CLIP
        # Unreachable nodes carry depth -1; treat them as depth 0.
        row[col["depth"]] = max(node.depth, 0) / max_depth
        row[col["in_loop"]] = float(node.in_loop)
        row[col["loop_header"]] = float(node.id in loop_headers)
        row[col["rel_line"]] = min(max(node.line, 0) / num_lines, 1.0)
    return out


_TIERS: dict[str, tuple[tuple[str, ...], Callable[[GraphRecord], np.ndarray]]] = {
    "S": (TIER_S_NAMES, _tier_s),
}


def _lookup(tier: str) -> tuple[tuple[str, ...], Callable[[GraphRecord], np.ndarray]]:
    try:
        return _TIERS[tier]
    except KeyError:
        raise ValueError(f"unknown feature tier {tier!r}; available: {sorted(_TIERS)}") from None


def feature_names(tier: str) -> tuple[str, ...]:
    return _lookup(tier)[0]


def feature_dim(tier: str) -> int:
    return len(feature_names(tier))


def featurize(graph: GraphRecord, tier: str = "S") -> np.ndarray:
    """Return the ``[num_nodes, feature_dim(tier)]`` feature matrix for ``graph``."""
    return _lookup(tier)[1](graph)
