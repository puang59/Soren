"""Structural analyses shared by the CFG builder and the synthetic generator."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import networkx as nx

Edge = tuple[int, int]


def _adjacency(num_nodes: int, edges: Sequence[Edge]) -> list[list[int]]:
    out: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in edges:
        out[u].append(v)
    for vs in out:
        vs.sort()
    return out


def bfs_depths(num_nodes: int, edges: Sequence[Edge], entry: int) -> list[int]:
    """Shortest-path distance from ``entry`` to every node; -1 for unreachable nodes."""
    adjacency = _adjacency(num_nodes, edges)
    depth = [-1] * num_nodes
    depth[entry] = 0
    queue = deque([entry])
    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if depth[v] < 0:
                depth[v] = depth[u] + 1
                queue.append(v)
    return depth


def find_back_edges(num_nodes: int, edges: Sequence[Edge], entry: int) -> list[Edge]:
    """Edges whose target is an ancestor of the source in a DFS from ``entry``."""
    adjacency = _adjacency(num_nodes, edges)
    white, grey, black = 0, 1, 2
    colour = [white] * num_nodes
    back: list[Edge] = []
    colour[entry] = grey
    stack: list[tuple[int, int]] = [(entry, 0)]
    while stack:
        u, i = stack[-1]
        if i < len(adjacency[u]):
            stack[-1] = (u, i + 1)
            v = adjacency[u][i]
            if colour[v] == white:
                colour[v] = grey
                stack.append((v, 0))
            elif colour[v] == grey:
                back.append((u, v))
        else:
            colour[u] = black
            stack.pop()
    return sorted(back)


def nodes_in_cycles(num_nodes: int, edges: Sequence[Edge]) -> set[int]:
    """Nodes that belong to a non-trivial strongly connected component."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edges)
    in_cycle: set[int] = set()
    for component in nx.strongly_connected_components(graph):
        if len(component) > 1:
            in_cycle |= component
    return in_cycle
