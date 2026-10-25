"""Structural analyses shared by the CFG builder and the synthetic generator.

All three are classic graph searches, written out here:

* :func:`bfs_depths`: breadth-first search gives the fewest edges from ENTRY to each node;
* :func:`find_back_edges`: depth-first search finds the edges that close a loop;
* :func:`nodes_in_cycles`: Kosaraju's algorithm finds the nodes that lie on some cycle.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

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


def strongly_connected_components(num_nodes: int, edges: Sequence[Edge]) -> list[set[int]]:
    """Groups of nodes that can all reach each other, by Kosaraju's algorithm.

    1. Run DFS over the whole graph and list nodes in the order they *finish* (all their
       descendants done).
    2. Reverse every edge. Take nodes in reverse finishing order; each DFS from a node not yet
       assigned collects exactly one component.

    The first pass puts a component that others can reach before them; on the reversed graph
    a search from it cannot leak into a different component. Both passes are iterative, so
    deep graphs do not hit Python's recursion limit.
    """
    forward = _adjacency(num_nodes, edges)
    finished: list[int] = []
    seen = [False] * num_nodes
    for root in range(num_nodes):
        if seen[root]:
            continue
        seen[root] = True
        stack = [(root, 0)]
        while stack:
            u, i = stack[-1]
            if i < len(forward[u]):
                stack[-1] = (u, i + 1)
                v = forward[u][i]
                if not seen[v]:
                    seen[v] = True
                    stack.append((v, 0))
            else:
                finished.append(u)
                stack.pop()

    backward = _adjacency(num_nodes, [(v, u) for u, v in edges])
    component_of = [-1] * num_nodes
    components: list[set[int]] = []
    for root in reversed(finished):
        if component_of[root] >= 0:
            continue
        component_of[root] = len(components)
        members, todo = {root}, [root]
        while todo:
            u = todo.pop()
            for v in backward[u]:
                if component_of[v] < 0:
                    component_of[v] = len(components)
                    members.add(v)
                    todo.append(v)
        components.append(members)
    return components


def nodes_in_cycles(num_nodes: int, edges: Sequence[Edge]) -> set[int]:
    """Nodes that belong to a strongly connected component of more than one node.

    A node lies on a cycle exactly when it can reach another node that can reach it back. The
    graphs have no self-loops, so single-node components are never cycles.
    """
    in_cycle: set[int] = set()
    for component in strongly_connected_components(num_nodes, edges):
        if len(component) > 1:
            in_cycle |= component
    return in_cycle
