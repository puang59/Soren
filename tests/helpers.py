"""Small hand-built graphs shared across tests."""

from __future__ import annotations

from soren.data.graph_utils import bfs_depths, find_back_edges, nodes_in_cycles
from soren.data.schema import GraphRecord, Node


def make_graph(
    kinds: list[str],
    edges: list[tuple[int, int]],
    vuln: list[int],
    sample_id: str = "g",
    lines: list[int] | None = None,
) -> GraphRecord:
    """Build a record from node kinds and edges; node ``i`` sits on line ``i + 1`` by default."""
    n = len(kinds)
    entry = kinds.index("ENTRY")
    exit_ = kinds.index("EXIT")
    depths = bfs_depths(n, edges, entry)
    loops = nodes_in_cycles(n, edges)
    lines = lines or [i + 1 for i in range(n)]
    nodes = [
        Node(id=i, line=lines[i], kind=kinds[i], code=f"s{i};", depth=depths[i], in_loop=i in loops)
        for i in range(n)
    ]
    return GraphRecord(
        sample_id=sample_id,
        nodes=nodes,
        edges=list(edges),
        back_edges=find_back_edges(n, edges, entry),
        entry=entry,
        exit=exit_,
        vuln_nodes=list(vuln),
        source_lines=[f"s{i};" for i in range(max(lines))],
    ).validate()


def line_graph(n_inner: int = 3, vuln: int | None = None) -> GraphRecord:
    """ENTRY -> s1 -> ... -> s_n -> EXIT."""
    kinds = ["ENTRY"] + ["ASSIGN"] * n_inner + ["EXIT"]
    edges = [(i, i + 1) for i in range(n_inner + 1)]
    return make_graph(kinds, edges, [vuln if vuln is not None else n_inner], sample_id="line")


def diamond_graph(vuln: int = 3) -> GraphRecord:
    """0 ENTRY -> 1 BRANCH -> {2, 3} -> 4 ASSIGN -> 5 EXIT."""
    kinds = ["ENTRY", "BRANCH", "ASSIGN", "CALL", "ASSIGN", "EXIT"]
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]
    return make_graph(kinds, edges, [vuln], sample_id="diamond")


def loop_graph(vuln: int = 3) -> GraphRecord:
    """0 ENTRY -> 1 LOOP -> 2 -> 3 -> 1 (back edge); 1 -> 4 RETURN -> 5 EXIT."""
    kinds = ["ENTRY", "LOOP", "ASSIGN", "CALL", "RETURN", "EXIT"]
    edges = [(0, 1), (1, 2), (2, 3), (3, 1), (1, 4), (4, 5)]
    return make_graph(kinds, edges, [vuln], sample_id="loop")
