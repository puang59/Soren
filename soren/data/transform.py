"""Graph rewrites applied when records are built."""

from __future__ import annotations

from dataclasses import replace

from soren.data.graph_utils import bfs_depths, find_back_edges, nodes_in_cycles
from soren.data.schema import GraphRecord, Node


def max_out_degree(record: GraphRecord) -> int:
    return max(len(record.successors(node.id)) for node in record.nodes)


def limit_out_degree(record: GraphRecord, k: int) -> GraphRecord:
    """Rewrite nodes with more than ``k`` successors as a chain of dispatch nodes.

    The environment offers a fixed number of move actions, and a ``switch`` with many cases
    has more successors than that. Such a node keeps its first ``k - 1`` successors and gains
    one extra successor, a dispatch node on the same line, which takes over the rest (and is
    split again if it is still too wide). This is how a switch reads as an if/else chain: no
    case becomes unreachable, later cases just take more moves to reach.

    Dispatch nodes copy the kind, line and code of the node they continue, are never
    vulnerable, and are appended after the existing nodes so that no id changes. Records that
    are already narrow enough are returned unchanged.
    """
    if k < 2:
        raise ValueError("k must be at least 2 to chain dispatch nodes")
    if max_out_degree(record) <= k:
        return record

    nodes = [replace(node) for node in record.nodes]
    edges: list[tuple[int, int]] = []
    for node in record.nodes:
        source = node.id
        remaining = list(record.successors(node.id))
        while len(remaining) > k:
            dispatch = Node(
                id=len(nodes),
                line=node.line,
                kind=node.kind,
                code=node.code,
                calls=list(node.calls),
                ops=list(node.ops),
            )
            nodes.append(dispatch)
            edges += [(source, target) for target in remaining[: k - 1]]
            edges.append((source, dispatch.id))
            source, remaining = dispatch.id, remaining[k - 1 :]
        edges += [(source, target) for target in remaining]

    edges = sorted(edges)
    depths = bfs_depths(len(nodes), edges, record.entry)
    in_loop = nodes_in_cycles(len(nodes), edges)
    for node in nodes:
        node.depth = depths[node.id]
        node.in_loop = node.id in in_loop
    return GraphRecord(
        sample_id=record.sample_id,
        nodes=nodes,
        edges=edges,
        back_edges=find_back_edges(len(nodes), edges, record.entry),
        entry=record.entry,
        exit=record.exit,
        vuln_nodes=list(record.vuln_nodes),
        source_lines=list(record.source_lines),
        project=record.project,
        commit_id=record.commit_id,
        cwe=record.cwe,
    ).validate()
