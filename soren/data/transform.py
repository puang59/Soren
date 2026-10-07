"""Graph rewrites applied when records are built."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

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
    ).validate(labelled=bool(record.vuln_nodes))


def shuffle_labels(record: GraphRecord, rng: np.random.Generator) -> GraphRecord:
    """Replace the vulnerable nodes with the same number of random reachable statements.

    A leakage control: a policy trained on shuffled labels has nothing real to learn, so if it
    still beats chance, something other than the features is giving the label away.
    """
    reachable = record.reachable_from_entry()
    candidates = sorted(
        n.id for n in record.nodes if n.kind not in ("ENTRY", "EXIT") and n.id in reachable
    )
    count = min(len(record.vuln_nodes), len(candidates))
    chosen = sorted(int(i) for i in rng.choice(candidates, size=count, replace=False))
    return replace(record, vuln_nodes=chosen, features=dict(record.features)).validate()


def permute_node_ids(record: GraphRecord, rng: np.random.Generator) -> GraphRecord:
    """Relabel the nodes with a random permutation of their ids, changing nothing else.

    A leakage control: lines, edges, labels and features all move with their node, so a policy
    that does not read node ids must behave identically on the result.
    """
    perm = [int(i) for i in rng.permutation(record.num_nodes)]
    nodes: list[Node | None] = [None] * record.num_nodes
    for node in record.nodes:
        nodes[perm[node.id]] = replace(node, id=perm[node.id])
    features = {}
    for key, rows in record.features.items():
        moved: list = [None] * record.num_nodes
        for old_id, row in enumerate(rows):
            moved[perm[old_id]] = row
        features[key] = moved
    return GraphRecord(
        sample_id=record.sample_id,
        nodes=nodes,  # type: ignore[arg-type]
        edges=sorted((perm[u], perm[v]) for u, v in record.edges),
        back_edges=sorted((perm[u], perm[v]) for u, v in record.back_edges),
        entry=perm[record.entry],
        exit=perm[record.exit],
        vuln_nodes=sorted(perm[v] for v in record.vuln_nodes),
        source_lines=list(record.source_lines),
        project=record.project,
        commit_id=record.commit_id,
        cwe=record.cwe,
        features=features,
    ).validate()
