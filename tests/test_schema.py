import dataclasses

import pytest
from helpers import diamond_graph, line_graph, loop_graph, make_graph

from soren.data.graph_utils import bfs_depths, find_back_edges, nodes_in_cycles
from soren.data.schema import (
    GraphRecord,
    SchemaError,
    read_jsonl,
    resolve_kind,
    write_jsonl,
)


def test_json_round_trip():
    record = loop_graph()
    record.features["S"] = [[0.0, 1.0]] * record.num_nodes
    assert GraphRecord.from_json(record.to_json()) == record


def test_jsonl_round_trip(tmp_path):
    records = [line_graph(), diamond_graph(), loop_graph()]
    path = tmp_path / "nested" / "graphs.jsonl"
    assert write_jsonl(records, path) == 3
    assert read_jsonl(path) == records


def test_from_dict_reports_missing_keys():
    with pytest.raises(SchemaError, match="malformed graph record"):
        GraphRecord.from_dict({"sample_id": "x"})


def test_successors_are_ordered_by_line_not_edge_order():
    kinds = ["ENTRY", "BRANCH", "ASSIGN", "ASSIGN", "EXIT"]
    edges = [(0, 1), (1, 3), (1, 2), (2, 4), (3, 4)]
    record = make_graph(kinds, edges, [2])
    assert record.successors(1) == (2, 3)
    # Swapping source lines swaps the canonical order, whatever the node ids are.
    swapped = make_graph(kinds, edges, [2], lines=[1, 2, 4, 3, 5])
    assert swapped.successors(1) == (3, 2)


def test_successor_order_is_independent_of_labels():
    assert diamond_graph(vuln=2).successors(1) == diamond_graph(vuln=3).successors(1)


def test_predecessors_and_back_edges():
    record = loop_graph()
    assert record.predecessors(1) == (0, 3)
    assert record.back_edges == [(3, 1)]
    assert record.is_back_edge(3, 1)
    assert not record.is_back_edge(1, 2)


def test_to_networkx():
    record = diamond_graph(vuln=3)
    graph = record.to_networkx()
    assert graph.number_of_nodes() == 6
    assert set(graph.edges) == set(record.edges)
    assert graph.nodes[3]["vulnerable"] and not graph.nodes[2]["vulnerable"]
    assert graph.nodes[1]["kind"] == "BRANCH"


def _mutated(record: GraphRecord, **changes) -> GraphRecord:
    return dataclasses.replace(record, **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"edges": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 9)]}, "references a missing node"),
        ({"edges": [(0, 1), (1, 2), (2, 3), (3, 4), (2, 2)]}, "self-loop"),
        ({"edges": [(0, 1), (1, 2), (2, 3), (3, 4), (0, 1)]}, "duplicate edge"),
        ({"back_edges": [(4, 0)]}, "is not an edge"),
        ({"vuln_nodes": []}, "vuln_nodes is empty"),
        ({"vuln_nodes": [7]}, "is not a node"),
        ({"vuln_nodes": [0]}, "ENTRY/EXIT cannot be vulnerable"),
        ({"vuln_nodes": [2, 2]}, "duplicates"),
        ({"entry": 9}, "entry 9 is not a node"),
        ({"entry": 1}, "single ENTRY"),
        ({"features": {"S": [[0.0]]}}, "rows for 5 nodes"),
    ],
)
def test_validate_rejects(changes, message):
    with pytest.raises(SchemaError, match=message):
        _mutated(line_graph(), **changes).validate()


def test_validate_rejects_unreachable_vulnerable_node():
    kinds = ["ENTRY", "ASSIGN", "ASSIGN", "EXIT"]
    record = make_graph(kinds, [(0, 1), (1, 3), (2, 3)], [1])
    with pytest.raises(SchemaError, match="reachable from entry"):
        _mutated(record, vuln_nodes=[2]).validate()


def test_validate_rejects_bad_node_ids_and_kinds():
    record = line_graph()
    bad_id = [dataclasses.replace(n, id=n.id + 1) for n in record.nodes]
    with pytest.raises(SchemaError, match="has id"):
        _mutated(record, nodes=bad_id).validate()
    bad_kind = [dataclasses.replace(n, kind="LOOPY") if n.id == 1 else n for n in record.nodes]
    with pytest.raises(SchemaError, match="unknown kind"):
        _mutated(record, nodes=bad_kind).validate()


def test_resolve_kind_priority():
    assert resolve_kind(["ASSIGN", "CALL"]) == "ASSIGN"
    assert resolve_kind(["CALL", "BRANCH", "ASSIGN"]) == "BRANCH"
    assert resolve_kind(["RETURN", "CALL"]) == "RETURN"
    assert resolve_kind(["JUMP", "ASSIGN"]) == "JUMP"
    assert resolve_kind(["DECL", "CALL"]) == "CALL"
    assert resolve_kind(["DECL"]) == "DECL"
    assert resolve_kind([]) == "OTHER"
    assert resolve_kind(["ENTRY", "CALL"]) == "ENTRY"
    with pytest.raises(SchemaError):
        resolve_kind(["WHATEVER"])


def test_graph_utils():
    edges = [(0, 1), (1, 2), (2, 3), (3, 1), (1, 4), (4, 5)]
    assert bfs_depths(7, edges, 0) == [0, 1, 2, 3, 2, 3, -1]
    assert find_back_edges(6, edges, 0) == [(3, 1)]
    assert nodes_in_cycles(6, edges) == {1, 2, 3}
