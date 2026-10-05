import numpy as np
import pytest
from helpers import diamond_graph, line_graph, loop_graph, make_graph

from soren.data.features import TIER_S_NAMES, feature_dim, feature_names, featurize
from soren.data.schema import NODE_KINDS

COL = {name: i for i, name in enumerate(TIER_S_NAMES)}


def test_shape_dtype_and_range():
    for graph in (line_graph(), diamond_graph(), loop_graph()):
        feats = featurize(graph, "S")
        assert feats.shape == (graph.num_nodes, feature_dim("S"))
        assert feats.dtype == np.float32
        assert not np.isnan(feats).any()
        assert feats.min() >= 0.0 and feats.max() <= 1.0


def test_width_is_fixed_and_named():
    assert feature_dim("S") == len(NODE_KINDS) + 6
    assert feature_names("S") == TIER_S_NAMES


def test_kind_one_hot_covers_every_kind():
    inner = [k for k in NODE_KINDS if k not in ("ENTRY", "EXIT")]
    kinds = ["ENTRY", *inner, "EXIT"]
    edges = [(i, i + 1) for i in range(len(kinds) - 1)]
    graph = make_graph(kinds, edges, [1])
    feats = featurize(graph, "S")
    one_hot = feats[:, : len(NODE_KINDS)]
    assert (one_hot.sum(axis=1) == 1.0).all()
    for node in graph.nodes:
        assert feats[node.id, COL[f"kind_{node.kind.lower()}"]] == 1.0


def test_degrees_are_clipped_and_scaled():
    feats = featurize(diamond_graph(), "S")
    assert feats[1, COL["out_degree"]] == pytest.approx(2 / 4)
    assert feats[4, COL["in_degree"]] == pytest.approx(2 / 4)
    assert feats[0, COL["in_degree"]] == 0.0
    assert feats[5, COL["out_degree"]] == 0.0

    # A node with six successors saturates at 1.0.
    kinds = ["ENTRY", "SWITCH", *["ASSIGN"] * 6, "EXIT"]
    edges = [(0, 1)] + [(1, i) for i in range(2, 8)] + [(i, 8) for i in range(2, 8)]
    wide = featurize(make_graph(kinds, edges, [2]), "S")
    assert wide[1, COL["out_degree"]] == 1.0
    assert wide[8, COL["in_degree"]] == 1.0


def test_depth_loop_and_position_features():
    graph = loop_graph()
    feats = featurize(graph, "S")
    assert feats[0, COL["depth"]] == 0.0
    assert feats[:, COL["depth"]].max() == 1.0
    assert list(feats[:, COL["in_loop"]]) == [0, 1, 1, 1, 0, 0]
    assert list(feats[:, COL["loop_header"]]) == [0, 1, 0, 0, 0, 0]
    rel = feats[:, COL["rel_line"]]
    assert (np.diff(rel) > 0).all() and rel[-1] == 1.0


def test_features_ignore_labels_and_node_ids():
    a = featurize(diamond_graph(vuln=2), "S")
    b = featurize(diamond_graph(vuln=3), "S")
    np.testing.assert_array_equal(a, b)


def test_unknown_tier():
    with pytest.raises(ValueError, match="unknown feature tier"):
        featurize(line_graph(), "Z")
