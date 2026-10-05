import numpy as np
import pytest

from soren.data.features import featurize
from soren.data.schema import read_jsonl
from soren.data.synthetic import (
    DANGEROUS_CALLS,
    SyntheticConfig,
    generate_dataset,
    generate_graph,
    main,
)


def _is_dangerous(node) -> bool:
    return any(call in DANGEROUS_CALLS for call in node.calls)


def test_graphs_are_valid_and_structured():
    cfg = SyntheticConfig(min_nodes=8, max_nodes=60)
    for graph in generate_dataset(300, seed=1, cfg=cfg):
        graph.validate()  # also asserts a reachable vulnerable node
        assert graph.reachable_from_entry() == set(range(graph.num_nodes))
        assert len(graph.vuln_nodes) == 1
        assert graph.successors(graph.exit) == ()
        assert max(len(graph.successors(n.id)) for n in graph.nodes) <= 4
        # Every node sits on its own source line and the line holds its code.
        assert len({n.line for n in graph.nodes}) == graph.num_nodes
        for node in graph.nodes:
            if node.code:
                assert graph.source_lines[node.line - 1].strip() == node.code


def test_node_counts_track_the_requested_range():
    small = generate_dataset(200, seed=0, cfg=SyntheticConfig(min_nodes=8, max_nodes=12))
    large = generate_dataset(200, seed=0, cfg=SyntheticConfig(min_nodes=60, max_nodes=80))
    small_sizes = [g.num_nodes for g in small]
    large_sizes = [g.num_nodes for g in large]
    assert min(small_sizes) >= 6 and max(small_sizes) <= 20
    assert 50 <= np.mean(large_sizes) <= 95


def test_contains_branches_and_loops():
    graphs = generate_dataset(100, seed=3)
    kinds = {node.kind for graph in graphs for node in graph.nodes}
    assert {"ENTRY", "EXIT", "ASSIGN", "CALL", "BRANCH", "LOOP", "RETURN"} <= kinds
    assert any(graph.back_edges for graph in graphs)


def test_generation_is_deterministic():
    a = generate_dataset(20, seed=7)
    b = generate_dataset(20, seed=7)
    c = generate_dataset(20, seed=8)
    assert a == b
    assert a != c


def _call_rates(signal: float, n: int = 1500) -> tuple[float, float]:
    cfg = SyntheticConfig(signal=signal)
    vuln_hits, other_hits, others = 0, 0, 0
    for graph in generate_dataset(n, seed=11, cfg=cfg):
        vuln = graph.vuln_nodes[0]
        vuln_hits += _is_dangerous(graph.nodes[vuln])
        for node in graph.nodes:
            if node.id != vuln and node.kind in ("ASSIGN", "CALL", "DECL"):
                others += 1
                other_hits += _is_dangerous(node)
    return vuln_hits / n, other_hits / others


def test_full_signal_separates_the_vulnerable_node():
    vuln_rate, other_rate = _call_rates(signal=1.0)
    assert vuln_rate == pytest.approx(0.9, abs=0.03)
    assert other_rate == pytest.approx(0.1, abs=0.02)


def test_zero_signal_is_indistinguishable():
    vuln_rate, other_rate = _call_rates(signal=0.0)
    assert vuln_rate == pytest.approx(other_rate, abs=0.03)


def test_structure_does_not_depend_on_signal():
    """The same seed gives the same CFG whatever the signal strength."""
    for strong, weak in zip(
        generate_dataset(30, seed=5, cfg=SyntheticConfig(signal=1.0)),
        generate_dataset(30, seed=5, cfg=SyntheticConfig(signal=1.0, hit_rate=0.5)),
        strict=True,
    ):
        assert strong.edges == weak.edges
        assert strong.vuln_nodes == weak.vuln_nodes


def test_tier_s_features_work_on_synthetic_graphs():
    for graph in generate_dataset(20, seed=2):
        feats = featurize(graph, "S")
        assert feats.shape[0] == graph.num_nodes
        assert feats.min() >= 0.0 and feats.max() <= 1.0


def test_config_validation():
    with pytest.raises(ValueError):
        SyntheticConfig(min_nodes=2)
    with pytest.raises(ValueError):
        SyntheticConfig(signal=1.5)


def test_cli_writes_jsonl(tmp_path, capsys):
    out = tmp_path / "syn.jsonl"
    main(["--out", str(out), "--n", "12", "--seed", "4", "--max-nodes", "20"])
    records = read_jsonl(out)
    assert len(records) == 12
    assert records == generate_dataset(12, seed=4, cfg=SyntheticConfig(max_nodes=20))
    assert "wrote 12 graphs" in capsys.readouterr().out


def test_generate_graph_default_config():
    graph = generate_graph(np.random.default_rng(0))
    assert graph.sample_id == "syn"
