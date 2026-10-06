import importlib.util
import json
from pathlib import Path

import pytest
from helpers import diamond_graph, line_graph, loop_graph, make_graph

from soren.data.schema import write_jsonl
from soren.data.stats import graph_statistics, out_degree_table, smallest_k
from soren.data.transform import limit_out_degree, max_out_degree
from soren.env.cfg_nav_env import CFGNavEnv, EnvConfig
from soren.env.policies import OraclePolicy, rollout

ROOT = Path(__file__).parent.parent


def switch_graph(cases: int, vuln_case: int):
    """ENTRY -> SWITCH -> case_1 .. case_n -> RETURN -> EXIT; node ids 2 .. n+1 are the cases."""
    kinds = ["ENTRY", "SWITCH", *["ASSIGN"] * cases, "RETURN", "EXIT"]
    join, exit_ = cases + 2, cases + 3
    edges = [(0, 1), *[(1, c) for c in range(2, cases + 2)]]
    edges += [(c, join) for c in range(2, cases + 2)] + [(join, exit_)]
    return make_graph(kinds, edges, [vuln_case + 1], sample_id=f"switch{cases}")


def reachable_cases(record, node=1, seen=None):
    """Case nodes reachable from the switch by following dispatch nodes only."""
    seen = seen or set()
    found = []
    for succ in record.successors(node):
        if record.nodes[succ].kind == "SWITCH" and succ not in seen:
            seen.add(succ)
            found += reachable_cases(record, succ, seen)
        elif record.nodes[succ].kind == "ASSIGN":
            found.append(succ)
    return found


def test_narrow_graphs_are_returned_unchanged():
    for graph in (line_graph(), diamond_graph(), loop_graph()):
        assert limit_out_degree(graph, 2) is graph
    assert max_out_degree(diamond_graph()) == 2


def test_wide_switch_becomes_a_dispatch_chain():
    wide = switch_graph(cases=10, vuln_case=9)
    assert max_out_degree(wide) == 10
    narrow = limit_out_degree(wide, 4)
    assert max_out_degree(narrow) == 4
    # 10 cases with 3 kept per level: 3 + 3 + 4, so two dispatch nodes are added.
    assert narrow.num_nodes == wide.num_nodes + 2
    added = narrow.nodes[wide.num_nodes :]
    assert all(n.kind == "SWITCH" and n.line == wide.nodes[1].line for n in added)
    assert all(n.code == wide.nodes[1].code for n in added)
    # No original node changed id, kind or line, and labels are untouched.
    for before, after in zip(wide.nodes, narrow.nodes, strict=False):
        assert (before.id, before.kind, before.line, before.code) == (
            after.id, after.kind, after.line, after.code,
        )  # fmt: skip
    assert narrow.vuln_nodes == wide.vuln_nodes
    assert (narrow.entry, narrow.exit) == (wide.entry, wide.exit)
    assert sorted(reachable_cases(narrow)) == list(range(2, 12))  # every case still reachable
    assert not narrow.back_edges


def test_exact_chain_structure():
    narrow = limit_out_degree(switch_graph(cases=5, vuln_case=5), 3)
    dispatch = 9  # appended after ENTRY, SWITCH, five cases, RETURN, EXIT
    assert narrow.num_nodes == 10
    assert set(narrow.successors(1)) == {2, 3, dispatch}
    assert set(narrow.successors(dispatch)) == {4, 5, 6}
    assert narrow.nodes[dispatch].depth == 2 and narrow.nodes[6].depth == 3
    assert narrow.nodes[2].depth == 2


def test_rewritten_graph_is_usable_by_the_environment():
    wide = switch_graph(cases=12, vuln_case=12)
    with pytest.raises(ValueError, match="exceeds k=6"):
        CFGNavEnv([wide], EnvConfig(k=6))
    narrow = limit_out_degree(wide, 6)
    result = rollout(CFGNavEnv([narrow], EnvConfig(k=6)), OraclePolicy(), seed=0)
    assert result.success
    # The last case sits behind two dispatch nodes: three extra moves compared with a direct hop.
    assert len(result.actions) == 1 + 3 + 1
    with pytest.raises(ValueError, match="at least 2"):
        limit_out_degree(wide, 1)


def test_out_degree_table_and_smallest_k():
    graphs = [line_graph(8), line_graph(8), diamond_graph(), switch_graph(cases=5, vuln_case=1)]
    table = out_degree_table(graphs)
    assert [row["out_degree"] for row in table] == [0, 1, 2, 3, 4, 5]
    assert table[-1]["nodes_covered"] == 1.0 and table[-1]["graphs_covered"] == 1.0
    assert table[1]["graphs_covered"] == 0.5  # the two line graphs
    assert table[2]["graphs_covered"] == 0.75  # plus the diamond
    assert sum(row["nodes"] for row in table) == sum(g.num_nodes for g in graphs)
    assert smallest_k(graphs, node_coverage=0.9) == 1
    assert smallest_k(graphs, node_coverage=0.97) == 2
    assert smallest_k(graphs, node_coverage=1.0) == 5


def test_graph_statistics():
    graphs = [line_graph(3, vuln=3), diamond_graph(vuln=3), loop_graph(vuln=3)]
    stats = graph_statistics(graphs)
    assert stats["graphs"] == 3
    assert stats["nodes"]["p50"] == 6 and stats["nodes"]["p100"] == 6
    assert stats["single_vulnerable_node_share"] == 1.0
    assert stats["distance_entry_to_vulnerable"]["p50"] == 3  # 3, 2 and 3 moves
    assert stats["vulnerable_node_fraction"]["mean"] == pytest.approx((1 / 5 + 1 / 6 + 1 / 6) / 3)
    assert stats["kinds"]["ENTRY"] == 3 and stats["kinds"]["LOOP"] == 1
    assert stats["smallest_k_for_99_percent_of_nodes"] == 2
    json.dumps(stats)  # everything is serialisable
    with pytest.raises(ValueError):
        graph_statistics([])


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stats_and_inspect_scripts(tmp_path, capsys):
    graphs = [line_graph(30), diamond_graph(), switch_graph(cases=8, vuln_case=2)]
    source = tmp_path / "graphs_train.jsonl"
    write_jsonl(graphs, source)
    out = tmp_path / "stats.json"
    _load("06_graph_stats").main(
        [
            "--graphs",
            str(source),
            "--output",
            str(out),
            "--env-config",
            str(ROOT / "configs" / "env.yaml"),
        ]
    )
    stats = json.loads(out.read_text())
    assert stats["env"] == {"k": 6, "graphs_wider_than_k": 1, "graphs_at_step_cap": 0}
    printed = capsys.readouterr().out
    assert "configured K = 6: 1 graphs are wider" in printed

    _load("inspect_graph").main(["--graphs", str(source), "--ids", "diamond"])
    printed = capsys.readouterr().out
    assert "== diamond" in printed and "6 nodes, 1 vulnerable" in printed
    assert "BRANCH -> L3, L4" in printed
    assert printed.count(">>") == 1
    _load("inspect_graph").main(["--graphs", str(source), "--sample", "2", "--max-nodes", "10"])
    assert capsys.readouterr().out.count("== ") == 1  # only the diamond is that small
