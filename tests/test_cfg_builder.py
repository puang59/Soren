import json
from pathlib import Path

import pytest

from soren.data.cfg_builder import (
    CfgBuildError,
    LineCfg,
    build_line_cfg,
    load_joern_methods,
    select_method,
)
from soren.data.features import TIER_L_NAMES, featurize

FIXTURES = Path(__file__).parent / "fixtures" / "joern"
METHODS = load_joern_methods([FIXTURES / "snippets.jsonl"])


def build(name: str, extension: str = ".c") -> LineCfg:
    method = select_method(METHODS[name])
    return build_line_cfg(method, (FIXTURES / "src" / f"{name}{extension}").read_text())


def line_edges(cfg: LineCfg) -> set[tuple]:
    """Edges written with source lines, ENTRY and EXIT, which is how they are checked by hand."""

    def label(node_id: int):
        node = cfg.nodes[node_id]
        return node.kind if node.kind in ("ENTRY", "EXIT") else node.line

    return {(label(u), label(v)) for u, v in cfg.edges}


def kinds(cfg: LineCfg) -> dict[int, str]:
    return {n.line: n.kind for n in cfg.nodes if n.kind not in ("ENTRY", "EXIT")}


def test_straight_line():
    cfg = build("straight")
    assert line_edges(cfg) == {("ENTRY", 3), (3, 4), (4, 5), (5, "EXIT")}
    assert kinds(cfg) == {3: "ASSIGN", 4: "ASSIGN", 5: "RETURN"}
    assert cfg.back_edges == []
    assert [n.depth for n in cfg.nodes] == [0, 1, 2, 3, 4]
    assert cfg.nodes[1].code == "int b = a + 1;"


def test_if_else():
    cfg = build("if_else")
    assert line_edges(cfg) == {
        ("ENTRY", 3), (3, 4), (4, 5), (4, 7), (5, 9), (7, 9), (9, "EXIT"),
    }  # fmt: skip
    assert kinds(cfg) == {3: "ASSIGN", 4: "BRANCH", 5: "ASSIGN", 7: "ASSIGN", 9: "RETURN"}
    assert not any(n.in_loop for n in cfg.nodes)


def test_while_loop():
    cfg = build("while_loop")
    assert line_edges(cfg) == {
        ("ENTRY", 3), (3, 4), (4, 5), (5, 6), (6, 4), (4, 8), (8, "EXIT"),
    }  # fmt: skip
    assert kinds(cfg)[4] == "LOOP"
    assert {cfg.nodes[n].line for n in range(len(cfg.nodes)) if cfg.nodes[n].in_loop} == {4, 5, 6}
    assert [(cfg.nodes[u].line, cfg.nodes[v].line) for u, v in cfg.back_edges] == [(6, 4)]


def test_for_loop_header_collapses_to_one_node():
    """Initialiser, condition and increment share a line, so they are one node."""
    cfg = build("for_loop")
    assert line_edges(cfg) == {("ENTRY", 3), (3, 4), (4, 5), (5, 4), (4, 7), (7, "EXIT")}
    assert kinds(cfg) == {3: "ASSIGN", 4: "LOOP", 5: "ASSIGN", 7: "RETURN"}
    assert not any(u == v for u, v in cfg.edges)


def test_switch_with_break_and_fallthrough():
    cfg = build("switch_case")
    assert line_edges(cfg) == {
        ("ENTRY", 3), (3, 4),
        (4, 5), (4, 8), (4, 10),      # dispatch to case 0, case 1 and default
        (5, 6), (6, 7), (7, 13),      # case 0 ... break
        (8, 9), (9, 10),              # case 1 falls through into default
        (10, 11), (11, 13), (13, "EXIT"),
    }  # fmt: skip
    assert kinds(cfg)[4] == "SWITCH" and kinds(cfg)[7] == "JUMP"
    assert kinds(cfg)[5] == "OTHER"  # a bare case label
    assert len(cfg.nodes) == 12


def test_early_return():
    cfg = build("early_return")
    assert line_edges(cfg) == {
        ("ENTRY", 3), (3, 4), (3, 5), (4, "EXIT"), (5, 6), (6, "EXIT"),
    }  # fmt: skip
    assert kinds(cfg) == {3: "BRANCH", 4: "RETURN", 5: "CALL", 6: "RETURN"}
    assert cfg.nodes[cfg.line_to_node[5]].calls == ["use"]


def test_goto():
    cfg = build("goto_out")
    assert line_edges(cfg) == {
        ("ENTRY", 3), (3, 4), (4, 5), (4, 6),
        (5, 7),                        # goto out
        (6, 7), (7, 8), (8, 9), (9, "EXIT"),
    }  # fmt: skip
    assert kinds(cfg)[5] == "JUMP" and kinds(cfg)[7] == "OTHER"  # the label line
    assert kinds(cfg)[6] == "ASSIGN" and cfg.nodes[cfg.line_to_node[6]].calls == ["work"]


def test_nested_loops_with_break():
    cfg = build("nested_loops")
    assert line_edges(cfg) == {
        ("ENTRY", 3), (3, 4),
        (4, 5), (4, 12),               # for: into the body, or out
        (5, 6), (5, 4),                # while: into the body, or back to the for increment
        (6, 7), (6, 8),
        (7, 4),                        # break leaves the while
        (8, 9), (9, 5),
        (12, "EXIT"),
    }  # fmt: skip
    assert kinds(cfg) == {
        3: "ASSIGN", 4: "LOOP", 5: "LOOP", 6: "BRANCH", 7: "JUMP", 8: "ASSIGN", 9: "ASSIGN",
        12: "RETURN",
    }  # fmt: skip
    assert {n.line for n in cfg.nodes if n.in_loop} == {4, 5, 6, 7, 8, 9}
    assert len(cfg.back_edges) == 3


def test_cpp_method_is_built_from_the_cpp_parse():
    assert set(METHODS["method"]) == {".cpp"}
    cfg = build("method", ".cpp")
    assert line_edges(cfg) == {("ENTRY", 3), (3, 4), (3, 5), (4, "EXIT"), (5, "EXIT")}


def test_nodes_carry_what_the_featurizers_need():
    cfg = build("nested_loops")
    record = cfg.to_record(
        "nested", [cfg.line_to_node[8]], project="p", commit_id="c", cwe="CWE-119"
    )
    assert record.sample_id == "nested" and record.cwe == "CWE-119"
    node = record.nodes[cfg.line_to_node[4]]
    assert "<operator>.lessThan" in node.ops and "<operator>.postIncrement" in node.ops
    feats = featurize(record, "L")
    row = feats[cfg.line_to_node[6]]
    assert row[TIER_L_NAMES.index("kind_branch")] == 1.0
    assert row[TIER_L_NAMES.index("comparison")] == 1.0
    assert feats[cfg.line_to_node[8], TIER_L_NAMES.index("arith_op")] == 1.0


# multi_line needs statements merged across lines and macro_heavy is a parse failure; both
# are covered with the span handling.
SIMPLE = [n for n in METHODS if n not in ("macro_heavy", "method", "multi_line")]


@pytest.mark.parametrize("name", SIMPLE)
def test_every_snippet_gives_a_valid_record(name):
    cfg = build(name)
    assert cfg.nodes[0].kind == "ENTRY" and cfg.nodes[-1].kind == "EXIT"
    assert [n.id for n in cfg.nodes] == list(range(len(cfg.nodes)))
    assert cfg.nodes[-1].depth > 0  # EXIT is reachable
    statements = [n for n in cfg.nodes[1:-1]]
    assert [n.line for n in statements] == sorted(n.line for n in statements)
    record = cfg.to_record(name, [statements[0].id])
    assert record.reachable_from_entry() == set(range(record.num_nodes))
    assert max(len(record.successors(n.id)) for n in record.nodes) <= 3


def test_select_method_takes_the_first_function_and_prefers_c():
    first = {"file": "s.c", "line": 1, "nodes": [1, 2, 3]}
    later = {"file": "s.c", "line": 40, "nodes": [1, 2, 3, 4, 5]}
    assert select_method({".c": [later, first]}) is first
    cpp_same = {"file": "s.cpp", "line": 1, "nodes": [1, 2, 3]}
    cpp_bigger = {"file": "s.cpp", "line": 1, "nodes": [1, 2, 3, 4]}
    assert select_method({".c": [first], ".cpp": [cpp_same]}) is first
    assert select_method({".c": [first], ".cpp": [cpp_bigger]}) is cpp_bigger
    assert select_method({".cpp": [cpp_same]}) is cpp_same
    assert select_method({}) is None


def test_nodes_without_a_line_are_routed_around():
    method = {
        "file": "x.c", "name": "f", "line": 1, "line_end": 4,
        "nodes": [
            [1, "METHOD", 1, "", ""], [9, "METHOD_RETURN", 1, "", ""],
            [2, "CALL", 2, "<operator>.assignment", "a = 1"],
            [3, "IDENTIFIER", -1, "", "ghost"],
            [4, "RETURN", 3, "", "return a;"],
        ],
        "edges": [[1, 2], [2, 3], [3, 4], [4, 9]],
        "controls": [], "stmts": [],
    }  # fmt: skip
    cfg = build_line_cfg(method, "int f() {\n  a = 1;\n  return a;\n}\n")
    assert line_edges(cfg) == {("ENTRY", 2), (2, 3), (3, "EXIT")}


def test_unusable_methods_raise():
    empty = {
        "file": "x.c", "name": "f", "line": 1, "line_end": 1,
        "nodes": [[1, "METHOD", 1, "", ""], [9, "METHOD_RETURN", 1, "", ""]],
        "edges": [[1, 9]], "controls": [], "stmts": [],
    }  # fmt: skip
    with pytest.raises(CfgBuildError, match="no statements"):
        build_line_cfg(empty, "int f();\n")
    with pytest.raises(CfgBuildError, match="METHOD"):
        build_line_cfg({**empty, "nodes": []}, "")


def test_fixture_matches_the_snippet_sources():
    """Guards against editing a snippet without regenerating the fixture."""
    files = {
        json.loads(line)["file"] for line in (FIXTURES / "snippets.jsonl").read_text().splitlines()
    }
    assert files == {p.name for p in (FIXTURES / "src").iterdir()}
