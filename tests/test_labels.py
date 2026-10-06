from pathlib import Path

from soren.data.cfg_builder import build_line_cfg, load_joern_methods, select_method
from soren.data.labels import align_flaw_lines

FIXTURES = Path(__file__).parent / "fixtures" / "joern"
METHODS = load_joern_methods([FIXTURES / "snippets.jsonl"])


def build(name: str):
    source = (FIXTURES / "src" / f"{name}.c").read_text()
    return build_line_cfg(select_method(METHODS[name]), source), source.split("\n")


def test_flaw_line_maps_to_its_node():
    cfg, lines = build("early_return")
    # Index 4 is line 5: `use(p);`
    result = align_flaw_lines(cfg, [4], [lines[4]])
    assert result.ok and result.vuln_nodes == [cfg.line_to_node[5]]
    assert (result.mapped, result.unmapped) == (1, 0)
    assert cfg.nodes[result.vuln_nodes[0]].code == "use(p);"
    record = cfg.to_record("s", result.vuln_nodes)
    assert record.vuln_nodes == result.vuln_nodes


def test_several_flaw_lines_give_several_nodes_without_duplicates():
    cfg, lines = build("while_loop")
    result = align_flaw_lines(cfg, [4, 5, 4], [lines[4], lines[5], lines[4]])
    assert result.ok
    assert [cfg.nodes[n].line for n in result.vuln_nodes] == [5, 6]
    assert result.mapped == 3


def test_any_line_of_a_multi_line_statement_maps_to_its_first_line_node():
    cfg, lines = build("multi_line")
    for index in (6, 7, 8):  # the three lines of the memcpy call
        result = align_flaw_lines(cfg, [index], [lines[index]])
        assert result.ok, index
        assert cfg.nodes[result.vuln_nodes[0]].line == 7
    middle_of_condition = align_flaw_lines(cfg, [4], [lines[4]])
    assert cfg.nodes[middle_of_condition.vuln_nodes[0]].line == 4


def test_off_by_one_index_is_caught_by_the_text_check():
    cfg, lines = build("early_return")
    # The text is `use(p);` (index 4) but the index points one line too late.
    shifted = align_flaw_lines(cfg, [5], [lines[4]])
    assert not shifted.ok and shifted.reason == "text_mismatch"
    assert shifted.vuln_nodes == []
    # One-based indices from a careless release are caught the same way.
    one_based = align_flaw_lines(cfg, [5, 6], [lines[4], lines[5]])
    assert one_based.reason == "text_mismatch"


def test_text_check_ignores_whitespace_only():
    cfg, _ = build("early_return")
    assert align_flaw_lines(cfg, [4], ["\tuse( p ) ;  "]).ok
    assert align_flaw_lines(cfg, [4], ["use(q);"]).reason == "text_mismatch"


def test_index_out_of_range_or_mismatched_lengths():
    cfg, lines = build("straight")
    assert align_flaw_lines(cfg, [99], ["x"]).reason == "text_mismatch"
    assert align_flaw_lines(cfg, [2, 3], [lines[2]]).reason == "text_mismatch"
    # Without texts there is nothing to verify, and a line outside the function has no node.
    assert align_flaw_lines(cfg, [99]).reason == "no_flaw_on_node"


def test_flaw_lines_without_a_node_are_not_labels():
    cfg, lines = build("if_else")
    # Index 5 is `} else {` and index 0 the signature: neither is a statement node.
    result = align_flaw_lines(cfg, [5, 0], [lines[5], lines[0]])
    assert result.reason == "no_flaw_on_node"
    assert (result.mapped, result.unmapped) == (0, 2)
    mixed = align_flaw_lines(cfg, [5, 4], [lines[5], lines[4]])
    assert mixed.ok and (mixed.mapped, mixed.unmapped) == (1, 1)
    assert [cfg.nodes[n].line for n in mixed.vuln_nodes] == [5]


def test_flaw_in_code_joern_left_out_is_rejected():
    cfg, lines = build("macro_heavy")
    # Index 6 is inside the inactive #ifdef branch.
    assert "copy_name" in lines[6]
    assert align_flaw_lines(cfg, [6], [lines[6]]).reason == "no_flaw_on_node"


def test_unreachable_flaw_is_rejected():
    method = {
        "file": "x.c", "name": "f", "line": 1, "line_end": 5,
        "nodes": [
            [1, "METHOD", 1, "", ""], [9, "METHOD_RETURN", 1, "", ""],
            [2, "RETURN", 2, "", "return 0;"],
            [3, "CALL", 3, "dead", "dead()"],
            [4, "RETURN", 4, "", "return 1;"],
        ],
        "edges": [[1, 2], [2, 9], [3, 4], [4, 9]],  # line 3 is dead code after the return
        "controls": [], "stmts": [],
    }  # fmt: skip
    source = "int f() {\n  return 0;\n  dead();\n  return 1;\n}\n"
    cfg = build_line_cfg(method, source)
    assert align_flaw_lines(cfg, [2], ["  dead();"]).reason == "flaw_unreachable"
    # A reachable flaw line next to an unreachable one keeps the sample.
    both = align_flaw_lines(cfg, [1, 2], ["  return 0;", "  dead();"])
    assert both.ok and len(both.vuln_nodes) == 2
