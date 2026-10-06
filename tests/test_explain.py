from helpers import diamond_graph

from soren.viz.explain import describe_flags, explain_trace, fix_diff
from soren.viz.trace import Trace, TraceStep


def trace(steps, success, reason):
    return Trace("diamond", "ppo", 0, steps, {"success": success, "end_reason": reason})


def test_fix_diff_marks_removed_and_added_lines():
    before = "int f(char *s) {\n  char b[8];\n  strcpy(b, s);\n  return 0;\n}"
    after = before.replace("strcpy(b, s);", "strncpy(b, s, 7);")
    rows = fix_diff(before, after, context=1)
    assert [(r["kind"], r["line"]) for r in rows] == [
        ("context", 2),
        ("removed", 3),
        ("added", None),
        ("context", 4),
    ]
    assert rows[1]["text"].strip() == "strcpy(b, s);" and "strncpy" in rows[2]["text"]
    assert fix_diff(before, before) == []


def test_explanations_state_what_happened():
    graph = diamond_graph(vuln=3)  # node 3 is on line 4
    graph.nodes[2].code, graph.nodes[2].calls = "memcpy(d, s, n);", ["memcpy"]
    assert describe_flags(graph.nodes[2])[0] == "calls a memory or string API (memcpy)"

    right = [TraceStep(0, 0, "MOVE_0", 1, -0.01), TraceStep(1, 1, "MOVE_1", 3, -0.01)]
    right.append(TraceStep(2, 3, "DECLARE", 3, 1.0))
    text = " ".join(explain_trace(graph, trace(right, True, "correct")))
    assert "declared line 4" in text and "the declaration is correct" in text

    wrong = [TraceStep(0, 0, "MOVE_0", 1, -0.01), TraceStep(1, 1, "MOVE_0", 2, -0.01)]
    wrong.append(TraceStep(2, 2, "DECLARE", 2, -0.5))
    sentences = explain_trace(graph, trace(wrong, False, "wrong_declare"))
    assert "declared line 3" in sentences[0] and "it changed line 4" in sentences[0]
    assert "looks risky" in sentences[1] and "memcpy" in sentences[1]
    assert "never reached a line the fix changed" in sentences[2] and "2 moves" in sentences[2]

    passed = [TraceStep(0, 0, "MOVE_0", 1, -0.01), TraceStep(1, 1, "MOVE_1", 3, -0.01)]
    passed += [TraceStep(2, 3, "MOVE_0", 4, -0.01), TraceStep(3, 4, "DECLARE", 4, -0.5)]
    sentences = explain_trace(graph, trace(passed, False, "wrong_declare"))
    assert "stood on line 4" in sentences[-1] and "at step 2 and moved on" in sentences[-1]

    oracle = explain_trace(graph, trace(right, True, "correct"), oracle_stop=True)
    assert oracle[0].startswith("This baseline does not decide")
