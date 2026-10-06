import numpy as np
import pytest
from helpers import diamond_graph, loop_graph

from soren.baselines import DFS
from soren.data.synthetic import SyntheticConfig, generate_dataset
from soren.env.cfg_nav_env import EnvConfig
from soren.viz.render import (
    COLOUR_CORRECT,
    COLOUR_CURRENT,
    COLOUR_TRUTH,
    COLOUR_UNVISITED,
    COLOUR_WRONG,
    COLOURS_VISITED,
    action_labels,
    cfg_dot,
    node_label,
    outcome_text,
    replay_state,
    source_html,
    step_table,
)
from soren.viz.trace import Trace, TraceStep

K = EnvConfig().k


def rng():
    return np.random.default_rng(0)


def manual_trace() -> Trace:
    """On the diamond graph: 0 -> 1 -> 2, back to 1, -> 3, declare (correct)."""
    steps = [
        TraceStep(0, 0, "MOVE_0", 1, -0.01),
        TraceStep(1, 1, "MOVE_0", 2, -0.01),
        TraceStep(2, 2, "BACKTRACK", 1, -0.01),
        TraceStep(3, 1, "MOVE_1", 3, -0.01),
        TraceStep(4, 3, "DECLARE", 3, 1.0),
    ]
    outcome = {
        "success": True,
        "declared_node": 3,
        "return": 0.96,
        "end_reason": "correct",
        "nodes_inspected": 4,
    }
    return Trace("diamond", "manual", 0, steps, outcome)


def node_line(dot: str, node: int) -> str:
    return next(line for line in dot.splitlines() if line.strip().startswith(f"n{node} ["))


def test_replay_state_tracks_position_stack_and_visits():
    trace = manual_trace()
    start = replay_state(trace, 0)
    assert (start.current, start.stack, start.visits, start.finished) == (0, [], {0: 1}, False)

    deep = replay_state(trace, 2)
    assert deep.current == 2 and deep.stack == [0, 1]
    assert deep.taken_edges == {(0, 1), (1, 2)}

    back = replay_state(trace, 3)
    assert back.current == 1 and back.stack == [0]
    assert back.visits == {0: 1, 1: 2, 2: 1}

    end = replay_state(trace, 5)
    assert end.current == 3 and end.declared == [3] and end.finished
    assert end.running_return == pytest.approx(0.96)
    assert replay_state(trace, 99).step == 5 and replay_state(trace, -3).step == 0


def test_replay_state_matches_the_environment_for_dfs():
    for graph in generate_dataset(15, seed=6, cfg=SyntheticConfig(min_nodes=8, max_nodes=30)):
        trace = DFS().trace(graph, 200, rng())
        state = replay_state(trace, len(trace.steps))
        assert state.current == trace.nodes[-1]
        assert len(state.visits) == trace.outcome["nodes_inspected"]
        assert state.running_return == pytest.approx(trace.outcome["return"])
        assert state.declared == [trace.outcome["declared_node"]]


def test_cfg_dot_colours_follow_the_replay():
    graph, trace = diamond_graph(vuln=3), manual_trace()
    dot = cfg_dot(graph, trace, 3)
    assert dot.startswith("digraph cfg {") and dot.rstrip().endswith("}")
    assert COLOUR_CURRENT in node_line(dot, 1)
    assert COLOURS_VISITED[0] in node_line(dot, 2)
    assert COLOUR_UNVISITED in node_line(dot, 3)
    assert "penwidth=2.5" in node_line(dot, 0)  # on the path stack
    assert "penwidth" not in node_line(dot, 2)
    assert f'n0 -> n1 [color="{"#1f6fb2"}", penwidth=2];' in dot
    assert "n3 -> n4;" in dot
    for u, v in graph.edges:
        assert f"n{u} -> n{v}" in dot


def test_cfg_dot_marks_declarations_and_hides_truth_until_asked():
    graph, trace = diamond_graph(vuln=3), manual_trace()
    hidden = cfg_dot(graph, trace, 2)
    assert COLOUR_TRUTH not in hidden and "peripheries" not in hidden
    shown = cfg_dot(graph, trace, 2, show_truth=True)
    assert COLOUR_TRUTH in node_line(shown, 3) and "peripheries=2" in node_line(shown, 3)
    assert COLOUR_TRUTH not in node_line(shown, 2)

    assert COLOUR_CORRECT in node_line(cfg_dot(graph, trace, 5), 3)
    wrong = Trace(
        "diamond", "m", 0, [TraceStep(0, 0, "MOVE_0", 1, 0), TraceStep(1, 1, "DECLARE", 1, -0.5)]
    )
    assert COLOUR_WRONG in node_line(cfg_dot(graph, wrong, 2), 1)


def test_cfg_dot_draws_back_edges_dashed_and_escapes_labels():
    graph = loop_graph()
    graph.nodes[2].code = 'printf("a\\n");'
    trace = Trace("loop", "m", 0)
    dot = cfg_dot(graph, trace, 0)
    assert "n3 -> n1 [style=dashed];" in dot
    assert '\\"a\\\\n\\"' in node_line(dot, 2)
    assert node_label(graph, 0) == "ENTRY"
    graph.nodes[3].code = "x = " + "y + " * 20 + "z;"
    assert node_label(graph, 3).endswith("…") and len(node_label(graph, 3)) < 45


def test_source_html_marks_lines():
    graph, trace = diamond_graph(vuln=3), manual_trace()
    html_hidden = source_html(graph, trace, 3)
    rows = html_hidden.split("<div style=")[2:]  # one per source line
    assert len(rows) == len(graph.source_lines)
    assert "255, 179, 71" in rows[1]  # line 2 holds the current node
    assert "90, 148, 200" in rows[2]  # line 3 was visited
    assert "transparent;border" in rows[3]  # line 4 not visited yet
    assert "◆" not in html_hidden
    shown = source_html(graph, trace, 3, show_truth=True)
    assert shown.count("◆") == 1 and "◆" in shown.split("<div style=")[5]  # line 4 = node 3
    assert "123, 211, 137" in source_html(graph, trace, 5).split("<div style=")[5]


def test_source_html_escapes_code():
    graph = diamond_graph()
    graph.source_lines[1] = "if (a < b && c > d) {"
    assert "a &lt; b &amp;&amp; c &gt; d" in source_html(graph, Trace("diamond", "m", 0), 0)


def test_step_table_and_labels():
    graph, trace = diamond_graph(vuln=3), manual_trace()
    table = step_table(graph, trace, 3)
    assert table["action"].tolist() == ["MOVE_0", "MOVE_0", "BACKTRACK"]
    assert table["line"].tolist() == [1, 2, 3] and table["to line"].tolist() == [2, 3, 2]
    assert table["return"].tolist() == pytest.approx([-0.01, -0.02, -0.03])
    assert len(step_table(graph, trace)) == 5 and step_table(graph, trace, 0).empty

    labels = action_labels(graph, 1, K + 2)
    assert labels[:2] == ["move → line 3", "move → line 4"]
    assert labels[-2:] == ["backtrack", "declare"] and len(labels) == K + 2
    text = outcome_text(graph, trace)
    assert "found the vulnerable statement" in text and "4 of 6 nodes" in text
