from pathlib import Path

import numpy as np
import pytest
from helpers import diamond_graph, loop_graph
from streamlit.testing.v1 import AppTest

from soren.baselines import BFS, DFS
from soren.data.schema import write_jsonl
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

APP = Path(__file__).parent.parent / "soren" / "viz" / "app.py"
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


@pytest.fixture
def demo(tmp_path):
    graphs = generate_dataset(6, seed=4, cfg=SyntheticConfig(min_nodes=8, max_nodes=20))
    graphs_path = tmp_path / "val.jsonl"
    write_jsonl(graphs, graphs_path)
    traces = tmp_path / "traces"
    for searcher in (DFS(), BFS()):
        for graph in graphs:
            searcher.trace(graph, 100, rng()).save(
                traces / searcher.name / f"{graph.sample_id}.json"
            )
    # One policy-style trace with probabilities, to exercise the probability chart.
    trace = DFS().trace(graphs[0], 100, rng())
    for step in trace.steps:
        step.probs = [1.0 / (K + 2)] * (K + 2)
        step.value = 0.5
    trace.method = "ppo"
    trace.save(traces / "ppo" / f"{graphs[0].sample_id}.json")
    return graphs, graphs_path, traces


def run_app(monkeypatch, graphs_path, traces) -> AppTest:
    monkeypatch.setenv("SOREN_GRAPHS", str(graphs_path))
    monkeypatch.setenv("SOREN_TRACES", str(traces))
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()
    assert not app.exception, app.exception
    return app


def test_app_replays_an_episode(monkeypatch, demo):
    graphs, graphs_path, traces = demo
    app = run_app(monkeypatch, graphs_path, traces)
    assert app.selectbox(key="method").options == ["bfs", "dfs", "ppo"]
    options = app.selectbox(key="graph").options
    assert [o.split(" · ")[0] for o in options] == [g.sample_id for g in graphs]
    assert options[0] == f"{graphs[0].sample_id} · {graphs[0].num_nodes} nodes"
    assert app.session_state["step"] == 0
    assert any("Step 0 of" in c.value for c in app.caption)

    app.button(key="next").click().run()
    app.button(key="next").click().run()
    assert not app.exception and app.session_state["step"] == 2
    assert len(app.dataframe[0].value) == 2
    app.button(key="prev").click().run()
    assert app.session_state["step"] == 1
    app.button(key="first").click().run()
    assert app.session_state["step"] == 0


def test_app_shows_the_outcome_at_the_end_and_resets_on_a_new_episode(monkeypatch, demo):
    graphs, graphs_path, traces = demo
    app = run_app(monkeypatch, graphs_path, traces)
    total = len(Trace.load(traces / "bfs" / f"{graphs[0].sample_id}.json").steps)
    app.slider(key="step").set_value(total).run()
    assert not app.exception
    assert "bfs found the vulnerable statement" in app.success[0].value

    app.selectbox(key="graph").select(graphs[1].sample_id).run()
    assert app.session_state["step"] == 0  # a new episode starts from the beginning
    app.selectbox(key="method").select("ppo").run()
    assert not app.exception
    assert [o.split(" · ")[0] for o in app.selectbox(key="graph").options] == [graphs[0].sample_id]
    assert any("Policy at this step" in c.value for c in app.caption)


def test_app_filters_and_ground_truth_toggle(monkeypatch, demo):
    graphs, graphs_path, traces = demo
    app = run_app(monkeypatch, graphs_path, traces)
    app.radio(key="outcome").set_value("failure").run()
    assert not app.exception
    assert "No episode matches the filters." in [i.value for i in app.info]
    app.radio(key="outcome").set_value("all").run()
    app.toggle(key="truth").set_value(True).run()
    assert not app.exception
    assert any("Ground truth is shown" in c.value for c in app.caption)


def test_app_explains_missing_data(monkeypatch, tmp_path, demo):
    _, graphs_path, _ = demo
    app = run_app(monkeypatch, tmp_path / "missing.jsonl", tmp_path / "none")
    assert "Graph file not found" in app.info[0].value
    app = run_app(monkeypatch, graphs_path, tmp_path / "none")
    assert "Traces directory not found" in app.info[0].value
    (tmp_path / "empty").mkdir()
    app = run_app(monkeypatch, graphs_path, tmp_path / "empty")
    assert "No traces in" in app.info[0].value


def test_compare_mode_shows_two_methods_advancing_together(monkeypatch, demo):
    graphs, graphs_path, traces = demo
    app = run_app(monkeypatch, graphs_path, traces)
    compare = app.selectbox(key="compare")
    assert compare.options == ["(none)", "dfs", "ppo"]  # bfs is the primary method
    assert not app.metric  # single view has no side-by-side counters

    compare.select("dfs").run()
    assert not app.exception
    assert [h.value for h in app.subheader] == ["bfs", "dfs"]
    assert app.session_state["step"] == 0
    assert [m.value for m in app.metric][::2] == [f"1 of {graphs[0].num_nodes}"] * 2

    bfs = Trace.load(traces / "bfs" / f"{graphs[0].sample_id}.json")
    dfs = Trace.load(traces / "dfs" / f"{graphs[0].sample_id}.json")
    longest = max(len(bfs.steps), len(dfs.steps))
    assert app.slider(key="step").max == longest

    app.button(key="next").click().run()
    app.button(key="next").click().run()
    actions = [m.value for m in app.metric][1::2]
    assert actions == [f"2 of {len(bfs.steps)}", f"2 of {len(dfs.steps)}"]

    # At the end both outcomes are shown; the shorter episode waited at its last step.
    app.slider(key="step").set_value(longest).run()
    assert not app.exception
    banners = [b.value for b in (*app.success, *app.error)]
    assert any(b.startswith("bfs ") for b in banners) and any(b.startswith("dfs ") for b in banners)
    actions = [m.value for m in app.metric][1::2]
    assert actions == [
        f"{len(bfs.steps)} of {len(bfs.steps)}",
        f"{len(dfs.steps)} of {len(dfs.steps)}",
    ]

    app.selectbox(key="compare").select("(none)").run()
    assert not app.exception and not app.metric


def test_compare_option_is_hidden_when_no_other_method_has_the_graph(monkeypatch, tmp_path, demo):
    graphs, graphs_path, _ = demo
    only = tmp_path / "only"
    DFS().trace(graphs[0], 100, rng()).save(only / "dfs" / f"{graphs[0].sample_id}.json")
    app = run_app(monkeypatch, graphs_path, only)
    assert not [box for box in app.selectbox if box.key == "compare"]
