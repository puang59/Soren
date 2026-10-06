import numpy as np

from soren.baselines import DFS
from soren.data.synthetic import SyntheticConfig, generate_dataset
from soren.demo import main, render_frame


def test_frames_follow_the_episode_and_hide_the_truth_until_the_end():
    graph = generate_dataset(1, seed=7, cfg=SyntheticConfig(min_nodes=12, max_nodes=18))[0]
    trace = DFS().trace(graph, 200, np.random.default_rng(0))
    start = render_frame(graph, trace, 0, colour=False)
    assert "step 0/" in start and "next action: MOVE_0" in start and "◆" not in start.split("\n")[3]
    assert start.count("▶") == 2  # the current line and the legend
    end = render_frame(graph, trace, len(trace.steps), show_truth=True, colour=False)
    assert "FOUND the vulnerable statement" in end and "✔" in end
    truth_line = graph.nodes[graph.vuln_nodes[0]].line
    assert end.split("\n")[2 + truth_line].lstrip().startswith("◆")
    assert "\033[" in render_frame(graph, trace, 1) and "\033[" not in start


def test_synthetic_demo_runs_without_data_or_checkpoint(capsys):
    main(["--synthetic", "--no-colour", "--compare", "bfs", "--checkpoint", "missing.zip"])
    out = capsys.readouterr().out
    assert "DFS on syn_" in out and "BFS on syn_" in out and "nodes inspected" in out
