import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from helpers import make_graph

from soren.baselines import BFS, DFS, HeuristicFirst, HeuristicScorer, ThresholdRule, make_baseline
from soren.baselines.heuristic import (
    BASELINE_FACTORIES,
    DEFAULT_WEIGHTS,
    best_threshold,
    threshold_curve,
)
from soren.data.schema import write_jsonl
from soren.data.synthetic import SyntheticConfig, generate_dataset
from soren.env.cfg_nav_env import EnvConfig
from soren.env.rewards import RewardConfig
from soren.eval.metrics import compute_metrics
from soren.eval.runner import load_results, run_evaluation

ROOT = Path(__file__).parent.parent
R = RewardConfig()
GRAPHS = generate_dataset(80, seed=17, cfg=SyntheticConfig(min_nodes=10, max_nodes=40))


def rng(seed=0):
    return np.random.default_rng(seed)


def scored_graph(vuln: int = 5):
    """
    0 ENTRY -> 1 if -> 2 x = y;            -> 4 memcpy(...) -> 6 EXIT
                    -> 3 buf[i] = *p + 1;  -> 5 strcpy(...) -> 6
    """
    kinds = ["ENTRY", "BRANCH", "ASSIGN", "ASSIGN", "CALL", "CALL", "EXIT"]
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6)]
    graph = make_graph(kinds, edges, [vuln], sample_id="scored")
    code = [
        "",
        "if (n > 0) {",
        "x = y;",
        "buf[i] = *p + 1;",
        "memcpy(d, s, n);",
        "strcpy(d, s);",
        "",
    ]
    calls = [[], [], [], [], ["memcpy"], ["strcpy"], []]
    for node, text, callees in zip(graph.nodes, code, calls, strict=True):
        node.code, node.calls = text, callees
    return graph


def test_scorer_is_a_scaled_weighted_sum():
    scores = HeuristicScorer().scores(scored_graph())
    total = sum(DEFAULT_WEIGHTS.values())
    assert scores[0] == 0.0 and scores[2] == 0.0  # ENTRY and a plain assignment
    assert scores[4] == pytest.approx(3.0 / total)  # dangerous call only
    assert scores[3] == pytest.approx((2.0 + 1.0 + 0.5) / total)  # subscript, deref, arithmetic
    assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_scorer_weights_are_configurable_and_validated():
    only_calls = HeuristicScorer({"dangerous_call": 1.0})
    assert only_calls.scores(scored_graph()).tolist() == [0, 0, 0, 0, 1, 1, 0]
    with pytest.raises(ValueError, match="unknown lexical features"):
        HeuristicScorer({"in_degree": 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        HeuristicScorer({"dangerous_call": -1.0})


def test_scorer_ignores_labels():
    a, b = scored_graph(vuln=4), scored_graph(vuln=5)
    np.testing.assert_array_equal(HeuristicScorer().scores(a), HeuristicScorer().scores(b))


def test_heuristic_first_expands_the_highest_scoring_frontier_node():
    graph = scored_graph(vuln=5)
    result = HeuristicFirst().run(graph, 100, rng())
    # From the branch, node 3 (score 3.5) beats node 2 (0); then node 5 (3.0) beats node 2.
    assert result.visit_order == [0, 1, 3, 5]
    assert result.success and result.nodes_inspected == 4
    assert DFS().run(graph, 100, rng()).nodes_inspected == 7  # DFS explores the other arm first
    # With the scored nodes used up, ties at zero go to the earlier line (node 2), which
    # then exposes memcpy (node 4).
    assert HeuristicFirst().order(graph, rng()) == [0, 1, 3, 5, 2, 4, 6]


def test_threshold_rule():
    graph = scored_graph()
    rule = ThresholdRule(HeuristicScorer({"dangerous_call": 1.0}), 0.5)
    assert [rule(graph, n) for n in range(7)] == [False, False, False, False, True, True, False]


def test_protocol_b_can_be_wrong():
    scorer = HeuristicScorer({"dangerous_call": 1.0})
    graph = scored_graph(vuln=5)
    dfs = make_baseline("dfs", scorer=scorer, threshold=0.5)
    wrong = dfs.run(graph, 100, rng())
    # DFS reaches memcpy (node 4) first and declares it, but strcpy (node 5) is the flaw.
    assert dfs.name == "dfs+heuristic"
    assert not wrong.success and wrong.declared_node == 4
    assert wrong.end_reason == "wrong_declare"
    assert wrong.cumulative_reward == pytest.approx(3 * R.step + R.wrong)

    right = make_baseline("dfs", scorer=scorer, threshold=0.5).run(scored_graph(vuln=4), 100, rng())
    assert right.success and right.declared_node == 4

    bfs = make_baseline("bfs", scorer=scorer, threshold=0.5).run(graph, 100, rng())
    assert bfs.method == "bfs+heuristic" and not bfs.success and bfs.declared_node == 4
    assert bfs.end_reason == "wrong_declare"
    assert bfs.cumulative_reward == pytest.approx(4 * R.step + R.wrong)


def test_protocol_b_walks_past_the_target_when_the_score_is_too_low():
    scorer = HeuristicScorer({"dangerous_call": 1.0})
    graph = scored_graph(vuln=3)  # the flaw has no dangerous call
    result = make_baseline("line_order", scorer=scorer, threshold=0.5).run(graph, 100, rng())
    assert not result.success and result.declared_node == 4
    assert result.first_hit_step == 3  # it stood on the flaw and moved on
    never = make_baseline("line_order", scorer=scorer, threshold=2.0).run(graph, 100, rng())
    assert never.declared_node is None and never.end_reason == "exhausted"
    assert never.cumulative_reward == pytest.approx(6 * R.step + R.timeout)


def test_protocol_a_is_unchanged_by_default():
    for name in BASELINE_FACTORIES:
        searcher = make_baseline(name)
        assert searcher.name == name
        results = [searcher.run(g, 20 * g.num_nodes, rng(1)) for g in GRAPHS[:20]]
        if name != "random_walk":
            assert all(r.success for r in results)
    with pytest.raises(ValueError, match="unknown baseline"):
        make_baseline("astar")
    with pytest.raises(ValueError, match="needs a threshold"):
        make_baseline("dfs", scorer=HeuristicScorer())


def test_tracing_protocol_b_matches_running_it():
    scorer = HeuristicScorer()
    for name in BASELINE_FACTORIES:
        searcher = make_baseline(name, scorer=scorer, threshold=0.3)
        for graph in GRAPHS[:10]:
            budget = EnvConfig().max_steps_for(graph.num_nodes)
            result = searcher.run(graph, budget, rng(2))
            trace = searcher.trace(graph, budget, rng(2)).validate()
            assert trace.outcome["success"] == result.success
            assert trace.outcome["declared_node"] == result.declared_node
            assert sum(s.reward for s in trace.steps) == pytest.approx(result.cumulative_reward)


def test_threshold_curve_and_best_threshold():
    scorer = HeuristicScorer()
    curve = threshold_curve("dfs", scorer, GRAPHS, [0.05, 0.3, 0.6, 0.99])
    assert curve["threshold"].tolist() == [0.05, 0.3, 0.6, 0.99]
    assert curve["success_rate"].between(0, 1).all()
    # A threshold nothing reaches means nothing is ever declared.
    assert curve.iloc[-1]["declared_rate"] == 0.0 and curve.iloc[-1]["success_rate"] == 0.0
    # On synthetic graphs the flaw is marked by a dangerous call (3 of 10.5 weight = 0.29).
    assert best_threshold(curve) == 0.05 or best_threshold(curve) == 0.3
    assert curve.set_index("threshold").loc[0.6, "success_rate"] == 0.0

    table = pd.DataFrame(
        {
            "threshold": [0.1, 0.2, 0.3],
            "success_rate": [0.5, 0.7, 0.7],
            "nodes_inspected": [5, 9, 6],
        }
    )
    assert best_threshold(table) == 0.3  # ties on success go to fewer nodes inspected


def test_heuristic_baselines_beat_blind_ones_on_the_synthetic_signal():
    scorer = HeuristicScorer({"dangerous_call": 1.0})
    searchers = [make_baseline("dfs"), make_baseline("heuristic_first", scorer=None)]
    searchers += [
        make_baseline(n, scorer=scorer, threshold=0.5) for n in ("dfs", "heuristic_first")
    ]
    frame = run_evaluation(searchers, GRAPHS)
    metrics = compute_metrics(frame).set_index("method")
    assert set(metrics.index) == {
        "dfs",
        "heuristic_first",
        "dfs+heuristic",
        "heuristic_first+heuristic",
    }
    # Under the oracle stop, following the scores reaches the flaw sooner than DFS.
    assert (
        metrics.loc["heuristic_first", "nodes_inspected_all"]
        < metrics.loc["dfs", "nodes_inspected_all"]
    )
    # Under Protocol B success is no longer guaranteed.
    assert 0.2 < metrics.loc["dfs+heuristic", "success_rate"] < 1.0


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tune_and_evaluate_scripts(tmp_path, capsys):
    graphs = tmp_path / "graphs_val.jsonl"
    write_jsonl(GRAPHS[:30], graphs)
    eval_config = tmp_path / "eval.yaml"
    eval_config.write_text("{}\n")
    curve = tmp_path / "curve.csv"
    env_config = str(ROOT / "configs" / "env.yaml")

    evaluate = _load("evaluate")
    base = ["--graphs", str(graphs), "--split", "val", "--out", str(tmp_path / "e.parquet")]
    base += ["--env-config", env_config, "--eval-config", str(eval_config), "--seeds", "0"]
    with pytest.raises(SystemExit):  # no thresholds tuned yet
        evaluate.main([*base, "--methods", "dfs", "--protocol-b"])

    _load("tune_thresholds").main(
        ["--graphs", str(graphs), "--methods", "dfs", "bfs", "--steps", "4", "--seeds", "0",
         "--env-config", env_config, "--eval-config", str(eval_config), "--curve", str(curve),
         "--write-config"]
    )  # fmt: skip
    stored = yaml.safe_load(eval_config.read_text())["protocol_b"]
    assert set(stored["heuristic"]) == {"dfs", "bfs"}
    assert stored["tuned_on"] == str(graphs)
    assert len(pd.read_csv(curve)) == 8

    evaluate.main([*base, "--methods", "dfs", "bfs", "--protocol-b"])
    assert set(load_results(tmp_path / "e.parquet")["method"]) == {
        "dfs",
        "bfs",
        "dfs+heuristic",
        "bfs+heuristic",
    }

    test_graphs = tmp_path / "graphs_test.jsonl"
    write_jsonl(GRAPHS[:3], test_graphs)
    with pytest.raises(SystemExit):
        _load("tune_thresholds").main(["--graphs", str(test_graphs)])
    capsys.readouterr()
    assert isinstance(BFS(), BFS)
