import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from helpers import diamond_graph, line_graph

from soren.agents.ppo import PPOConfig
from soren.agents.searcher import PolicySearcher
from soren.agents.train import train
from soren.baselines import BFS, DFS, LineOrder, RandomOrder, RandomWalk
from soren.data.schema import write_jsonl
from soren.data.synthetic import SyntheticConfig, generate_dataset
from soren.env.cfg_nav_env import EnvConfig
from soren.eval.metrics import (
    METRIC_COLUMNS,
    add_buckets,
    breakdown,
    compute_metrics,
    metrics_across_seeds,
)
from soren.eval.runner import (
    TestSplitError,
    graph_metadata,
    load_results,
    run_evaluation,
    save_results,
    shortest_path_to_vulnerable,
)

GRAPHS = generate_dataset(30, seed=8, cfg=SyntheticConfig(min_nodes=8, max_nodes=40))
ROOT = Path(__file__).parent.parent


def episode(**overrides) -> dict:
    base = {
        "method": "m",
        "seed": 0,
        "split": "val",
        "graph_id": "g",
        "project": "p",
        "cwe": "CWE-119",
        "num_nodes": 10,
        "num_vuln": 1,
        "dist_to_vuln": 2,
        "max_steps": 40,
        "success": True,
        "first_declare_correct": True,
        "nodes_inspected": 3,
        "actions_taken": 3,
        "cumulative_reward": 0.98,
        "first_hit_step": 2,
        "end_reason": "correct",
    }
    return {**base, **overrides}


def test_metrics_on_a_hand_built_table():
    table = pd.DataFrame(
        [
            episode(),  # optimal: 2 moves + declare
            episode(nodes_inspected=6, actions_taken=9, cumulative_reward=0.92),  # 8 moves for 2
            episode(
                success=False,
                first_declare_correct=False,
                nodes_inspected=4,
                actions_taken=5,
                cumulative_reward=-0.54,
                end_reason="wrong_declare",
            ),
            episode(
                success=False,
                first_declare_correct=False,
                nodes_inspected=7,
                actions_taken=40,
                cumulative_reward=-0.9,
                end_reason="timeout",
            ),
        ]
    )
    metrics = compute_metrics(table).iloc[0]
    assert metrics["episodes"] == 4
    assert metrics["success_rate"] == pytest.approx(0.5)
    assert metrics["localization_accuracy"] == pytest.approx(0.5)
    assert metrics["nodes_inspected_success"] == pytest.approx((3 + 6) / 2)
    assert metrics["actions_success"] == pytest.approx((3 + 9) / 2)
    # Failures are charged every node (10) and the whole budget (40).
    assert metrics["nodes_inspected_all"] == pytest.approx((3 + 6 + 10 + 10) / 4)
    assert metrics["actions_all"] == pytest.approx((3 + 9 + 40 + 40) / 4)
    assert metrics["normalised_cost"] == pytest.approx((0.3 + 0.6 + 1.0 + 1.0) / 4)
    assert metrics["cumulative_reward"] == pytest.approx((0.98 + 0.92 - 0.54 - 0.9) / 4)
    assert metrics["path_efficiency"] == pytest.approx((2 / 2 + 8 / 2) / 2)


def test_localization_accuracy_differs_from_success_with_a_declare_budget():
    table = pd.DataFrame([episode(first_declare_correct=False), episode()])
    metrics = compute_metrics(table).iloc[0]
    assert metrics["success_rate"] == 1.0
    assert metrics["localization_accuracy"] == 0.5


def test_metrics_group_by_method_and_handle_no_successes():
    table = pd.DataFrame(
        [episode(method="a"), episode(method="b", success=False, first_declare_correct=False)]
    )
    metrics = compute_metrics(table).set_index("method")
    assert list(metrics.columns) == list(METRIC_COLUMNS)
    assert metrics.loc["a", "success_rate"] == 1.0
    assert metrics.loc["b", "success_rate"] == 0.0
    assert np.isnan(metrics.loc["b", "nodes_inspected_success"])
    assert metrics.loc["b", "nodes_inspected_all"] == 10


def test_buckets_and_breakdowns():
    table = pd.DataFrame(
        [
            episode(num_nodes=20, dist_to_vuln=2, num_vuln=1),
            episode(num_nodes=21, dist_to_vuln=3, num_vuln=2),
            episode(num_nodes=100, dist_to_vuln=10, num_vuln=1, success=False),
            episode(num_nodes=101, dist_to_vuln=11, num_vuln=3, cwe="CWE-125"),
        ]
    )
    bucketed = add_buckets(table)
    assert bucketed["size_bucket"].tolist() == ["<=20", "21-50", "51-100", ">100"]
    assert bucketed["distance_bucket"].tolist() == ["1-2", "3-5", "6-10", ">10"]
    assert bucketed["vuln_bucket"].tolist() == ["single", "multiple", "single", "multiple"]

    by_size = breakdown(table, "size_bucket")
    assert set(by_size["size_bucket"]) == {"<=20", "21-50", "51-100", ">100"}
    assert by_size.set_index("size_bucket").loc["51-100", "success_rate"] == 0.0
    by_cwe = breakdown(table, "cwe").set_index("cwe")
    assert by_cwe.loc["CWE-119", "episodes"] == 3 and by_cwe.loc["CWE-125", "episodes"] == 1
    assert breakdown(table, "vuln_bucket").shape[0] == 2


def test_metrics_across_seeds():
    table = pd.DataFrame(
        [
            episode(seed=0),
            episode(seed=0),
            episode(seed=1),
            episode(seed=1, success=False, first_declare_correct=False),
        ]
    )
    summary = metrics_across_seeds(table).iloc[0]
    assert summary["seeds"] == 2
    assert summary["success_rate_mean"] == pytest.approx(0.75)
    assert summary["success_rate_std"] == pytest.approx(0.25)


def test_shortest_path_and_metadata():
    assert shortest_path_to_vulnerable(line_graph(4, vuln=3)) == 3
    assert shortest_path_to_vulnerable(diamond_graph(vuln=3)) == 2
    meta = graph_metadata(diamond_graph(vuln=3))
    assert meta == {
        "graph_id": "diamond",
        "project": "",
        "cwe": "",
        "num_nodes": 6,
        "num_vuln": 1,
        "dist_to_vuln": 2,
    }


def test_runner_rows_and_seed_handling():
    searchers = [DFS(), BFS(), LineOrder(), RandomWalk(), RandomOrder()]
    frame = run_evaluation(searchers, GRAPHS, seeds=[0, 1, 2])
    counts = frame.groupby("method")["seed"].nunique().to_dict()
    # Deterministic methods run once; seeded ones once per seed.
    assert counts == {"bfs": 1, "dfs": 1, "line_order": 1, "random_order": 3, "random_walk": 3}
    assert len(frame) == len(GRAPHS) * (3 + 2 * 3)
    assert set(frame["split"]) == {"val"}
    assert (frame["max_steps"] == [EnvConfig().max_steps_for(n) for n in frame["num_nodes"]]).all()
    assert frame.groupby(["method", "seed"])["graph_id"].nunique().eq(len(GRAPHS)).all()

    metrics = compute_metrics(frame).set_index("method")
    assert metrics.loc["dfs", "success_rate"] == 1.0
    assert metrics.loc["dfs", "localization_accuracy"] == 1.0  # oracle stop declares correctly
    assert metrics.loc["dfs", "path_efficiency"] >= 1.0
    assert 0.0 < metrics.loc["random_order", "normalised_cost"] <= 1.0


def test_runner_is_reproducible():
    searchers = [RandomWalk(), RandomOrder()]
    a = run_evaluation(searchers, GRAPHS, seeds=[4, 5])
    b = run_evaluation(searchers, GRAPHS, seeds=[4, 5])
    pd.testing.assert_frame_equal(a, b)


def test_test_split_is_refused_without_permission():
    with pytest.raises(TestSplitError, match="allow_test"):
        run_evaluation([DFS()], GRAPHS[:2], split="test")
    frame = run_evaluation([DFS()], GRAPHS[:2], split="test", allow_test=True)
    assert set(frame["split"]) == {"test"}


def test_results_round_trip(tmp_path):
    frame = run_evaluation([DFS(), RandomWalk()], GRAPHS[:5], seeds=[0, 1])
    path = save_results(frame, tmp_path / "nested" / "eval.parquet")
    pd.testing.assert_frame_equal(load_results(path), frame)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "evaluate_script", ROOT / "scripts" / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    cfg = PPOConfig(
        total_timesteps=256,
        n_envs=2,
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        net_arch=[16],
        vec_env="dummy",
        eval_freq=256,
    )
    run_dir = tmp_path_factory.mktemp("runs") / "seed3"
    return train(GRAPHS[:10], GRAPHS[10:14], run_dir, cfg, seed=3, tensorboard=False)


def test_policy_and_baselines_share_one_harness(checkpoint):
    searchers = [DFS(), PolicySearcher(checkpoint.model, name="ppo")]
    frame = run_evaluation(searchers, GRAPHS[:8], seeds=[0, 1], seed_overrides={1: 3})
    assert set(frame["method"]) == {"dfs", "ppo"}
    assert frame[frame["method"] == "ppo"]["seed"].unique().tolist() == [3]
    assert len(frame) == 16
    assert compute_metrics(frame).shape == (2, 1 + len(METRIC_COLUMNS))


def test_evaluate_script(tmp_path, checkpoint, capsys):
    script = _load_script()
    graphs = tmp_path / "val.jsonl"
    write_jsonl(GRAPHS[:6], graphs)
    out = tmp_path / "eval.parquet"
    script.main(
        [
            "--graphs", str(graphs),
            "--split", "val",
            "--out", str(out),
            "--methods", "dfs", "random_walk",
            "--checkpoints", str(checkpoint.best_model_path),
            "--seeds", "0", "1",
            "--env-config", str(ROOT / "configs" / "env.yaml"),
            "--breakdown", "size_bucket",
        ]
    )  # fmt: skip
    frame = load_results(out)
    assert set(frame["method"]) == {"dfs", "random_walk", "ppo"}
    assert frame[frame["method"] == "ppo"]["seed"].unique().tolist() == [3]  # from the path
    assert len(frame) == 6 * (1 + 2 + 1)
    printed = capsys.readouterr().out
    assert "success_rate" in printed and "by size_bucket:" in printed

    with pytest.raises(TestSplitError):
        script.main(
            ["--graphs", str(graphs), "--split", "test", "--out", str(out), "--methods", "dfs"]
        )
    assert script.checkpoint_seed("runs/a/best_model.zip", 7) == 7
