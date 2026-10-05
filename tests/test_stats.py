import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soren.baselines import BFS, DFS, RandomOrder, RandomWalk
from soren.data.synthetic import SyntheticConfig, generate_dataset
from soren.eval.runner import run_evaluation, save_results
from soren.eval.stats import (
    bootstrap_ci,
    comparison_table,
    confidence_intervals,
    mcnemar_success,
    per_graph,
    wilcoxon_nodes_inspected,
)

ROOT = Path(__file__).parent.parent


def episodes(method, successes, nodes, seed=0, num_nodes=20, max_steps=80) -> pd.DataFrame:
    n = len(successes)
    return pd.DataFrame(
        {
            "method": method,
            "seed": seed,
            "graph_id": [f"g{i}" for i in range(n)],
            "num_nodes": num_nodes,
            "max_steps": max_steps,
            "success": successes,
            "first_declare_correct": successes,
            "nodes_inspected": nodes,
            "actions_taken": nodes,
            "cumulative_reward": [1.0 if s else -0.5 for s in successes],
        }
    )


def test_bootstrap_ci_of_a_constant_is_degenerate():
    assert bootstrap_ci([3.0] * 20) == (3.0, 3.0, 3.0)


def test_bootstrap_ci_matches_the_normal_approximation():
    rng = np.random.default_rng(0)
    values = rng.normal(10.0, 2.0, size=400)
    mean, low, high = bootstrap_ci(values, n_resamples=5000)
    standard_error = values.std(ddof=1) / np.sqrt(values.size)
    assert mean == pytest.approx(values.mean())
    assert low == pytest.approx(mean - 1.96 * standard_error, abs=0.03)
    assert high == pytest.approx(mean + 1.96 * standard_error, abs=0.03)


def test_bootstrap_ci_for_a_proportion():
    values = [1.0] * 70 + [0.0] * 30
    mean, low, high = bootstrap_ci(values)
    assert mean == 0.7
    assert 0.59 < low < 0.64 and 0.76 < high < 0.80  # near the binomial interval


def test_bootstrap_ci_is_seeded_and_widens_with_confidence():
    values = np.random.default_rng(1).normal(size=50)
    assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)
    _, low95, high95 = bootstrap_ci(values, confidence=0.95)
    _, low99, high99 = bootstrap_ci(values, confidence=0.99)
    assert low99 < low95 < high95 < high99
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], confidence=1.5)


def test_per_graph_averages_seeds_and_charges_failures():
    frame = pd.concat(
        [
            episodes("m", [True, True], [4, 6], seed=0),
            episodes("m", [True, False], [8, 5], seed=1),
        ]
    )
    table = per_graph(frame).set_index("graph_id")
    assert table.loc["g0", "nodes_inspected"] == 6.0  # (4 + 8) / 2
    assert table.loc["g1", "success"] == 0.5
    assert table.loc["g1", "nodes_inspected"] == (6 + 20) / 2  # the failure is charged 20 nodes
    assert table.loc["g1", "actions_taken"] == (6 + 80) / 2  # and the whole 80-step budget
    assert table.loc["g1", "normalised_cost"] == pytest.approx((0.3 + 1.0) / 2)


def test_confidence_intervals_table():
    frame = pd.concat(
        [
            episodes("a", [True] * 30, [5] * 30),
            episodes("b", [True] * 15 + [False] * 15, [9] * 30),
        ]
    )
    table = confidence_intervals(frame, n_resamples=2000).set_index(["method", "metric"])
    assert table.loc[("a", "success"), ["mean", "low", "high"]].tolist() == [1.0, 1.0, 1.0]
    b = table.loc[("b", "success")]
    assert b["mean"] == 0.5 and b["low"] < 0.5 < b["high"] and b["graphs"] == 30
    assert (table["low"] <= table["mean"]).all() and (table["mean"] <= table["high"]).all()


def test_wilcoxon_detects_a_consistent_difference():
    rng = np.random.default_rng(0)
    base = rng.integers(5, 15, size=40)
    frame = pd.concat(
        [
            episodes("a", [True] * 40, base),
            episodes("b", [True] * 40, base + rng.integers(1, 4, size=40)),
        ]
    )
    result = wilcoxon_nodes_inspected(frame, "a", "b")
    assert result["graphs"] == 40
    assert result["median_difference"] < 0 and result["p_value"] < 1e-6
    # Swapping the methods flips the sign and keeps the p-value.
    swapped = wilcoxon_nodes_inspected(frame, "b", "a")
    assert swapped["median_difference"] == -result["median_difference"]
    assert swapped["p_value"] == pytest.approx(result["p_value"])


def test_wilcoxon_on_noise_and_on_identical_methods():
    rng = np.random.default_rng(1)
    frame = pd.concat(
        [
            episodes("a", [True] * 60, rng.integers(5, 15, size=60)),
            episodes("b", [True] * 60, rng.integers(5, 15, size=60)),
            episodes("c", [True] * 60, [7] * 60),
            episodes("d", [True] * 60, [7] * 60),
        ]
    )
    assert wilcoxon_nodes_inspected(frame, "a", "b")["p_value"] > 0.05
    assert wilcoxon_nodes_inspected(frame, "c", "d")["p_value"] == 1.0


def test_mcnemar_counts_discordant_graphs():
    a = [True] * 20 + [True] * 12 + [False] * 2 + [False] * 6
    b = [True] * 20 + [False] * 12 + [True] * 2 + [False] * 6
    frame = pd.concat([episodes("a", a, [5] * 40), episodes("b", b, [5] * 40)])
    result = mcnemar_success(frame, "a", "b")
    assert (result["a_only"], result["b_only"]) == (12, 2)
    assert result["p_value"] == pytest.approx(0.0129, abs=1e-3)  # exact binomial, 2 of 14
    same = mcnemar_success(
        pd.concat([episodes("a", a, [5] * 40), episodes("b", a, [5] * 40)]), "a", "b"
    )
    assert same["p_value"] == 1.0


def test_paired_tests_use_only_shared_graphs_and_reject_unknown_methods():
    frame = pd.concat([episodes("a", [True] * 10, [5] * 10), episodes("b", [True] * 6, [6] * 6)])
    assert wilcoxon_nodes_inspected(frame, "a", "b")["graphs"] == 6
    with pytest.raises(ValueError, match="no episodes for method 'zzz'"):
        mcnemar_success(frame, "a", "zzz")


def test_comparison_table_on_real_evaluation_results():
    graphs = generate_dataset(60, seed=2, cfg=SyntheticConfig(min_nodes=10, max_nodes=40))
    frame = run_evaluation([DFS(), BFS(), RandomWalk(), RandomOrder()], graphs, seeds=[0, 1, 2])
    table = comparison_table(frame, reference="dfs", n_resamples=1000)
    assert table["method"].tolist() == ["dfs", "bfs", "random_order", "random_walk"]
    reference = table.iloc[0]
    assert np.isnan(reference["p_success"]) and np.isnan(reference["p_nodes"])
    assert reference["success"].startswith("1.000 [1.000, 1.000]")
    others = table.iloc[1:]
    assert others["p_success"].between(0, 1).all() and others["p_nodes"].between(0, 1).all()
    # RandomWalk fails on many graphs that DFS solves.
    assert table.set_index("method").loc["random_walk", "p_success"] < 0.01
    with pytest.raises(ValueError, match="reference method"):
        comparison_table(frame, reference="ppo")


def test_summarize_results_script(tmp_path, capsys):
    graphs = generate_dataset(25, seed=3, cfg=SyntheticConfig(min_nodes=10, max_nodes=30))
    frame = run_evaluation([DFS(), RandomWalk()], graphs, seeds=[0, 1])
    results = save_results(frame, tmp_path / "eval.parquet")
    spec = importlib.util.spec_from_file_location(
        "summarize", ROOT / "scripts" / "summarize_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "table.csv"
    module.main(
        ["--results", str(results), "--reference", "dfs", "--resamples", "500", "--out", str(out)]
    )
    printed = capsys.readouterr().out
    assert "p-values vs dfs" in printed and "spread across seeds" in printed
    written = pd.read_csv(out)
    assert written["method"].tolist() == ["dfs", "random_walk"]
