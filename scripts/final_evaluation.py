#!/usr/bin/env python3
"""The single test-set evaluation: every method, both protocols, with statistics.

Example:
    python scripts/final_evaluation.py --allow-test
    python scripts/final_evaluation.py --allow-test --runs runs/v2 --explain

Run this once, after the configuration is frozen. It evaluates, on the test split:

* the baselines under Protocol A (oracle stop) and Protocol B (heuristic threshold stop)
* the final PPO checkpoints, and the three- and five-declaration checkpoints
* the node classifier's top-k ranking, with the heuristic score and a random ranking
* each PPO strategy on randomly reassigned labels, as its own chance reference

and writes per-episode results, metric tables and confidence intervals to ``--out``.
``--explain`` prints how the headline numbers and tests are calculated.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from soren.agents.searcher import PolicySearcher, evaluate_policy
from soren.agents.train import load_model
from soren.baselines.heuristic import BASELINE_FACTORIES, HeuristicScorer, make_baseline
from soren.baselines.node_classifier import NodeClassifier, chance_top_k, top_k_accuracy
from soren.config import load_config, load_yaml
from soren.data.schema import read_jsonl
from soren.data.transform import shuffle_labels
from soren.env.cfg_nav_env import EnvConfig
from soren.env.rewards import RewardConfig
from soren.eval.metrics import breakdown, compute_metrics, metrics_across_seeds
from soren.eval.runner import TestSplitError, run_evaluation, save_results
from soren.eval.stats import (
    bootstrap_ci,
    comparison_table,
    confidence_intervals,
    mcnemar_success,
    per_graph,
    wilcoxon_nodes_inspected,
)
from soren.mathlog import log

PPO_GROUPS = (  # method name, checkpoint pattern under --runs, declare budget
    ("ppo", "final/base/seed*/best_model.npz", 1),
    ("ppo_top3", "ablations/declares_3/seed*/best_model.npz", 3),
    ("ppo_top5", "ablations/declares_5/seed*/best_model.npz", 5),
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphs", default="data/processed/graphs_test.jsonl")
    parser.add_argument("--out", default="experiments/test")
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--runs", default="runs/v2", help="directory holding the trained runs")
    parser.add_argument("--classifiers", default="classifier/tier_L_seed*.npz", help="under --runs")
    parser.add_argument("--explain", action="store_true", help="show how the numbers are computed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_test:
        raise TestSplitError("this evaluates the test split; pass --allow-test to confirm")

    env = load_config(EnvConfig, args.env_config, section="env")
    reward = load_config(RewardConfig, args.env_config, section="reward")
    thresholds = load_yaml(args.eval_config)["protocol_b"]["heuristic"]
    graphs = read_jsonl(args.graphs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scorer = HeuristicScorer()
    searchers = [make_baseline(name, env, reward) for name in sorted(BASELINE_FACTORIES)]
    searchers += [
        make_baseline(name, env, reward, scorer, thresholds[name])
        for name in sorted(BASELINE_FACTORIES)
    ]
    seed_overrides: dict[int, int] = {}
    models: dict[str, list] = {}
    for method, pattern, budget in PPO_GROUPS:
        method_env = replace(env, max_declares=budget)
        for seed, path in enumerate(sorted(Path(args.runs).glob(pattern))):
            model = load_model(path)
            models.setdefault(method, []).append((model, method_env))
            seed_overrides[len(searchers)] = seed
            searchers.append(PolicySearcher(model, method_env, reward, name=method))

    frame = run_evaluation(
        searchers, graphs, env, seeds=args.seeds, split="test", allow_test=True,
        seed_overrides=seed_overrides,
    )  # fmt: skip
    save_results(frame, out / "results.parquet")
    compute_metrics(frame).round(4).to_csv(out / "metrics.csv", index=False)
    metrics_across_seeds(frame).round(4).to_csv(out / "metrics_across_seeds.csv", index=False)
    confidence_intervals(frame).round(4).to_csv(out / "confidence_intervals.csv", index=False)
    declaring = frame[frame["method"].str.contains(r"\+heuristic") | (frame["method"] == "ppo")]
    comparison_table(declaring, reference="ppo").to_csv(out / "comparison_vs_ppo.csv", index=False)
    for column in ("size_bucket", "vuln_bucket", "cwe"):
        breakdown(declaring, column).round(4).to_csv(out / f"by_{column}.csv", index=False)

    # Ranking references: what the features support when every node is visible.
    ranking = {"random": chance_top_k(graphs)}
    ranking["heuristic"] = top_k_accuracy([scorer.scores(g) for g in graphs], graphs)
    classifier_scores = [
        NodeClassifier.load(path).top_k(graphs)
        for path in sorted(Path(args.runs).glob(args.classifiers))
    ]
    if classifier_scores:
        ranking["classifier"] = pd.DataFrame(classifier_scores).mean().to_dict()
        ranking["classifier_seeds"] = len(classifier_scores)

    # Each PPO strategy on random labels: its own chance level, free of selection effects.
    random_labels = {}
    for method, entries in models.items():
        scores = []
        for model, method_env in entries:
            for k in range(5):
                rng = np.random.default_rng(5000 + k)
                shuffled = [shuffle_labels(g, rng) for g in graphs]
                results = evaluate_policy(model, shuffled, method_env, reward)
                scores.append(np.mean([r.success for r in results]))
        random_labels[method] = float(np.mean(scores))

    summary = {
        "graphs": len(graphs),
        "ppo_seeds": {method: len(entries) for method, entries in models.items()},
        "ranking_top_k": ranking,
        "ppo_success_on_random_labels": random_labels,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    columns = ["method", "episodes", "localization_accuracy", "success_rate"]
    columns += ["nodes_inspected_success", "cumulative_reward"]
    print(compute_metrics(frame)[columns].round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}/")
    if args.explain:
        explain_headline(frame, declaring)


def explain_headline(frame: pd.DataFrame, declaring: pd.DataFrame) -> None:
    """Show, step by step, how the headline success rate and the tests against PPO arise."""
    log.enabled = True
    log.heading("How the headline numbers are calculated")
    table = per_graph(frame)
    ppo = table[table["method"] == "ppo"]
    episodes = frame[frame["method"] == "ppo"]
    log.section("success rate of the PPO agent")
    log.text(
        f"{len(episodes)} episodes = {ppo.shape[0]} graphs × "
        f"{episodes['seed'].nunique()} trained seeds; one deterministic episode each"
    )
    log.formula(
        "per graph",
        "s_g = (episodes on g that end in a correct declaration) / seeds",
        None,
        "values in {0, 0.2, …, 1}",
    )
    log.formula(
        "success rate",
        "S = Σ_g s_g / G",
        f"{ppo['success'].sum():.1f} / {len(ppo)}",
        f"{ppo['success'].mean():.4f}",
    )
    bootstrap_ci(ppo["success"].to_numpy(), explain="PPO success rate")
    for method in sorted(m for m in declaring["method"].unique() if m != "ppo"):
        mcnemar_success(declaring, "ppo", method, explain=True)
    wilcoxon_nodes_inspected(declaring, "ppo", "dfs+heuristic", explain=True)


if __name__ == "__main__":
    main()
