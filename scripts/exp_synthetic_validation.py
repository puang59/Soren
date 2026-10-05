#!/usr/bin/env python3
"""Validate PPO on synthetic graphs, with and without a feature signal (issue #12).

Three conditions share the same graph structures and labels and differ only in how the
vulnerable statement is marked:

* ``clean``  the vulnerable statement always carries a dangerous call and nothing else does
* ``noisy``  it carries one with probability 0.9, every other statement with probability 0.1
* ``none``   it carries one as often as any other statement: features say nothing

For each condition a PPO agent is trained, the best validation checkpoint is evaluated on
held-out graphs, and BFS, DFS and RandomWalk are run on the same graphs under Protocol A.

Example:
    python scripts/exp_synthetic_validation.py --timesteps 500000 --out runs/exp_synthetic
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from soren.agents.ppo import PPOConfig
from soren.agents.searcher import evaluate_policy
from soren.agents.train import load_model, train
from soren.baselines import BFS, DFS, RandomWalk
from soren.config import load_config
from soren.data.features import TIER_L_NAMES, featurize
from soren.data.schema import GraphRecord
from soren.data.synthetic import SyntheticConfig, generate_dataset
from soren.env.cfg_nav_env import EnvConfig
from soren.env.rewards import RewardConfig
from soren.eval.metrics import summarize

CONDITIONS = {
    "clean": {"signal": 1.0, "hit_rate": 1.0, "decoy_rate": 0.0},
    "noisy": {"signal": 1.0, "hit_rate": 0.9, "decoy_rate": 0.1},
    "none": {"signal": 0.0, "hit_rate": 0.9, "decoy_rate": 0.1},
}
CANDIDATE_KINDS = ("ASSIGN", "CALL", "DECL")
DANGEROUS = TIER_L_NAMES.index("dangerous_call")


def feature_ceiling(graphs: list[GraphRecord], cfg: SyntheticConfig) -> float:
    """Best success rate any single declaration can reach from the features alone.

    Given every node's features, the best guess is a statement with a dangerous call if there
    is one. The ceiling is the posterior probability of that guess, averaged over graphs.
    """
    p_vuln, p_other = cfg.vuln_call_rate, cfg.decoy_rate
    total = 0.0
    for graph in graphs:
        flagged = featurize(graph, "L")[:, DANGEROUS]
        candidates = [n.id for n in graph.nodes if n.kind in CANDIDATE_KINDS]
        weights = []
        for node in candidates:
            if flagged[node]:
                weights.append(p_vuln / p_other if p_other else np.inf)
            else:
                weights.append((1 - p_vuln) / (1 - p_other) if p_other < 1 else np.inf)
        weights = np.array(weights)
        if np.isinf(weights).any():
            total += 1.0 / np.isinf(weights).sum()
        elif weights.sum() == 0:
            total += 1.0 / len(weights)
        else:
            total += weights.max() / weights.sum()
    return total / len(graphs)


def run_baselines(graphs, env_cfg, reward_cfg, seed) -> dict[str, dict[str, float]]:
    out = {}
    for searcher in (DFS(env_cfg, reward_cfg), BFS(reward_cfg), RandomWalk(env_cfg, reward_cfg)):
        rng = np.random.default_rng(seed)
        results = [searcher.run(g, env_cfg.max_steps_for(g.num_nodes), rng) for g in graphs]
        out[searcher.name] = summarize(results)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="runs/exp_synthetic")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--min-nodes", type=int, default=SyntheticConfig.min_nodes)
    parser.add_argument("--max-nodes", type=int, default=SyntheticConfig.max_nodes)
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--ent-coef", type=float)
    parser.add_argument("--max-declares", type=int)
    args = parser.parse_args(argv)

    ppo_cfg = replace(load_config(PPOConfig, args.config), total_timesteps=args.timesteps)
    if args.ent_coef is not None:
        ppo_cfg = replace(ppo_cfg, ent_coef=args.ent_coef)
    env_cfg = load_config(EnvConfig, args.env_config, section="env")
    if args.max_declares is not None:
        env_cfg = replace(env_cfg, max_declares=args.max_declares)
    reward_cfg = load_config(RewardConfig, args.env_config, section="reward")
    out_dir = Path(args.out)
    report: dict[str, dict] = {}

    for name in args.conditions:
        syn = SyntheticConfig(
            min_nodes=args.min_nodes, max_nodes=args.max_nodes, **CONDITIONS[name]
        )
        # Fixed dataset seeds: every condition sees the same structures and labels.
        train_graphs = generate_dataset(args.n_train, seed=100, cfg=syn, prefix="train")
        val_graphs = generate_dataset(args.n_val, seed=101, cfg=syn, prefix="val")
        test_graphs = generate_dataset(args.n_test, seed=102, cfg=syn, prefix="test")

        entry = {
            "synthetic": CONDITIONS[name],
            "feature_ceiling": feature_ceiling(test_graphs, syn),
            "mean_nodes": float(np.mean([g.num_nodes for g in test_graphs])),
            "baselines": run_baselines(test_graphs, env_cfg, reward_cfg, seed=0),
            "ppo": {},
        }
        for seed in args.seeds:
            run_dir = out_dir / name / f"seed{seed}"
            result = train(train_graphs, val_graphs, run_dir, ppo_cfg, env_cfg, reward_cfg, seed)
            model = load_model(result.best_model_path)
            entry["ppo"][str(seed)] = {
                "best_val": result.best_metrics,
                "test": summarize(evaluate_policy(model, test_graphs, env_cfg, reward_cfg)),
            }
            print(f"[{name} seed {seed}] test: {json.dumps(entry['ppo'][str(seed)]['test'])}")
        report[name] = entry
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print()
    print("| condition | method | success | nodes inspected | actions | return |")
    print("|---|---|---|---|---|---|")
    for name, entry in report.items():
        rows = [(f"ppo (seed {s})", r["test"]) for s, r in entry["ppo"].items()]
        rows += list(entry["baselines"].items())
        for method, m in rows:
            print(
                f"| {name} | {method} | {m['success_rate']:.3f} | {m['nodes_inspected']:.2f} "
                f"| {m['actions_taken']:.2f} | {m['cumulative_reward']:.3f} |"
            )
        print(f"| {name} | feature ceiling | {entry['feature_ceiling']:.3f} | | | |")


if __name__ == "__main__":
    main()
