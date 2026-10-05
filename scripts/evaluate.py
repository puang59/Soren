#!/usr/bin/env python3
"""Evaluate baselines and trained checkpoints on a JSONL split of graph records.

Example:
    python scripts/evaluate.py --graphs data/synthetic/val.jsonl --split val \\
        --methods dfs bfs random_walk random_order line_order \\
        --checkpoints runs/synthetic-demo/best_model.zip \\
        --out runs/synthetic-demo/eval_val.parquet

Writes one row per (method, seed, graph) to the output file and prints the metrics table.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import replace

from soren.agents.searcher import PolicySearcher
from soren.agents.train import load_model
from soren.baselines.heuristic import BASELINE_FACTORIES, HeuristicScorer, make_baseline
from soren.config import load_config, load_yaml
from soren.data.schema import read_jsonl
from soren.env.cfg_nav_env import EnvConfig
from soren.env.rewards import RewardConfig
from soren.eval.metrics import breakdown, compute_metrics
from soren.eval.runner import run_evaluation, save_results


def checkpoint_seed(path: str, fallback: int) -> int:
    """Read the training seed from a path such as ``runs/x/seed3/best_model.zip``."""
    match = re.search(r"seed[_-]?(\d+)", path)
    return int(match.group(1)) if match else fallback


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphs", required=True, help="graph records (JSONL)")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--out", required=True, help="per-episode results (Parquet)")
    parser.add_argument("--methods", nargs="*", default=[], choices=sorted(BASELINE_FACTORIES))
    parser.add_argument("--checkpoints", nargs="*", default=[], help="trained models to evaluate")
    parser.add_argument("--policy-name", default="ppo", help="method name for the checkpoints")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--max-declares", type=int, help="override the declare budget (Top-k)")
    parser.add_argument("--breakdown", nargs="*", default=[], help="extra tables, e.g. size_bucket")
    parser.add_argument(
        "--protocol-b",
        action="store_true",
        help="also run each baseline with the heuristic threshold stop (Protocol B)",
    )
    parser.add_argument(
        "--eval-config", default="configs/eval.yaml", help="holds the Protocol B thresholds"
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="permit evaluating the test split; do this once, after freezing the configuration",
    )
    args = parser.parse_args(argv)
    if not args.methods and not args.checkpoints:
        parser.error("nothing to evaluate: pass --methods and/or --checkpoints")

    env_config = load_config(EnvConfig, args.env_config, section="env")
    if args.max_declares is not None:
        env_config = replace(env_config, max_declares=args.max_declares)
    reward_config = load_config(RewardConfig, args.env_config, section="reward")

    searchers = [make_baseline(name, env_config, reward_config) for name in args.methods]
    if args.protocol_b:
        thresholds = (load_yaml(args.eval_config).get("protocol_b") or {}).get("heuristic") or {}
        missing = [name for name in args.methods if name not in thresholds]
        if missing:
            parser.error(
                f"no Protocol B threshold for {', '.join(missing)} in {args.eval_config}; "
                "run scripts/tune_thresholds.py on the validation split first"
            )
        scorer = HeuristicScorer()
        searchers += [
            make_baseline(name, env_config, reward_config, scorer, thresholds[name])
            for name in args.methods
        ]
    seed_overrides = {}
    for index, path in enumerate(args.checkpoints):
        seed_overrides[len(searchers)] = checkpoint_seed(path, index)
        searchers.append(
            PolicySearcher(load_model(path), env_config, reward_config, name=args.policy_name)
        )

    frame = run_evaluation(
        searchers,
        read_jsonl(args.graphs),
        env_config,
        seeds=args.seeds,
        split=args.split,
        allow_test=args.allow_test,
        seed_overrides=seed_overrides,
    )
    path = save_results(frame, args.out)
    print(f"wrote {len(frame)} episodes to {path}\n")
    print(compute_metrics(frame).round(3).to_string(index=False))
    for column in args.breakdown:
        print(f"\nby {column}:")
        print(breakdown(frame, column).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
