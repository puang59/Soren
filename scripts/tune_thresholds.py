#!/usr/bin/env python3
"""Tune the Protocol B declaration threshold of each baseline on the validation split.

Example:
    python scripts/tune_thresholds.py --graphs data/processed/graphs_val.jsonl \\
        --curve runs/protocol_b/threshold_curve.csv --write-config

For every baseline and every candidate threshold, the baseline walks in its own order and
declares at the first node whose heuristic score reaches the threshold. The threshold with the
highest success rate is kept (ties: fewer nodes inspected). Never run this on the test split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from soren.baselines.heuristic import (
    BASELINE_FACTORIES,
    HeuristicScorer,
    best_threshold,
    threshold_curve,
)
from soren.config import load_config, load_yaml
from soren.data.schema import read_jsonl
from soren.env.cfg_nav_env import EnvConfig
from soren.env.rewards import RewardConfig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphs", required=True, help="validation graphs (JSONL)")
    parser.add_argument("--methods", nargs="*", default=sorted(BASELINE_FACTORIES))
    parser.add_argument("--steps", type=int, default=19, help="thresholds between 0.05 and 0.95")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--curve", help="write the full threshold curve to this CSV file")
    parser.add_argument(
        "--write-config", action="store_true", help="store the thresholds in --eval-config"
    )
    args = parser.parse_args(argv)
    if "test" in Path(args.graphs).stem.split("_"):
        parser.error("thresholds are tuned on the validation split, never on test")

    env_config = load_config(EnvConfig, args.env_config, section="env")
    reward_config = load_config(RewardConfig, args.env_config, section="reward")
    graphs = read_jsonl(args.graphs)
    scorer = HeuristicScorer()
    candidates = np.round(np.linspace(0.05, 0.95, args.steps), 4)

    curves = []
    chosen: dict[str, float] = {}
    for method in args.methods:
        curve = threshold_curve(
            method, scorer, graphs, candidates, env_config, reward_config, args.seeds
        )
        curves.append(curve)
        chosen[method] = best_threshold(curve)
        best = curve[curve["threshold"] == chosen[method]].iloc[0]
        print(
            f"{method:<16} threshold {chosen[method]:.2f}  success {best['success_rate']:.3f}  "
            f"nodes inspected {best['nodes_inspected']:.2f}"
        )

    if args.curve:
        path = Path(args.curve)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(curves, ignore_index=True).to_csv(path, index=False)
        print(f"wrote {path}")
    if args.write_config:
        config = load_yaml(args.eval_config)
        config.setdefault("protocol_b", {})[scorer.name] = chosen
        config["protocol_b"]["tuned_on"] = str(args.graphs)
        Path(args.eval_config).write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"wrote thresholds to {args.eval_config}")


if __name__ == "__main__":
    main()
