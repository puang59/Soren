#!/usr/bin/env python3
"""Record replayable episode traces for baselines and trained checkpoints.

Example:
    python scripts/make_traces.py --graphs data/synthetic/val.jsonl --limit 20 \\
        --methods dfs bfs --checkpoints runs/synthetic-demo/best_model.zip \\
        --out runs/synthetic-demo/traces

Writes ``<out>/<method>/<graph_id>.json``, one trace per method and graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from soren.agents.searcher import PolicySearcher
from soren.agents.train import load_model
from soren.baselines import BFS, DFS, LineOrder, RandomOrder, RandomWalk
from soren.config import load_config
from soren.data.schema import read_jsonl
from soren.env.cfg_nav_env import EnvConfig
from soren.env.rewards import RewardConfig

BASELINES = {
    "dfs": lambda env, reward: DFS(env, reward),
    "random_walk": lambda env, reward: RandomWalk(env, reward),
    "bfs": lambda env, reward: BFS(reward),
    "random_order": lambda env, reward: RandomOrder(reward),
    "line_order": lambda env, reward: LineOrder(reward),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphs", required=True, help="graph records (JSONL)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--methods", nargs="*", default=[], choices=sorted(BASELINES))
    parser.add_argument("--checkpoints", nargs="*", default=[], help="trained models to trace")
    parser.add_argument("--limit", type=int, help="trace only the first N graphs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-config", default="configs/env.yaml")
    args = parser.parse_args(argv)
    if not args.methods and not args.checkpoints:
        parser.error("nothing to trace: pass --methods and/or --checkpoints")

    env_config = load_config(EnvConfig, args.env_config, section="env")
    reward_config = load_config(RewardConfig, args.env_config, section="reward")
    searchers = [BASELINES[name](env_config, reward_config) for name in args.methods]
    for index, path in enumerate(args.checkpoints):
        name = "ppo" if len(args.checkpoints) == 1 else f"ppo_{index}"
        searchers.append(PolicySearcher(load_model(path), env_config, reward_config, name=name))

    graphs = read_jsonl(args.graphs)[: args.limit]
    out_dir = Path(args.out)
    for searcher in searchers:
        rng = np.random.default_rng(args.seed)
        for graph in graphs:
            budget = env_config.max_steps_for(graph.num_nodes)
            trace = searcher.trace(graph, budget, rng)
            trace.save(out_dir / searcher.name / f"{graph.sample_id}.json")
        print(f"{searcher.name}: wrote {len(graphs)} traces to {out_dir / searcher.name}")


if __name__ == "__main__":
    main()
