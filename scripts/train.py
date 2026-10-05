#!/usr/bin/env python3
"""Train a masked PPO agent on a JSONL dataset of graph records.

Example:
    python -m soren.data.synthetic --out data/synthetic/train.jsonl --n 2000 --seed 0
    python -m soren.data.synthetic --out data/synthetic/val.jsonl --n 200 --seed 1
    python scripts/train.py --train data/synthetic/train.jsonl --val data/synthetic/val.jsonl \\
        --name synthetic-demo --timesteps 300000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from soren.agents.ppo import PPOConfig
from soren.agents.train import train
from soren.config import load_config
from soren.data.schema import read_jsonl
from soren.env.cfg_nav_env import EnvConfig
from soren.env.rewards import RewardConfig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True, help="training graphs (JSONL)")
    parser.add_argument("--val", required=True, help="validation graphs (JSONL)")
    parser.add_argument("--name", required=True, help="run name; output goes to <out-dir>/<name>")
    parser.add_argument("--config", default="configs/ppo.yaml", help="PPO hyperparameters")
    parser.add_argument("--env-config", default="configs/env.yaml", help="env and reward settings")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, help="override total_timesteps")
    parser.add_argument("--n-envs", type=int, help="override n_envs")
    parser.add_argument("--vec-env", choices=["subproc", "dummy"], help="override vec_env")
    args = parser.parse_args(argv)

    ppo_config = load_config(PPOConfig, args.config)
    overrides = {
        "total_timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "vec_env": args.vec_env,
    }
    ppo_config = replace(ppo_config, **{k: v for k, v in overrides.items() if v is not None})
    env_config = load_config(EnvConfig, args.env_config, section="env")
    reward_config = load_config(RewardConfig, args.env_config, section="reward")

    train_graphs = read_jsonl(args.train)
    val_graphs = read_jsonl(args.val)
    run_dir = Path(args.out_dir) / args.name
    print(
        f"training on {len(train_graphs)} graphs, validating on {len(val_graphs)}; run: {run_dir}"
    )

    result = train(
        train_graphs, val_graphs, run_dir, ppo_config, env_config, reward_config, seed=args.seed
    )
    print("best validation metrics:")
    print(json.dumps(result.best_metrics, indent=2))
    print(f"best checkpoint: {result.best_model_path}")


if __name__ == "__main__":
    main()
