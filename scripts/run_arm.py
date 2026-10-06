#!/usr/bin/env python3
"""Train one experiment arm and evaluate its best checkpoint on the validation split.

Example:
    python scripts/run_arm.py --group sweep --name wrong1.0 --seed 0 \\
        --set reward.wrong=-1.0 --set ppo.ent_coef=0.03

An arm is the default configuration plus ``--set section.key=value`` overrides, where the
section is ``ppo``, ``env`` or ``reward``. Output goes to ``runs/<group>/<name>/seed<seed>/``;
``metrics.json`` there holds the validation metrics of the best checkpoint.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from soren.agents.ppo import PPOConfig
from soren.agents.searcher import PolicySearcher
from soren.agents.train import load_model, train
from soren.config import load_config
from soren.data.schema import read_jsonl
from soren.data.transform import shuffle_labels
from soren.env.cfg_nav_env import EnvConfig
from soren.env.curriculum import CurriculumConfig
from soren.env.rewards import RewardConfig
from soren.eval.metrics import compute_metrics
from soren.eval.runner import run_evaluation, save_results


def apply_overrides(configs: dict[str, object], overrides: list[str]) -> dict[str, object]:
    for item in overrides:
        target, _, raw = item.partition("=")
        section, _, key = target.partition(".")
        if section not in configs or not key or not raw:
            raise SystemExit(f"bad override {item!r}; use section.key=value")
        if not hasattr(configs[section], key):
            raise SystemExit(f"{section} has no setting {key!r}")
        configs[section] = replace(configs[section], **{key: yaml.safe_load(raw)})
    return configs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--train", default="data/processed/graphs_train.jsonl")
    parser.add_argument("--val", default="data/processed/graphs_val.jsonl")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument(
        "--shuffle-labels",
        action="store_true",
        help="leakage control: train and validate on randomly reassigned labels",
    )
    args = parser.parse_args(argv)

    configs = {
        "ppo": load_config(PPOConfig, args.config),
        "env": load_config(EnvConfig, args.env_config, section="env"),
        "reward": load_config(RewardConfig, args.env_config, section="reward"),
        "curriculum": load_config(CurriculumConfig, args.env_config, section="curriculum"),
    }
    configs = apply_overrides(configs, args.overrides)
    if args.timesteps:
        configs["ppo"] = replace(configs["ppo"], total_timesteps=args.timesteps)

    train_graphs, val_graphs = read_jsonl(args.train), read_jsonl(args.val)
    if args.shuffle_labels:
        rng = np.random.default_rng(args.seed)
        train_graphs = [shuffle_labels(g, rng) for g in train_graphs]
        val_graphs = [shuffle_labels(g, rng) for g in val_graphs]

    run_dir = Path(args.out_dir) / args.group / args.name / f"seed{args.seed}"
    result = train(
        train_graphs,
        val_graphs,
        run_dir,
        configs["ppo"],
        configs["env"],
        configs["reward"],
        seed=args.seed,
        curriculum=configs["curriculum"],
        tensorboard=False,
    )
    model = load_model(result.best_model_path)
    searcher = PolicySearcher(model, configs["env"], configs["reward"], name=args.name)
    frame = run_evaluation(
        [searcher], val_graphs, configs["env"], split="val", seed_overrides={0: args.seed}
    )
    save_results(frame, run_dir / "eval_val.parquet")
    metrics = compute_metrics(frame).iloc[0].to_dict()
    metrics.update(
        arm=args.name,
        seed=args.seed,
        overrides=args.overrides,
        best_timesteps=result.best_metrics.get("timesteps"),
        train_success=float(
            compute_metrics(
                run_evaluation([searcher], train_graphs[:300], configs["env"], split="train")
            ).iloc[0]["success_rate"]
        ),
    )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(
        f"{args.name} seed {args.seed}: val success {metrics['success_rate']:.3f}, "
        f"train success {metrics['train_success']:.3f}, "
        f"nodes inspected {metrics['nodes_inspected_all']:.1f}"
    )


if __name__ == "__main__":
    main()
