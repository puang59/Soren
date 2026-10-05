"""Train a masked PPO agent and select the best checkpoint on the validation split."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList

from soren.agents.callbacks import DiagnosticsCallback, ValidationCallback
from soren.agents.ppo import PPOConfig, build_model, make_vec_env
from soren.data.schema import GraphRecord
from soren.env.cfg_nav_env import EnvConfig
from soren.env.curriculum import CurriculumConfig
from soren.env.rewards import RewardConfig


@dataclass
class TrainResult:
    run_dir: Path
    model: MaskablePPO
    best_model_path: Path
    final_model_path: Path
    best_metrics: dict[str, Any]
    history: list[dict[str, Any]]


def train(
    train_graphs: Sequence[GraphRecord],
    val_graphs: Sequence[GraphRecord],
    run_dir: str | Path,
    ppo_config: PPOConfig | None = None,
    env_config: EnvConfig | None = None,
    reward_config: RewardConfig | None = None,
    seed: int = 0,
    curriculum: CurriculumConfig | None = None,
    tensorboard: bool = True,
) -> TrainResult:
    """Train on ``train_graphs``; evaluate on ``val_graphs`` every ``eval_freq`` timesteps.

    Everything needed to reproduce the run is written to ``run_dir``: the resolved
    configuration, the best and final checkpoints, the validation history and TensorBoard logs.
    """
    ppo_config = ppo_config or PPOConfig()
    env_config = env_config or EnvConfig()
    # The shaping term must discount with the same gamma as the learner.
    reward_config = replace(reward_config or RewardConfig(), gamma=ppo_config.gamma)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "train_graphs": len(train_graphs),
                "val_graphs": len(val_graphs),
                "ppo": asdict(ppo_config),
                "env": asdict(env_config),
                "reward": asdict(reward_config),
                "curriculum": asdict(curriculum or CurriculumConfig()),
            },
            indent=2,
        )
    )

    venv = make_vec_env(train_graphs, env_config, reward_config, ppo_config, seed, curriculum)
    try:
        model = build_model(
            venv, ppo_config, seed, tensorboard_log=str(run_dir / "tb") if tensorboard else None
        )
        validation = ValidationCallback(
            val_graphs, env_config, reward_config, ppo_config.eval_freq, run_dir
        )
        callbacks = CallbackList([validation, DiagnosticsCallback(env_config.k)])
        model.learn(total_timesteps=ppo_config.total_timesteps, callback=callbacks)
        final_path = run_dir / "final_model.zip"
        model.save(final_path)
    finally:
        venv.close()

    return TrainResult(
        run_dir=run_dir,
        model=model,
        best_model_path=run_dir / "best_model.zip",
        final_model_path=final_path,
        best_metrics=validation.best_metrics or {},
        history=validation.history,
    )


def load_model(path: str | Path, device: str = "cpu") -> MaskablePPO:
    return MaskablePPO.load(path, device=device)
