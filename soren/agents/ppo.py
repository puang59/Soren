"""Masked PPO: configuration, vectorised environments and model construction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize

from soren.data.schema import GraphRecord
from soren.env.cfg_nav_env import CFGNavEnv, EnvConfig
from soren.env.curriculum import CurriculumConfig, CurriculumSampler
from soren.env.rewards import RewardConfig

_ACTIVATIONS = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}


@dataclass
class PPOConfig:
    total_timesteps: int = 2_000_000
    n_envs: int = 16
    n_steps: int = 512
    batch_size: int = 512
    n_epochs: int = 10
    learning_rate: float = 3e-4
    lr_schedule: str = "linear"  # "linear" decays to zero; "constant" keeps it fixed
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "tanh"
    normalize_reward: bool = True
    vec_env: str = "subproc"  # "subproc" or "dummy"
    device: str = "cpu"  # the MLP policy is faster on CPU than on a GPU
    eval_freq: int = 50_000
    """Timesteps between evaluations on the validation split."""

    def __post_init__(self) -> None:
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}")
        if self.lr_schedule not in ("linear", "constant"):
            raise ValueError("lr_schedule must be 'linear' or 'constant'")
        if self.vec_env not in ("subproc", "dummy"):
            raise ValueError("vec_env must be 'subproc' or 'dummy'")


def learning_rate_schedule(cfg: PPOConfig) -> float | Callable[[float], float]:
    if cfg.lr_schedule == "constant":
        return cfg.learning_rate
    return lambda progress_remaining: cfg.learning_rate * progress_remaining


def make_vec_env(
    graphs: Sequence[GraphRecord],
    env_config: EnvConfig,
    reward_config: RewardConfig,
    cfg: PPOConfig,
    seed: int = 0,
    curriculum: CurriculumConfig | None = None,
) -> VecEnv:
    """Build ``cfg.n_envs`` monitored environments over ``graphs``.

    With an enabled ``curriculum``, each environment gets its own size-capped sampler.
    """
    graphs = list(graphs)
    use_curriculum = curriculum is not None and curriculum.enabled

    def factory() -> Monitor:
        sampler = CurriculumSampler(graphs, curriculum) if use_curriculum else None
        return Monitor(CFGNavEnv(graphs, env_config, reward_config, sampler))

    factories = [factory] * cfg.n_envs
    venv: VecEnv = SubprocVecEnv(factories) if cfg.vec_env == "subproc" else DummyVecEnv(factories)
    venv.seed(seed)
    if cfg.normalize_reward:
        # Observations are already scaled to [0, 1]; only rewards are normalised.
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=cfg.gamma)
    return venv


def build_model(
    venv: VecEnv, cfg: PPOConfig, seed: int = 0, tensorboard_log: str | None = None
) -> MaskablePPO:
    return MaskablePPO(
        "MlpPolicy",
        venv,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        learning_rate=learning_rate_schedule(cfg),
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs={
            "net_arch": {"pi": list(cfg.net_arch), "vf": list(cfg.net_arch)},
            "activation_fn": _ACTIVATIONS[cfg.activation],
        },
        seed=seed,
        device=cfg.device,
        tensorboard_log=tensorboard_log,
        verbose=0,
    )
