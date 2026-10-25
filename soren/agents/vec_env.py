"""Run several copies of the environment side by side, and normalise their rewards.

PPO learns from batches of experience. Stepping ``n`` environments together gives ``n``
transitions per policy call, and episodes from different functions in the same batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from soren.env.cfg_nav_env import CFGNavEnv
from soren.mathlog import log, num


class VecEnv:
    """Steps a list of environments one after another and restarts each when it finishes.

    When an episode ends, ``step`` returns the first observation of the next episode in its
    place and puts the last one in ``info["terminal_observation"]``, so the batch never has a
    hole in it.
    """

    def __init__(self, envs: Sequence[CFGNavEnv], seed: int = 0) -> None:
        if not envs:
            raise ValueError("VecEnv needs at least one environment")
        self.envs = list(envs)
        self.seed = seed
        self.num_envs = len(self.envs)
        self.obs_dim = self.envs[0].observation_space.shape[0]
        self.n_actions = self.envs[0].action_space.n

    def reset(self) -> np.ndarray:
        # Environment i starts from seed + i, so the copies sample different functions.
        return np.stack([env.reset(seed=self.seed + i)[0] for i, env in enumerate(self.envs)])

    def action_masks(self) -> np.ndarray:
        return np.stack([env.action_masks() for env in self.envs])

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        observations, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions, strict=True):
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            if done:
                info = {
                    **info,
                    "terminal_observation": obs,
                    # A timeout is not a real end: the value of the state is not zero.
                    "truncated": truncated and not terminated,
                }
                obs, _ = env.reset()
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return np.stack(observations), np.array(rewards), np.array(dones), infos


class RunningMeanStd:
    """Mean and variance of a stream of numbers, updated batch by batch.

    Combining the running ``(mean, var, count)`` with a batch's ``(μ_b, σ²_b, n_b)``
    (Chan et al., 1979):

        δ = μ_b − mean,   N = count + n_b
        mean ← mean + δ·n_b/N
        var  ← (var·count + σ²_b·n_b + δ²·count·n_b/N) / N
    """

    def __init__(self) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, batch: np.ndarray) -> None:
        batch_mean, batch_var, n = float(np.mean(batch)), float(np.var(batch)), batch.size
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.var = (
            self.var * self.count + batch_var * n + delta**2 * self.count * n / total
        ) / total
        self.count = total


class RewardNormalizer:
    """Divide rewards by the running standard deviation of the discounted return.

    Raw rewards here range from −0.03 to +1. Scaling them so that returns have unit variance
    keeps the value targets, and so the size of the gradients, in a steady range. The rewards
    are only scaled, never shifted, so their sign and the best policy do not change.

        G ← γ·G + r          running discounted return of each environment (0 at a new episode)
        r̃ = clip(r / √(Var[G] + ε), −10, 10)
    """

    def __init__(self, num_envs: int, gamma: float, clip: float = 10.0, eps: float = 1e-8):
        self.returns = np.zeros(num_envs)
        self.gamma = gamma
        self.clip = clip
        self.eps = eps
        self.rms = RunningMeanStd()

    def __call__(self, rewards: np.ndarray, dones: np.ndarray, explain: bool = False) -> np.ndarray:
        previous = self.returns
        self.returns = self.returns * self.gamma + rewards
        self.rms.update(self.returns)
        scale = np.sqrt(self.rms.var + self.eps)
        normalised = np.clip(rewards / scale, -self.clip, self.clip)
        if explain:
            log.section("reward normalisation (environment 0, this step)")
            log.formula(
                "running return",
                "G ← γ·G + r",
                f"{self.gamma}×{num(previous[0], 4)} + {num(rewards[0], 4)}",
                f"{self.returns[0]:.4f}",
            )
            log.formula(
                "scale",
                "σ = √(Var[G] + ε)",
                f"√({self.rms.var:.4f} + 1e-8)",
                f"{scale:.4f}   (over {self.rms.count:,.0f} values of G seen so far)",
            )
            log.formula(
                "normalised reward",
                "r̃ = clip(r / σ, −10, 10)",
                f"{num(rewards[0], 4)} / {scale:.4f}",
                f"{normalised[0]:.4f}",
            )
        self.returns[dones] = 0.0
        return normalised
