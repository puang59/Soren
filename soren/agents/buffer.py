"""Store one rollout of experience and turn its rewards into advantages.

A rollout is ``n_steps`` steps of ``n_envs`` environments. After it, every step gets an
*advantage* (how much better the action was than the critic expected) and a *return target*
(what the critic should have predicted), both computed by Generalised Advantage Estimation
(GAE; Schulman et al., 2016):

    δ_t = r_t + γ·V(s_{t+1})·(1 − done_t) − V(s_t)       one-step surprise (TD error)
    A_t = δ_t + γ·λ·(1 − done_t)·A_{t+1}                  discounted sum of future surprises
    R_t = A_t + V(s_t)                                    target for the critic

``λ`` trades bias for variance: ``λ = 0`` trusts the critic fully (``A_t = δ_t``), ``λ = 1``
uses the actual rewards to the end of the episode.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from soren.mathlog import log, num


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    lam: float,
    explain: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and returns for arrays shaped ``[n_steps, n_envs]``, walking backwards.

    ``dones[t]`` is True when the episode ended at step ``t``; ``last_values`` is ``V`` of the
    observation after the final step.
    """
    n_steps = len(rewards)
    advantages = np.zeros_like(rewards, dtype=np.float64)
    next_advantage = np.zeros(rewards.shape[1])
    for t in reversed(range(n_steps)):
        next_value = last_values if t == n_steps - 1 else values[t + 1]
        alive = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * alive - values[t]
        advantages[t] = next_advantage = delta + gamma * lam * alive * next_advantage
        if explain and t >= n_steps - 2:
            _explain_step(t, rewards, values, dones, next_value, delta, advantages, gamma, lam)
    return advantages, advantages + values


def _explain_step(t, rewards, values, dones, next_value, delta, advantages, gamma, lam) -> None:
    r, v, nv, d = rewards[t, 0], values[t, 0], next_value[0], int(dones[t, 0])
    a_next = 0.0 if t == len(rewards) - 1 else advantages[t + 1, 0]
    log.section(f"GAE at step t={t} of environment 0 (computed backwards from the last step)")
    log.formula(
        "TD error",
        "δ_t = r_t + γ·V(s_{t+1})·(1 − done) − V(s_t)",
        f"{num(r, 4)} + {gamma}×{num(nv, 4)}×{1 - d} − {num(v, 4)}",
        f"{delta[0]:.4f}",
    )
    log.formula(
        "advantage",
        "A_t = δ_t + γλ·(1 − done)·A_{t+1}",
        f"{num(delta[0], 4)} + {gamma}×{lam}×{1 - d}×{num(a_next, 4)}",
        f"{advantages[t, 0]:.4f}",
    )
    log.formula(
        "return target",
        "R_t = A_t + V(s_t)",
        f"{num(advantages[t, 0], 4)} + {num(v, 4)}",
        f"{advantages[t, 0] + v:.4f}",
    )


class RolloutBuffer:
    def __init__(self, n_steps: int, n_envs: int, obs_dim: int, n_actions: int) -> None:
        self.n_steps, self.n_envs = n_steps, n_envs
        self.obs = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.masks = np.zeros((n_steps, n_envs, n_actions), dtype=bool)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.log_probs = np.zeros((n_steps, n_envs))
        self.values = np.zeros((n_steps, n_envs))
        self.rewards = np.zeros((n_steps, n_envs))
        self.dones = np.zeros((n_steps, n_envs), dtype=bool)
        self.advantages = np.zeros((n_steps, n_envs))
        self.returns = np.zeros((n_steps, n_envs))

    def add(self, t: int, obs, masks, actions, log_probs, values, rewards, dones) -> None:
        self.obs[t], self.masks[t], self.actions[t] = obs, masks, actions
        self.log_probs[t], self.values[t] = log_probs, values
        self.rewards[t], self.dones[t] = rewards, dones

    def finish(self, last_values: np.ndarray, gamma: float, lam: float, explain: bool = False):
        self.advantages, self.returns = compute_gae(
            self.rewards, self.values, self.dones, last_values, gamma, lam, explain
        )

    def minibatches(self, batch_size: int, rng: np.random.Generator) -> Iterator[dict]:
        """The whole rollout, shuffled, in minibatches of ``batch_size`` transitions."""
        total = self.n_steps * self.n_envs
        flat = {
            "obs": self.obs.reshape(total, -1),
            "masks": self.masks.reshape(total, -1),
            "actions": self.actions.reshape(total),
            "old_log_probs": self.log_probs.reshape(total),
            "old_values": self.values.reshape(total),
            "advantages": self.advantages.reshape(total),
            "returns": self.returns.reshape(total),
        }
        order = rng.permutation(total)
        for start in range(0, total, batch_size):
            index = order[start : start + batch_size]
            yield {key: array[index] for key, array in flat.items()}
