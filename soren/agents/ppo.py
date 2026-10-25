"""Proximal Policy Optimisation with action masking, written from scratch.

PPO (Schulman et al., 2017) repeats two phases:

1. **Collect** ``n_steps`` steps in each of ``n_envs`` environments with the current policy,
   storing observation, action, ``log π_old(a|s)``, ``V(s)`` and reward. Then compute an
   advantage ``A_t`` and a return target ``R_t`` for every step (``soren.agents.buffer``).
2. **Learn** from that batch for ``n_epochs`` passes in shuffled minibatches, minimising

       L = L_clip + c_v · L_V − c_e · H

   L_clip = −mean( min( r·Â, clip(r, 1−ε, 1+ε)·Â ) )   with r = π_new(a|s) / π_old(a|s)
   L_V    = mean( (R − V(s))² )                        the critic's squared error
   H      = mean( −Σ_a π(a|s)·log π(a|s) )             entropy bonus: keep exploring

The ratio ``r`` says how much more (or less) likely the action has become since the data was
collected. Multiplying by the advantage raises the probability of actions that did better
than expected and lowers the others. Clipping ``r`` to ``[1−ε, 1+ε]`` removes the incentive
to move further once the policy has changed by ``ε`` on a sample, so one batch of data cannot
push the policy too far: that is the "proximal" part.

Gradients are derived by hand (see :func:`ppo_loss`) and backpropagated through the network
by ``soren.nn``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from soren.agents.buffer import RolloutBuffer
from soren.agents.policy import ActorCritic
from soren.agents.vec_env import RewardNormalizer, VecEnv
from soren.data.schema import GraphRecord
from soren.env.cfg_nav_env import CFGNavEnv, EnvConfig
from soren.env.curriculum import CurriculumConfig, CurriculumSampler
from soren.env.rewards import RewardConfig
from soren.mathlog import log, num
from soren.nn import Adam, clip_grad_norm, entropy

_ACTIVATIONS = ("tanh", "relu")


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
    adam_eps: float = 1e-5
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "tanh"
    normalize_reward: bool = True
    eval_freq: int = 50_000
    """Timesteps between evaluations on the validation split."""

    def __post_init__(self) -> None:
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}")
        if self.lr_schedule not in ("linear", "constant"):
            raise ValueError("lr_schedule must be 'linear' or 'constant'")


def learning_rate_schedule(cfg: PPOConfig) -> float | Callable[[float], float]:
    """``lr = lr₀ · (1 − timesteps / total)`` for "linear"; a constant otherwise."""
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
    """``cfg.n_envs`` environments over ``graphs``; each gets its own curriculum if enabled."""
    graphs = list(graphs)
    use_curriculum = curriculum is not None and curriculum.enabled
    envs = [
        CFGNavEnv(
            graphs,
            env_config,
            reward_config,
            CurriculumSampler(graphs, curriculum) if use_curriculum else None,
        )
        for _ in range(cfg.n_envs)
    ]
    return VecEnv(envs, seed)


# ------------------------------------------------------------------------------ the loss


def ppo_loss(
    policy: ActorCritic,
    batch: dict[str, np.ndarray],
    clip_range: float,
    ent_coef: float,
    vf_coef: float,
    explain: str | None = None,
) -> dict[str, float]:
    """Compute the PPO loss on a minibatch and fill ``policy.grads`` with its gradient.

    The gradient with respect to the actor's logits ``z`` (for the action ``a`` taken), per
    sample and divided by the batch size ``B``:

        ∂L_clip/∂log π(a)  = −r·Â   if the unclipped term is the smaller one, else 0
        ∂log π(a)/∂z_j     = 1[j = a] − p_j                       (derivative of log-softmax)
        ∂(−H)/∂z_j         = p_j · (log p_j + H)

    and with respect to the critic's output: ``∂L/∂V = c_v · 2·(V − R)``. Each network then
    backpropagates its own part.

    ``explain`` is a heading; with it, the calculation for one sample is printed.
    """
    obs, masks, actions = batch["obs"], batch["masks"], batch["actions"]
    old_log_p, returns = batch["old_log_probs"], batch["returns"]
    size = len(actions)
    rows = np.arange(size)

    # Advantages are normalised per minibatch: only their relative size matters.
    raw_adv = batch["advantages"]
    adv = raw_adv
    if size > 1:
        adv = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-8)

    log_p_all = policy.log_probs(obs, masks)
    p = np.exp(log_p_all)
    log_p = log_p_all[rows, actions]
    ratio = np.exp(log_p - old_log_p)
    clipped = np.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
    surrogate = np.minimum(ratio * adv, clipped * adv)
    policy_loss = -surrogate.mean()

    h = entropy(log_p_all)
    entropy_loss = -h.mean()

    values = policy.values(obs)
    value_loss = np.mean((returns - values) ** 2)
    loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss

    # Backward pass, by hand.
    unclipped_active = ratio * adv <= clipped * adv
    d_log_p = np.where(unclipped_active, -ratio * adv, 0.0) / size
    one_hot = np.zeros_like(p)
    one_hot[rows, actions] = 1.0
    safe_log_p = np.where(p > 0, log_p_all, 0.0)
    d_logits = d_log_p[:, None] * (one_hot - p)
    d_logits += ent_coef * p * (safe_log_p + h[:, None]) / size
    d_logits[~masks] = 0.0  # masked logits are constants, not network outputs
    policy.actor.backward(d_logits)
    policy.critic.backward((vf_coef * 2.0 * (values - returns) / size)[:, None])

    stats = {
        "loss": float(loss),
        "policy_loss": float(policy_loss),
        "value_loss": float(value_loss),
        "entropy": float(h.mean()),
        "approx_kl": float(np.mean((ratio - 1.0) - np.log(ratio))),
        "clip_fraction": float(np.mean(np.abs(ratio - 1.0) > clip_range)),
    }
    if explain:
        _explain_loss(
            explain, batch, raw_adv, adv, p, log_p_all, ratio, clipped, values, h, d_logits,
            stats, clip_range, ent_coef, vf_coef,
        )  # fmt: skip
    return stats


def _explain_loss(
    title, batch, raw_adv, adv, p, log_p_all, ratio, clipped, values, h, d_logits, stats,
    clip_range, ent_coef, vf_coef,
) -> None:  # fmt: skip
    # Show a sample where clipping is active if there is one, it is the interesting case;
    # otherwise one where the agent had a choice.
    outside = np.flatnonzero(np.abs(ratio - 1.0) > clip_range)
    choice = np.flatnonzero((p > 0).sum(axis=1) >= 2)
    i = int(outside[0]) if len(outside) else int(choice[0]) if len(choice) else 0
    a = int(batch["actions"][i])
    size = len(raw_adv)
    log.section(f"{title} (minibatch of B = {size}; sample {i} shown)")
    log.formula(
        "normalised advantage",
        "Â = (A − mean(A)) / (std(A) + 1e-8)",
        f"({num(raw_adv[i], 4)} − {num(raw_adv.mean(), 4)}) / {raw_adv.std():.4f}",
        f"{adv[i]:.4f}",
    )
    log.formula(
        "probability ratio",
        "r = exp(log π_new(a|s) − log π_old(a|s))",
        f"exp({num(log_p_all[i, a], 4)} − {num(batch['old_log_probs'][i], 4)})",
        f"{ratio[i]:.4f}",
    )
    log.formula(
        "clipped ratio",
        "clip(r, 1 − ε, 1 + ε)",
        f"clip({ratio[i]:.4f}, {1 - clip_range:.2f}, {1 + clip_range:.2f})",
        f"{clipped[i]:.4f}",
    )
    term = min(ratio[i] * adv[i], clipped[i] * adv[i])
    log.formula(
        "clipped surrogate",
        "ℓ = −min(r·Â, clip(r)·Â)",
        f"−min({ratio[i]:.4f}×{num(adv[i], 4)}, {clipped[i]:.4f}×{num(adv[i], 4)})",
        f"{-term:.4f}",
    )
    if abs(ratio[i] - 1.0) > clip_range:
        if clipped[i] * adv[i] < ratio[i] * adv[i]:
            log.note(
                "clip(r)·Â is the smaller term and it is a constant, so this sample adds no",
                f"gradient: π(a|s) has already moved more than ε = {clip_range} the way Â asks.",
            )
        else:
            log.note(
                "r·Â is the smaller term, so the gradient still acts: π(a|s) moved the wrong",
                "way for this advantage, and PPO pulls it back.",
            )
    log.formula(
        "value error",
        "(R − V(s))²",
        f"({num(batch['returns'][i], 4)} − {num(values[i], 4)})²",
        f"{(batch['returns'][i] - values[i]) ** 2:.4f}",
    )
    valid = [int(j) for j in np.flatnonzero(p[i] > 0)]
    terms = " + ".join(f"{p[i, j]:.3f}×{num(log_p_all[i, j])}" for j in valid)
    log.formula("entropy", "H = −Σ p·log p", f"−({terms})", f"{h[i]:.4f}")
    if np.allclose(ratio, 1.0):
        log.note(
            "The policy has not changed since the data was collected, so r = 1 everywhere and",
            "L_clip = −mean(Â) = 0 (Â has mean 0). Its gradient is not zero, though.",
        )
    log.text(f"Averaged over all {size} samples of the minibatch:")
    log.formula(
        "total loss",
        "L = L_clip + c_v·L_V − c_e·H",
        f"{num(stats['policy_loss'], 4)} + {vf_coef}×{stats['value_loss']:.4f}"
        f" − {ent_coef}×{stats['entropy']:.4f}",
        f"{stats['loss']:.4f}",
    )
    grads = ", ".join(f"{d_logits[i, j]:+.2e}" for j in valid)
    log.formula(
        "gradient at the logits",
        "∂L/∂z_j = [−r·Â·(1[j=a] − p_j) + c_e·p_j·(log p_j + H)] / B",
        f"for the valid actions {valid} (a = {a}): [{grads}]",
    )
    log.note("Backpropagation carries this gradient through the actor's layers (soren/nn).")


# ---------------------------------------------------------------------------- the agent


class Callback:
    """Hooks into training. Return False from ``on_step`` to stop early."""

    def on_training_start(self, model: PPO) -> None: ...

    def on_step(self, model: PPO, actions: np.ndarray, dones: np.ndarray, infos: list) -> bool:
        return True

    def on_rollout_end(self, model: PPO) -> None: ...

    def on_update_end(self, model: PPO) -> None: ...

    def on_training_end(self, model: PPO) -> None: ...


class PPO:
    def __init__(self, venv: VecEnv, cfg: PPOConfig | None = None, seed: int = 0) -> None:
        self.venv = venv
        self.cfg = cfg = cfg or PPOConfig()
        self.policy = ActorCritic(venv.obs_dim, venv.n_actions, cfg.net_arch, cfg.activation, seed)
        self.optimizer = Adam(
            self.policy.params, self.policy.grads, cfg.learning_rate, eps=cfg.adam_eps
        )
        self.schedule = learning_rate_schedule(cfg)
        self.rng = np.random.default_rng(seed)
        self.normalizer = (
            RewardNormalizer(venv.num_envs, cfg.gamma) if cfg.normalize_reward else None
        )
        self.buffer = RolloutBuffer(cfg.n_steps, venv.num_envs, venv.obs_dim, venv.n_actions)
        self.num_timesteps = 0
        self.total_timesteps = cfg.total_timesteps
        self.updates = 0
        self.explain_every = 0
        """With the math log on, explain every this many updates (0: never)."""
        self.metrics: dict[str, float] = {}
        """Numbers recorded during the current update, for the console and the CSV log."""
        self._obs: np.ndarray | None = None

    def record(self, key: str, value: float) -> None:
        self.metrics[key] = float(value)

    def predict(
        self, obs: np.ndarray, action_masks: np.ndarray, deterministic: bool = True
    ) -> tuple[int, None]:
        return self.policy.predict(obs, action_masks, deterministic)

    def save(self, path: Any) -> Any:
        return self.policy.save(path, num_timesteps=self.num_timesteps)

    # ------------------------------------------------------------------ loop

    def learn(self, total_timesteps: int, callbacks: Sequence[Callback] = ()) -> PPO:
        self.total_timesteps = total_timesteps
        self._obs = self.venv.reset()
        for cb in callbacks:
            cb.on_training_start(self)
        while self.num_timesteps < total_timesteps:
            self.updates += 1
            self.metrics = {}
            explain = bool(
                log.enabled and self.explain_every and (self.updates - 1) % self.explain_every == 0
            )
            if explain:
                log.heading(
                    f"PPO update {self.updates} · timesteps {self.num_timesteps:,} → "
                    f"{self.num_timesteps + self.cfg.n_steps * self.venv.num_envs:,}"
                )
            if not self.collect_rollouts(callbacks, explain):
                break
            self.train(explain)
            for cb in callbacks:
                cb.on_update_end(self)
        for cb in callbacks:
            cb.on_training_end(self)
        return self

    def collect_rollouts(self, callbacks: Sequence[Callback], explain: bool = False) -> bool:
        cfg, venv, policy = self.cfg, self.venv, self.policy
        show_action = explain
        for t in range(cfg.n_steps):
            masks = venv.action_masks()
            actions, log_p, values = policy.act(self._obs, masks, self.rng)
            if show_action and masks.sum(axis=1).max() >= 3:
                _explain_action(policy, self._obs, masks, actions, log_p, values, t)
                show_action = False
            next_obs, rewards, dones, infos = venv.step(actions)
            self.num_timesteps += venv.num_envs
            if self.normalizer is not None:
                rewards = self.normalizer(rewards, dones, explain and t == cfg.n_steps - 1)
            for i, info in enumerate(infos):
                if dones[i] and info.get("truncated"):
                    # Out of steps, not a real ending: add the value the episode still had.
                    rewards[i] += cfg.gamma * policy.value(info["terminal_observation"])
            self.buffer.add(t, self._obs, masks, actions, log_p, values, rewards, dones)
            self._obs = next_obs
            for cb in callbacks:
                if cb.on_step(self, actions, dones, infos) is False:
                    return False
        self.buffer.finish(policy.values(self._obs), cfg.gamma, cfg.gae_lambda, explain)
        for cb in callbacks:
            cb.on_rollout_end(self)
        return True

    def train(self, explain: bool = False) -> None:
        cfg, policy = self.cfg, self.policy
        progress = max(0.0, 1.0 - self.num_timesteps / self.total_timesteps)
        lr = self.schedule(progress) if callable(self.schedule) else self.schedule
        if explain:
            log.section("learning rate")
            if callable(self.schedule):
                log.formula(
                    "linear decay",
                    "lr = lr₀ · (1 − timesteps / total)",
                    f"{cfg.learning_rate} × "
                    f"(1 − {self.num_timesteps:,} / {self.total_timesteps:,})",
                    f"{lr:.3e}",
                )
            else:
                log.formula("constant", "lr = lr₀", None, f"{lr:.3e}")

        totals: dict[str, list[float]] = {}
        norms = []
        for epoch in range(cfg.n_epochs):
            epoch_stats: dict[str, list[float]] = {}
            for index, batch in enumerate(self.buffer.minibatches(cfg.batch_size, self.rng)):
                show = explain and epoch == 0 and index == 0
                title = None
                if explain and index == 0 and epoch in (0, cfg.n_epochs - 1):
                    title = f"PPO loss, epoch {epoch + 1} of {cfg.n_epochs}"
                stats = ppo_loss(
                    policy, batch, cfg.clip_range, cfg.ent_coef, cfg.vf_coef, explain=title
                )
                norm, scale = clip_grad_norm(policy.grads, cfg.max_grad_norm)
                norms.append(norm)
                if show:
                    explain_clip(norm, scale, cfg.max_grad_norm)
                    before = adam_snapshot(self.optimizer)
                self.optimizer.step(lr)
                if show:
                    explain_adam(self.optimizer, before, lr)
                for key, value in stats.items():
                    epoch_stats.setdefault(key, []).append(value)
                    totals.setdefault(key, []).append(value)
            if explain:
                if epoch == 0:
                    log.section(f"{cfg.n_epochs} passes over the rollout (mean over minibatches)")
                log.text(
                    f"epoch {epoch + 1:>2}: "
                    + "  ".join(
                        f"{key} {np.mean(epoch_stats[key]):+.4f}"
                        for key in ("policy_loss", "value_loss", "entropy", "approx_kl")
                    )
                    + f"  clip_fraction {np.mean(epoch_stats['clip_fraction']):.3f}"
                )

        values, returns = self.buffer.values.ravel(), self.buffer.returns.ravel()
        var_returns = np.var(returns)
        ev = float("nan") if var_returns == 0 else 1.0 - np.var(returns - values) / var_returns
        if explain:
            log.section("how well the critic predicted this rollout")
            log.formula(
                "explained variance",
                "EV = 1 − Var(R − V) / Var(R)",
                f"1 − {np.var(returns - values):.4f} / {var_returns:.4f}",
                f"{ev:.3f}   (1 = perfect, 0 = no better than predicting the mean)",
            )
        for key, value in totals.items():
            self.record(f"train/{key}", float(np.mean(value)))
        self.record("train/explained_variance", ev)
        self.record("train/grad_norm", float(np.mean(norms)))
        self.record("train/learning_rate", lr)


# ------------------------------------------------------------------ explanation helpers


def _explain_action(policy, obs, masks, actions, log_p, values, t) -> None:
    e = int(np.argmax(masks.sum(axis=1)))  # the environment with the most choices
    probs = np.exp(policy.log_probs(obs[e : e + 1], masks[e : e + 1])[0])
    valid = [int(j) for j in np.flatnonzero(masks[e])]
    log.section(f"choosing an action (environment {e}, step {t} of the rollout)")
    log.text(f"valid actions (mask): {valid}   (0-5 move to a successor, 6 backtrack, 7 declare)")
    log.formula(
        "policy",
        "π(·|s) = softmax(masked logits)",
        "[" + ", ".join(f"a{j}: {probs[j]:.3f}" for j in valid) + "]",
    )
    log.formula(
        "sampled action",
        "a ~ π(·|s)",
        f"a = {actions[e]},  log π(a|s) = {log_p[e]:.4f},  V(s) = {values[e]:.4f}",
    )


def explain_clip(norm: float, scale: float, max_norm: float) -> None:
    log.section("gradient clipping")
    log.formula(
        "global gradient norm",
        "‖g‖ = √(Σ g²) over every weight",
        None,
        f"{norm:.4f}",
    )
    if scale < 1.0:
        log.formula(
            "scale down",
            "g ← g · max_norm / ‖g‖",
            f"g × {max_norm} / {norm:.4f}",
            f"g × {scale:.4f}",
        )
    else:
        log.text(f"‖g‖ ≤ max_norm = {max_norm}: gradients are left as they are")


def adam_snapshot(opt: Adam) -> dict[str, float]:
    """Before-values of one weight: the first bias of the actor's output layer."""
    i = len(opt.params) // 2 - 1  # actor's last bias (actor params come first)
    return {
        "i": i,
        "p": float(opt.params[i].flat[0]),
        "g": float(opt.grads[i].flat[0]),
        "m": float(opt.m[i].flat[0]),
        "v": float(opt.v[i].flat[0]),
    }


def explain_adam(opt: Adam, before: dict[str, float], lr: float) -> None:
    i, g = before["i"], before["g"]
    b1, b2, t = opt.beta1, opt.beta2, opt.t
    m, v, p = float(opt.m[i].flat[0]), float(opt.v[i].flat[0]), float(opt.params[i].flat[0])
    m_hat, v_hat = m / (1 - b1**t), v / (1 - b2**t)
    log.section(f"Adam step (step t = {t}) for one weight: the bias of action 0 in the actor")
    log.formula("gradient", "g", None, f"{g:+.3e}")
    log.formula(
        "first moment", "m ← β₁·m + (1 − β₁)·g", f"{b1}×{before['m']:+.3e} + {1 - b1:.1f}×{g:+.3e}",
        f"{m:+.3e}",
    )  # fmt: skip
    log.formula(
        "second moment",
        "v ← β₂·v + (1 − β₂)·g²",
        f"{b2}×{before['v']:.3e} + {1 - b2:.3f}×{g * g:.3e}",
        f"{v:.3e}",
    )
    log.formula(
        "bias correction",
        "m̂ = m/(1 − β₁ᵗ),  v̂ = v/(1 − β₂ᵗ)",
        None,
        f"m̂ = {m_hat:+.3e},  v̂ = {v_hat:.3e}",
    )
    log.formula(
        "update",
        "θ ← θ − lr·m̂/(√v̂ + ε)",
        f"{before['p']:+.6f} − {lr:.2e}×{m_hat:+.3e}/({np.sqrt(v_hat):.3e} + {opt.eps:g})",
        f"{p:+.6f}",
    )
