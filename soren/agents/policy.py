"""The actor-critic network: what the agent uses to choose actions and to judge states.

Two separate MLPs read the same observation vector:

* the **actor** (policy) outputs one logit per action; after masking and softmax these are the
  probabilities ``π(a | s)`` of each action;
* the **critic** (value function) outputs one number ``V(s)``: the return the agent expects
  to collect from state ``s`` onwards. PPO uses it to tell whether an action turned out
  better or worse than expected.

With the defaults, each is ``248 → 256 → 256 → out`` with tanh activations.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from soren.nn import MLP, log_softmax, masked_logits, sample


class ActorCritic:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        net_arch: Sequence[int] = (256, 256),
        activation: str = "tanh",
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.config = {
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "net_arch": list(net_arch),
            "activation": activation,
            "seed": seed,
        }
        # A small output gain makes every action about equally likely at the start, so the
        # agent explores before it has learned anything.
        self.actor = MLP([obs_dim, *net_arch, n_actions], activation, rng, output_gain=0.01)
        self.critic = MLP([obs_dim, *net_arch, 1], activation, rng, output_gain=1.0)

    @property
    def params(self) -> list[np.ndarray]:
        return self.actor.params + self.critic.params

    @property
    def grads(self) -> list[np.ndarray]:
        return self.actor.grads + self.critic.grads

    # ---------------------------------------------------------------- forward

    def log_probs(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """``log π(a | s)`` for every action, shape ``[batch, n_actions]``."""
        logits = self.actor.forward(np.asarray(obs, dtype=np.float64))
        return log_softmax(masked_logits(logits, masks))

    def values(self, obs: np.ndarray) -> np.ndarray:
        """``V(s)`` per row of ``obs``."""
        return self.critic.forward(np.asarray(obs, dtype=np.float64))[:, 0]

    def act(
        self,
        obs: np.ndarray,
        masks: np.ndarray,
        rng: np.random.Generator | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Choose an action per row: returns ``(actions, log π(action | s), V(s))``.

        Training samples from ``π`` so the agent keeps trying alternatives; evaluation takes
        the most probable action.
        """
        log_p = self.log_probs(obs, masks)
        if deterministic:
            actions = log_p.argmax(axis=1)
        else:
            actions = sample(np.exp(log_p), rng or np.random.default_rng())
        chosen = log_p[np.arange(len(actions)), actions]
        return actions, chosen, self.values(obs)

    # ------------------------------------------------- single-state helpers

    def predict(
        self, obs: np.ndarray, action_masks: np.ndarray, deterministic: bool = True
    ) -> tuple[int, None]:
        """The action for one observation. Returns ``(action, None)``."""
        actions, _, _ = self.act(
            np.asarray(obs)[None], np.asarray(action_masks)[None], deterministic=deterministic
        )
        return int(actions[0]), None

    def action_probs(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.exp(self.log_probs(np.asarray(obs)[None], np.asarray(mask)[None])[0])

    def value(self, obs: np.ndarray) -> float:
        return float(self.values(np.asarray(obs)[None])[0])

    # ------------------------------------------------------------ save / load

    def save(self, path: str | Path, **meta: Any) -> Path:
        """Write the weights and the configuration to one ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"actor_{i}": p for i, p in enumerate(self.actor.params)}
        arrays |= {f"critic_{i}": p for i, p in enumerate(self.critic.params)}
        header = json.dumps({**self.config, "meta": meta})
        with open(path, "wb") as fh:
            np.savez(fh, config=np.array(header), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> ActorCritic:
        with np.load(path, allow_pickle=False) as data:
            config = json.loads(str(data["config"]))
            config.pop("meta", None)
            policy = cls(**config)
            for name, net in (("actor", policy.actor), ("critic", policy.critic)):
                for i, p in enumerate(net.params):
                    p[...] = data[f"{name}_{i}"]
        return policy
