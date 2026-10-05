"""Training callbacks: validation with checkpoint selection, and behaviour diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from soren.agents.searcher import evaluate_policy
from soren.data.schema import GraphRecord
from soren.env.cfg_nav_env import EnvConfig
from soren.env.rewards import RewardConfig
from soren.eval.metrics import summarize


class ValidationCallback(BaseCallback):
    """Evaluate the deterministic policy on the whole validation split at fixed intervals.

    The best checkpoint is chosen by success rate, with ties broken by fewer nodes inspected.
    """

    def __init__(
        self,
        graphs: Sequence[GraphRecord],
        env_config: EnvConfig,
        reward_config: RewardConfig,
        eval_freq: int,
        run_dir: Path,
    ) -> None:
        super().__init__()
        self.graphs = list(graphs)
        self.env_config = env_config
        self.reward_config = reward_config
        self.eval_freq = eval_freq
        self.run_dir = Path(run_dir)
        self.history: list[dict[str, Any]] = []
        self.best_key: tuple[float, float] | None = None
        self.best_metrics: dict[str, Any] | None = None
        self._next_eval = eval_freq

    def evaluate(self) -> dict[str, Any]:
        results = evaluate_policy(self.model, self.graphs, self.env_config, self.reward_config)
        metrics: dict[str, Any] = {"timesteps": self.num_timesteps, **summarize(results)}
        self.history.append(metrics)
        for key, value in metrics.items():
            if key != "timesteps":
                self.logger.record(f"val/{key}", value)

        key = (metrics["success_rate"], -metrics["nodes_inspected"])
        if self.best_key is None or key > self.best_key:
            self.best_key = key
            self.best_metrics = metrics
            self.model.save(self.run_dir / "best_model")
        (self.run_dir / "val_history.json").write_text(json.dumps(self.history, indent=2))
        return metrics

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_freq
            self.evaluate()
        return True

    def _on_training_end(self) -> None:
        if not self.history or self.history[-1]["timesteps"] != self.num_timesteps:
            self.evaluate()


class DiagnosticsCallback(BaseCallback):
    """Log what the policy does during training, to spot degenerate behaviour early.

    Per rollout: the share of ``MOVE``, ``BACKTRACK`` and ``DECLARE`` actions, and how
    episodes ended (correct, wrong declare, timeout, dead end).
    """

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self._actions: Counter[str] = Counter()
        self._endings: Counter[str] = Counter()

    def _on_step(self) -> bool:
        actions = np.asarray(self.locals["actions"]).reshape(-1)
        self._actions["move"] += int(np.sum(actions < self.k))
        self._actions["backtrack"] += int(np.sum(actions == self.k))
        self._actions["declare"] += int(np.sum(actions == self.k + 1))
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                self._endings[info.get("end_reason") or "unknown"] += 1
        return True

    def _on_rollout_end(self) -> None:
        total_actions = sum(self._actions.values())
        for name in ("move", "backtrack", "declare"):
            self.logger.record(
                f"behaviour/frac_{name}", self._actions[name] / max(total_actions, 1)
            )
        total_endings = sum(self._endings.values())
        for name in ("correct", "wrong_declare", "timeout", "dead_end"):
            self.logger.record(f"behaviour/end_{name}", self._endings[name] / max(total_endings, 1))
        self._actions.clear()
        self._endings.clear()
