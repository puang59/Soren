"""Training callbacks: validation with checkpoint selection, behaviour diagnostics, and logs."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from soren.agents.ppo import PPO, Callback
from soren.agents.searcher import evaluate_policy
from soren.data.schema import GraphRecord
from soren.env.cfg_nav_env import EnvConfig
from soren.env.rewards import RewardConfig
from soren.eval.metrics import summarize


class ValidationCallback(Callback):
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
        verbose: bool = False,
    ) -> None:
        self.graphs = list(graphs)
        self.env_config = env_config
        self.reward_config = reward_config
        self.eval_freq = eval_freq
        self.run_dir = Path(run_dir)
        self.verbose = verbose
        self.history: list[dict[str, Any]] = []
        self.best_key: tuple[float, float] | None = None
        self.best_metrics: dict[str, Any] | None = None
        self._next_eval = eval_freq

    def evaluate(self, model: PPO) -> dict[str, Any]:
        results = evaluate_policy(model, self.graphs, self.env_config, self.reward_config)
        metrics: dict[str, Any] = {"timesteps": model.num_timesteps, **summarize(results)}
        self.history.append(metrics)
        for key, value in metrics.items():
            if key != "timesteps":
                model.record(f"val/{key}", value)

        key = (metrics["success_rate"], -metrics["nodes_inspected"])
        improved = self.best_key is None or key > self.best_key
        if improved:
            self.best_key = key
            self.best_metrics = metrics
            model.save(self.run_dir / "best_model.npz")
        (self.run_dir / "val_history.json").write_text(json.dumps(self.history, indent=2))
        if self.verbose:
            hits = sum(r.success for r in results)
            print(
                f"  validation at {model.num_timesteps:,} steps: success = {hits}/{len(results)}"
                f" = {metrics['success_rate']:.3f}, nodes inspected = "
                f"{metrics['nodes_inspected']:.1f} on average"
                + ("  ← best so far, saved best_model.npz" if improved else "")
            )
        return metrics

    def on_step(self, model: PPO, actions, dones, infos) -> bool:
        if model.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_freq
            self.evaluate(model)
        return True

    def on_training_end(self, model: PPO) -> None:
        if not self.history or self.history[-1]["timesteps"] != model.num_timesteps:
            self.evaluate(model)


class DiagnosticsCallback(Callback):
    """Record what the policy does during training, to spot degenerate behaviour early.

    Per rollout: the share of ``MOVE``, ``BACKTRACK`` and ``DECLARE`` actions, how episodes
    ended (correct, wrong declare, timeout, dead end) and their mean unshaped return.
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self._actions: Counter[str] = Counter()
        self._endings: Counter[str] = Counter()
        self._returns: list[float] = []
        self._caps: list[int] = []

    def on_step(self, model: PPO, actions, dones, infos) -> bool:
        actions = np.asarray(actions).reshape(-1)
        self._actions["move"] += int(np.sum(actions < self.k))
        self._actions["backtrack"] += int(np.sum(actions == self.k))
        self._actions["declare"] += int(np.sum(actions == self.k + 1))
        for done, info in zip(dones, infos, strict=True):
            if done:
                self._endings[info.get("end_reason") or "unknown"] += 1
                self._returns.append(float(info["episode_return"]))
                if "curriculum_cap" in info:
                    self._caps.append(info["curriculum_cap"])
        return True

    def on_rollout_end(self, model: PPO) -> None:
        total_actions = sum(self._actions.values())
        for name in ("move", "backtrack", "declare"):
            model.record(f"behaviour/frac_{name}", self._actions[name] / max(total_actions, 1))
        total_endings = sum(self._endings.values())
        model.record("behaviour/episodes", total_endings)
        for name in ("correct", "wrong_declare", "timeout", "dead_end"):
            model.record(f"behaviour/end_{name}", self._endings[name] / max(total_endings, 1))
        if self._returns:
            model.record("behaviour/episode_return", float(np.mean(self._returns)))
        if self._caps:
            model.record("curriculum/cap", float(np.mean(self._caps)))
        self._actions.clear()
        self._endings.clear()
        self._returns.clear()
        self._caps.clear()


class ProgressLog(Callback):
    """One line per update in the terminal, and every recorded number in ``train_log.csv``."""

    def __init__(self, run_dir: Path | None = None, verbose: bool = True) -> None:
        self.path = None if run_dir is None else Path(run_dir) / "train_log.csv"
        self.verbose = verbose
        self.rows: list[dict[str, float]] = []
        self._start = 0.0

    def on_training_start(self, model: PPO) -> None:
        self._start = time.perf_counter()
        if self.verbose:
            per_update = model.cfg.n_steps * model.venv.num_envs
            updates = -(-model.total_timesteps // per_update)
            print(
                f"PPO: {updates} updates of {model.cfg.n_steps} steps × {model.venv.num_envs} "
                f"environments = {per_update:,} transitions each, {model.cfg.n_epochs} epochs "
                f"in minibatches of {model.cfg.batch_size}"
            )
            print(
                "  columns: success = training episodes ending in a correct declaration; return ="
                " mean unshaped\n  episode return; π-loss, V-loss, entropy = the three terms of"
                " the PPO loss; KL = how far\n  the policy moved this update; clip = share of"
                " samples clipped; EV = explained variance of\n  the critic; lr = learning rate"
            )
            print(
                "  update  timesteps  success  return  π-loss  V-loss  entropy     KL  clip   "
                "EV       lr  steps/s"
            )

    def on_update_end(self, model: PPO) -> None:
        m = model.metrics
        elapsed = time.perf_counter() - self._start
        row = {"update": model.updates, "timesteps": model.num_timesteps, **m}
        self.rows.append(row)
        if self.path is not None:
            columns = sorted({key for r in self.rows for key in r}, key=_column_order)
            with open(self.path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=columns)
                writer.writeheader()
                writer.writerows(self.rows)
        if self.verbose:
            print(
                f"  {model.updates:>6}  {model.num_timesteps:>9,}"
                f"  {m.get('behaviour/end_correct', float('nan')):>7.1%}"
                f"  {m.get('behaviour/episode_return', float('nan')):>+6.3f}"
                f"  {m['train/policy_loss']:>+6.3f}  {m['train/value_loss']:>6.3f}"
                f"  {m['train/entropy']:>7.3f}  {m['train/approx_kl']:>5.3f}"
                f"  {m['train/clip_fraction']:>4.2f}  {m['train/explained_variance']:>+4.2f}"
                f"  {m['train/learning_rate']:>7.1e}  {model.num_timesteps / elapsed:>7,.0f}"
            )


def _column_order(key: str) -> tuple[int, str]:
    order = ("update", "timesteps")
    return (order.index(key), "") if key in order else (len(order), key)
