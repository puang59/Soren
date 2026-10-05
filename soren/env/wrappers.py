"""Environment wrappers and samplers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from soren.env.cfg_nav_env import CFGNavEnv
from soren.env.curriculum import CurriculumConfig, CurriculumSampler
from soren.viz.trace import Trace, TraceStep

__all__ = ["CurriculumConfig", "CurriculumSampler", "TraceRecorder"]


class TraceRecorder(gym.Wrapper):
    """Record each episode as a :class:`~soren.viz.trace.Trace` without altering it.

    Call :meth:`annotate` before a step to attach the policy's action probabilities and value
    estimate to that step. The finished or in-progress trace is available as :attr:`trace`.
    """

    def __init__(self, env: CFGNavEnv, method: str = "") -> None:
        super().__init__(env)
        self.method = method
        self.trace: Trace | None = None
        self._pending: tuple[list[float] | None, float | None] = (None, None)

    @property
    def core(self) -> CFGNavEnv:
        return self.env.unwrapped  # type: ignore[return-value]

    def action_masks(self) -> np.ndarray:
        return self.core.action_masks()

    def annotate(self, probs: Any = None, value: float | None = None) -> None:
        probs = None if probs is None else [float(p) for p in probs]
        self._pending = (probs, None if value is None else float(value))

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.trace = Trace(
            graph_id=info["graph_id"], method=self.method, start_node=info["current_node"]
        )
        self._pending = (None, None)
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.trace is None:
            raise RuntimeError("call reset() before step()")
        core = self.core
        node = core.node
        mask = [bool(m) for m in core.action_masks()]
        probs, value = self._pending
        self._pending = (None, None)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.trace.steps.append(
            TraceStep(
                t=len(self.trace.steps),
                node=node,
                action=info["action_name"],
                next_node=info["current_node"],
                reward=float(info["reward_unshaped"]),
                mask=mask,
                probs=probs,
                value=value,
            )
        )
        if terminated or truncated:
            self.trace.outcome = {
                "success": bool(info["is_success"]),
                "declared_node": info["declared_node"],
                "return": float(info["episode_return"]),
                "end_reason": info["end_reason"],
                "nodes_inspected": info["visited_unique"],
            }
        return obs, reward, terminated, truncated, info
