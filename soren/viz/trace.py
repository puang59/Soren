"""Episode traces: a method-independent record of one traversal.

A trace lists what a method did step by step, so the visualizer can replay the learned agent
and the baselines in the same way. Policy traces also carry the action probabilities and the
value estimate at each step; baselines leave those empty.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

INSPECT = "INSPECT"
"""Action name used by order-based baselines, which jump to a node rather than walk to it."""


@dataclass
class TraceStep:
    t: int
    node: int
    """Node the action was taken from."""
    action: str
    next_node: int
    """Node the method stands on after the action."""
    reward: float
    """Unshaped reward."""
    mask: list[bool] | None = None
    probs: list[float] | None = None
    value: float | None = None


@dataclass
class Trace:
    graph_id: str
    method: str
    start_node: int
    steps: list[TraceStep] = field(default_factory=list)
    outcome: dict[str, Any] = field(default_factory=dict)
    """``success``, ``declared_node``, ``return``, ``end_reason``, ``nodes_inspected``."""

    @property
    def nodes(self) -> list[int]:
        """The node occupied before the first step and after each step."""
        return [self.start_node, *(step.next_node for step in self.steps)]

    def validate(self) -> Trace:
        position = self.start_node
        for index, step in enumerate(self.steps):
            if step.t != index:
                raise ValueError(f"step {index} has t={step.t}")
            if step.node != position:
                raise ValueError(f"step {index} starts at {step.node}, expected {position}")
            position = step.next_node
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        return cls(
            graph_id=data["graph_id"],
            method=data["method"],
            start_node=data["start_node"],
            steps=[TraceStep(**step) for step in data.get("steps", [])],
            outcome=dict(data.get("outcome", {})),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> Trace:
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    @classmethod
    def load(cls, path: str | Path) -> Trace:
        return cls.from_json(Path(path).read_text())
