"""Load YAML configuration files into dataclasses."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return its top-level mapping (empty if the file is empty)."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level, got {type(data).__name__}")
    return data


def from_dict(cls: type[T], data: dict[str, Any] | None) -> T:
    """Build dataclass ``cls`` from ``data``, rejecting keys that are not fields of ``cls``."""
    data = data or {}
    names = {f.name for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {', '.join(unknown)}")
    return cls(**data)


def load_config(cls: type[T], path: str | Path, section: str | None = None) -> T:
    """Load dataclass ``cls`` from a YAML file, optionally from one top-level ``section``."""
    data = load_yaml(path)
    if section is not None:
        data = data.get(section) or {}
    return from_dict(cls, data)
