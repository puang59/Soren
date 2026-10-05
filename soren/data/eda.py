"""Exploratory statistics over the filtered BigVul functions (used by ``notebooks/01_eda``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from soren.data.filtering import FilterConfig, filter_bigvul


def vulnerable_cwe_counts(frame: pd.DataFrame) -> pd.Series:
    """Vulnerable functions per CWE before any other filter, most frequent first."""
    vulnerable = frame[frame["vul"] == 1]
    return vulnerable["cwe"].replace("", "(none)").value_counts()


def attrition_by_cwe(
    frame: pd.DataFrame, cwes: Sequence[str], config: FilterConfig | None = None
) -> pd.DataFrame:
    """Rows surviving each filter step, one column per CWE.

    Each CWE is filtered on its own, so the last row is the number of usable functions that
    choosing that CWE alone would give.
    """
    config = config or FilterConfig()
    vulnerable = frame[frame["vul"] == 1]
    columns = {}
    for cwe in cwes:
        _, attrition = filter_bigvul(vulnerable, replace(config, cwes=[cwe]))
        columns[cwe] = {entry["step"]: entry["rows"] for entry in attrition[2:]}
    return pd.DataFrame(columns)


def relative_flaw_positions(filtered: pd.DataFrame) -> np.ndarray:
    """Position of every flaw line within its function: 0 is the first line, 1 the last."""
    positions = []
    for indices, num_lines in zip(
        filtered["flaw_line_indices"], filtered["num_lines"], strict=True
    ):
        positions.extend(index / max(num_lines - 1, 1) for index in indices)
    return np.asarray(positions, dtype=float)


def first_flaw_positions(filtered: pd.DataFrame) -> np.ndarray:
    """Relative position of the first flaw line of each function."""
    return np.asarray(
        [
            min(indices) / max(num_lines - 1, 1)
            for indices, num_lines in zip(
                filtered["flaw_line_indices"], filtered["num_lines"], strict=True
            )
        ],
        dtype=float,
    )


def position_summary(positions: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Share of flaw lines per position decile, next to the uniform share."""
    counts, edges = np.histogram(positions, bins=bins, range=(0.0, 1.0))
    return pd.DataFrame(
        {
            "from": edges[:-1].round(2),
            "to": edges[1:].round(2),
            "share": counts / max(counts.sum(), 1),
            "uniform": 1.0 / bins,
        }
    )


def dataset_summary(filtered: pd.DataFrame) -> pd.Series:
    flaw_counts = filtered["flaw_line_indices"].map(len)
    return pd.Series(
        {
            "functions": len(filtered),
            "projects": filtered["project"].nunique(),
            "fixing commits": filtered["commit_id"].nunique(),
            "median lines": float(filtered["num_lines"].median()),
            "mean lines": float(filtered["num_lines"].mean()),
            "median flaw lines": float(flaw_counts.median()),
            "mean flaw lines": float(flaw_counts.mean()),
            "single flaw line (share)": float((flaw_counts == 1).mean()),
            "largest project (share)": float(
                filtered["project"].value_counts(normalize=True).iloc[0]
            ),
        }
    )
