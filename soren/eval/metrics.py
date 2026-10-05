"""Summary metrics over episode results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd

from soren.baselines.base import EpisodeResult


def summarize(results: Sequence[EpisodeResult]) -> dict[str, float]:
    """Aggregate episode results into headline numbers.

    Step counts are averaged over all episodes, so failures count with however many steps
    they used before stopping.
    """
    if not results:
        raise ValueError("no results to summarize")
    n = len(results)
    summary = {
        "episodes": float(n),
        "success_rate": float(np.mean([r.success for r in results])),
        "nodes_inspected": float(np.mean([r.nodes_inspected for r in results])),
        "actions_taken": float(np.mean([r.actions_taken for r in results])),
        "cumulative_reward": float(np.mean([r.cumulative_reward for r in results])),
    }
    for reason, count in sorted(Counter(r.end_reason for r in results).items()):
        summary[f"end_{reason}"] = count / n
    return summary


# ------------------------------------------------------------------ per-episode tables

SIZE_BUCKETS: tuple[tuple[int, str], ...] = ((20, "<=20"), (50, "21-50"), (100, "51-100"))
DISTANCE_BUCKETS: tuple[tuple[int, str], ...] = ((2, "1-2"), (5, "3-5"), (10, "6-10"))

METRIC_COLUMNS: tuple[str, ...] = (
    "episodes",
    "localization_accuracy",
    "success_rate",
    "nodes_inspected_success",
    "nodes_inspected_all",
    "actions_success",
    "actions_all",
    "normalised_cost",
    "cumulative_reward",
    "path_efficiency",
)


def _bucket(values: pd.Series, buckets: tuple[tuple[int, str], ...], overflow: str) -> pd.Series:
    labels = pd.Series(overflow, index=values.index, dtype=object)
    for upper, label in reversed(buckets):
        labels[values <= upper] = label
    return labels


def add_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the breakdown columns ``size_bucket``, ``distance_bucket`` and ``vuln_bucket``."""
    out = frame.copy()
    out["size_bucket"] = _bucket(out["num_nodes"], SIZE_BUCKETS, ">100")
    out["distance_bucket"] = _bucket(out["dist_to_vuln"], DISTANCE_BUCKETS, ">10")
    out["vuln_bucket"] = np.where(out["num_vuln"] > 1, "multiple", "single")
    return out


def _episode_metrics(group: pd.DataFrame) -> pd.Series:
    success = group["success"].astype(bool)
    solved = group[success]
    # Failures are charged the full budget: every node for inspections, max_steps for actions.
    nodes_all = group["nodes_inspected"].where(success, group["num_nodes"])
    actions_all = group["actions_taken"].where(success, group["max_steps"])
    # Moves used over the shortest possible number of moves; 1.0 is optimal. The final
    # declaration is not a move.
    reachable = solved[solved["dist_to_vuln"] > 0]
    efficiency = (reachable["actions_taken"] - 1) / reachable["dist_to_vuln"]
    return pd.Series(
        {
            "episodes": len(group),
            "localization_accuracy": group["first_declare_correct"].mean(),
            "success_rate": success.mean(),
            "nodes_inspected_success": solved["nodes_inspected"].mean(),
            "nodes_inspected_all": nodes_all.mean(),
            "actions_success": solved["actions_taken"].mean(),
            "actions_all": actions_all.mean(),
            "normalised_cost": (nodes_all / group["num_nodes"]).mean(),
            "cumulative_reward": group["cumulative_reward"].mean(),
            "path_efficiency": efficiency.mean(),
        }
    )


def compute_metrics(frame: pd.DataFrame, by: Sequence[str] = ("method",)) -> pd.DataFrame:
    """Aggregate a per-episode table into one row of metrics per group.

    * ``localization_accuracy``: share of episodes whose first declaration is correct
    * ``success_rate``: share ending in a correct declaration within the step budget
    * ``nodes_inspected_success`` / ``actions_success``: means over successful episodes
    * ``nodes_inspected_all`` / ``actions_all``: means over all episodes, with a failure
      counted as inspecting every node and using the whole step budget
    * ``normalised_cost``: nodes inspected (failures charged in full) over graph size
    * ``cumulative_reward``: mean unshaped return
    * ``path_efficiency``: moves used over the shortest path length, successes only
    """
    keys = list(by)
    rows = [
        {**dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True)), **metrics}
        for key, metrics in (
            (key, _episode_metrics(group)) for key, group in frame.groupby(keys, sort=True)
        )
    ]
    out = pd.DataFrame(rows, columns=[*keys, *METRIC_COLUMNS])
    out["episodes"] = out["episodes"].astype(int)
    return out


def breakdown(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """Metrics per method within each value of ``by``.

    ``by`` is one of ``size_bucket``, ``distance_bucket``, ``vuln_bucket``, ``cwe`` or any
    other column of the per-episode table.
    """
    if by in ("size_bucket", "distance_bucket", "vuln_bucket") and by not in frame.columns:
        frame = add_buckets(frame)
    return compute_metrics(frame, by=(by, "method"))


def metrics_across_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric across seeds, per method."""
    per_seed = compute_metrics(frame, by=("method", "seed"))
    columns = [c for c in METRIC_COLUMNS if c != "episodes"]
    grouped = per_seed.groupby("method")[columns]
    out = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=0).add_suffix("_std"))
    out.insert(0, "seeds", per_seed.groupby("method")["seed"].nunique())
    return out.reset_index()
