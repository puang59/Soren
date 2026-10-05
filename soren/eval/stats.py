"""Statistics over per-episode evaluation tables.

The unit of analysis is the graph. When a method has several seeds, its episodes on one graph
are averaged first, so that resampling and paired tests treat each graph as one observation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

EPISODE_COLUMNS: tuple[str, ...] = (
    "success",
    "localization_accuracy",
    "nodes_inspected",
    "actions_taken",
    "normalised_cost",
    "cumulative_reward",
)
"""Per-graph quantities with a confidence interval. Step counts charge failures in full."""


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean of ``values`` with a percentile bootstrap confidence interval.

    Returns ``(mean, low, high)``.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    rng = np.random.default_rng(seed)
    indices = rng.integers(values.size, size=(n_resamples, values.size))
    means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(values.mean()), float(low), float(high)


def per_graph(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(method, graph)``: episode quantities averaged over seeds.

    A failed episode is charged every node and the whole step budget, as in
    :func:`soren.eval.metrics.compute_metrics`.
    """
    success = frame["success"].astype(bool)
    nodes = frame["nodes_inspected"].where(success, frame["num_nodes"])
    episodes = pd.DataFrame(
        {
            "method": frame["method"],
            "graph_id": frame["graph_id"],
            "success": success.astype(float),
            "localization_accuracy": frame["first_declare_correct"].astype(float),
            "nodes_inspected": nodes.astype(float),
            "actions_taken": frame["actions_taken"]
            .where(success, frame["max_steps"])
            .astype(float),
            "normalised_cost": nodes / frame["num_nodes"],
            "cumulative_reward": frame["cumulative_reward"].astype(float),
        }
    )
    return episodes.groupby(["method", "graph_id"], sort=True).mean().reset_index()


def confidence_intervals(
    frame: pd.DataFrame, confidence: float = 0.95, n_resamples: int = 10_000, seed: int = 0
) -> pd.DataFrame:
    """Bootstrap confidence intervals over graphs, in long form.

    One row per ``(method, metric)`` with columns ``mean``, ``low``, ``high`` and ``graphs``.
    """
    rows = []
    for method, group in per_graph(frame).groupby("method", sort=True):
        for column in EPISODE_COLUMNS:
            mean, low, high = bootstrap_ci(group[column].to_numpy(), confidence, n_resamples, seed)
            rows.append(
                {
                    "method": method,
                    "metric": column,
                    "mean": mean,
                    "low": low,
                    "high": high,
                    "graphs": len(group),
                }
            )
    return pd.DataFrame(rows)


def _paired(frame: pd.DataFrame, method_a: str, method_b: str) -> pd.DataFrame:
    table = per_graph(frame)
    a = table[table["method"] == method_a].set_index("graph_id")
    b = table[table["method"] == method_b].set_index("graph_id")
    if a.empty or b.empty:
        missing = method_a if a.empty else method_b
        raise ValueError(f"no episodes for method {missing!r}")
    common = a.index.intersection(b.index)
    if common.empty:
        raise ValueError(f"{method_a!r} and {method_b!r} share no graphs")
    return a.loc[common].join(b.loc[common], lsuffix="_a", rsuffix="_b")


def wilcoxon_nodes_inspected(frame: pd.DataFrame, method_a: str, method_b: str) -> dict[str, float]:
    """Wilcoxon signed-rank test on nodes inspected, paired by graph.

    ``median_difference`` is ``a - b``: negative means ``method_a`` inspects fewer nodes.
    """
    pairs = _paired(frame, method_a, method_b)
    difference = (pairs["nodes_inspected_a"] - pairs["nodes_inspected_b"]).to_numpy()
    if np.allclose(difference, 0.0):
        statistic, p_value = 0.0, 1.0  # identical on every graph: nothing to test
    else:
        result = scipy_stats.wilcoxon(difference, zero_method="wilcox")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "graphs": float(len(pairs)),
        "median_difference": float(np.median(difference)),
        "mean_difference": float(np.mean(difference)),
        "statistic": statistic,
        "p_value": p_value,
    }


def mcnemar_success(frame: pd.DataFrame, method_a: str, method_b: str) -> dict[str, float]:
    """Exact McNemar test on success, paired by graph.

    With several seeds, a method counts as succeeding on a graph when it succeeds in at least
    half of them. Only discordant graphs carry information: ``a_only`` and ``b_only`` count
    the graphs solved by one method and not the other.
    """
    pairs = _paired(frame, method_a, method_b)
    a = pairs["success_a"] >= 0.5
    b = pairs["success_b"] >= 0.5
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    discordant = a_only + b_only
    p_value = 1.0
    if discordant:
        p_value = float(scipy_stats.binomtest(a_only, discordant, 0.5).pvalue)
    return {
        "graphs": float(len(pairs)),
        "a_only": float(a_only),
        "b_only": float(b_only),
        "p_value": p_value,
    }


def comparison_table(
    frame: pd.DataFrame,
    reference: str,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Headline table: each method's success and cost with CIs, and tests against ``reference``.

    The p-values compare every other method with ``reference`` on the same graphs.
    """
    intervals = confidence_intervals(frame, confidence, n_resamples, seed)
    methods = sorted(intervals["method"].unique())
    if reference not in methods:
        raise ValueError(f"reference method {reference!r} not in results: {methods}")

    def cell(method: str, metric: str) -> str:
        row = intervals[(intervals["method"] == method) & (intervals["metric"] == metric)].iloc[0]
        return f"{row['mean']:.3f} [{row['low']:.3f}, {row['high']:.3f}]"

    rows = []
    for method in [reference, *(m for m in methods if m != reference)]:
        row = {
            "method": method,
            "success": cell(method, "success"),
            "nodes_inspected": cell(method, "nodes_inspected"),
            "normalised_cost": cell(method, "normalised_cost"),
            "cumulative_reward": cell(method, "cumulative_reward"),
            "p_success": np.nan,
            "p_nodes": np.nan,
        }
        if method != reference:
            row["p_success"] = mcnemar_success(frame, reference, method)["p_value"]
            row["p_nodes"] = wilcoxon_nodes_inspected(frame, reference, method)["p_value"]
        rows.append(row)
    return pd.DataFrame(rows)
