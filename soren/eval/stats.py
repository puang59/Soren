"""Statistics over per-episode evaluation tables.

The unit of analysis is the graph. When a method has several seeds, its episodes on one graph
are averaged first, so that resampling and paired tests treat each graph as one observation.

The tests are implemented here rather than imported, each with its formula:

* :func:`bootstrap_ci`: percentile bootstrap confidence interval of a mean;
* :func:`binomial_test`: exact two-sided binomial test, which is McNemar's test on the
  discordant pairs;
* :func:`wilcoxon_signed_rank`: Wilcoxon signed-rank test on paired differences.

``tests/test_stats.py`` checks them against SciPy when it is installed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from soren.mathlog import log

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
    explain: str | None = None,
) -> tuple[float, float, float]:
    """Mean of ``values`` with a percentile bootstrap confidence interval.

    Draw ``n`` values *with replacement* from the ``n`` observed ones, take the mean, and
    repeat ``n_resamples`` times. The spread of those means estimates how much the mean would
    vary on another sample of the same size; the middle 95% of them is the interval.

    Returns ``(mean, low, high)``. ``explain`` names the quantity and prints the steps.
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
    if explain:
        log.section(f"bootstrap confidence interval: {explain}")
        log.formula(
            "mean", "x̄ = Σ x_i / n", f"{values.sum():.3f} / {values.size}", f"{values.mean():.4f}"
        )
        log.text(
            f"resample {values.size} values with replacement, {n_resamples:,} times; "
            f"first resample means: " + ", ".join(f"{m:.3f}" for m in means[:5]) + ", …"
        )
        log.formula(
            f"{confidence:.0%} interval",
            f"[{tail:.1%} quantile, {1 - tail:.1%} quantile] of the resampled means",
            None,
            f"[{low:.4f}, {high:.4f}]",
        )
    return float(values.mean()), float(low), float(high)


# ---------------------------------------------------------------------- test statistics


def binomial_test(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided p-value of ``k`` successes in ``n`` trials with success chance ``p``.

        P(X = i) = C(n, i) · pⁱ · (1 − p)ⁿ⁻ⁱ
        p-value  = Σ P(X = i) over every i with P(X = i) ≤ P(X = k)

    That is, the chance of an outcome at least as unlikely as the one observed.
    """
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    pmf = [math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    threshold = pmf[k] * (1 + 1e-7)  # tolerance for rounding, as SciPy uses
    return min(1.0, sum(q for q in pmf if q <= threshold))


def rank_with_ties(values: np.ndarray) -> np.ndarray:
    """Ranks 1…n of ``values``; tied values share the average of the ranks they span."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2 + 1  # mean of ranks start+1 … end+1
        start = end + 1
    return ranks


def _exact_signed_rank_cdf(statistic: float, n: int) -> float:
    """``P(T ≤ statistic)`` for the signed-rank sum ``T`` of ``n`` untied pairs.

    Under the null hypothesis every rank is positive or negative with probability ½, so all
    2ⁿ sign patterns are equally likely. ``counts[s]`` is the number of patterns whose
    positive ranks sum to ``s``; it is built one rank at a time.
    """
    total = n * (n + 1) // 2
    counts = np.zeros(total + 1)
    counts[0] = 1
    for rank in range(1, n + 1):
        counts[rank:] = counts[rank:] + counts[:-rank].copy()
    return float(counts[: int(math.floor(statistic)) + 1].sum() / 2**n)


def wilcoxon_signed_rank(differences: Sequence[float] | np.ndarray, explain: str | None = None):
    """Two-sided Wilcoxon signed-rank test. Returns ``(statistic, p_value)``.

    1. Drop zero differences; ``n`` pairs remain.
    2. Rank the absolute differences (ties share their average rank).
    3. ``T⁺`` = sum of ranks of positive differences, ``T⁻`` = the same for negative ones.
       The statistic is ``T = min(T⁺, T⁻)``.
    4. If nothing differs, ``T⁺`` would be about ``n(n+1)/4``. The p-value says how unusual the
       observed ``T⁺`` is:

       * up to 50 pairs with no ties or zeros: exactly, ``p = 2·P(T ≤ t)``;
       * otherwise by the normal approximation
         ``z = (T⁺ − n(n+1)/4) / √(n(n+1)(2n+1)/24 − Σ(t³ − t)/48)`` and ``p = 2·(1 − Φ(|z|))``,
         where each ``t`` is the size of a group of tied ranks.
    """
    # Rounding first lets equal differences tie: averages over seeds such as 0.2 come out as
    # 0.1999999999999993 and 0.20000000000000018.
    d = np.round(np.asarray(differences, dtype=float), 9)
    original = d.size
    zeros = int(np.sum(d == 0))
    d = d[d != 0]
    n = d.size
    if n == 0:
        return 0.0, 1.0
    ranks = rank_with_ties(np.abs(d))
    t_plus = float(ranks[d > 0].sum())
    t_minus = float(ranks[d < 0].sum())
    statistic = min(t_plus, t_minus)
    _, tie_sizes = np.unique(np.abs(d), return_counts=True)
    has_ties = bool(np.any(tie_sizes > 1))

    if original <= 50 and not has_ties and zeros == 0:
        method = "exact"
        p_value = min(1.0, 2.0 * _exact_signed_rank_cdf(statistic, n))
    else:
        method = "normal approximation"
        mean = n * (n + 1) / 4
        tie_term = float(np.sum(tie_sizes**3 - tie_sizes))
        sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48)
        z = (t_plus - mean) / sd
        p_value = math.erfc(abs(z) / math.sqrt(2))  # = 2·(1 − Φ(|z|))

    if explain:
        log.section(f"Wilcoxon signed-rank test: {explain}")
        log.text(f"{original} paired differences; {zeros} are zero and dropped, n = {n} remain")
        shown = np.argsort(np.abs(d))[:6]
        log.table(
            ["difference", "|difference|", "rank", "sign"],
            [[float(d[i]), float(abs(d[i])), float(ranks[i]), "+" if d[i] > 0 else "−"]
             for i in shown],
        )  # fmt: skip
        log.note(f"(the {len(shown)} smallest of {n}; equal |differences| share a rank)")
        log.formula("positive rank sum", "T⁺ = Σ rank where d > 0", None, f"{t_plus:.1f}")
        log.formula("negative rank sum", "T⁻ = Σ rank where d < 0", None, f"{t_minus:.1f}")
        if method == "exact":
            log.formula(
                "p-value (exact)", "p = 2·P(T ≤ min(T⁺, T⁻)) over all 2ⁿ sign patterns",
                None, f"{p_value:.4g}",
            )  # fmt: skip
        else:
            log.formula(
                "expected T⁺",
                "μ = n(n+1)/4 if neither method is better",
                f"{n}×{n + 1}/4",
                f"{mean:.1f}",
            )
            log.formula(
                "standard deviation",
                "σ = √(n(n+1)(2n+1)/24 − Σ(t³−t)/48)",
                f"√({n}×{n + 1}×{2 * n + 1}/24 − {tie_term:.0f}/48)",
                f"{sd:.3f}",
            )
            log.formula("z score", "z = (T⁺ − μ) / σ", f"({t_plus:.1f} − {mean:.1f}) / {sd:.3f}",
                        f"{z:.3f}")  # fmt: skip
            log.formula("p-value", "p = 2·(1 − Φ(|z|))", None, f"{p_value:.4g}")
    return statistic, p_value


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


def wilcoxon_nodes_inspected(
    frame: pd.DataFrame, method_a: str, method_b: str, explain: bool = False
) -> dict[str, float]:
    """Wilcoxon signed-rank test on nodes inspected, paired by graph.

    ``median_difference`` is ``a - b``: negative means ``method_a`` inspects fewer nodes.
    """
    pairs = _paired(frame, method_a, method_b)
    difference = (pairs["nodes_inspected_a"] - pairs["nodes_inspected_b"]).to_numpy()
    if np.allclose(difference, 0.0):
        statistic, p_value = 0.0, 1.0  # identical on every graph: nothing to test
    else:
        label = f"nodes inspected, {method_a} − {method_b}" if explain else None
        statistic, p_value = wilcoxon_signed_rank(difference, explain=label)
    return {
        "graphs": float(len(pairs)),
        "median_difference": float(np.median(difference)),
        "mean_difference": float(np.mean(difference)),
        "statistic": statistic,
        "p_value": p_value,
    }


def mcnemar_success(
    frame: pd.DataFrame, method_a: str, method_b: str, explain: bool = False
) -> dict[str, float]:
    """Exact McNemar test on success, paired by graph.

    With several seeds, a method counts as succeeding on a graph when it succeeds in at least
    half of them. Only discordant graphs carry information: ``a_only`` and ``b_only`` count
    the graphs solved by one method and not the other. If both methods were equally good,
    each discordant graph would be equally likely to go either way, so ``a_only`` would follow
    a Binomial(discordant, ½) distribution; the p-value is :func:`binomial_test` on it.
    """
    pairs = _paired(frame, method_a, method_b)
    a = pairs["success_a"] >= 0.5
    b = pairs["success_b"] >= 0.5
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    discordant = a_only + b_only
    p_value = 1.0
    if discordant:
        p_value = binomial_test(a_only, discordant, 0.5)
    if explain:
        both = int((a & b).sum())
        neither = int((~a & ~b).sum())
        log.section(f"McNemar test on success: {method_a} vs {method_b}")
        log.table(
            ["", f"{method_b} solves", f"{method_b} fails"],
            [[f"{method_a} solves", both, a_only], [f"{method_a} fails", b_only, neither]],
        )
        log.text(
            f"Only the {discordant} graphs where exactly one method succeeds say which is better."
        )
        if discordant:
            k = min(a_only, b_only)
            log.formula(
                "binomial probabilities",
                "P(X = i) = C(n, i)·0.5ⁿ,  n = a_only + b_only",
                f"n = {a_only} + {b_only} = {discordant}",
            )
            log.formula(
                "p-value",
                "p = Σ P(X = i) over i at least as extreme as observed",
                f"2 × Σ_(i ≤ {k}) C({discordant}, i)·0.5^{discordant}"
                if a_only != b_only
                else "every outcome is at least as extreme as a tie",
                f"{p_value:.4g}",
            )
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
