"""Summary metrics over episode results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

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
