"""Assemble processed graph records from the filtered functions and Joern's export."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from soren.data.cfg_builder import (
    MIN_COVERAGE,
    CfgBuildError,
    build_line_cfg,
    rejection_reason,
    select_method,
)
from soren.data.features import featurize
from soren.data.labels import align_flaw_lines
from soren.data.schema import GraphRecord, SchemaError

STAGES: tuple[str, ...] = (
    "filtered_functions",
    "parsed_by_joern",
    "cfg_built",
    "cfg_usable",
    "labels_aligned",
    "size_within_bounds",
)


@dataclass
class BuildConfig:
    min_nodes: int = 5
    max_nodes: int = 300
    """Bounds on the number of CFG nodes, ENTRY and EXIT included."""
    min_coverage: float = MIN_COVERAGE


def build_graphs(
    filtered: pd.DataFrame,
    methods: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    config: BuildConfig | None = None,
) -> tuple[list[GraphRecord], list[dict[str, int | str]], dict[str, int]]:
    """Turn every filtered function into a :class:`GraphRecord`, or record why it was dropped.

    Returns ``(records, attrition, reasons)``. ``attrition`` lists the functions surviving
    each stage, in the order of :data:`STAGES`; ``reasons`` counts the drops by cause.
    """
    config = config or BuildConfig()
    survived = Counter({"filtered_functions": len(filtered)})
    reasons: Counter[str] = Counter()
    records: list[GraphRecord] = []

    for row in filtered.itertuples(index=False):
        method = select_method(methods.get(row.sample_id, {}))
        if method is None:
            reasons["not_parsed"] += 1
            continue
        survived["parsed_by_joern"] += 1

        try:
            cfg = build_line_cfg(method, row.func_before)
        except CfgBuildError:
            reasons["no_statements"] += 1
            continue
        survived["cfg_built"] += 1

        reason = rejection_reason(cfg, config.min_coverage)
        if reason is not None:
            reasons[reason] += 1
            continue
        survived["cfg_usable"] += 1

        alignment = align_flaw_lines(cfg, list(row.flaw_line_indices), list(row.flaw_lines))
        if not alignment.ok:
            reasons[alignment.reason or "unaligned"] += 1
            continue
        survived["labels_aligned"] += 1

        if len(cfg.nodes) < config.min_nodes:
            reasons["too_few_nodes"] += 1
            continue
        if len(cfg.nodes) > config.max_nodes:
            reasons["too_many_nodes"] += 1
            continue

        try:
            record = cfg.to_record(
                row.sample_id,
                alignment.vuln_nodes,
                project=row.project,
                commit_id=row.commit_id,
                cwe=row.cwe,
            )
        except SchemaError:
            reasons["invalid_record"] += 1
            continue
        # Features are computed on demand from the record, not stored; this only checks that
        # every tier can be computed for the graph.
        for tier in ("S", "L"):
            if not np.isfinite(featurize(record, tier)).all():
                raise ValueError(f"{row.sample_id}: non-finite Tier {tier} features")
        survived["size_within_bounds"] += 1
        records.append(record)

    attrition = [{"step": stage, "rows": int(survived[stage])} for stage in STAGES]
    return records, attrition, dict(sorted(reasons.items()))
