#!/usr/bin/env python3
"""Tabulate the arms of an experiment group from their ``metrics.json`` files.

Example:
    python scripts/collect_runs.py --group sweep --out experiments/sweep.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COLUMNS = [
    "success_rate",
    "localization_accuracy",
    "train_success",
    "nodes_inspected_success",
    "actions_success",
    "cumulative_reward",
]


def collect(group_dir: Path) -> pd.DataFrame:
    rows = [json.loads(path.read_text()) for path in sorted(group_dir.glob("*/seed*/metrics.json"))]
    if not rows:
        raise SystemExit(f"no metrics.json files under {group_dir}")
    frame = pd.DataFrame(rows)
    grouped = frame.groupby("arm", sort=False)
    table = grouped[COLUMNS].mean()
    table.insert(0, "seeds", grouped["seed"].nunique())
    table.insert(2, "success_std", grouped["success_rate"].std(ddof=0))
    return table.sort_values("success_rate", ascending=False).reset_index()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", required=True)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", help="also write the table to this CSV file")
    args = parser.parse_args(argv)
    table = collect(Path(args.runs) / args.group)
    print(table.round(3).to_string(index=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
