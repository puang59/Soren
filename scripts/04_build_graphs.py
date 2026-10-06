#!/usr/bin/env python3
"""Build processed graph records from the filtered functions and Joern's export.

Example:
    python scripts/04_build_graphs.py

For each function: pick the Joern method, collapse it to a line-level CFG, reject bad parses,
align the flaw lines to nodes and check the size. Writes ``data/processed/graphs_all.jsonl``
and adds the per-stage counts and drop reasons to ``data/processed/attrition.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from soren.config import from_dict, load_yaml
from soren.data.build import BuildConfig, build_graphs
from soren.data.cfg_builder import load_joern_methods
from soren.data.schema import write_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--filtered", default="data/interim/filtered.parquet")
    parser.add_argument("--joern", default="data/interim/joern", help="directory of batch exports")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output", default="data/processed/graphs_all.jsonl")
    parser.add_argument("--attrition", default="data/processed/attrition.json")
    args = parser.parse_args(argv)

    exports = sorted(Path(args.joern).glob("batch_*.jsonl"))
    if not exports:
        parser.error(f"no batch_*.jsonl files in {args.joern}; run scripts/03_run_joern.sh first")
    config = from_dict(BuildConfig, load_yaml(args.config).get("graphs"))
    filtered = pd.read_parquet(args.filtered)
    records, steps, reasons = build_graphs(filtered, load_joern_methods(exports), config)
    count = write_jsonl(records, args.output)

    attrition_path = Path(args.attrition)
    log = json.loads(attrition_path.read_text()) if attrition_path.exists() else {}
    log["graph_steps"] = steps
    log["graph_drop_reasons"] = reasons
    attrition_path.parent.mkdir(parents=True, exist_ok=True)
    attrition_path.write_text(json.dumps(log, indent=2))

    width = max(len(str(step["step"])) for step in steps)
    for step in steps:
        print(f"{step['step']:<{width}}  {step['rows']:>6}")
    print("dropped:", ", ".join(f"{name} {n}" for name, n in reasons.items()) or "nothing")
    print(f"wrote {count} graphs to {args.output}")


if __name__ == "__main__":
    main()
