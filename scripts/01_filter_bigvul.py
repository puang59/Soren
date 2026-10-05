#!/usr/bin/env python3
"""Filter BigVul down to usable vulnerable functions and log the attrition.

Example:
    python scripts/01_filter_bigvul.py --input data/raw/MSR_data_cleaned.csv

Writes ``data/interim/filtered.parquet`` and ``data/processed/attrition.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soren.config import load_yaml
from soren.data.bigvul import load_bigvul
from soren.data.filtering import FilterConfig, filter_bigvul


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="BigVul CSV or canonical Parquet")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output", default="data/interim/filtered.parquet")
    parser.add_argument("--attrition", default="data/processed/attrition.json")
    args = parser.parse_args(argv)

    settings = load_yaml(args.config)
    config = FilterConfig(
        cwes=list(settings.get("cwes", FilterConfig().cwes)),
        min_lines=int(settings.get("min_lines", FilterConfig.min_lines)),
        max_lines=int(settings.get("max_lines", FilterConfig.max_lines)),
    )
    filtered, attrition = filter_bigvul(load_bigvul(args.input), config)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(output, index=False)
    attrition_path = Path(args.attrition)
    attrition_path.parent.mkdir(parents=True, exist_ok=True)
    attrition_path.write_text(json.dumps({"cwes": config.cwes, "steps": attrition}, indent=2))

    width = max(len(str(entry["step"])) for entry in attrition)
    for entry in attrition:
        print(f"{entry['step']:<{width}}  {entry['rows']:>8}")
    print(f"wrote {len(filtered)} functions to {output}")
    for cwe, count in filtered["cwe"].value_counts().items():
        print(f"  {cwe}: {count}")


if __name__ == "__main__":
    main()
