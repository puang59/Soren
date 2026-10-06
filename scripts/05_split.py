#!/usr/bin/env python3
"""Split the processed graphs into train, validation and test sets by fixing commit.

Example:
    python scripts/05_split.py

Writes ``data/processed/graphs_{train,val,test}.jsonl`` and ``data/processed/split.json``.
Fails if any commit or function body would be shared between splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soren.config import from_dict, load_yaml
from soren.data.schema import read_jsonl, write_jsonl
from soren.data.split import SplitConfig, split_by_commit, split_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="data/processed/graphs_all.jsonl")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)

    config = from_dict(SplitConfig, load_yaml(args.config).get("split"))
    splits = split_by_commit(read_jsonl(args.input), config)
    out_dir = Path(args.out_dir)
    for name, records in splits.items():
        write_jsonl(records, out_dir / f"graphs_{name}.jsonl")
    summary = split_summary(splits)
    (out_dir / "split.json").write_text(
        json.dumps({"seed": config.seed, "splits": summary}, indent=2)
    )
    for name, info in summary.items():
        cwes = ", ".join(f"{cwe} {n}" for cwe, n in info["by_cwe"].items())
        print(f"{name:<5} {info['graphs']:>5} graphs ({info['share']:.1%})  "
              f"{info['commits']:>4} commits  {cwes}")  # fmt: skip
    print(f"wrote graphs_{{train,val,test}}.jsonl to {out_dir}")


if __name__ == "__main__":
    main()
