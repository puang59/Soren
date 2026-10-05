#!/usr/bin/env python3
"""Write one C file per filtered function, in batches ready for Joern.

Example:
    python scripts/02_write_sources.py --input data/interim/filtered.parquet

Writes ``data/interim/src/batch_XXXX/<sample_id>.c`` and ``data/interim/src/manifest.csv``.
"""

from __future__ import annotations

import argparse

import pandas as pd

from soren.data.sources import BATCH_SIZE, write_sources


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="data/interim/filtered.parquet")
    parser.add_argument("--out", default="data/interim/src")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)

    frame = pd.read_parquet(args.input, columns=["sample_id", "func_before"])
    manifest = write_sources(frame, args.out, args.batch_size)
    print(f"wrote {len(manifest)} files in {manifest['batch'].nunique()} batches to {args.out}")


if __name__ == "__main__":
    main()
