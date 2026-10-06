#!/usr/bin/env python3
"""Add statement embeddings (Tier E features) to the processed graphs.

Example:
    python scripts/07_embed_statements.py --model .tools/codebert-base

Encodes every statement with a frozen pretrained code model, fits PCA on the training split
only, and rewrites ``graphs_{train,val,test}.jsonl`` in place with the reduced embeddings in
each record's ``features["embed"]``. The projection is saved next to them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from soren.data.embeddings import (
    DEFAULT_DIM,
    DEFAULT_MODEL,
    Projection,
    attach_embeddings,
    encode_texts,
    statement_texts,
)
from soren.data.schema import read_jsonl, write_jsonl

SPLITS = ("train", "val", "test")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="data/processed", help="holds graphs_<split>.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model name or local directory")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    directory = Path(args.dir)
    graphs = {split: read_jsonl(directory / f"graphs_{split}.jsonl") for split in SPLITS}
    texts = {s: [t for g in graphs[s] for t in statement_texts(g)] for s in SPLITS}
    flat = [t for s in SPLITS for t in texts[s]]
    print(f"encoding {len(flat)} statements ({len(set(flat))} distinct) with {args.model}")
    vectors = encode_texts(flat, args.model, args.batch_size)

    bounds = np.cumsum([0, *(len(texts[s]) for s in SPLITS)])
    per_split = {s: vectors[bounds[i] : bounds[i + 1]] for i, s in enumerate(SPLITS)}
    projection = Projection.fit(per_split["train"], args.dim)  # training split only
    projection.save(directory / "embed_projection.npz")
    for split in SPLITS:
        attach_embeddings(graphs[split], per_split[split], projection)
        write_jsonl(graphs[split], directory / f"graphs_{split}.jsonl")
        print(f"{split}: {len(graphs[split])} graphs")
    print(f"wrote embeddings to {directory}/graphs_*.jsonl")


if __name__ == "__main__":
    main()
