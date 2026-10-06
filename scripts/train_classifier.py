#!/usr/bin/env python3
"""Train the supervised node classifier and report its top-k accuracy.

Example:
    python scripts/train_classifier.py --out runs/classifier/model.pt

The classifier ranks every statement of a function by predicted probability of being
vulnerable. It is a reference for how much the features can say, not a traversal method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soren.baselines.heuristic import HeuristicScorer
from soren.baselines.node_classifier import NodeClassifier, chance_top_k, top_k_accuracy
from soren.data.schema import read_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", default="data/processed/graphs_train.jsonl")
    parser.add_argument("--val", default="data/processed/graphs_val.jsonl")
    parser.add_argument("--out", default="runs/classifier/model.pt")
    parser.add_argument("--tier", default="L", choices=["S", "L"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    train, val = read_jsonl(args.train), read_jsonl(args.val)
    classifier = NodeClassifier(args.tier, args.hidden, args.epochs, seed=args.seed).fit(train, val)
    path = classifier.save(args.out)
    heuristic = HeuristicScorer()
    report = {
        "classifier_val": classifier.top_k(val),
        "classifier_train": classifier.top_k(train),
        "heuristic_val": top_k_accuracy([heuristic.scores(g) for g in val], val),
        "chance_val": chance_top_k(val),
    }
    Path(path).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"{'':<18}{'top-1':>8}{'top-3':>8}{'top-5':>8}")
    for name, metrics in report.items():
        print(f"{name:<18}{metrics['top1']:>8.3f}{metrics['top3']:>8.3f}{metrics['top5']:>8.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
