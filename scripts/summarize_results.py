#!/usr/bin/env python3
"""Summarise an evaluation results file: confidence intervals, tests and seed spread.

Example:
    python scripts/summarize_results.py --results runs/demo/eval_val.parquet --reference ppo

Prints each method's success rate and search cost with bootstrap confidence intervals over
graphs, and p-values for every method against the reference (McNemar on success, Wilcoxon
signed-rank on nodes inspected, both paired by graph).
"""

from __future__ import annotations

import argparse

from soren.eval.metrics import metrics_across_seeds
from soren.eval.runner import load_results
from soren.eval.stats import comparison_table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, help="per-episode results (Parquet)")
    parser.add_argument("--reference", default="ppo", help="method the others are tested against")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--out", help="also write the comparison table to this CSV file")
    args = parser.parse_args(argv)

    frame = load_results(args.results)
    table = comparison_table(frame, args.reference, args.confidence, args.resamples)
    print(f"mean [{args.confidence:.0%} bootstrap CI over graphs]; p-values vs {args.reference}\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}", na_rep="-"))

    seeds = metrics_across_seeds(frame)
    multi = seeds[seeds["seeds"] > 1]
    if not multi.empty:
        columns = ["method", "seeds", "success_rate_mean", "success_rate_std"]
        columns += ["nodes_inspected_all_mean", "nodes_inspected_all_std"]
        print("\nspread across seeds:\n")
        print(multi[columns].round(4).to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
