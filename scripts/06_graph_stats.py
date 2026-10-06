#!/usr/bin/env python3
"""Describe the training graphs: sizes, branching and labels.

Example:
    python scripts/06_graph_stats.py

Writes ``data/processed/stats.json``. The statistics come from the training split only, so
nothing about validation or test graphs feeds into the environment's settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from soren.config import load_config
from soren.data.schema import read_jsonl
from soren.data.stats import graph_statistics
from soren.env.cfg_nav_env import EnvConfig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphs", default="data/processed/graphs_train.jsonl")
    parser.add_argument("--output", default="data/processed/stats.json")
    parser.add_argument("--env-config", default="configs/env.yaml")
    args = parser.parse_args(argv)

    records = read_jsonl(args.graphs)
    stats = graph_statistics(records)
    env = load_config(EnvConfig, args.env_config, section="env")
    budgets = [env.max_steps_for(g.num_nodes) for g in records]
    stats["env"] = {
        "k": env.k,
        "graphs_wider_than_k": sum(
            max(len(g.successors(n.id)) for n in g.nodes) > env.k for g in records
        ),
        "graphs_at_step_cap": sum(b == env.max_steps_cap for b in budgets),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))

    nodes, vuln = stats["nodes"], stats["vulnerable_nodes"]
    print(f"{stats['graphs']} graphs")
    print(f"nodes: median {nodes['p50']:.0f}, p95 {nodes['p95']:.0f}, max {nodes['p100']:.0f}")
    print(f"vulnerable nodes: median {vuln['p50']:.0f}, mean {vuln['mean']:.2f}; "
          f"single {stats['single_vulnerable_node_share']:.1%}")  # fmt: skip
    distance = stats["distance_entry_to_vulnerable"]
    print(f"moves from entry to nearest vulnerable node: median {distance['p50']:.0f}, "
          f"p95 {distance['p95']:.0f}")  # fmt: skip
    print("out-degree  nodes covered  graphs covered")
    for row in stats["out_degree"][:9]:
        print(f"{row['out_degree']:>10}  {row['nodes_covered']:>13.4f}  "
              f"{row['graphs_covered']:>14.3f}")  # fmt: skip
    print(f"smallest K covering 99% of nodes: {stats['smallest_k_for_99_percent_of_nodes']}")
    wider, capped = stats["env"]["graphs_wider_than_k"], stats["env"]["graphs_at_step_cap"]
    print(f"configured K = {env.k}: {wider} graphs are wider")
    print(f"{capped} graphs hit the {env.max_steps_cap}-step cap")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
