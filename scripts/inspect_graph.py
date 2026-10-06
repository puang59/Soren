#!/usr/bin/env python3
"""Print graphs as text next to their source, for spot-checking the pipeline by eye.

Example:
    python scripts/inspect_graph.py --graphs data/processed/graphs_train.jsonl --sample 5

Each source line is shown with the node that covers it, its kind and where control can go
next (as line numbers). Lines of vulnerable nodes are marked with ``>>``.
"""

from __future__ import annotations

import argparse

import numpy as np

from soren.data.schema import GraphRecord, read_jsonl


def describe(record: GraphRecord) -> str:
    line_of = {node.id: node.line for node in record.nodes}

    def target(node_id: int) -> str:
        kind = record.nodes[node_id].kind
        return kind if kind in ("ENTRY", "EXIT") else f"L{line_of[node_id]}"

    by_line: dict[int, list] = {}
    for node in record.nodes:
        if node.kind not in ("ENTRY", "EXIT"):
            by_line.setdefault(node.line, []).append(node)

    out = [
        f"== {record.sample_id}  {record.project}  {record.cwe}  "
        f"{record.num_nodes} nodes, {len(record.vuln_nodes)} vulnerable",
        f"   ENTRY -> {', '.join(target(s) for s in record.successors(record.entry))}",
    ]
    for number, text in enumerate(record.source_lines, start=1):
        nodes = by_line.get(number, [])
        mark = ">>" if any(n.id in record.vuln_set for n in nodes) else "  "
        if nodes:
            node = nodes[0]
            flow = ", ".join(target(s) for s in record.successors(node.id)) or "-"
            extra = f" (+{len(nodes) - 1} dispatch)" if len(nodes) > 1 else ""
            note = f"{node.kind:<7}-> {flow}{extra}"
        else:
            note = ""
        out.append(f"{mark}{number:>4} | {text[:70]:<70} | {note}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--ids", nargs="*", help="sample ids to print")
    parser.add_argument("--sample", type=int, default=3, help="random graphs to print")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, help="only sample graphs up to this size")
    args = parser.parse_args(argv)

    records = read_jsonl(args.graphs)
    if args.ids:
        chosen = [r for r in records if r.sample_id in set(args.ids)]
    else:
        pool = [r for r in records if args.max_nodes is None or r.num_nodes <= args.max_nodes]
        rng = np.random.default_rng(args.seed)
        picks = rng.choice(len(pool), size=min(args.sample, len(pool)), replace=False)
        chosen = [pool[i] for i in sorted(picks)]
    print("\n\n".join(describe(record) for record in chosen))


if __name__ == "__main__":
    main()
