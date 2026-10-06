#!/usr/bin/env python3
"""Export episodes for the web visualizer in ``web-app/``.

Example:
    python scripts/export_webapp.py

Runs the trained agent and the baselines on test functions and writes everything the page
needs (source, control flow graph, labels, and each method's step-by-step trace) to
``web-app/data/episodes.js``. The page is static: it reads that one file and needs no server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from soren.baselines.heuristic import make_baseline
from soren.config import load_config
from soren.data.schema import GraphRecord, read_jsonl
from soren.env.cfg_nav_env import EnvConfig
from soren.env.rewards import RewardConfig
from soren.viz.render import action_labels
from soren.viz.trace import Trace

BASELINES = ("dfs", "bfs", "line_order", "heuristic_first")
METHOD_NAMES = {
    "ppo": "PPO agent",
    "dfs": "Depth-first search",
    "bfs": "Breadth-first search",
    "line_order": "Line order",
    "heuristic_first": "Heuristic-first",
}


def trace_payload(graph: GraphRecord, trace: Trace) -> dict:
    steps = []
    for step in trace.steps:
        entry = {
            "node": step.node,
            "next": step.next_node,
            "action": step.action,
            "reward": round(step.reward, 4),
        }
        if step.probs is not None:
            labels = action_labels(graph, step.node, len(step.probs))
            mask = step.mask or [True] * len(labels)
            entry["options"] = [
                {"label": label, "p": round(float(p), 4)}
                for label, p, ok in zip(labels, step.probs, mask, strict=True)
                if ok
            ]
            entry["value"] = round(float(step.value), 3)
        steps.append(entry)
    return {
        "start": trace.start_node,
        "steps": steps,
        "success": bool(trace.outcome.get("success")),
        "end_reason": trace.outcome.get("end_reason"),
    }


def graph_payload(graph: GraphRecord) -> dict:
    return {
        "id": graph.sample_id,
        "project": graph.project,
        "cwe": graph.cwe,
        "source": graph.source_lines,
        "nodes": [
            {"id": n.id, "line": n.line, "kind": n.kind, "depth": n.depth, "code": n.code}
            for n in graph.nodes
        ],
        "edges": [list(edge) for edge in graph.edges],
        "back_edges": [list(edge) for edge in graph.back_edges],
        "entry": graph.entry,
        "exit": graph.exit,
        "vulnerable": graph.vuln_nodes,
        "traces": {},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphs", default="data/processed/graphs_test.jsonl")
    parser.add_argument("--checkpoint", default="runs/final/base/seed1/best_model.zip")
    parser.add_argument("--out", default="web-app/data/episodes.js")
    parser.add_argument("--max-nodes", type=int, default=32, help="skip larger functions")
    parser.add_argument("--max-lines", type=int, default=60)
    parser.add_argument("--env-config", default="configs/env.yaml")
    args = parser.parse_args(argv)

    env = load_config(EnvConfig, args.env_config, section="env")
    reward = load_config(RewardConfig, args.env_config, section="reward")
    searchers = {name: make_baseline(name, env, reward) for name in BASELINES}
    if Path(args.checkpoint).is_file():
        from soren.agents.searcher import PolicySearcher
        from soren.agents.train import load_model

        policy = PolicySearcher(load_model(args.checkpoint), env, reward, name="ppo")
        searchers = {"ppo": policy, **searchers}
    else:
        print(f"no checkpoint at {args.checkpoint}; exporting baselines only")

    graphs = [
        g
        for g in read_jsonl(args.graphs)
        if g.num_nodes <= args.max_nodes and len(g.source_lines) <= args.max_lines
    ]
    episodes = []
    for graph in graphs:
        payload = graph_payload(graph)
        budget = env.max_steps_for(graph.num_nodes)
        for name, searcher in searchers.items():
            trace = searcher.trace(graph, budget, np.random.default_rng(0))
            payload["traces"][name] = trace_payload(graph, trace)
        episodes.append(payload)

    data = {"methods": {name: METHOD_NAMES[name] for name in searchers}, "episodes": episodes}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # A script that sets a global, so the page also works when opened straight from disk.
    out.write_text("window.SOREN_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {len(episodes)} functions x {len(searchers)} methods to {out}")
    print(f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
