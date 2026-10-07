"""The data the web visualizer in ``web-app/`` reads: methods, graphs and replayable traces.

Shared by ``scripts/export_webapp.py``, which writes the static episodes file, and
``soren.serve``, which analyses a pasted function on request.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from soren.baselines.base import Searcher
from soren.baselines.heuristic import HeuristicScorer, make_baseline
from soren.config import load_config, load_yaml
from soren.data.schema import GraphRecord
from soren.env.cfg_nav_env import EnvConfig
from soren.env.rewards import RewardConfig
from soren.viz.render import action_labels
from soren.viz.trace import Trace

BASELINES = ("dfs", "bfs", "line_order", "heuristic_first")
BASELINE_NAMES = {
    "dfs": "Depth-first search",
    "bfs": "Breadth-first search",
    "line_order": "Line order",
    "heuristic_first": "Heuristic-first",
}
DECLARES = "must declare"
ORACLE = "stops at the answer"


@dataclass
class Methods:
    """The methods the page offers, in menu order, and what it says about each."""

    env: EnvConfig
    searchers: dict[str, Searcher]
    names: dict[str, str]
    info: dict[str, dict[str, Any]]
    scorer: HeuristicScorer

    def header(self) -> dict[str, Any]:
        return {"methods": self.names, "method_info": self.info}

    def needs_labels(self, name: str) -> bool:
        """An oracle stop is defined by the ground truth, so it cannot run without one."""
        return self.info[name]["rule"] == "oracle"

    def traces(self, graph: GraphRecord, labelled: bool = True) -> dict[str, dict[str, Any]]:
        budget = self.env.max_steps_for(graph.num_nodes)
        return {
            name: trace_payload(graph, searcher.trace(graph, budget, np.random.default_rng(0)))
            for name, searcher in self.searchers.items()
            if labelled or not self.needs_labels(name)
        }


def load_methods(
    checkpoint: str | Path,
    env_config: str | Path = "configs/env.yaml",
    eval_config: str | Path = "configs/eval.yaml",
) -> Methods:
    """The trained agent (when ``checkpoint`` exists) and every baseline under both protocols."""
    env = load_config(EnvConfig, env_config, section="env")
    reward = load_config(RewardConfig, env_config, section="reward")
    thresholds = load_yaml(eval_config)["protocol_b"]["heuristic"]
    scorer = HeuristicScorer()
    searchers: dict[str, Searcher] = {}
    names: dict[str, str] = {}
    info: dict[str, dict[str, Any]] = {}
    if Path(checkpoint).is_file():
        from soren.agents.searcher import PolicySearcher
        from soren.agents.train import load_model

        searchers["ppo"] = PolicySearcher(load_model(checkpoint), env, reward, name="ppo")
        names["ppo"] = "PPO agent"
        info["ppo"] = {"rule": "policy"}
    # Declaring baselines first: they are the fair comparison for the agent.
    for name in BASELINES:
        declaring = make_baseline(name, env, reward, scorer, thresholds[name])
        searchers[declaring.name] = declaring
        names[declaring.name] = f"{BASELINE_NAMES[name]} ({DECLARES})"
        info[declaring.name] = {"rule": "threshold", "threshold": thresholds[name]}
    for name in BASELINES:
        searchers[name] = make_baseline(name, env, reward)
        names[name] = f"{BASELINE_NAMES[name]} ({ORACLE})"
        info[name] = {"rule": "oracle"}
    return Methods(env, searchers, names, info, scorer)


def trace_payload(graph: GraphRecord, trace: Trace) -> dict[str, Any]:
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


def graph_payload(graph: GraphRecord, scorer: HeuristicScorer) -> dict[str, Any]:
    """One function as the page draws it, without background details or traces."""
    return {
        "id": graph.sample_id,
        "project": graph.project,
        "cwe": graph.cwe,
        "commit": graph.commit_id,
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
        "scores": [round(float(score), 3) for score in scorer.scores(graph)],
        "traces": {},
    }
