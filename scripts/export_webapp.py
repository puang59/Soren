#!/usr/bin/env python3
"""Export episodes for the web visualizer in ``web-app/``.

Example:
    python scripts/export_webapp.py

Runs the trained agent and the baselines on test functions and writes everything the page
needs (source, control flow graph, labels, and each method's step-by-step trace) to
``web-app/data/episodes.js``. The page is static: it reads that one file and needs no server.

Each baseline is exported twice: under Protocol B, where it declares at the first statement
whose heuristic score reaches its tuned threshold and can be wrong like the agent, and under
Protocol A, where it is stopped on reaching a vulnerable statement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from soren.data.schema import GraphRecord, read_jsonl
from soren.viz.explain import CWE_NAMES, fix_diff
from soren.viz.webapp import graph_payload, load_methods


def load_background(raw_dir: str, filtered: str) -> dict[str, dict]:
    """CVE, commit and fixed function body per sample, where the raw dataset is available."""
    background: dict[str, dict] = {}
    if Path(filtered).is_file():
        frame = pd.read_parquet(filtered, columns=["sample_id", "func_before", "func_after"])
        for row in frame.itertuples(index=False):
            background[row.sample_id] = {"before": row.func_before, "after": row.func_after}
    files = sorted(Path(raw_dir).glob("*.parquet"))
    if files:
        columns = ["CVE ID", "CVE Page", "commit_message", "codeLink"]
        raw = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
        # Sample ids are positions in this concatenation, as assigned by load_bigvul.
        for sample_id, entry in background.items():
            row = raw.iloc[int(sample_id.rsplit("_", 1)[1])]
            entry.update(
                cve=row["CVE ID"] if isinstance(row["CVE ID"], str) else "",
                cve_url=row["CVE Page"] if isinstance(row["CVE Page"], str) else "",
                commit_url=row["codeLink"] if isinstance(row["codeLink"], str) else "",
                commit_message=(
                    row["commit_message"].strip()[:700]
                    if isinstance(row["commit_message"], str)
                    else ""
                ),
            )
    return background


def background_payload(graph: GraphRecord, background: dict) -> dict:
    info = background.get(graph.sample_id, {})
    return {
        "cwe_name": CWE_NAMES.get(graph.cwe, ""),
        "cve": info.get("cve", ""),
        "cve_url": info.get("cve_url", ""),
        "commit_url": info.get("commit_url", ""),
        "commit_message": info.get("commit_message", ""),
        "fix": fix_diff(info["before"], info["after"]) if info.get("after") else [],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphs", default="data/processed/graphs_test.jsonl")
    parser.add_argument("--checkpoint", default="runs/final/base/seed1/best_model.zip")
    parser.add_argument("--out", default="web-app/data/episodes.js")
    parser.add_argument("--max-nodes", type=int, default=32, help="skip larger functions")
    parser.add_argument("--max-lines", type=int, default=60)
    parser.add_argument("--env-config", default="configs/env.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml", help="Protocol B thresholds")
    parser.add_argument("--raw", default="data/raw/bigvul_hf", help="for CVE and commit details")
    parser.add_argument("--filtered", default="data/interim/filtered.parquet", help="for the fix")
    args = parser.parse_args(argv)

    methods = load_methods(args.checkpoint, args.env_config, args.eval_config)
    if "ppo" not in methods.searchers:
        print(f"no checkpoint at {args.checkpoint}; exporting baselines only")

    graphs = [
        g
        for g in read_jsonl(args.graphs)
        if g.num_nodes <= args.max_nodes and len(g.source_lines) <= args.max_lines
    ]
    background = load_background(args.raw, args.filtered)
    episodes = []
    for graph in graphs:
        payload = graph_payload(graph, methods.scorer)
        payload.update(background_payload(graph, background))
        payload["traces"] = methods.traces(graph)
        episodes.append(payload)

    data = {**methods.header(), "episodes": episodes}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # A script that sets a global, so the page also works when opened straight from disk.
    out.write_text("window.SOREN_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {len(episodes)} functions x {len(methods.searchers)} methods to {out}")
    print(f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
