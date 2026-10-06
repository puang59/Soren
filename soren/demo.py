"""Terminal demo: watch a method walk a function's control flow graph.

    python -m soren.demo                      # the trained agent on a test function
    python -m soren.demo --method dfs         # a baseline on the same function
    python -m soren.demo --compare dfs        # the agent, then DFS, then a summary
    python -m soren.demo --step               # advance with Enter instead of a timer
    python -m soren.demo --synthetic          # no dataset or checkpoint needed

The source is shown with the current line highlighted, visited lines marked, and the policy's
action probabilities under it. The ground truth stays hidden until the method has declared.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from soren.baselines.heuristic import BASELINE_FACTORIES, make_baseline
from soren.data.schema import GraphRecord, read_jsonl
from soren.data.synthetic import SyntheticConfig, generate_dataset
from soren.env.cfg_nav_env import EnvConfig
from soren.viz.render import action_labels, replay_state
from soren.viz.trace import Trace

DEFAULT_GRAPHS = "data/processed/graphs_test.jsonl"
DEFAULT_CHECKPOINT = "runs/final/base/seed1/best_model.zip"
WIDTH = 78

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
CURRENT, VISITED = "\033[30;43m", "\033[36m"  # black on yellow; cyan
CORRECT, WRONG, TRUTH = "\033[30;42m", "\033[30;41m", "\033[35m"  # on green; on red; magenta


def paint(text: str, code: str, colour: bool) -> str:
    return f"{code}{text}{RESET}" if colour and code else text


def render_frame(
    graph: GraphRecord, trace: Trace, step: int, show_truth: bool = False, colour: bool = True
) -> str:
    """One screen of the demo, as text: header, annotated source, and what happens next."""
    state = replay_state(trace, step)
    total = len(trace.steps)
    line_of = {node.id: node.line for node in graph.nodes}
    current_line = line_of[state.current]
    visited = {line_of[n] for n in state.visits}
    declared = {line_of[n]: n in graph.vuln_set for n in state.declared}
    truth = {line_of[n] for n in graph.vuln_nodes} if show_truth else set()

    out = [
        paint(f" {trace.method.upper()} on {graph.sample_id} ", BOLD, colour)
        + f" {graph.project or 'synthetic'} · {graph.cwe} · {graph.num_nodes} nodes",
        f" step {step}/{total} · nodes inspected {len(state.visits)}"
        f" · return {state.running_return:+.2f}",
        "",
    ]
    for number, text in enumerate(graph.source_lines, start=1):
        body = text.replace("\t", "    ")[: WIDTH - 10]
        mark, code = " ", ""
        if number in visited:
            mark, code = "·", VISITED
        if number == current_line and not state.finished:
            mark, code = "▶", CURRENT
        if number in declared:
            mark, code = ("✔", CORRECT) if declared[number] else ("✘", WRONG)
        flag = paint("◆", TRUTH, colour) if number in truth else " "
        out.append(f" {flag}{mark} {number:>3} │ " + paint(f"{body:<{WIDTH - 10}}", code, colour))
    out.append("")

    if step < total:
        upcoming = trace.steps[step]
        out.append(f" next action: {paint(upcoming.action, BOLD, colour)}")
        if upcoming.probs is not None:
            labels = action_labels(graph, upcoming.node, len(upcoming.probs))
            mask = upcoming.mask or [True] * len(labels)
            options = [
                (p, name) for p, name, ok in zip(upcoming.probs, labels, mask, strict=True) if ok
            ]
            for probability, name in sorted(options, reverse=True)[:4]:
                bar = "█" * round(probability * 30)
                out.append(f"   {name:<20} {probability:5.2f} {bar}")
            out.append(paint(f"   value estimate {upcoming.value:+.2f}", DIM, colour))
    else:
        outcome = trace.outcome
        verdict = "FOUND the vulnerable statement" if outcome.get("success") else "MISSED"
        code = CORRECT if outcome.get("success") else WRONG
        out.append(
            " "
            + paint(f" {verdict} ", code, colour)
            + f" {total} actions, {outcome.get('nodes_inspected')} of {graph.num_nodes} nodes"
            f" inspected ({outcome.get('end_reason')})"
        )
    out.append(
        paint(
            " ▶ current   · visited   ✔/✘ declared   ◆ ground truth (shown at the end)",
            DIM,
            colour,
        )
    )
    return "\n".join(out)


def play(graph: GraphRecord, trace: Trace, delay: float, step_mode: bool, colour: bool) -> None:
    """Animate a trace, then reveal the ground truth."""
    interactive = sys.stdout.isatty()
    frames = [(step, False) for step in range(len(trace.steps) + 1)]
    frames.append((len(trace.steps), True))
    for index, (step, reveal) in enumerate(frames):
        if interactive:
            print("\033[H\033[2J", end="")
        elif index:
            print("\n" + "─" * WIDTH)
        print(render_frame(graph, trace, step, reveal, colour))
        if index == len(frames) - 1:
            break
        if step_mode and interactive:
            input(paint(" press Enter", DIM, colour))
        elif interactive:
            time.sleep(delay)


def make_searcher(method: str, checkpoint: str, env: EnvConfig):
    if method != "ppo":
        return make_baseline(method, env)
    from soren.agents.searcher import PolicySearcher
    from soren.agents.train import load_model

    if not Path(checkpoint).is_file():
        raise SystemExit(
            f"no checkpoint at {checkpoint}; train one (see the README), pass --checkpoint, "
            "or run a baseline with --method dfs"
        )
    return PolicySearcher(load_model(checkpoint), env, name="ppo")


def choose_graph(
    graphs: list[GraphRecord], searcher, env: EnvConfig, want_success: bool
) -> GraphRecord:
    """A function small enough to fit on screen where the outcome is the one asked for."""
    rng = np.random.default_rng(0)
    candidates = [g for g in graphs if 10 <= g.num_nodes <= 26 and len(g.source_lines) <= 34]
    for graph in candidates or graphs:
        trace = searcher.trace(graph, env.max_steps_for(graph.num_nodes), rng)
        if bool(trace.outcome.get("success")) == want_success and len(trace.steps) >= 3:
            return graph
    return (candidates or graphs)[0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graphs", default=DEFAULT_GRAPHS, help="graph records (JSONL)")
    parser.add_argument("--graph", help="sample id to show; chosen automatically if omitted")
    parser.add_argument("--method", default="ppo", choices=["ppo", *sorted(BASELINE_FACTORIES)])
    parser.add_argument("--compare", choices=sorted(BASELINE_FACTORIES), help="then run this too")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--failure", action="store_true", help="pick a function the method misses")
    parser.add_argument("--delay", type=float, default=0.9, help="seconds per step")
    parser.add_argument("--step", action="store_true", help="advance with Enter")
    parser.add_argument("--no-colour", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="use generated graphs and DFS")
    parser.add_argument("--list", action="store_true", help="list suitable sample ids and exit")
    args = parser.parse_args(argv)

    env = EnvConfig()
    colour = not args.no_colour
    if args.synthetic or not Path(args.graphs).is_file():
        if not args.synthetic:
            print(f"{args.graphs} not found; using synthetic graphs", file=sys.stderr)
        graphs = generate_dataset(40, seed=7, cfg=SyntheticConfig(min_nodes=12, max_nodes=22))
        if args.method == "ppo" and not Path(args.checkpoint).is_file():
            args.method = "dfs"
    else:
        graphs = read_jsonl(args.graphs)
    if "E" in env.feature_tier and not graphs[0].features:
        raise SystemExit("these graphs have no embeddings; use a Tier L configuration")

    searcher = make_searcher(args.method, args.checkpoint, env)
    if args.list:
        rng = np.random.default_rng(0)
        for graph in [g for g in graphs if 10 <= g.num_nodes <= 26 and len(g.source_lines) <= 34]:
            trace = searcher.trace(graph, env.max_steps_for(graph.num_nodes), rng)
            verdict = "success" if trace.outcome.get("success") else "failure"
            size, actions = graph.num_nodes, len(trace.steps)
            print(f"{graph.sample_id}  {size:>3} nodes  {actions:>3} actions  {verdict}")
        return

    by_id = {graph.sample_id: graph for graph in graphs}
    if args.graph and args.graph not in by_id:
        raise SystemExit(f"no graph {args.graph!r} in {args.graphs}; try --list")
    graph = (
        by_id[args.graph] if args.graph else choose_graph(graphs, searcher, env, not args.failure)
    )

    budget = env.max_steps_for(graph.num_nodes)
    traces = [searcher.trace(graph, budget, np.random.default_rng(0))]
    if args.compare:
        other = make_searcher(args.compare, args.checkpoint, env)
        traces.append(other.trace(graph, budget, np.random.default_rng(0)))
    for index, trace in enumerate(traces):
        if index and sys.stdout.isatty():
            input(paint(f" press Enter to watch {trace.method}", DIM, colour))
        play(graph, trace, args.delay, args.step, colour)
    if len(traces) > 1:
        print()
        for trace in traces:
            outcome = trace.outcome
            verdict = "found it" if outcome.get("success") else "missed"
            print(
                f" {trace.method:<16} {verdict:<9} {len(trace.steps):>3} actions, "
                f"{outcome.get('nodes_inspected'):>3} nodes inspected"
            )
        if any(name in BASELINE_FACTORIES for name in (args.method, args.compare)):
            print(
                paint(
                    " baselines stop automatically on reaching the flaw; the agent must decide",
                    DIM,
                    colour,
                )
            )


if __name__ == "__main__":
    main()
