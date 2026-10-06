"""Turn a graph and a position in a trace into things to draw.

Everything here is a pure function of ``(graph, trace, step)``. The terminal demo and the
exporter for the web visualizer build on it. ``step`` counts actions already taken: 0 is the
start, ``len(trace.steps)`` the end.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field

import pandas as pd

from soren.data.schema import GraphRecord
from soren.viz.trace import Trace

# Fill colours are light enough for black text in both light and dark app themes.
COLOUR_CURRENT = "#ffb347"
COLOUR_UNVISITED = "#f4f4f4"
COLOURS_VISITED = ("#d6e6f5", "#aecde8", "#7fb0d9", "#5a94c8")  # by visit count, capped
COLOUR_CORRECT = "#7bd389"
COLOUR_WRONG = "#f28b82"
COLOUR_TRUTH = "#8e44ad"
COLOUR_EDGE = "#8a8a8a"
COLOUR_EDGE_TAKEN = "#1f6fb2"
LABEL_WIDTH = 34


@dataclass
class ReplayState:
    """Where a replay stands after ``step`` actions."""

    step: int
    current: int
    visits: dict[int, int] = field(default_factory=dict)
    stack: list[int] = field(default_factory=list)
    declared: list[int] = field(default_factory=list)
    taken_edges: set[tuple[int, int]] = field(default_factory=set)
    running_return: float = 0.0
    finished: bool = False


def replay_state(trace: Trace, step: int) -> ReplayState:
    """Replay the first ``step`` actions of ``trace``."""
    step = max(0, min(step, len(trace.steps)))
    state = ReplayState(step=step, current=trace.start_node, visits={trace.start_node: 1})
    for record in trace.steps[:step]:
        state.running_return += record.reward
        if record.action == "DECLARE":
            state.declared.append(record.node)
            continue
        moved = record.next_node != record.node  # a wasted (masked) move goes nowhere
        if record.action == "BACKTRACK" and state.stack:
            state.stack.pop()
        elif record.action.startswith("MOVE_") and moved:
            state.stack.append(record.node)
            state.taken_edges.add((record.node, record.next_node))
        if moved:
            state.visits[record.next_node] = state.visits.get(record.next_node, 0) + 1
        state.current = record.next_node
    state.finished = step == len(trace.steps)
    return state


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def node_label(graph: GraphRecord, node_id: int) -> str:
    node = graph.nodes[node_id]
    if node.kind in ("ENTRY", "EXIT"):
        return node.kind
    code = " ".join(node.code.split())
    if len(code) > LABEL_WIDTH:
        code = code[: LABEL_WIDTH - 1] + "…"
    return f"{node.line}: {code}" if code else f"{node.line}: {node.kind}"


def cfg_dot(graph: GraphRecord, trace: Trace, step: int, show_truth: bool = False) -> str:
    """Graphviz source for the CFG, coloured for the given point in the replay.

    * current node: orange
    * visited nodes: blue, darker with more visits
    * nodes on the path stack: bold outline
    * declared node: green if correct, red if wrong
    * ground truth (only if ``show_truth``): purple double outline
    """
    state = replay_state(trace, step)
    lines = [
        "digraph cfg {",
        '  rankdir=TB; bgcolor="transparent";',
        '  node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=13, '
        'color="#555555", fontcolor="#111111"];',
        f'  edge [color="{COLOUR_EDGE}", arrowsize=0.7];',
    ]
    on_stack = set(state.stack)
    for node in graph.nodes:
        visits = state.visits.get(node.id, 0)
        fill = COLOUR_UNVISITED
        if visits:
            fill = COLOURS_VISITED[min(visits, len(COLOURS_VISITED)) - 1]
        if node.id == state.current:
            fill = COLOUR_CURRENT
        if node.id in state.declared:
            fill = COLOUR_CORRECT if node.id in graph.vuln_set else COLOUR_WRONG
        attrs = [f'label="{_escape(node_label(graph, node.id))}"', f'fillcolor="{fill}"']
        if node.id in on_stack or node.id == state.current:
            attrs.append("penwidth=2.5")
        if show_truth and node.id in graph.vuln_set:
            attrs += [f'color="{COLOUR_TRUTH}"', "peripheries=2", "penwidth=2.5"]
        lines.append(f"  n{node.id} [{', '.join(attrs)}];")
    for u, v in graph.edges:
        attrs = []
        if (u, v) in state.taken_edges:
            attrs += [f'color="{COLOUR_EDGE_TAKEN}"', "penwidth=2"]
        if graph.is_back_edge(u, v):
            attrs.append("style=dashed")
        suffix = f" [{', '.join(attrs)}]" if attrs else ""
        lines.append(f"  n{u} -> n{v}{suffix};")
    lines.append("}")
    return "\n".join(lines)


def source_html(graph: GraphRecord, trace: Trace, step: int, show_truth: bool = False) -> str:
    """The function's source as HTML, with the current, visited and flaw lines marked."""
    state = replay_state(trace, step)
    line_of = {node.id: node.line for node in graph.nodes}
    current_line = line_of[state.current]
    visited_lines = {line_of[n] for n in state.visits}
    declared_lines = {line_of[n]: n in graph.vuln_set for n in state.declared}
    truth_lines = {line_of[n] for n in graph.vuln_nodes} if show_truth else set()

    rows = []
    for number, text in enumerate(graph.source_lines, start=1):
        background = "transparent"
        if number in visited_lines:
            background = "rgba(90, 148, 200, 0.22)"
        if number == current_line:
            background = "rgba(255, 179, 71, 0.55)"
        if number in declared_lines:
            background = (
                "rgba(123, 211, 137, 0.6)" if declared_lines[number] else "rgba(242, 139, 130, 0.6)"
            )
        marker = "◆" if number in truth_lines else "&nbsp;"
        border = f"3px solid {COLOUR_TRUTH}" if number in truth_lines else "3px solid transparent"
        rows.append(
            f'<div style="background:{background};border-left:{border};padding:0 6px;'
            f'white-space:pre;">'
            f'<span style="opacity:0.55;display:inline-block;width:3em;text-align:right;">'
            f"{number}</span> "
            f'<span style="color:{COLOUR_TRUTH};">{marker}</span> '
            f"{html.escape(text) or '&nbsp;'}</div>"
        )
    return (
        '<div style="font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;'
        'line-height:1.5;overflow-x:auto;">' + "".join(rows) + "</div>"
    )


def step_table(graph: GraphRecord, trace: Trace, step: int | None = None) -> pd.DataFrame:
    """Actions taken so far: step, source line, action, reward and running return."""
    steps = trace.steps if step is None else trace.steps[:step]
    running = 0.0
    rows = []
    for record in steps:
        running += record.reward
        rows.append(
            {
                "step": record.t + 1,
                "line": graph.nodes[record.node].line,
                "action": record.action,
                "to line": graph.nodes[record.next_node].line,
                "reward": round(record.reward, 4),
                "return": round(running, 4),
            }
        )
    return pd.DataFrame(rows, columns=["step", "line", "action", "to line", "reward", "return"])


def action_labels(graph: GraphRecord, node_id: int, num_actions: int) -> list[str]:
    """Readable names for a policy's actions at ``node_id``: where each move leads."""
    successors = graph.successors(node_id)
    labels = []
    for slot in range(num_actions - 2):
        if slot < len(successors):
            labels.append(f"move → line {graph.nodes[successors[slot]].line}")
        else:
            labels.append(f"move {slot} (n/a)")
    return [*labels, "backtrack", "declare"]


def outcome_text(graph: GraphRecord, trace: Trace) -> str:
    outcome = trace.outcome
    verdict = "found the vulnerable statement" if outcome.get("success") else "did not find it"
    reason = outcome.get("end_reason") or "unknown"
    return (
        f"{trace.method} {verdict} ({reason}): {len(trace.steps)} actions, "
        f"{outcome.get('nodes_inspected', '?')} of {graph.num_nodes} nodes inspected, "
        f"return {outcome.get('return', 0.0):.3f}"
    )
