"""Plain-language accounts of an episode, built only from recorded facts.

Two questions come up when watching a replay: why is this function vulnerable, and why did the
method get it right or wrong? The first is answered with the fix itself (a diff between the
function before and after the fixing commit). The second is answered from the trace: where the
method declared, what the fix actually changed, whether the method ever stood on a changed
line, and which features the statements involved carry. Nothing here is inferred or guessed.
"""

from __future__ import annotations

import difflib
from typing import Any

from soren.data.features import lexical_flags
from soren.data.schema import GraphRecord, Node
from soren.eval.runner import shortest_path_to_vulnerable
from soren.viz.trace import Trace

CWE_NAMES = {
    "CWE-119": "Improper restriction of operations within the bounds of a memory buffer",
    "CWE-125": "Out-of-bounds read",
}

FLAG_PHRASES = {
    "dangerous_call": "calls a memory or string API",
    "alloc_call": "allocates memory",
    "free_call": "frees memory",
    "array_subscript": "indexes an array",
    "pointer_deref": "dereferences a pointer",
    "pointer_arith": "does pointer arithmetic",
    "arith_op": "does arithmetic",
    "comparison": "compares values",
    "uses_sizeof": "uses sizeof",
    "has_cast": "casts a value",
    "length_like_ident": "uses a length or index variable",
}


def describe_flags(node: Node) -> list[str]:
    """What the lexical features see in a statement, as short phrases."""
    flags = lexical_flags(node)
    phrases = []
    for name, phrase in FLAG_PHRASES.items():
        if flags[name]:
            if name == "dangerous_call":
                phrase += f" ({', '.join(node.calls)})"
            phrases.append(phrase)
    return phrases


def fix_diff(before: str, after: str, context: int = 2) -> list[dict[str, Any]]:
    """The fixing commit's change to the function, as rows for display.

    Each row has ``kind`` (``context``, ``removed``, ``added`` or ``gap``), the line number in
    the vulnerable version where there is one, and the text.
    """
    a, b = before.rstrip("\n").split("\n"), after.rstrip("\n").split("\n")
    rows: list[dict[str, Any]] = []
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for group_index, group in enumerate(matcher.get_grouped_opcodes(context)):
        if group_index:
            rows.append({"kind": "gap", "line": None, "text": ""})
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                rows += [{"kind": "context", "line": i + 1, "text": a[i]} for i in range(i1, i2)]
                continue
            if tag in ("replace", "delete"):
                rows += [{"kind": "removed", "line": i + 1, "text": a[i]} for i in range(i1, i2)]
            if tag in ("replace", "insert"):
                rows += [{"kind": "added", "line": None, "text": b[j]} for j in range(j1, j2)]
    return rows


def _lines(graph: GraphRecord, nodes: list[int]) -> str:
    numbers = sorted({graph.nodes[n].line for n in nodes})
    if len(numbers) == 1:
        return f"line {numbers[0]}"
    shown = ", ".join(str(n) for n in numbers[:6])
    more = f" and {len(numbers) - 6} more" if len(numbers) > 6 else ""
    return f"lines {shown}{more}"


def explain_trace(graph: GraphRecord, trace: Trace, oracle_stop: bool = False) -> list[str]:
    """Why the episode ended the way it did, one fact per sentence."""
    vulnerable = graph.vuln_set
    truth = _lines(graph, graph.vuln_nodes)
    statements = graph.num_nodes - 2
    visited_at: dict[int, int] = {}
    position = trace.start_node
    for step in trace.steps:
        if step.action != "DECLARE":
            position = step.next_node
            visited_at.setdefault(position, step.t + 1)
    inspected = len({trace.start_node, *visited_at} - {graph.entry, graph.exit})
    declared = [step.node for step in trace.steps if step.action == "DECLARE"]
    out: list[str] = []

    if oracle_stop:
        out.append(
            "This baseline does not decide where the flaw is. It walks in a fixed order and is "
            "stopped automatically the moment it stands on a line the fix changed."
        )
    reason = trace.outcome.get("end_reason")
    if trace.outcome.get("success"):
        node = graph.nodes[declared[-1]]
        out.append(
            f"It declared line {node.line} after looking at {inspected} of {statements} "
            f"statements. The fix changed {truth}, so the declaration is correct."
        )
        seen = describe_flags(node)
        if seen and not oracle_stop:
            out.append(f"What the features see on that line: it {', '.join(seen)}.")
        return out

    if declared:
        node = graph.nodes[declared[-1]]
        out.append(
            f"It declared line {node.line} after looking at {inspected} of {statements} "
            f"statements. The fix did not touch that line; it changed {truth}."
        )
        seen = describe_flags(node)
        if seen:
            out.append(
                f"Line {node.line} looks risky to the features: it {', '.join(seen)}. "
                "Looking risky is not the same as being the flaw."
            )
        else:
            out.append(f"Line {node.line} sets none of the lexical flags the agent can see.")
    elif reason == "timeout":
        out.append(
            f"It used its whole step budget ({len(trace.steps)} actions) without declaring, "
            f"after looking at {inspected} of {statements} statements. The fix changed {truth}."
        )
    else:
        out.append(
            f"It looked at {inspected} of {statements} statements and declared nothing. "
            f"The fix changed {truth}."
        )

    passed = sorted((visited_at[n], n) for n in vulnerable if n in visited_at)
    if passed:
        step, node_id = passed[0]
        node = graph.nodes[node_id]
        seen = describe_flags(node)
        detail = f"it {', '.join(seen)}" if seen else "it sets none of the lexical flags"
        out.append(
            f"It stood on line {node.line}, which the fix changed, at step {step} and moved on. "
            f"What the features see there: {detail}."
        )
    else:
        distance = shortest_path_to_vulnerable(graph)
        out.append(
            f"It never reached a line the fix changed. The nearest one is {distance} "
            f"move{'s' if distance != 1 else ''} from the function's entry."
        )
    return out
