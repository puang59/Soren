"""Build line-level control flow graphs from Joern's export.

Joern's CFG has one node per expression: ``buf[i] = foo(x)`` alone is several nodes. The
agent moves between statements and the ground truth is given per line, so the CFG is collapsed
to one node per source line. Two Joern nodes on different lines joined by a CFG edge give an
edge between those lines; flow within a line disappears.

The input is the JSON written by ``joern/export_cfg.sc``, one object per method.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from soren.data.graph_utils import bfs_depths, find_back_edges, nodes_in_cycles
from soren.data.schema import GraphRecord, Node, resolve_kind

OPERATOR_PREFIX = "<operator>."
CONTROL_KINDS = {
    "IF": "BRANCH",
    "WHILE": "LOOP",
    "FOR": "LOOP",
    "DO": "LOOP",
    "SWITCH": "SWITCH",
    "BREAK": "JUMP",
    "CONTINUE": "JUMP",
    "GOTO": "JUMP",
}
ASSIGNMENT_OPERATORS = {
    OPERATOR_PREFIX + name
    for name in ("preIncrement", "postIncrement", "preDecrement", "postDecrement")
}
EXTENSION_PREFERENCE = (".c", ".cpp")


class CfgBuildError(ValueError):
    """Raised when a Joern method cannot be turned into a usable line-level graph."""


@dataclass
class LineCfg:
    """A line-level CFG without labels. Node 0 is ENTRY and the last node is EXIT."""

    nodes: list[Node]
    edges: list[tuple[int, int]]
    back_edges: list[tuple[int, int]]
    source_lines: list[str]
    line_to_node: dict[int, int] = field(default_factory=dict)
    """Source line -> id of the node that covers it. ENTRY and EXIT are not included."""

    @property
    def entry(self) -> int:
        return 0

    @property
    def exit(self) -> int:
        return len(self.nodes) - 1

    def to_record(self, sample_id: str, vuln_nodes: list[int], **metadata: str) -> GraphRecord:
        """Attach labels and metadata (``project``, ``commit_id``, ``cwe``) and validate."""
        return GraphRecord(
            sample_id=sample_id,
            nodes=self.nodes,
            edges=self.edges,
            back_edges=self.back_edges,
            entry=self.entry,
            exit=self.exit,
            vuln_nodes=sorted(vuln_nodes),
            source_lines=self.source_lines,
            **metadata,
        ).validate()


def _is_assignment(name: str) -> bool:
    return name.startswith(OPERATOR_PREFIX + "assignment") or name in ASSIGNMENT_OPERATORS


def _canonical_lines(method: dict[str, Any]) -> dict[int, int]:
    """Map each source line to the line of the node that represents it (identity for now)."""
    return {}


def build_line_cfg(method: dict[str, Any], source: str) -> LineCfg:
    """Collapse one exported Joern method to a line-level CFG.

    ``source`` is the text of the file the method was parsed from; node code is taken from it.
    """
    source_lines = source.rstrip("\n").split("\n")
    canonical = _canonical_lines(method)

    def canon(line: int) -> int:
        return canonical.get(line, line)

    entry_id = exit_id = None
    line_of: dict[int, int] = {}  # Joern node id -> canonical line, for nodes with a line
    calls: dict[int, list[str]] = defaultdict(list)
    ops: dict[int, list[str]] = defaultdict(list)
    constructs: dict[int, set[str]] = defaultdict(set)

    for node_id, label, line, name, _code in method["nodes"]:
        if label == "METHOD":
            entry_id = node_id
            continue
        if label == "METHOD_RETURN":
            exit_id = node_id
            continue
        if line < 1:
            continue  # no position: flow through it is routed around below
        line = canon(line)
        line_of[node_id] = line
        constructs[line].add("OTHER")
        if label == "RETURN":
            constructs[line].add("RETURN")
        elif label == "CALL":
            if name.startswith(OPERATOR_PREFIX):
                if name not in ops[line]:
                    ops[line].append(name)
                if _is_assignment(name):
                    constructs[line].add("ASSIGN")
            else:
                if name not in calls[line]:
                    calls[line].append(name)
                constructs[line].add("CALL")
    if entry_id is None or exit_id is None:
        raise CfgBuildError("method has no METHOD or METHOD_RETURN node")

    for control_type, own_line, header_first, _header_last in method["controls"]:
        kind = CONTROL_KINDS.get(control_type)
        anchor = canon(header_first if header_first > 0 else own_line)
        if kind and anchor in constructs:
            constructs[anchor].add(kind)
    for label, first, _last in method["stmts"]:
        if label == "LOCAL" and canon(first) in constructs:
            constructs[canon(first)].add("DECL")

    lines = sorted(constructs)
    if not lines:
        raise CfgBuildError("method has no statements with a line number")
    node_of_line = {line: index + 1 for index, line in enumerate(lines)}
    exit_node = len(lines) + 1

    def group(joern_id: int) -> int | None:
        if joern_id == entry_id:
            return 0
        if joern_id == exit_id:
            return exit_node
        line = line_of.get(joern_id)
        return None if line is None else node_of_line[line]

    successors: dict[int, list[int]] = defaultdict(list)
    for source_id, target_id in method["edges"]:
        successors[source_id].append(target_id)

    def grouped_targets(joern_id: int) -> set[int]:
        """Groups reached from ``joern_id``, passing through nodes that have no line."""
        found: set[int] = set()
        seen = {joern_id}
        stack = list(successors[joern_id])
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            target = group(current)
            if target is None:
                stack.extend(successors[current])
            else:
                found.add(target)
        return found

    edges: set[tuple[int, int]] = set()
    for joern_id in [entry_id, *line_of]:
        origin = group(joern_id)
        for target in grouped_targets(joern_id):
            if target != origin:
                edges.add((origin, target))
    edge_list = sorted(edges)

    def text(line: int) -> str:
        return source_lines[line - 1].strip() if 0 < line <= len(source_lines) else ""

    num_nodes = len(lines) + 2
    depths = bfs_depths(num_nodes, edge_list, 0)
    in_loop = nodes_in_cycles(num_nodes, edge_list)
    nodes = [Node(id=0, line=max(method["line"], 1), kind="ENTRY")]
    for line in lines:
        node_id = node_of_line[line]
        nodes.append(
            Node(
                id=node_id,
                line=line,
                kind=resolve_kind(constructs[line]),
                code=text(line),
                depth=depths[node_id],
                in_loop=node_id in in_loop,
                calls=calls[line],
                ops=ops[line],
            )
        )
    last_line = method["line_end"] if method["line_end"] > 0 else len(source_lines)
    nodes.append(Node(id=exit_node, line=last_line, kind="EXIT", depth=depths[exit_node]))

    return LineCfg(
        nodes=nodes,
        edges=edge_list,
        back_edges=find_back_edges(num_nodes, edge_list, 0),
        source_lines=source_lines,
        line_to_node=dict(node_of_line),
    )


# ---------------------------------------------------------------------- loading exports


def _sample_and_extension(file_name: str) -> tuple[str, str]:
    path = Path(file_name)
    return path.stem, path.suffix


def load_joern_methods(paths: Iterable[str | Path]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Read export files into ``{sample_id: {extension: [methods]}}``."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    method = json.loads(line)
                    sample_id, extension = _sample_and_extension(method["file"])
                    grouped[sample_id][extension].append(method)
    return {sample: dict(by_ext) for sample, by_ext in grouped.items()}


def _flow_nodes(method: dict[str, Any]) -> int:
    return len(method["nodes"])


def select_method(candidates: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    """Choose the method that represents a sample.

    A source file should hold one function, but a parser can split it or find extra ones. Per
    extension, the function that starts first is taken (the largest, on a tie). Across the C and
    C++ parses the one with more CFG nodes wins, and C is preferred when they are equal.
    """
    best: dict[str, Any] | None = None
    for extension in EXTENSION_PREFERENCE:
        methods = candidates.get(extension)
        if not methods:
            continue
        chosen = min(methods, key=lambda m: (m["line"], -_flow_nodes(m)))
        if best is None or _flow_nodes(chosen) > _flow_nodes(best):
            best = chosen
    return best
