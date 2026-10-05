"""The processed graph record: the contract between the data pipeline and the RL side.

A :class:`GraphRecord` is one function's line-level control flow graph together with its
ground-truth vulnerable nodes. Everything downstream of CFG extraction (featurizers, the
environment, baselines, the visualizer) reads only this format.

Records are treated as immutable once built: adjacency is cached on first use.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import networkx as nx

NODE_KINDS: tuple[str, ...] = (
    "ENTRY",
    "EXIT",
    "ASSIGN",
    "CALL",
    "BRANCH",
    "LOOP",
    "SWITCH",
    "RETURN",
    "JUMP",
    "DECL",
    "OTHER",
)

# When one source line carries several constructs, the earliest kind in this list wins:
# control structure > return > jump > assignment > call > declaration > other.
KIND_PRIORITY: tuple[str, ...] = (
    "BRANCH",
    "LOOP",
    "SWITCH",
    "RETURN",
    "JUMP",
    "ASSIGN",
    "CALL",
    "DECL",
    "OTHER",
)


class SchemaError(ValueError):
    """Raised when a :class:`GraphRecord` violates the schema."""


def resolve_kind(constructs: Iterable[str]) -> str:
    """Pick the node kind for a line that carries the given construct kinds."""
    present = set(constructs)
    unknown = present - set(NODE_KINDS)
    if unknown:
        raise SchemaError(f"unknown node kinds: {sorted(unknown)}")
    for kind in ("ENTRY", "EXIT"):
        if kind in present:
            return kind
    for kind in KIND_PRIORITY:
        if kind in present:
            return kind
    return "OTHER"


@dataclass
class Node:
    """One statement (source line) of the function.

    ``calls`` holds callee names and ``ops`` holds operator names (Joern style, for example
    ``<operator>.indirectIndexAccess``) found on the line, so featurizers never re-parse source.
    """

    id: int
    line: int
    kind: str
    code: str = ""
    depth: int = 0
    in_loop: bool = False
    calls: list[str] = field(default_factory=list)
    ops: list[str] = field(default_factory=list)


@dataclass
class GraphRecord:
    sample_id: str
    nodes: list[Node]
    edges: list[tuple[int, int]]
    entry: int
    exit: int
    vuln_nodes: list[int]
    back_edges: list[tuple[int, int]] = field(default_factory=list)
    source_lines: list[str] = field(default_factory=list)
    project: str = ""
    commit_id: str = ""
    cwe: str = ""
    features: dict[str, list[list[float]]] = field(default_factory=dict)

    # ------------------------------------------------------------------ structure

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @cached_property
    def _successors(self) -> list[tuple[int, ...]]:
        out: list[list[int]] = [[] for _ in self.nodes]
        for u, v in self.edges:
            out[u].append(v)
        # Canonical, label-independent order: by source line, ties broken by node id.
        return [tuple(sorted(vs, key=lambda v: (self.nodes[v].line, v))) for vs in out]

    @cached_property
    def _predecessors(self) -> list[tuple[int, ...]]:
        out: list[list[int]] = [[] for _ in self.nodes]
        for u, v in self.edges:
            out[v].append(u)
        return [tuple(sorted(us, key=lambda u: (self.nodes[u].line, u))) for us in out]

    @cached_property
    def _back_edge_set(self) -> frozenset[tuple[int, int]]:
        return frozenset((u, v) for u, v in self.back_edges)

    @cached_property
    def vuln_set(self) -> frozenset[int]:
        return frozenset(self.vuln_nodes)

    def successors(self, node: int) -> tuple[int, ...]:
        """Successors of ``node`` in canonical order (by line number, then id)."""
        return self._successors[node]

    def predecessors(self, node: int) -> tuple[int, ...]:
        return self._predecessors[node]

    def is_back_edge(self, u: int, v: int) -> bool:
        return (u, v) in self._back_edge_set

    def reachable_from_entry(self) -> set[int]:
        seen = {self.entry}
        queue = deque([self.entry])
        while queue:
            u = queue.popleft()
            for v in self.successors(u):
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        return seen

    def to_networkx(self) -> nx.DiGraph:
        graph = nx.DiGraph(sample_id=self.sample_id, entry=self.entry, exit=self.exit)
        for node in self.nodes:
            attrs = asdict(node)
            attrs.pop("id")
            graph.add_node(node.id, vulnerable=node.id in self.vuln_set, **attrs)
        for u, v in self.edges:
            graph.add_edge(u, v, back=self.is_back_edge(u, v))
        return graph

    # ----------------------------------------------------------------- validation

    def validate(self) -> GraphRecord:
        """Check structural invariants; raise :class:`SchemaError` on the first violation."""
        n = self.num_nodes
        if n < 3:
            raise SchemaError(f"{self.sample_id}: need at least 3 nodes, got {n}")
        for index, node in enumerate(self.nodes):
            if node.id != index:
                raise SchemaError(f"{self.sample_id}: node at index {index} has id {node.id}")
            if node.kind not in NODE_KINDS:
                raise SchemaError(f"{self.sample_id}: node {index} has unknown kind {node.kind!r}")

        for name, value in (("entry", self.entry), ("exit", self.exit)):
            if not 0 <= value < n:
                raise SchemaError(f"{self.sample_id}: {name} {value} is not a node")
        entries = [node.id for node in self.nodes if node.kind == "ENTRY"]
        exits = [node.id for node in self.nodes if node.kind == "EXIT"]
        if entries != [self.entry]:
            raise SchemaError(f"{self.sample_id}: expected a single ENTRY node at {self.entry}")
        if exits != [self.exit]:
            raise SchemaError(f"{self.sample_id}: expected a single EXIT node at {self.exit}")

        seen: set[tuple[int, int]] = set()
        for u, v in self.edges:
            if not (0 <= u < n and 0 <= v < n):
                raise SchemaError(f"{self.sample_id}: edge ({u}, {v}) references a missing node")
            if u == v:
                raise SchemaError(f"{self.sample_id}: self-loop on node {u}")
            if (u, v) in seen:
                raise SchemaError(f"{self.sample_id}: duplicate edge ({u}, {v})")
            seen.add((u, v))
        for edge in self.back_edges:
            if tuple(edge) not in seen:
                raise SchemaError(f"{self.sample_id}: back edge {tuple(edge)} is not an edge")

        if not self.vuln_nodes:
            raise SchemaError(f"{self.sample_id}: vuln_nodes is empty")
        if len(set(self.vuln_nodes)) != len(self.vuln_nodes):
            raise SchemaError(f"{self.sample_id}: vuln_nodes contains duplicates")
        for v in self.vuln_nodes:
            if not 0 <= v < n:
                raise SchemaError(f"{self.sample_id}: vulnerable node {v} is not a node")
            if v in (self.entry, self.exit):
                raise SchemaError(f"{self.sample_id}: ENTRY/EXIT cannot be vulnerable")
        if not self.vuln_set & self.reachable_from_entry():
            raise SchemaError(f"{self.sample_id}: no vulnerable node is reachable from entry")

        for tier, matrix in self.features.items():
            if len(matrix) != n:
                raise SchemaError(
                    f"{self.sample_id}: features[{tier!r}] has {len(matrix)} rows for {n} nodes"
                )
            if len({len(row) for row in matrix}) > 1:
                raise SchemaError(f"{self.sample_id}: features[{tier!r}] has ragged rows")
        return self

    # -------------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "project": self.project,
            "commit_id": self.commit_id,
            "cwe": self.cwe,
            "source_lines": list(self.source_lines),
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [[u, v] for u, v in self.edges],
            "back_edges": [[u, v] for u, v in self.back_edges],
            "entry": self.entry,
            "exit": self.exit,
            "vuln_nodes": list(self.vuln_nodes),
            "features": {tier: [list(row) for row in rows] for tier, rows in self.features.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphRecord:
        try:
            return cls(
                sample_id=data["sample_id"],
                project=data.get("project", ""),
                commit_id=data.get("commit_id", ""),
                cwe=data.get("cwe", ""),
                source_lines=list(data.get("source_lines", [])),
                nodes=[Node(**node) for node in data["nodes"]],
                edges=[(int(u), int(v)) for u, v in data["edges"]],
                back_edges=[(int(u), int(v)) for u, v in data.get("back_edges", [])],
                entry=data["entry"],
                exit=data["exit"],
                vuln_nodes=list(data["vuln_nodes"]),
                features={k: [list(r) for r in v] for k, v in data.get("features", {}).items()},
            )
        except (KeyError, TypeError) as exc:
            raise SchemaError(f"malformed graph record: {exc}") from exc

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> GraphRecord:
        return cls.from_dict(json.loads(text))


def write_jsonl(records: Iterable[GraphRecord], path: str | Path) -> int:
    """Write one record per line; return the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.to_json())
            fh.write("\n")
            count += 1
    return count


def iter_jsonl(path: str | Path) -> Iterator[GraphRecord]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield GraphRecord.from_json(line)


def read_jsonl(path: str | Path) -> list[GraphRecord]:
    return list(iter_jsonl(path))
