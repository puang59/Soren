"""Synthetic CFG generator.

Builds random structured programs (sequences, if/else, while loops, switches) and lowers them
to line-level control flow graphs, then plants one vulnerable statement. This lets the
environment, the baselines and the agent be developed and validated before the BigVul pipeline
exists.

The feature signal is a call to a "dangerous" API. ``signal`` interpolates the probability that
the vulnerable statement carries such a call between ``decoy_rate`` (``signal=0``: the
vulnerable node is statistically indistinguishable from any other statement) and ``hit_rate``
(``signal=1``). Every other statement carries one with probability ``decoy_rate``.

Run ``python -m soren.data.synthetic --help`` to write a dataset to JSONL.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from soren.data.graph_utils import bfs_depths, find_back_edges, nodes_in_cycles
from soren.data.schema import GraphRecord, Node, write_jsonl

DANGEROUS_CALLS = ("memcpy", "strcpy", "sprintf", "memmove", "strcat")
BENIGN_CALLS = ("log_msg", "update_state", "check_flags", "notify", "release_lock")
VARIABLES = ("i", "n", "len", "ret", "tmp", "count", "off", "val")


@dataclass
class SyntheticConfig:
    min_nodes: int = 10
    max_nodes: int = 40
    max_depth: int = 3
    signal: float = 1.0
    hit_rate: float = 0.9
    decoy_rate: float = 0.1
    p_if: float = 0.22
    p_while: float = 0.12
    p_switch: float = 0.05

    def __post_init__(self) -> None:
        if not 4 <= self.min_nodes <= self.max_nodes:
            raise ValueError("need 4 <= min_nodes <= max_nodes")
        for name in ("signal", "hit_rate", "decoy_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    @property
    def vuln_call_rate(self) -> float:
        """Probability that the vulnerable statement carries a dangerous call."""
        return self.decoy_rate + self.signal * (self.hit_rate - self.decoy_rate)


class _Builder:
    """Emits nodes, edges and source lines while walking a randomly generated program."""

    def __init__(self, rng: np.random.Generator, cfg: SyntheticConfig, budget: int) -> None:
        self.rng = rng
        self.cfg = cfg
        self.budget = budget
        self.nodes: list[Node] = []
        self.edges: list[tuple[int, int]] = []
        self.lines: list[str] = []
        self.simple: list[int] = []  # ids of plain statements, candidates for the label

    # -- emission helpers ---------------------------------------------------

    def _line(self, text: str, indent: int) -> int:
        self.lines.append("    " * indent + text)
        return len(self.lines)

    def _node(self, kind: str, code: str, indent: int, preds: list[int], **attrs) -> int:
        node_id = len(self.nodes)
        self.nodes.append(Node(id=node_id, line=self._line(code, indent), kind=kind, code=code))
        for key, value in attrs.items():
            setattr(self.nodes[node_id], key, value)
        for pred in preds:
            self.edges.append((pred, node_id))
        self.budget -= 1
        return node_id

    def _var(self) -> str:
        return str(self.rng.choice(VARIABLES))

    def _condition(self) -> str:
        op = str(self.rng.choice(["<", ">", "==", "!=", "<="]))
        return f"{self._var()} {op} {self._var()}"

    # -- statements ---------------------------------------------------------

    def _simple(self, indent: int, preds: list[int]) -> list[int]:
        roll = self.rng.random()
        if roll < 0.5:
            code = f"{self._var()} = {self._var()} + {int(self.rng.integers(1, 9))};"
            ops = ["<operator>.assignment", "<operator>.addition"]
            node = self._node("ASSIGN", code, indent, preds, ops=ops)
        elif roll < 0.85:
            callee = str(self.rng.choice(BENIGN_CALLS))
            node = self._node("CALL", f"{callee}({self._var()});", indent, preds, calls=[callee])
        else:
            node = self._node("DECL", f"int {self._var()}_{len(self.nodes)};", indent, preds)
        self.simple.append(node)
        return [node]

    def _if(self, indent: int, depth: int, preds: list[int]) -> list[int]:
        cond = self._node("BRANCH", f"if ({self._condition()}) {{", indent, preds)
        exits = self._block(indent + 1, depth + 1, [cond], self._sub_budget())
        if self.budget > 0 and self.rng.random() < 0.5:
            self._line("} else {", indent)
            exits += self._block(indent + 1, depth + 1, [cond], self._sub_budget())
        else:
            exits.append(cond)
        self._line("}", indent)
        return exits

    def _while(self, indent: int, depth: int, preds: list[int]) -> list[int]:
        header = self._node("LOOP", f"while ({self._condition()}) {{", indent, preds)
        for tail in self._block(indent + 1, depth + 1, [header], self._sub_budget()):
            self.edges.append((tail, header))
        self._line("}", indent)
        return [header]

    def _switch(self, indent: int, depth: int, preds: list[int]) -> list[int]:
        head = self._node("SWITCH", f"switch ({self._var()}) {{", indent, preds)
        exits = [head]  # no default case: control may fall past the switch
        for case in range(int(self.rng.integers(2, 4))):
            self._line(f"case {case}:", indent)
            exits += self._block(indent + 1, depth + 1, [head], max(1, self._sub_budget() // 2))
            self._line("break;", indent + 1)
        self._line("}", indent)
        return exits

    def _sub_budget(self) -> int:
        return int(self.rng.integers(1, max(2, min(self.budget, 6) + 1)))

    def _block(self, indent: int, depth: int, preds: list[int], budget: int) -> list[int]:
        """Emit at least one statement; return the nodes whose control falls out of the block."""
        cfg = self.cfg
        emitted = 0
        while emitted == 0 or (budget > 0 and self.budget > 0):
            before = len(self.nodes)
            roll = self.rng.random()
            compound = depth < cfg.max_depth and self.budget >= 3 and budget >= 2
            if compound and roll < cfg.p_if:
                preds = self._if(indent, depth, preds)
            elif compound and roll < cfg.p_if + cfg.p_while:
                preds = self._while(indent, depth, preds)
            elif compound and roll < cfg.p_if + cfg.p_while + cfg.p_switch:
                preds = self._switch(indent, depth, preds)
            else:
                preds = self._simple(indent, preds)
            used = len(self.nodes) - before
            emitted += used
            budget -= used
        return preds


def _make_dangerous(node: Node, rng: np.random.Generator) -> None:
    callee = str(rng.choice(DANGEROUS_CALLS))
    node.kind = "CALL"
    node.calls = [callee]
    node.ops = []
    node.code = f"{callee}(buf, src, {rng.choice(VARIABLES)});"


def generate_graph(
    rng: np.random.Generator, cfg: SyntheticConfig | None = None, sample_id: str = "syn"
) -> GraphRecord:
    """Generate one validated synthetic :class:`GraphRecord`."""
    cfg = cfg or SyntheticConfig()
    target = int(rng.integers(cfg.min_nodes, cfg.max_nodes + 1))
    # ENTRY, the final return and EXIT are added around the generated body.
    builder = _Builder(rng, cfg, budget=max(1, target - 3))

    entry_line = builder._line("int f(char *buf, char *src) {", 0)
    builder.nodes.append(Node(id=0, line=entry_line, kind="ENTRY"))
    exits = builder._block(1, 0, [0], builder.budget)
    ret = builder._node("RETURN", "return 0;", 1, exits)
    exit_id = len(builder.nodes)
    builder.nodes.append(Node(id=exit_id, line=builder._line("}", 0), kind="EXIT"))
    builder.edges.append((ret, exit_id))

    # The label is drawn only after the structure is fixed, so node ids, line numbers and
    # successor order cannot depend on it.
    vuln = int(rng.choice(builder.simple))
    for node_id in builder.simple:
        rate = cfg.vuln_call_rate if node_id == vuln else cfg.decoy_rate
        if rng.random() < rate:
            node = builder.nodes[node_id]
            indent = len(builder.lines[node.line - 1]) - len(builder.lines[node.line - 1].lstrip())
            _make_dangerous(node, rng)
            builder.lines[node.line - 1] = " " * indent + node.code

    n = len(builder.nodes)
    depths = bfs_depths(n, builder.edges, 0)
    loops = nodes_in_cycles(n, builder.edges)
    for node in builder.nodes:
        node.depth = depths[node.id]
        node.in_loop = node.id in loops

    return GraphRecord(
        sample_id=sample_id,
        project="synthetic",
        cwe="SYNTHETIC",
        source_lines=builder.lines,
        nodes=builder.nodes,
        edges=builder.edges,
        back_edges=find_back_edges(n, builder.edges, 0),
        entry=0,
        exit=exit_id,
        vuln_nodes=[vuln],
    ).validate()


def generate_dataset(
    n: int, seed: int = 0, cfg: SyntheticConfig | None = None, prefix: str = "syn"
) -> list[GraphRecord]:
    """Generate ``n`` graphs deterministically from ``seed``.

    Each graph draws from its own stream, so graph ``i`` has the same structure and label
    whatever the signal settings are.
    """
    return [
        generate_graph(np.random.default_rng([seed, i]), cfg, sample_id=f"{prefix}_{i:06d}")
        for i in range(n)
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write a synthetic CFG dataset to JSONL.")
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument("--n", type=int, default=1000, help="number of graphs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-nodes", type=int, default=SyntheticConfig.min_nodes)
    parser.add_argument("--max-nodes", type=int, default=SyntheticConfig.max_nodes)
    parser.add_argument("--signal", type=float, default=SyntheticConfig.signal)
    parser.add_argument("--hit-rate", type=float, default=SyntheticConfig.hit_rate)
    parser.add_argument("--decoy-rate", type=float, default=SyntheticConfig.decoy_rate)
    args = parser.parse_args(argv)
    cfg = SyntheticConfig(
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        signal=args.signal,
        hit_rate=args.hit_rate,
        decoy_rate=args.decoy_rate,
    )
    count = write_jsonl(generate_dataset(args.n, args.seed, cfg), args.out)
    print(f"wrote {count} graphs to {args.out}")


if __name__ == "__main__":
    main()
