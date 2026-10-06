"""Map ground-truth flaw lines onto CFG nodes.

A flaw line is a 0-based index into the function's lines. It becomes a label only if a node
covers that line, and only after the text at that position has been checked against the
recorded flaw-line text: an index that is off by one must not silently label a neighbour.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from soren.data.cfg_builder import LineCfg


@dataclass
class Alignment:
    vuln_nodes: list[int] = field(default_factory=list)
    reason: str | None = None
    """Why the sample must be dropped, or ``None`` if it is usable:

    * ``text_mismatch``: a flaw line's text is not at the recorded index
    * ``no_flaw_on_node``: no flaw line falls on a statement of the graph
    * ``flaw_unreachable``: the labelled statements cannot be reached from ENTRY
    """
    mapped: int = 0
    """Flaw lines that landed on a node."""
    unmapped: int = 0
    """Flaw lines with no node: declarations, lone ``else`` lines, code Joern left out."""

    @property
    def ok(self) -> bool:
        return self.reason is None


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def align_flaw_lines(
    cfg: LineCfg, flaw_line_indices: Sequence[int], flaw_lines: Sequence[str] | None = None
) -> Alignment:
    """Find the nodes that cover the flaw lines.

    ``flaw_lines`` holds the text of each flaw line, parallel to ``flaw_line_indices``. When
    given, each text must match the source at its index (ignoring whitespace), and must be part
    of the code of the node it maps to.
    """
    if flaw_lines is not None and len(flaw_lines) != len(flaw_line_indices):
        return Alignment(reason="text_mismatch")

    result = Alignment()
    nodes: set[int] = set()
    for position, index in enumerate(flaw_line_indices):
        line = int(index) + 1
        if flaw_lines is not None:
            expected = _squash(flaw_lines[position])
            in_range = 0 < line <= len(cfg.source_lines)
            if not in_range or _squash(cfg.source_lines[line - 1]) != expected:
                return Alignment(reason="text_mismatch")
        node = cfg.line_to_node.get(line)
        if node is None:
            result.unmapped += 1
            continue
        if flaw_lines is not None and expected not in _squash(cfg.nodes[node].code):
            # The line belongs to the node's span but the node's text does not contain it.
            return Alignment(reason="text_mismatch")
        result.mapped += 1
        nodes.add(node)

    if not nodes:
        result.reason = "no_flaw_on_node"
        return result
    result.vuln_nodes = sorted(nodes)
    if all(cfg.nodes[node].depth < 0 for node in nodes):
        result.reason = "flaw_unreachable"
    return result
