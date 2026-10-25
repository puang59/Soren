"""Print calculations the way they are written on paper: formula, numbers, result.

Training, evaluation and the walkthrough (``python -m soren.walkthrough``) call this when
``--explain`` is on, so every number the program reports can be traced to the formula that
produced it::

    advantage   A_t = δ_t + γλ·(1 − done)·A_{t+1}
                    = -0.108 + 0.99×0.95×1×0.042
                    = -0.068

Nothing here computes anything; the callers pass the numbers they actually used.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any, TextIO

RESET, BOLD, DIM, CYAN, YELLOW = "\033[0m", "\033[1m", "\033[2m", "\033[36m", "\033[33m"


def num(x: float, places: int = 3) -> str:
    """A number for the middle of a formula: negative values get brackets."""
    text = f"{x:.{places}f}"
    return f"({text})" if x < 0 else text


class MathLog:
    """Writes headings, formulas and tables to a stream; does nothing while disabled."""

    def __init__(self, enabled: bool = False, stream: TextIO | None = None) -> None:
        self.enabled = enabled
        self._stream = stream

    @property
    def stream(self) -> TextIO:
        return self._stream or sys.stdout

    @property
    def colour(self) -> bool:
        return self.stream.isatty() and not os.environ.get("NO_COLOR")

    def paint(self, text: str, code: str) -> str:
        return f"{code}{text}{RESET}" if self.colour else text

    def write(self, text: str = "") -> None:
        if self.enabled:
            print(text, file=self.stream, flush=True)

    def heading(self, title: str) -> None:
        """A numbered part of an explanation."""
        self.write()
        self.write(self.paint(f"══ {title} " + "═" * max(0, 74 - len(title)), BOLD))

    def section(self, title: str) -> None:
        self.write()
        self.write(self.paint(f"── {title}", YELLOW))

    def text(self, *lines: str) -> None:
        for line in lines:
            self.write(f"   {line}")

    def note(self, *lines: str) -> None:
        for line in lines:
            self.write(self.paint(f"   {line}", DIM))

    def formula(
        self, name: str, symbolic: str, numeric: str | None = None, result: Any = None
    ) -> None:
        """``name  symbolic`` then ``= numeric`` and ``= result`` aligned under it."""
        head = f"   {name:<22} "
        if not symbolic:  # just the arithmetic, on one line
            parts = [str(p) for p in (numeric, result) if p is not None and str(p) != ""]
            self.write(head + "= " + " = ".join(parts))
            return
        self.write(head + self.paint(symbolic, CYAN))
        pad = " " * len(head)
        for part in (numeric, result):
            if part is not None and str(part) != "":
                self.write(f"{pad}= {part}")

    def table(self, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
        """A plain text table; numbers are right-aligned, text left-aligned."""
        cells = [[str(h) for h in header]] + [[_cell(v) for v in row] for row in rows]
        widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
        numeric = [
            all(isinstance(row[i], int | float) for row in rows) if rows else False
            for i in range(len(header))
        ]

        def line(row: list[str]) -> str:
            parts = [
                cell.rjust(widths[i]) if numeric[i] else cell.ljust(widths[i])
                for i, cell in enumerate(row)
            ]
            return "   " + "  ".join(parts).rstrip()

        self.write(self.paint(line(cells[0]), BOLD))
        self.write("   " + "  ".join("─" * w for w in widths))
        for row in cells[1:]:
            self.write(line(row))


def _cell(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


log = MathLog()
"""The shared log. Scripts switch it on with ``log.enabled = True`` for ``--explain``."""
