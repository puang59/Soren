"""Reduce BigVul to usable, labelled, de-duplicated vulnerable functions.

Filters run in a fixed order and the surviving row count is recorded after each one, so the
attrition table in the report can be regenerated from ``attrition.json``.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass, field

import pandas as pd

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_TRIVIAL_LINE = re.compile(r"^[\s{}();]*$")


@dataclass
class FilterConfig:
    cwes: list[str] = field(default_factory=lambda: ["CWE-119", "CWE-125"])
    min_lines: int = 5
    max_lines: int = 300


def strip_comments(code: str) -> str:
    """Remove ``/* */`` and ``//`` comments, keeping the line structure intact."""
    code = _BLOCK_COMMENT.sub(lambda m: "\n" * m.group().count("\n"), code)
    return _LINE_COMMENT.sub("", code)


def is_trivial_line(line: str) -> bool:
    """True for a single line with no code: blank, braces or punctuation only, or a comment.

    Judges the line in isolation; use :func:`code_line_mask` for whole functions, where a line
    may sit inside a multi-line comment.
    """
    stripped = line.strip()
    if stripped.startswith(("//", "/*", "*/")) or stripped == "*" or stripped.startswith("* "):
        return True
    return bool(_TRIVIAL_LINE.match(strip_comments(line)))


def code_line_mask(code: str) -> list[bool]:
    """For each line of ``code``, whether it carries any code once comments are removed."""
    without_comments = strip_comments(code).split("\n")
    return [not _TRIVIAL_LINE.match(line) for line in without_comments]


def derive_flaw_lines(before: str, after: str) -> list[int]:
    """0-based indices of the lines of ``before`` that the fix deleted or changed."""
    before_lines = before.split("\n")
    after_lines = after.split("\n")
    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    indices: list[int] = []
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            indices.extend(range(i1, i2))
    return indices


def normalise_body(code: str) -> str:
    """Canonical form for duplicate detection: no comments, whitespace collapsed."""
    return re.sub(r"\s+", " ", strip_comments(code)).strip()


def body_hash(code: str) -> str:
    return hashlib.sha1(normalise_body(code).encode("utf-8")).hexdigest()


def _flaw_indices(row: pd.Series) -> list[int]:
    """Flaw-line indices for a row: as given, or derived from the diff when absent."""
    given = [int(i) for i in row["flaw_line_indices"]]
    if given:
        return sorted(set(given))
    if row["func_after"]:
        return derive_flaw_lines(row["func_before"], row["func_after"])
    return []


def _code_flaw_indices(code: str, indices: list[int]) -> list[int]:
    """Keep only the flaw lines that carry code."""
    mask = code_line_mask(code)
    return [i for i in indices if 0 <= i < len(mask) and mask[i]]


def filter_bigvul(
    frame: pd.DataFrame, config: FilterConfig | None = None
) -> tuple[pd.DataFrame, list[dict[str, int | str]]]:
    """Apply the filters in order; return the surviving rows and the attrition log.

    The result gains ``num_lines`` and ``body_hash``; ``flaw_line_indices`` and ``flaw_lines``
    are restricted to lines that carry code.
    """
    config = config or FilterConfig()
    attrition: list[dict[str, int | str]] = []

    def record(step: str, current: pd.DataFrame) -> None:
        attrition.append({"step": step, "rows": int(len(current))})

    frame = frame.reset_index(drop=True)
    record("loaded", frame)

    frame = frame[frame["vul"] == 1]
    record("vulnerable", frame)

    frame = frame[frame["cwe"].isin(config.cwes)]
    record("selected_cwes", frame)

    frame = frame.copy()
    frame["flaw_line_indices"] = [_flaw_indices(row) for _, row in frame.iterrows()]
    frame = frame[frame["flaw_line_indices"].map(len) > 0]
    record("has_flaw_lines", frame)

    frame = frame.copy()
    frame["flaw_line_indices"] = [
        _code_flaw_indices(code, indices)
        for code, indices in zip(frame["func_before"], frame["flaw_line_indices"], strict=True)
    ]
    frame = frame[frame["flaw_line_indices"].map(len) > 0]
    record("has_code_flaw_lines", frame)

    frame = frame.copy()
    frame["num_lines"] = [code.count("\n") + 1 for code in frame["func_before"]]
    frame = frame[frame["num_lines"].between(config.min_lines, config.max_lines)]
    record("line_count", frame)

    frame = frame.copy()
    frame["body_hash"] = [body_hash(code) for code in frame["func_before"]]
    frame = frame.drop_duplicates(subset="body_hash", keep="first")
    record("deduplicated", frame)

    frame = frame.copy()
    frame["flaw_lines"] = [
        [code.split("\n")[i] for i in indices]
        for code, indices in zip(frame["func_before"], frame["flaw_line_indices"], strict=True)
    ]
    return frame.reset_index(drop=True), attrition
