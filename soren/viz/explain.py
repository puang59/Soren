"""What the fixing commit changed, for showing why a function is vulnerable."""

from __future__ import annotations

import difflib
from typing import Any

CWE_NAMES = {
    "CWE-119": "Improper restriction of operations within the bounds of a memory buffer",
    "CWE-125": "Out-of-bounds read",
}


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
