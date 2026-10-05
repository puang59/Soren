"""Write each function to its own C file, batched for Joern.

Each file holds only the pre-fix function body, starting at line 1. That makes Joern's line
numbers line up with the flaw-line indices: ``joern_line = flaw_line_index + 1``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BATCH_SIZE = 500
"""Functions per directory; one Joern run per batch amortises the JVM start-up."""


def batch_name(index: int) -> str:
    return f"batch_{index:04d}"


def write_sources(
    frame: pd.DataFrame, out_dir: str | Path, batch_size: int = BATCH_SIZE
) -> pd.DataFrame:
    """Write ``<out_dir>/batch_XXXX/<sample_id>.c`` for every row of ``frame``.

    Returns the manifest (``sample_id``, ``batch``, ``path`` relative to ``out_dir``), which
    is also written to ``<out_dir>/manifest.csv``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be unique")
    out_dir = Path(out_dir)
    rows = []
    for position, (sample_id, code) in enumerate(
        zip(frame["sample_id"], frame["func_before"], strict=True)
    ):
        batch = batch_name(position // batch_size)
        relative = Path(batch) / f"{sample_id}.c"
        path = out_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # Normalise line endings so line numbers match the indices computed on "\n" splits.
        text = code.replace("\r\n", "\n").replace("\r", "\n")
        path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        rows.append({"sample_id": sample_id, "batch": batch, "path": str(relative)})
    manifest = pd.DataFrame(rows, columns=["sample_id", "batch", "path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest
