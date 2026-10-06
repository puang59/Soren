"""Write each function to its own source file, batched for Joern.

Each file holds only the pre-fix function body, starting at line 1. That makes Joern's line
numbers line up with the flaw-line indices: ``joern_line = flaw_line_index + 1``.

Every function is written twice, as ``.c`` and as ``.cpp``. BigVul mixes C and C++ (Chrome and
Android are largely C++), and Joern picks its parser by file extension: a C++ method such as
``void Foo::bar()`` yields no function when parsed as C, while C code that uses ``new`` or
``class`` as an identifier fails as C++. Parsing both ways and keeping whichever works
recovers 97% of the filtered functions, against 65% for C alone.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BATCH_SIZE = 500
"""Functions per directory; one Joern run per batch amortises the JVM start-up."""
EXTENSIONS: tuple[str, ...] = (".c", ".cpp")
"""Each function is written once per extension; ``.c`` is preferred when both parse."""


def batch_name(index: int) -> str:
    return f"batch_{index:04d}"


def write_sources(
    frame: pd.DataFrame, out_dir: str | Path, batch_size: int = BATCH_SIZE
) -> pd.DataFrame:
    """Write ``<out_dir>/batch_XXXX/<sample_id>{.c,.cpp}`` for every row of ``frame``.

    Returns the manifest (``sample_id``, ``batch``, ``path`` of the ``.c`` file relative to
    ``out_dir``), which is also written to ``<out_dir>/manifest.csv``.
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
        # Normalise line endings so line numbers match the indices computed on "\n" splits.
        text = code.replace("\r\n", "\n").replace("\r", "\n")
        text = text if text.endswith("\n") else text + "\n"
        (out_dir / batch).mkdir(parents=True, exist_ok=True)
        for extension in EXTENSIONS:
            (out_dir / batch / f"{sample_id}{extension}").write_text(text, encoding="utf-8")
        rows.append(
            {"sample_id": sample_id, "batch": batch, "path": f"{batch}/{sample_id}{EXTENSIONS[0]}"}
        )
    manifest = pd.DataFrame(rows, columns=["sample_id", "batch", "path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest
