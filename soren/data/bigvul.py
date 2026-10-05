"""Load BigVul and normalise its columns.

BigVul (Fan et al., MSR 2020) is distributed in more than one form, and the column names
differ between them: the original ``MSR_data_cleaned.csv`` and the LineVul-preprocessed
release, which adds ``flaw_line`` and ``flaw_line_index``. Everything downstream works on the
canonical columns defined here, so the release only matters in this module.

Canonical columns:

==================== ===========================================================
``sample_id``        stable id derived from the row's position in the source file
``project``          project name
``commit_id``        fixing commit
``cwe``              ``"CWE-<n>"``, or ``""`` when unknown
``vul``              1 for a vulnerable function, 0 otherwise
``func_before``      function body before the fix
``func_after``       function body after the fix (``""`` if the release lacks it)
``flaw_lines``       text of the lines removed or changed by the fix (may be empty)
``flaw_line_indices`` 0-based indices of those lines within ``func_before``
==================== ===========================================================
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

CANONICAL_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "project",
    "commit_id",
    "cwe",
    "vul",
    "func_before",
    "func_after",
    "flaw_lines",
    "flaw_line_indices",
)

# Canonical name -> names it goes by in the known releases, in order of preference.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "project": ("project",),
    "commit_id": ("commit_id",),
    "cwe": ("CWE ID", "cwe_id", "cwe"),
    "vul": ("vul", "target"),
    "func_before": ("func_before", "processed_func"),
    "func_after": ("func_after",),
    "flaw_lines": ("flaw_line", "flaw_lines"),
    "flaw_line_indices": ("flaw_line_index", "flaw_line_indices"),
}
REQUIRED: tuple[str, ...] = ("project", "commit_id", "cwe", "vul", "func_before")
FLAW_LINE_SEPARATOR = "/~/"


class BigVulFormatError(ValueError):
    """Raised when a BigVul file lacks the columns needed to build the canonical frame."""


def resolve_columns(available: Iterable[str]) -> dict[str, str]:
    """Map canonical names to the source columns present in ``available``.

    Raises :class:`BigVulFormatError` naming every required column that cannot be found. Flaw
    lines are optional only when ``func_after`` is present, since they can then be derived
    from the diff.
    """
    available = list(available)
    present = set(available)
    mapping = {
        canonical: next(alias for alias in aliases if alias in present)
        for canonical, aliases in COLUMN_ALIASES.items()
        if any(alias in present for alias in aliases)
    }
    missing = [name for name in REQUIRED if name not in mapping]
    has_flaw_lines = "flaw_lines" in mapping and "flaw_line_indices" in mapping
    if not has_flaw_lines and "func_after" not in mapping:
        missing.append("func_after (or flaw_line and flaw_line_index)")
    if missing:
        details = "; ".join(
            f"{name} (looked for: {', '.join(COLUMN_ALIASES.get(name, (name,)))})"
            for name in missing
        )
        raise BigVulFormatError(
            f"BigVul file is missing required columns: {details}. "
            f"Columns found: {', '.join(sorted(map(str, available)))}"
        )
    return mapping


def normalise_cwe(value: object) -> str:
    """``"CWE-119"``, ``"119"`` and ``119.0`` all become ``"CWE-119"``; missing becomes ``""``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    match = re.search(r"\d+", str(value))
    return f"CWE-{int(match.group())}" if match else ""


def parse_flaw_lines(value: object) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(v) for v in value]
    if not isinstance(value, str) or not value:
        return []
    return value.split(FLAW_LINE_SEPARATOR)


def parse_flaw_line_indices(value: object) -> list[int]:
    if isinstance(value, list | tuple):
        return [int(v) for v in value]
    if isinstance(value, int | float) and not pd.isna(value):
        return [int(value)]
    if not isinstance(value, str) or not value.strip():
        return []
    return [int(part) for part in re.findall(r"\d+", value)]


def canonicalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical frame from a raw BigVul frame, preserving row order."""
    mapping = resolve_columns(raw.columns)
    n = len(raw)
    out = pd.DataFrame(index=range(n))
    out["sample_id"] = [f"bigvul_{i:06d}" for i in range(n)]
    out["project"] = raw[mapping["project"]].fillna("").astype(str).to_numpy()
    out["commit_id"] = raw[mapping["commit_id"]].fillna("").astype(str).to_numpy()
    out["cwe"] = [normalise_cwe(v) for v in raw[mapping["cwe"]]]
    out["vul"] = (
        pd.to_numeric(raw[mapping["vul"]], errors="coerce").fillna(0).astype(int).to_numpy()
    )
    out["func_before"] = raw[mapping["func_before"]].fillna("").astype(str).to_numpy()
    if "func_after" in mapping:
        out["func_after"] = raw[mapping["func_after"]].fillna("").astype(str).to_numpy()
    else:
        out["func_after"] = ""
    if "flaw_lines" in mapping and "flaw_line_indices" in mapping:
        out["flaw_lines"] = [parse_flaw_lines(v) for v in raw[mapping["flaw_lines"]]]
        out["flaw_line_indices"] = [
            parse_flaw_line_indices(v) for v in raw[mapping["flaw_line_indices"]]
        ]
    else:
        out["flaw_lines"] = [[] for _ in range(n)]
        out["flaw_line_indices"] = [[] for _ in range(n)]
    return out[list(CANONICAL_COLUMNS)]


def load_bigvul(path: str | Path) -> pd.DataFrame:
    """Load a BigVul CSV (any known release) or a canonical Parquet file.

    Only the needed columns are read from CSV, which keeps memory in check: the full file
    carries the patch and CVE metadata for every row.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        missing = [c for c in CANONICAL_COLUMNS if c not in frame.columns]
        if missing:
            raise BigVulFormatError(f"{path}: not a canonical file, missing {missing}")
        for column in ("flaw_lines", "flaw_line_indices"):
            frame[column] = [list(v) for v in frame[column]]
        frame["flaw_line_indices"] = [[int(i) for i in v] for v in frame["flaw_line_indices"]]
        return frame[list(CANONICAL_COLUMNS)]

    header = pd.read_csv(path, nrows=0).columns
    mapping = resolve_columns(header)
    raw = pd.read_csv(path, usecols=sorted(set(mapping.values())), low_memory=False)
    return canonicalise(raw)


def convert_to_parquet(source: str | Path, destination: str | Path) -> pd.DataFrame:
    """Load ``source`` and write the canonical frame to ``destination`` as Parquet."""
    frame = load_bigvul(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return frame
