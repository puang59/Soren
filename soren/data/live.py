"""Turn one pasted function into a graph, by running Joern on it.

The offline pipeline does this in batches (``scripts/02`` to ``04``); this module does the same
steps for a single function, without labels, so it can be analysed on request. Like the
pipeline, it parses the function as C and as C++ and keeps whichever parse is better.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from soren.data.build import BuildConfig
from soren.data.cfg_builder import (
    CfgBuildError,
    build_line_cfg,
    load_joern_methods,
    rejection_reason,
    select_method,
)
from soren.data.schema import GraphRecord
from soren.data.sources import EXTENSIONS
from soren.data.transform import limit_out_degree, max_out_degree

ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCRIPT = ROOT / "joern" / "export_cfg.sc"
SAMPLE_ID = "pasted"
MAX_CHARS = 20_000


class AnalysisError(ValueError):
    """The function could not be analysed; the message is written for the person who pasted it."""


def find_joern(joern_home: str | Path | None = None) -> Path | None:
    """The directory holding ``joern`` and ``joern-parse``, looked up as the pipeline does."""
    candidates = [joern_home, os.environ.get("JOERN_HOME"), ROOT / ".tools" / "joern-cli"]
    on_path = shutil.which("joern")
    if on_path:
        candidates.append(Path(on_path).parent)
    for candidate in candidates:
        if candidate and (Path(candidate) / "joern-parse").is_file():
            return Path(candidate)
    return None


def run_joern(code: str, joern_home: Path, timeout: float = 120.0) -> Path:
    """Parse ``code`` as C and C++ and return the export file, inside a fresh temp directory."""
    work = Path(tempfile.mkdtemp(prefix="soren-live-"))
    (work / "src").mkdir()
    for extension in EXTENSIONS:
        (work / "src" / f"{SAMPLE_ID}{extension}").write_text(code, encoding="utf-8")
    out = work / "out.jsonl"
    commands = [
        [str(joern_home / "joern-parse"), str(work / "src"), "--language", "c", "-o", "cpg.bin"],
        [str(joern_home / "joern"), "--script", str(EXPORT_SCRIPT),
         "--param", "cpgFile=cpg.bin", "--param", f"outFile={out}"],
    ]  # fmt: skip
    try:
        for command in commands:
            # Joern keeps a workspace in the current directory, so it runs inside ``work``.
            subprocess.run(command, cwd=work, check=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise AnalysisError("Joern took too long on this function.") from None
    except subprocess.CalledProcessError:
        raise AnalysisError("Joern could not parse this code.") from None
    return out


def normalise(code: str) -> str:
    text = code.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not text.strip():
        raise AnalysisError("Paste a C or C++ function first.")
    if len(text) > MAX_CHARS:
        raise AnalysisError(f"That is too long: the limit is {MAX_CHARS:,} characters.")
    return text + "\n"


def build_record(code: str, export: str | Path, config: BuildConfig | None = None) -> GraphRecord:
    """The unlabelled graph of the function in ``code``, from Joern's ``export`` of it."""
    config = config or BuildConfig()
    method = select_method(load_joern_methods([export]).get(SAMPLE_ID, {}))
    if method is None:
        raise AnalysisError("No function found. Paste one complete C or C++ function.")
    try:
        cfg = build_line_cfg(method, code)
    except CfgBuildError:
        raise AnalysisError("The function has no statements to walk.") from None
    reason = rejection_reason(cfg, config.min_coverage)
    if reason == "low_coverage":
        raise AnalysisError(
            "Most of the function is missing from the parse, usually because of macros or "
            "preprocessor branches."
        )
    if reason is not None:
        raise AnalysisError("The control flow of this function could not be recovered.")
    if len(cfg.nodes) < config.min_nodes:
        raise AnalysisError("The function is too short: it needs at least three statements.")
    if len(cfg.nodes) > config.max_nodes:
        raise AnalysisError(
            f"The function is too long: {len(cfg.nodes) - 2} statements, and the limit is "
            f"{config.max_nodes - 2}."
        )
    record = GraphRecord(
        sample_id=SAMPLE_ID,
        nodes=cfg.nodes,
        edges=cfg.edges,
        back_edges=cfg.back_edges,
        entry=cfg.entry,
        exit=cfg.exit,
        vuln_nodes=[],
        source_lines=cfg.source_lines,
    ).validate(labelled=False)
    if max_out_degree(record) > config.max_out_degree:
        record = limit_out_degree(record, config.max_out_degree)
    return record


def analyse(code: str, joern_home: Path, config: BuildConfig | None = None) -> GraphRecord:
    """Parse a pasted function with Joern and return its graph."""
    text = normalise(code)
    export = run_joern(text, joern_home)
    try:
        return build_record(text, export, config)
    finally:
        shutil.rmtree(export.parent, ignore_errors=True)
