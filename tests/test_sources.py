import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from soren.data.sources import batch_name, write_sources

ROOT = Path(__file__).parent.parent
BODY = "int f(int n) {\n    int x = n;\n    return x;\n}"


def frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"bigvul_{i:06d}" for i in range(n)],
            "func_before": [BODY.replace("f(", f"f{i}(") for i in range(n)],
        }
    )


def test_one_file_per_function_in_batches(tmp_path):
    manifest = write_sources(frame(7), tmp_path, batch_size=3)
    assert manifest["batch"].tolist() == ["batch_0000"] * 3 + ["batch_0001"] * 3 + ["batch_0002"]
    assert manifest["sample_id"].tolist() == [f"bigvul_{i:06d}" for i in range(7)]
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.c"))
    assert files == sorted(manifest["path"])
    assert len(files) == 7
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "batch_0000",
        "batch_0001",
        "batch_0002",
        "manifest.csv",
    ]
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "manifest.csv"), manifest)


def test_line_one_of_the_file_is_line_one_of_the_function(tmp_path):
    write_sources(frame(1), tmp_path)
    lines = (tmp_path / "batch_0000" / "bigvul_000000.c").read_text().split("\n")
    body_lines = BODY.replace("f(", "f0(").split("\n")
    # File line i + 1 is function line index i, so joern_line = flaw_line_index + 1.
    assert lines[: len(body_lines)] == body_lines
    assert lines[len(body_lines) :] == [""]  # exactly one trailing newline


def test_line_endings_are_normalised(tmp_path):
    data = pd.DataFrame({"sample_id": ["a", "b"], "func_before": ["x;\r\ny;\r\n", "p;\rq;"]})
    write_sources(data, tmp_path)
    assert (tmp_path / "batch_0000" / "a.c").read_bytes() == b"x;\ny;\n"
    assert (tmp_path / "batch_0000" / "b.c").read_bytes() == b"p;\nq;\n"


def test_rejects_duplicate_ids_and_bad_batch_size(tmp_path):
    duplicated = pd.DataFrame({"sample_id": ["a", "a"], "func_before": ["x;", "y;"]})
    with pytest.raises(ValueError, match="unique"):
        write_sources(duplicated, tmp_path)
    with pytest.raises(ValueError, match="batch_size"):
        write_sources(frame(1), tmp_path, batch_size=0)
    assert batch_name(12) == "batch_0012"


def test_script(tmp_path, capsys):
    source = tmp_path / "filtered.parquet"
    frame(5).assign(cwe="CWE-119").to_parquet(source)
    spec = importlib.util.spec_from_file_location(
        "write_sources", ROOT / "scripts" / "02_write_sources.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "src"
    module.main(["--input", str(source), "--out", str(out), "--batch-size", "2"])
    assert len(list(out.rglob("*.c"))) == 5
    assert "wrote 5 files in 3 batches" in capsys.readouterr().out
