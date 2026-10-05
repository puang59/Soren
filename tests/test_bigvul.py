import pandas as pd
import pytest

from soren.data.bigvul import (
    CANONICAL_COLUMNS,
    BigVulFormatError,
    canonicalise,
    convert_to_parquet,
    load_bigvul,
    normalise_cwe,
    parse_flaw_line_indices,
    parse_flaw_lines,
    resolve_columns,
)

BEFORE = "int f(char *s) {\n    char buf[8];\n    strcpy(buf, s);\n    return 0;\n}"
AFTER = "int f(char *s) {\n    char buf[8];\n    strncpy(buf, s, 7);\n    return 0;\n}"


def msr_frame() -> pd.DataFrame:
    """Shaped like the original MSR_data_cleaned.csv: no flaw-line columns."""
    return pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "CVE ID": ["CVE-1", "CVE-2", "CVE-3"],
            "CWE ID": ["CWE-119", None, "CWE-125"],
            "commit_id": ["aaa", "bbb", "ccc"],
            "project": ["linux", "chrome", None],
            "func_before": [BEFORE, "void g() {}", BEFORE],
            "func_after": [AFTER, "void g() {}", AFTER],
            "vul": [1, 0, 1],
            "patch": ["...", "...", "..."],
        }
    )


def linevul_frame() -> pd.DataFrame:
    """Shaped like the LineVul release: target, processed_func and flaw-line columns."""
    return pd.DataFrame(
        {
            "index": [10, 11],
            "CWE ID": ["CWE-119", "CWE-20"],
            "commit_id": ["aaa", "ddd"],
            "project": ["linux", "ffmpeg"],
            "processed_func": [BEFORE, "void h() {}"],
            "target": [1, 0],
            "flaw_line": ["    strcpy(buf, s);/~/    return 0;", None],
            "flaw_line_index": ["2,3", None],
        }
    )


def test_canonicalise_msr_release():
    frame = canonicalise(msr_frame())
    assert tuple(frame.columns) == CANONICAL_COLUMNS
    assert frame["sample_id"].tolist() == ["bigvul_000000", "bigvul_000001", "bigvul_000002"]
    assert frame["cwe"].tolist() == ["CWE-119", "", "CWE-125"]
    assert frame["vul"].tolist() == [1, 0, 1]
    assert frame["project"].tolist() == ["linux", "chrome", ""]
    assert frame.loc[0, "func_before"] == BEFORE and frame.loc[0, "func_after"] == AFTER
    assert frame["flaw_lines"].tolist() == [[], [], []]
    assert frame["flaw_line_indices"].tolist() == [[], [], []]


def test_canonicalise_linevul_release():
    frame = canonicalise(linevul_frame())
    assert tuple(frame.columns) == CANONICAL_COLUMNS
    assert frame["vul"].tolist() == [1, 0]
    assert frame.loc[0, "func_before"] == BEFORE
    assert frame["func_after"].tolist() == ["", ""]
    assert frame.loc[0, "flaw_lines"] == ["    strcpy(buf, s);", "    return 0;"]
    assert frame.loc[0, "flaw_line_indices"] == [2, 3]
    assert frame.loc[1, "flaw_lines"] == [] and frame.loc[1, "flaw_line_indices"] == []


def test_sample_ids_follow_row_position_not_the_index():
    shuffled = msr_frame().set_index(pd.Index([7, 3, 9]))
    assert canonicalise(shuffled)["sample_id"].tolist()[0] == "bigvul_000000"


@pytest.mark.parametrize("column", ["commit_id", "project", "CWE ID", "vul", "func_before"])
def test_missing_required_column_is_named(column):
    with pytest.raises(BigVulFormatError) as error:
        canonicalise(msr_frame().drop(columns=[column]))
    canonical = {"CWE ID": "cwe"}.get(column, column)
    assert canonical in str(error.value)
    assert "Columns found:" in str(error.value)


def test_needs_either_func_after_or_flaw_lines():
    with pytest.raises(BigVulFormatError, match="func_after"):
        canonicalise(msr_frame().drop(columns=["func_after"]))
    # The LineVul shape has no func_after but carries flaw lines, which is enough.
    canonicalise(linevul_frame())
    with pytest.raises(BigVulFormatError, match="func_after"):
        canonicalise(linevul_frame().drop(columns=["flaw_line_index"]))


def test_resolve_columns_prefers_the_first_alias():
    mapping = resolve_columns(
        ["project", "commit_id", "CWE ID", "vul", "target", "func_before", "func_after"]
    )
    assert mapping["vul"] == "vul"
    assert mapping["cwe"] == "CWE ID"


def test_normalise_cwe():
    assert normalise_cwe("CWE-119") == "CWE-119"
    assert normalise_cwe("119") == "CWE-119"
    assert normalise_cwe(119.0) == "CWE-119"
    assert normalise_cwe("cwe-020") == "CWE-20"
    assert normalise_cwe(None) == ""
    assert normalise_cwe(float("nan")) == ""
    assert normalise_cwe("NVD-CWE-Other") == ""


def test_parse_flaw_fields():
    assert parse_flaw_lines("a/~/b") == ["a", "b"]
    assert parse_flaw_lines(None) == [] and parse_flaw_lines(float("nan")) == []
    assert parse_flaw_line_indices("1,2, 5") == [1, 2, 5]
    assert parse_flaw_line_indices("[3, 4]") == [3, 4]
    assert parse_flaw_line_indices(7.0) == [7]
    assert parse_flaw_line_indices(None) == [] and parse_flaw_line_indices("") == []


def test_load_csv_reads_only_needed_columns(tmp_path):
    path = tmp_path / "MSR_data_cleaned.csv"
    msr_frame().to_csv(path, index=False)
    frame = load_bigvul(path)
    pd.testing.assert_frame_equal(frame, canonicalise(msr_frame()))


def test_load_csv_with_missing_column_fails_loudly(tmp_path):
    path = tmp_path / "bad.csv"
    msr_frame().drop(columns=["commit_id"]).to_csv(path, index=False)
    with pytest.raises(BigVulFormatError, match="commit_id"):
        load_bigvul(path)


def test_parquet_round_trip(tmp_path):
    csv = tmp_path / "linevul.csv"
    linevul_frame().to_csv(csv, index=False)
    parquet = tmp_path / "interim" / "bigvul.parquet"
    written = convert_to_parquet(csv, parquet)
    loaded = load_bigvul(parquet)
    pd.testing.assert_frame_equal(loaded, written)
    assert loaded.loc[0, "flaw_line_indices"] == [2, 3]
    assert isinstance(loaded.loc[0, "flaw_lines"], list)


def test_raw_parquet_is_canonicalised(tmp_path):
    path = tmp_path / "raw.parquet"
    msr_frame().to_parquet(path)
    pd.testing.assert_frame_equal(load_bigvul(path), canonicalise(msr_frame()))
    bad = tmp_path / "bad.parquet"
    msr_frame().drop(columns=["vul"]).to_parquet(bad)
    with pytest.raises(BigVulFormatError, match="vul"):
        load_bigvul(bad)


def test_directory_of_parquet_files_is_concatenated_in_name_order(tmp_path):
    """The Hugging Face mirror ships one Parquet file per split."""
    first, second = msr_frame().iloc[:1], msr_frame().iloc[1:]
    second.to_parquet(tmp_path / "a-test.parquet")
    first.to_parquet(tmp_path / "b-train.parquet")
    frame = load_bigvul(tmp_path)
    assert frame["commit_id"].tolist() == ["bbb", "ccc", "aaa"]
    assert frame["sample_id"].tolist() == ["bigvul_000000", "bigvul_000001", "bigvul_000002"]
    assert tuple(frame.columns) == CANONICAL_COLUMNS
    with pytest.raises(BigVulFormatError, match="no Parquet files"):
        load_bigvul(_empty_dir(tmp_path))


def _empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    return empty
