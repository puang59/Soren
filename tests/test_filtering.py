import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from soren.data.bigvul import canonicalise
from soren.data.filtering import (
    FilterConfig,
    body_hash,
    code_line_mask,
    derive_flaw_lines,
    filter_bigvul,
    is_trivial_line,
    strip_comments,
)

BEFORE = """int f(char *s, int n) {
    char buf[8];
    /* copy the input */
    strcpy(buf, s);
    return n;
}"""
AFTER = BEFORE.replace("strcpy(buf, s);", "strncpy(buf, s, 7);")


def row(**overrides) -> dict:
    base = {
        "project": "linux",
        "commit_id": "c0",
        "CWE ID": "CWE-119",
        "vul": 1,
        "func_before": BEFORE,
        "func_after": AFTER,
    }
    return {**base, **overrides}


def frame(rows: list[dict]) -> pd.DataFrame:
    return canonicalise(pd.DataFrame(rows))


def steps(attrition) -> dict[str, int]:
    return {entry["step"]: entry["rows"] for entry in attrition}


def test_strip_comments_preserves_line_numbers():
    code = "a; /* one\n two */ b;\n// gone\nc; // tail"
    assert strip_comments(code) == "a; \n b;\n\nc; "
    assert strip_comments(code).count("\n") == code.count("\n")


@pytest.mark.parametrize(
    "line", ["", "   ", "{", "}", "  } ", "});", "// note", "/* x */", " * doc", "*/"]
)
def test_trivial_lines(line):
    assert is_trivial_line(line)


@pytest.mark.parametrize(
    "line", ["x = 1;", "} else {", "return;", "*p = 0;", "f(); // call", "a * b;"]
)
def test_code_lines(line):
    assert not is_trivial_line(line)


def test_code_line_mask_handles_multi_line_comments():
    code = "int x;\n/* start\n   strcpy(a, b);\n end */\n}\ny = 1;"
    assert code_line_mask(code) == [True, False, False, False, False, True]


def test_derive_flaw_lines_from_the_diff():
    assert derive_flaw_lines(BEFORE, AFTER) == [3]
    # Lines only added by the fix leave nothing to point at.
    added = BEFORE.replace("    return n;", "    if (n < 0) return -1;\n    return n;")
    assert derive_flaw_lines(BEFORE, added) == []
    # A deletion and a change in different places.
    after = "\n".join(line for i, line in enumerate(AFTER.split("\n")) if i != 1)
    assert derive_flaw_lines(BEFORE, after) == [1, 3]


def test_body_hash_ignores_comments_and_whitespace():
    reformatted = BEFORE.replace("    ", "\t").replace("/* copy the input */", "// different words")
    assert body_hash(BEFORE) == body_hash(reformatted)
    assert body_hash(BEFORE) != body_hash(AFTER)


def test_filters_apply_in_order_with_attrition():
    other = BEFORE.replace("buf", "dst")
    rows = [
        row(),  # kept
        row(vul=0),  # not vulnerable
        row(**{"CWE ID": "CWE-20"}),  # CWE not selected
        row(func_after=BEFORE),  # no flaw lines: the fix changed nothing here
        row(func_after=BEFORE.replace("/* copy the input */", "/* copy */")),  # comment-only flaw
        row(func_before="int g() {\n  x();\n}", func_after="int g() {\n  y();\n}"),  # too short
        row(func_before=BEFORE.replace("    ", "  "), commit_id="c1"),  # duplicate of the first
        row(
            func_before=other,
            func_after=other.replace("strcpy", "strlcpy"),
            **{"CWE ID": "CWE-125"},
        ),
    ]
    filtered, attrition = filter_bigvul(frame(rows), FilterConfig())
    assert steps(attrition) == {
        "loaded": 8,
        "vulnerable": 7,
        "selected_cwes": 6,
        "has_flaw_lines": 5,
        "has_code_flaw_lines": 4,
        "line_count": 3,
        "deduplicated": 2,
    }
    assert filtered["sample_id"].tolist() == ["bigvul_000000", "bigvul_000007"]
    assert filtered["cwe"].tolist() == ["CWE-119", "CWE-125"]
    assert filtered["flaw_line_indices"].tolist() == [[3], [3]]
    assert filtered.loc[0, "flaw_lines"] == ["    strcpy(buf, s);"]
    assert filtered["num_lines"].tolist() == [6, 6]
    assert filtered["body_hash"].is_unique


def test_given_flaw_lines_are_used_and_restricted_to_code():
    linevul = pd.DataFrame(
        {
            "project": ["linux"],
            "commit_id": ["c0"],
            "CWE ID": ["CWE-119"],
            "target": [1],
            "processed_func": [BEFORE],
            "flaw_line": ["    /* copy the input */ /~/    strcpy(buf, s);/~/}"],
            "flaw_line_index": ["2,3,5"],
        }
    )
    filtered, _ = filter_bigvul(canonicalise(linevul))
    assert filtered.loc[0, "flaw_line_indices"] == [3]
    assert filtered.loc[0, "flaw_lines"] == ["    strcpy(buf, s);"]


def test_out_of_range_flaw_indices_are_dropped():
    linevul = pd.DataFrame(
        {
            "project": ["p"],
            "commit_id": ["c"],
            "CWE ID": ["CWE-119"],
            "target": [1],
            "processed_func": [BEFORE],
            "flaw_line": ["x"],
            "flaw_line_index": ["99"],
        }
    )
    filtered, attrition = filter_bigvul(canonicalise(linevul))
    assert filtered.empty and steps(attrition)["has_code_flaw_lines"] == 0


def test_line_count_bounds_are_configurable():
    filtered, _ = filter_bigvul(frame([row()]), FilterConfig(min_lines=7))
    assert filtered.empty
    filtered, _ = filter_bigvul(frame([row()]), FilterConfig(max_lines=5))
    assert filtered.empty
    filtered, _ = filter_bigvul(frame([row()]), FilterConfig(cwes=["CWE-787"]))
    assert filtered.empty


def test_script_writes_parquet_and_attrition(tmp_path, capsys):
    source = tmp_path / "raw.csv"
    pd.DataFrame([row(), row(vul=0)]).to_csv(source, index=False)
    out = tmp_path / "interim" / "filtered.parquet"
    attrition = tmp_path / "processed" / "attrition.json"

    script = Path(__file__).parent.parent / "scripts" / "01_filter_bigvul.py"
    spec = importlib.util.spec_from_file_location("filter_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = Path(__file__).parent.parent / "configs" / "data.yaml"
    module.main(
        [
            "--input",
            str(source),
            "--output",
            str(out),
            "--attrition",
            str(attrition),
            "--config",
            str(config),
        ]
    )

    written = pd.read_parquet(out)
    assert written["sample_id"].tolist() == ["bigvul_000000"]
    assert {"num_lines", "body_hash", "flaw_line_indices"} <= set(written.columns)
    log = json.loads(attrition.read_text())
    assert log["cwes"] == ["CWE-119", "CWE-125"]
    assert [entry["step"] for entry in log["steps"]][0] == "loaded"
    assert log["steps"][-1] == {"step": "deduplicated", "rows": 1}
    assert "wrote 1 functions" in capsys.readouterr().out
