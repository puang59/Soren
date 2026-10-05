import numpy as np
import pandas as pd
import pytest

from soren.data.bigvul import canonicalise
from soren.data.eda import (
    attrition_by_cwe,
    dataset_summary,
    first_flaw_positions,
    position_summary,
    relative_flaw_positions,
    vulnerable_cwe_counts,
)
from soren.data.filtering import filter_bigvul

BEFORE = """int f(char *s, int n) {
    char buf[8];
    int k = n;
    strcpy(buf, s);
    return k;
}"""


def raw() -> pd.DataFrame:
    def row(i, cwe, vul=1, changed=True):
        body = BEFORE.replace("buf", f"buf{i}")
        after = body.replace("strcpy", "strlcpy") if changed else body
        return {
            "project": f"p{i % 2}",
            "commit_id": f"c{i}",
            "CWE ID": cwe,
            "vul": vul,
            "func_before": body,
            "func_after": after,
        }

    rows = [row(0, "CWE-119"), row(1, "CWE-119"), row(2, "CWE-119", changed=False)]
    rows += [row(3, "CWE-125"), row(4, "CWE-20"), row(5, None), row(6, "CWE-119", vul=0)]
    return canonicalise(pd.DataFrame(rows))


def test_vulnerable_cwe_counts():
    counts = vulnerable_cwe_counts(raw())
    assert counts.to_dict() == {"CWE-119": 3, "CWE-125": 1, "CWE-20": 1, "(none)": 1}
    assert counts.index[0] == "CWE-119"


def test_attrition_by_cwe():
    table = attrition_by_cwe(raw(), ["CWE-119", "CWE-125", "CWE-787"])
    assert table.loc["selected_cwes"].to_dict() == {"CWE-119": 3, "CWE-125": 1, "CWE-787": 0}
    assert table.loc["has_flaw_lines"].to_dict() == {"CWE-119": 2, "CWE-125": 1, "CWE-787": 0}
    assert table.loc["deduplicated"].to_dict() == {"CWE-119": 2, "CWE-125": 1, "CWE-787": 0}
    assert table.index[0] == "selected_cwes" and table.index[-1] == "deduplicated"


def test_positions_and_summary():
    filtered, _ = filter_bigvul(raw())
    positions = relative_flaw_positions(filtered)
    assert positions.tolist() == pytest.approx([3 / 5] * 3)
    np.testing.assert_allclose(first_flaw_positions(filtered), positions)

    table = position_summary(np.array([0.0, 0.05, 0.55, 1.0]), bins=10)
    assert table["share"].tolist() == pytest.approx([0.5, 0, 0, 0, 0, 0.25, 0, 0, 0, 0.25])
    assert table["uniform"].eq(0.1).all()

    summary = dataset_summary(filtered)
    assert summary["functions"] == 3 and summary["projects"] == 2
    assert summary["median flaw lines"] == 1 and summary["single flaw line (share)"] == 1.0
    assert summary["median lines"] == 6
