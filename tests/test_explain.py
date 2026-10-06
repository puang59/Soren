from soren.viz.explain import fix_diff


def test_fix_diff_marks_removed_and_added_lines():
    before = "int f(char *s) {\n  char b[8];\n  strcpy(b, s);\n  return 0;\n}"
    after = before.replace("strcpy(b, s);", "strncpy(b, s, 7);")
    rows = fix_diff(before, after, context=1)
    assert [(r["kind"], r["line"]) for r in rows] == [
        ("context", 2),
        ("removed", 3),
        ("added", None),
        ("context", 4),
    ]
    assert rows[1]["text"].strip() == "strcpy(b, s);" and "strncpy" in rows[2]["text"]
    assert fix_diff(before, before) == []
