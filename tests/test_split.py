import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest
from helpers import line_graph

from soren.config import from_dict, load_yaml
from soren.data.schema import read_jsonl, write_jsonl
from soren.data.split import (
    LeakageError,
    SplitConfig,
    check_no_leakage,
    source_hash,
    split_by_commit,
    split_summary,
)

ROOT = Path(__file__).parent.parent


def graph(index: int, commit: str, cwe: str = "CWE-119"):
    record = line_graph(3 + index % 4)
    record.sample_id = f"g{index:04d}"
    record.commit_id = commit
    record.cwe = cwe
    record.source_lines = [f"int f{index}() {{", f"  return {index};", "}"]
    return record


def dataset(n_commits: int = 120):
    """Commits of one to four functions; every fourth commit is CWE-125."""
    records = []
    for commit in range(n_commits):
        cwe = "CWE-125" if commit % 4 == 0 else "CWE-119"
        for _ in range(1 + commit % 4):
            records.append(graph(len(records), f"commit{commit:03d}", cwe))
    return records


def test_whole_commits_stay_together():
    splits = split_by_commit(dataset())
    home = {}
    for name, records in splits.items():
        for record in records:
            assert home.setdefault(record.commit_id, name) == name
    assert sum(len(v) for v in splits.values()) == len(dataset())
    assert len({r.sample_id for v in splits.values() for r in v}) == len(dataset())


def test_ratios_are_approximated_overall_and_per_cwe():
    records = dataset()
    splits = split_by_commit(records)
    total = len(records)
    assert len(splits["train"]) / total == pytest.approx(0.8, abs=0.02)
    assert len(splits["val"]) / total == pytest.approx(0.1, abs=0.02)
    assert len(splits["test"]) / total == pytest.approx(0.1, abs=0.02)
    for cwe in ("CWE-119", "CWE-125"):
        in_cwe = sum(r.cwe == cwe for r in records)
        train = sum(r.cwe == cwe for r in splits["train"])
        assert train / in_cwe == pytest.approx(0.8, abs=0.04)
        assert any(r.cwe == cwe for r in splits["val"])
        assert any(r.cwe == cwe for r in splits["test"])


def test_split_is_deterministic_and_seed_dependent():
    records = dataset()
    ids = lambda splits: {k: [r.sample_id for r in v] for k, v in splits.items()}  # noqa: E731
    assert ids(split_by_commit(records)) == ids(split_by_commit(records))
    assert ids(split_by_commit(records)) != ids(split_by_commit(records, SplitConfig(seed=1)))
    # Each split keeps the input order.
    for records_in_split in split_by_commit(records).values():
        assert [r.sample_id for r in records_in_split] == sorted(
            r.sample_id for r in records_in_split
        )


def test_functions_without_a_commit_are_their_own_groups():
    records = [graph(i, "") for i in range(40)]
    splits = split_by_commit(records)
    assert len(splits["train"]) == 32 and len(splits["val"]) == 4 and len(splits["test"]) == 4


def test_leak_checks():
    a, b = graph(0, "c1"), graph(1, "c2")
    check_no_leakage({"train": [a], "val": [b], "test": []})
    with pytest.raises(LeakageError, match="commit c1 appears in both train and test"):
        check_no_leakage({"train": [a], "val": [], "test": [graph(2, "c1")]})
    # The same body under a different commit, differing only in comments and spacing.
    twin = dataclasses.replace(graph(3, "c9"))
    twin.source_lines = ["int f0()  {", "  return 0; // same thing", "}"]
    assert source_hash(twin) == source_hash(a)
    with pytest.raises(LeakageError, match="function body also appears in train"):
        check_no_leakage({"train": [a], "val": [twin], "test": []})


def test_duplicate_bodies_across_commits_fail_the_split():
    records = dataset(40)
    clone = graph(9999, "another_commit")
    clone.source_lines = list(records[0].source_lines)
    with pytest.raises(LeakageError):
        # Whether this raises depends on where the clone lands, so try seeds until it is
        # separated from its twin; with 20% outside train that happens within a few seeds.
        for seed in range(30):
            split_by_commit([*records, clone], SplitConfig(seed=seed))


def test_config():
    assert (
        from_dict(SplitConfig, load_yaml(ROOT / "configs" / "data.yaml")["split"]) == SplitConfig()
    )
    with pytest.raises(ValueError, match="sum to 1"):
        SplitConfig(train=0.9, val=0.2, test=0.1)
    with pytest.raises(ValueError, match="non-negative"):
        SplitConfig(train=1.2, val=-0.2, test=0.0)
    only_train = split_by_commit(dataset(20), SplitConfig(train=1.0, val=0.0, test=0.0))
    assert not only_train["val"] and not only_train["test"]


def test_summary_and_script(tmp_path, capsys):
    records = dataset(60)
    summary = split_summary(split_by_commit(records))
    assert sum(info["graphs"] for info in summary.values()) == len(records)
    assert sum(info["commits"] for info in summary.values()) == 60
    assert set(summary["train"]["by_cwe"]) == {"CWE-119", "CWE-125"}

    source = tmp_path / "graphs_all.jsonl"
    write_jsonl(records, source)
    spec = importlib.util.spec_from_file_location("split", ROOT / "scripts" / "05_split.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(
        [
            "--input",
            str(source),
            "--out-dir",
            str(tmp_path),
            "--config",
            str(ROOT / "configs" / "data.yaml"),
        ]
    )
    written = {
        name: read_jsonl(tmp_path / f"graphs_{name}.jsonl") for name in ("train", "val", "test")
    }
    assert sum(len(v) for v in written.values()) == len(records)
    assert json.loads((tmp_path / "split.json").read_text())["splits"]["train"]["graphs"] == len(
        written["train"]
    )
    assert "wrote graphs_{train,val,test}.jsonl" in capsys.readouterr().out
