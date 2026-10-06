import importlib.util
import json
from pathlib import Path

import pandas as pd

from soren.config import from_dict, load_yaml
from soren.data.build import STAGES, BuildConfig, build_graphs
from soren.data.cfg_builder import load_joern_methods
from soren.data.schema import read_jsonl
from soren.env.cfg_nav_env import CFGNavEnv
from soren.env.policies import OraclePolicy, rollout

ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures" / "joern"
METHODS = load_joern_methods([FIXTURES / "snippets.jsonl"])


def row(name: str, flaw_indices: list[int], extension: str = ".c") -> dict:
    source = (FIXTURES / "src" / f"{name}{extension}").read_text()
    lines = source.split("\n")
    return {
        "sample_id": name,
        "project": "proj",
        "commit_id": f"commit_{name}",
        "cwe": "CWE-119",
        "func_before": source,
        "flaw_line_indices": flaw_indices,
        "flaw_lines": [lines[i] for i in flaw_indices],
    }


def filtered() -> pd.DataFrame:
    return pd.DataFrame(
        [
            row("nested_loops", [7]),  # kept: `c += m;`
            row("multi_line", [7]),  # kept: middle line of the memcpy call
            row("method", [3], ".cpp"),  # kept: parsed only as C++
            row("switch_case", [5, 8]),  # kept: two flaw lines
            row("straight", [3]),  # kept at min_nodes=5
            row("if_else", [5]),  # dropped: the flaw line is `} else {`
            row("macro_heavy", [6]),  # dropped: low coverage
            {**row("straight", [3]), "sample_id": "never_parsed"},  # dropped: no Joern output
        ]
    )


def test_build_graphs_stages_and_reasons():
    records, attrition, reasons = build_graphs(filtered(), METHODS)
    assert [step["step"] for step in attrition] == list(STAGES)
    assert [step["rows"] for step in attrition] == [8, 7, 7, 6, 5, 5]
    assert reasons == {"low_coverage": 1, "no_flaw_on_node": 1, "not_parsed": 1}
    assert [r.sample_id for r in records] == [
        "nested_loops",
        "multi_line",
        "method",
        "switch_case",
        "straight",
    ]


def test_records_carry_labels_and_metadata():
    records, _, _ = build_graphs(filtered(), METHODS)
    by_id = {r.sample_id: r for r in records}
    nested = by_id["nested_loops"]
    assert (nested.project, nested.commit_id, nested.cwe) == (
        "proj",
        "commit_nested_loops",
        "CWE-119",
    )
    assert [nested.nodes[n].code for n in nested.vuln_nodes] == ["c += m;"]
    assert [by_id["multi_line"].nodes[n].line for n in by_id["multi_line"].vuln_nodes] == [7]
    assert len(by_id["switch_case"].vuln_nodes) == 2
    assert nested.features == {}  # features are computed on demand, not stored
    for record in records:
        record.validate()


def test_every_built_graph_is_solvable():
    records, _, _ = build_graphs(filtered(), METHODS)
    env = CFGNavEnv(records)
    for index in range(len(records)):
        assert rollout(env, OraclePolicy(), seed=0, options={"graph_index": index}).success


def test_size_bounds():
    _, attrition, reasons = build_graphs(filtered(), METHODS, BuildConfig(min_nodes=6))
    assert attrition[-1]["rows"] == 3
    assert reasons["too_few_nodes"] == 2  # `straight` and the C++ method have five nodes
    records, _, reasons = build_graphs(filtered(), METHODS, BuildConfig(max_nodes=6))
    assert reasons["too_many_nodes"] == 2
    assert {r.sample_id for r in records} == {"multi_line", "method", "straight"}


def test_coverage_threshold_is_configurable():
    _, _, reasons = build_graphs(filtered(), METHODS, BuildConfig(min_coverage=0.1))
    assert "low_coverage" not in reasons
    # With the parse accepted, the flaw inside the inactive #ifdef still has no node.
    assert reasons["no_flaw_on_node"] == 2


def test_text_mismatch_drops_the_sample_not_the_run():
    frame = filtered().iloc[:2].copy()
    frame.at[0, "flaw_line_indices"] = [8]  # points one line past the recorded text
    records, _, reasons = build_graphs(frame, METHODS)
    assert reasons == {"text_mismatch": 1}
    assert [r.sample_id for r in records] == ["multi_line"]


def test_config_defaults_match_the_yaml():
    settings = load_yaml(ROOT / "configs" / "data.yaml")
    assert from_dict(BuildConfig, settings["graphs"]) == BuildConfig()


def test_script_writes_graphs_and_extends_the_attrition_log(tmp_path, capsys):
    source = tmp_path / "filtered.parquet"
    filtered().to_parquet(source)
    joern = tmp_path / "joern"
    joern.mkdir()
    (joern / "batch_0000.jsonl").write_text((FIXTURES / "snippets.jsonl").read_text())
    attrition = tmp_path / "attrition.json"
    attrition.write_text(
        json.dumps({"cwes": ["CWE-119"], "steps": [{"step": "loaded", "rows": 9}]})
    )
    out = tmp_path / "graphs_all.jsonl"

    spec = importlib.util.spec_from_file_location("build", ROOT / "scripts" / "04_build_graphs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(
        ["--filtered", str(source), "--joern", str(joern), "--output", str(out),
         "--attrition", str(attrition), "--config", str(ROOT / "configs" / "data.yaml")]
    )  # fmt: skip
    assert len(read_jsonl(out)) == 5
    log = json.loads(attrition.read_text())
    assert log["steps"] == [{"step": "loaded", "rows": 9}]  # the filter's log is preserved
    assert log["graph_steps"][-1] == {"step": "size_within_bounds", "rows": 5}
    assert log["graph_drop_reasons"]["not_parsed"] == 1
    assert "wrote 5 graphs" in capsys.readouterr().out
