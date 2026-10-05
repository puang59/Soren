import importlib.util
from pathlib import Path

import numpy as np
import pytest
from helpers import diamond_graph, line_graph

from soren.agents.ppo import PPOConfig
from soren.agents.searcher import PolicySearcher, evaluate_policy, record_policy_trace
from soren.agents.train import train
from soren.baselines import BFS, DFS, LineOrder, RandomOrder, RandomWalk
from soren.data.schema import write_jsonl
from soren.data.synthetic import SyntheticConfig, generate_dataset
from soren.env.cfg_nav_env import CFGNavEnv, EnvConfig
from soren.env.rewards import RewardConfig
from soren.env.wrappers import TraceRecorder
from soren.viz.trace import INSPECT, Trace, TraceStep

GRAPHS = generate_dataset(20, seed=13, cfg=SyntheticConfig(min_nodes=8, max_nodes=30))
ROOT = Path(__file__).parent.parent
K = EnvConfig().k
R = RewardConfig()


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_trace_round_trips_through_json(tmp_path):
    trace = Trace(
        graph_id="g",
        method="ppo",
        start_node=0,
        steps=[
            TraceStep(0, 0, "MOVE_0", 1, -0.01, [True, False], [0.9, 0.1], 0.4),
            TraceStep(1, 1, "DECLARE", 1, 1.0, [False, True], None, None),
        ],
        outcome={"success": True, "declared_node": 1, "return": 0.99},
    )
    assert Trace.from_json(trace.to_json()) == trace
    path = trace.save(tmp_path / "nested" / "t.json")
    assert Trace.load(path) == trace
    assert trace.nodes == [0, 1, 1]
    trace.validate()


def test_validate_catches_inconsistent_steps():
    with pytest.raises(ValueError, match="has t="):
        Trace("g", "m", 0, [TraceStep(1, 0, "MOVE_0", 1, 0.0)]).validate()
    with pytest.raises(ValueError, match="starts at"):
        Trace("g", "m", 0, [TraceStep(0, 5, "MOVE_0", 1, 0.0)]).validate()


def test_recorder_captures_steps_masks_and_outcome():
    recorder = TraceRecorder(CFGNavEnv([diamond_graph(vuln=3)]), method="manual")
    recorder.reset(seed=0)
    recorder.step(0)
    recorder.annotate(probs=[0.25, 0.75], value=0.5)
    recorder.step(1)
    recorder.step(K + 1)
    trace = recorder.trace.validate()

    assert (trace.graph_id, trace.method, trace.start_node) == ("diamond", "manual", 0)
    assert [s.action for s in trace.steps] == ["MOVE_0", "MOVE_1", "DECLARE"]
    assert trace.nodes == [0, 1, 3, 3]
    assert [s.reward for s in trace.steps] == pytest.approx([R.step, R.step, R.correct])
    assert trace.steps[0].mask == [True] + [False] * (K + 1)
    assert trace.steps[0].probs is None and trace.steps[0].value is None
    assert trace.steps[1].probs == [0.25, 0.75] and trace.steps[1].value == 0.5
    assert trace.steps[2].probs is None  # an annotation applies to one step only
    assert trace.outcome == {
        "success": True,
        "declared_node": 3,
        "return": pytest.approx(2 * R.step + R.correct),
        "end_reason": "correct",
        "nodes_inspected": 3,
    }


def test_recorder_starts_a_fresh_trace_on_reset_and_exposes_masks():
    recorder = TraceRecorder(CFGNavEnv([diamond_graph()]))
    with pytest.raises(RuntimeError):
        recorder.step(0)
    recorder.reset(seed=0)
    recorder.step(0)
    np.testing.assert_array_equal(recorder.action_masks(), recorder.core.action_masks())
    recorder.reset(seed=0)
    assert recorder.trace.steps == []


def test_recorded_rewards_are_unshaped():
    env = CFGNavEnv([line_graph(4, vuln=3)], reward=RewardConfig(shaping=True))
    recorder = TraceRecorder(env)
    recorder.reset(seed=0)
    _, shaped, *_ = recorder.step(0)
    assert shaped != pytest.approx(R.step)
    assert recorder.trace.steps[0].reward == pytest.approx(R.step)


@pytest.mark.parametrize("searcher", [DFS(), RandomWalk(), BFS(), LineOrder(), RandomOrder()])
def test_tracing_a_baseline_does_not_change_its_outcome(searcher):
    for graph in GRAPHS:
        budget = EnvConfig().max_steps_for(graph.num_nodes)
        result = searcher.run(graph, budget, rng(3))
        trace = searcher.trace(graph, budget, rng(3)).validate()
        assert trace.method == searcher.name and trace.graph_id == graph.sample_id
        assert trace.outcome["success"] == result.success
        assert trace.outcome["declared_node"] == result.declared_node
        assert trace.outcome["nodes_inspected"] == result.nodes_inspected
        assert trace.outcome["return"] == pytest.approx(result.cumulative_reward)
        assert sum(step.reward for step in trace.steps) == pytest.approx(result.cumulative_reward)
        assert len(trace.steps) == result.actions_taken
        assert list(dict.fromkeys(trace.nodes)) == result.visit_order


def test_order_baselines_trace_as_inspect_jumps():
    trace = BFS().trace(diamond_graph(vuln=3), 100, rng())
    assert [s.action for s in trace.steps] == [INSPECT, INSPECT, INSPECT, "DECLARE"]
    assert trace.nodes == [0, 1, 2, 3, 3]
    assert all(s.probs is None and s.mask is None for s in trace.steps)

    failed = LineOrder().trace(line_graph(6, vuln=6), 2, rng())
    assert not failed.outcome["success"]
    assert failed.steps[-1].reward == pytest.approx(R.step + R.timeout)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    cfg = PPOConfig(
        total_timesteps=256,
        n_envs=2,
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        net_arch=[16],
        vec_env="dummy",
        eval_freq=256,
    )
    return train(GRAPHS[:12], GRAPHS[12:16], tmp_path_factory.mktemp("runs") / "t", cfg, seed=0)


def test_policy_trace_matches_evaluation_and_carries_policy_outputs(run):
    results = evaluate_policy(run.model, GRAPHS[:6])
    for graph, result in zip(GRAPHS[:6], results, strict=True):
        trace = record_policy_trace(run.model, graph).validate()
        assert trace.outcome["success"] == result.success
        assert trace.outcome["return"] == pytest.approx(result.cumulative_reward)
        assert len(trace.steps) == result.actions_taken
        for step in trace.steps:
            assert len(step.probs) == K + 2 and len(step.mask) == K + 2
            assert sum(step.probs) == pytest.approx(1.0, abs=1e-5)
            # Masked actions get no probability mass.
            assert all(p < 1e-6 for p, m in zip(step.probs, step.mask, strict=True) if not m)
            assert isinstance(step.value, float)

    searcher_trace = PolicySearcher(run.model, name="agent").trace(GRAPHS[0], 5, rng())
    assert searcher_trace.method == "agent" and len(searcher_trace.steps) <= 5


def test_make_traces_script(tmp_path, run, capsys):
    spec = importlib.util.spec_from_file_location(
        "make_traces", ROOT / "scripts" / "make_traces.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    graphs = tmp_path / "val.jsonl"
    write_jsonl(GRAPHS, graphs)
    out = tmp_path / "traces"
    module.main(
        [
            "--graphs", str(graphs),
            "--out", str(out),
            "--limit", "4",
            "--methods", "dfs", "bfs",
            "--checkpoints", str(run.best_model_path),
            "--env-config", str(ROOT / "configs" / "env.yaml"),
        ]
    )  # fmt: skip
    for method in ("dfs", "bfs", "ppo"):
        files = sorted((out / method).glob("*.json"))
        assert [f.stem for f in files] == [g.sample_id for g in GRAPHS[:4]]
        assert Trace.load(files[0]).validate().method == method
    assert "wrote 4 traces" in capsys.readouterr().out
