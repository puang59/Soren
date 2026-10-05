import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from helpers import line_graph

from soren.agents.ppo import PPOConfig
from soren.agents.train import train
from soren.config import load_config
from soren.data.synthetic import SyntheticConfig, generate_dataset
from soren.env.cfg_nav_env import CFGNavEnv
from soren.env.policies import OraclePolicy, RandomPolicy, rollout
from soren.env.wrappers import CurriculumConfig, CurriculumSampler


def sized_graphs(sizes):
    """Line graphs with exactly the given node counts."""
    graphs = []
    for i, size in enumerate(sizes):
        graph = line_graph(size - 2)
        graph.sample_id = f"g{i}_{size}"
        graphs.append(graph)
    return graphs


GRAPHS = sized_graphs([10, 15, 20, 25, 30, 40, 60, 90])


def sampled_sizes(sampler, n=400, seed=0):
    rng = np.random.default_rng(seed)
    return Counter(GRAPHS[sampler(rng)].num_nodes for _ in range(n))


def test_starts_with_small_graphs_only():
    sampler = CurriculumSampler(GRAPHS, CurriculumConfig(start_cap=20))
    assert sampler.cap == 20 and sampler.eligible == 3 and not sampler.finished
    assert set(sampled_sizes(sampler)) == {10, 15, 20}


def test_widens_when_the_rolling_success_rate_reaches_the_threshold():
    cfg = CurriculumConfig(start_cap=20, growth=1.5, success_threshold=0.8, window=10)
    sampler = CurriculumSampler(GRAPHS, cfg)
    for _ in range(9):
        sampler.report(True)
    assert sampler.cap == 20  # window not full yet
    sampler.report(True)
    assert sampler.cap == 30 and sampler.eligible == 5
    assert set(sampled_sizes(sampler)) == {10, 15, 20, 25, 30}


def test_does_not_widen_below_the_threshold_and_resets_after_widening():
    cfg = CurriculumConfig(start_cap=20, success_threshold=0.8, window=10)
    sampler = CurriculumSampler(GRAPHS, cfg)
    for i in range(50):
        sampler.report(i % 2 == 0)  # 50% success
    assert sampler.cap == 20
    streak = 0
    while sampler.cap == 20:
        sampler.report(True)
        streak += 1
    assert sampler.cap == 30 and streak <= 10  # widened once the window held 80% successes
    # The window was cleared: a few more successes are not enough on their own.
    for _ in range(9):
        sampler.report(True)
    assert sampler.cap == 30


def test_fixed_schedule_ignores_success():
    cfg = CurriculumConfig(start_cap=20, growth=2.0, widen_every=5)
    sampler = CurriculumSampler(GRAPHS, cfg)
    caps = []
    for _ in range(15):
        sampler.report(False)
        caps.append(sampler.cap)
    assert caps == [20] * 4 + [40] * 5 + [80] * 5 + [90]
    assert sampler.finished


def test_cap_reaches_the_largest_graph_and_then_sampling_is_uniform():
    sampler = CurriculumSampler(GRAPHS, CurriculumConfig(start_cap=20, widen_every=1))
    for _ in range(20):
        sampler.report(True)
    assert sampler.finished and sampler.cap == 90 and sampler.eligible == len(GRAPHS)
    counts = sampled_sizes(sampler, n=4000)
    assert set(counts) == {g.num_nodes for g in GRAPHS}
    assert max(counts.values()) < 1.3 * min(counts.values())
    sampler.report(True)  # a no-op once finished
    assert sampler.cap == 90


def test_a_widening_always_adds_graphs():
    # 21..59 are empty sizes: one widening must jump straight to a cap that adds a graph.
    graphs = sized_graphs([10, 20, 60])
    sampler = CurriculumSampler(graphs, CurriculumConfig(start_cap=20, growth=1.1, widen_every=1))
    sampler.report(True)
    assert sampler.eligible == 3


def test_start_cap_below_the_smallest_graph_is_raised():
    sampler = CurriculumSampler(GRAPHS, CurriculumConfig(start_cap=3))
    assert sampler.cap == 10 and sampler.eligible == 1


def test_environment_reports_outcomes_and_exposes_the_cap():
    cfg = CurriculumConfig(start_cap=20, window=5, success_threshold=1.0)
    sampler = CurriculumSampler(GRAPHS, cfg)
    env = CFGNavEnv(GRAPHS, sampler=sampler)
    caps = []
    for seed in range(5):
        result = rollout(env, OraclePolicy(), seed=seed)
        assert env.graph.num_nodes <= 20
        caps.append(result.info["curriculum_cap"])
    assert caps == [20, 20, 20, 20, 30]  # the fifth success fills the window
    assert env.reset(seed=0)[1]["curriculum_cap"] == 30


def test_failures_keep_the_environment_on_small_graphs():
    sampler = CurriculumSampler(GRAPHS, CurriculumConfig(start_cap=20, window=5))
    env = CFGNavEnv(GRAPHS, sampler=sampler)
    for seed in range(40):
        rollout(env, RandomPolicy(), seed=seed)
    assert sampler.cap == 20


def test_without_a_curriculum_sampling_is_uniform_and_info_is_unchanged():
    env = CFGNavEnv(GRAPHS)
    _, info = env.reset(seed=0)
    assert "curriculum_cap" not in info
    seen = {env.reset()[1]["graph_id"] for _ in range(200)}
    assert len(seen) == len(GRAPHS)


def test_config_validation_and_defaults():
    path = Path(__file__).parent.parent / "configs" / "env.yaml"
    assert load_config(CurriculumConfig, path, section="curriculum") == CurriculumConfig()
    assert CurriculumConfig().enabled is False
    for bad in ({"start_cap": 0}, {"growth": 1.0}, {"window": 0}, {"widen_every": 0}):
        with pytest.raises(ValueError):
            CurriculumConfig(**bad)
    with pytest.raises(ValueError):
        CurriculumSampler([])


def test_training_with_a_curriculum(tmp_path):
    graphs = generate_dataset(30, seed=0, cfg=SyntheticConfig(min_nodes=8, max_nodes=40))
    cfg = PPOConfig(
        total_timesteps=512,
        n_envs=2,
        n_steps=128,
        batch_size=64,
        n_epochs=1,
        net_arch=[16],
        vec_env="dummy",
        eval_freq=512,
    )
    curriculum = CurriculumConfig(enabled=True, start_cap=15, widen_every=20)
    result = train(
        graphs, graphs[:4], tmp_path / "c", cfg, curriculum=curriculum, tensorboard=False
    )
    config = json.loads((result.run_dir / "config.json").read_text())
    assert config["curriculum"]["enabled"] is True and config["curriculum"]["start_cap"] == 15
    assert result.history
