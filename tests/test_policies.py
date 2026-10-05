import numpy as np
import pytest
from helpers import diamond_graph, line_graph, loop_graph

from soren.data.synthetic import SyntheticConfig, generate_dataset
from soren.env.cfg_nav_env import CFGNavEnv, EnvConfig
from soren.env.policies import (
    DeclareImmediatelyPolicy,
    NeverDeclarePolicy,
    OraclePolicy,
    RandomPolicy,
    rollout,
)
from soren.env.rewards import RewardConfig, distance_to_vulnerable

GRAPHS = generate_dataset(150, seed=21, cfg=SyntheticConfig(min_nodes=8, max_nodes=60))


def pinned(index: int) -> dict:
    return {"graph_index": index}


def test_oracle_solves_every_graph_along_a_shortest_path():
    env = CFGNavEnv(GRAPHS)
    for index, graph in enumerate(GRAPHS):
        result = rollout(env, OraclePolicy(), seed=0, options=pinned(index))
        assert result.success
        assert result.info["end_reason"] == "correct"
        assert result.nodes[-1] in graph.vuln_set
        # Moves along a shortest path, plus the final declare. The forward distance from
        # ENTRY can only be longer than the undirected one.
        assert len(result.actions) - 1 >= distance_to_vulnerable(graph)[graph.entry]
        assert env.backtrack_action not in result.actions


def test_oracle_takes_the_shortest_branch():
    result = rollout(CFGNavEnv([diamond_graph(vuln=3)]), OraclePolicy(), seed=0)
    assert result.nodes == [0, 1, 3, 3]
    assert result.episode_return == pytest.approx(2 * RewardConfig().step + RewardConfig().correct)


def test_oracle_backtracks_when_started_past_the_target():
    env = CFGNavEnv([line_graph(4, vuln=1)])
    env.reset(seed=0)
    env.step(0)
    env.step(0)  # now at node 2, one step past the vulnerable node
    rng = np.random.default_rng(0)
    assert OraclePolicy().act(env, rng) == env.backtrack_action
    env.step(env.backtrack_action)
    assert OraclePolicy().act(env, rng) == env.declare_action


def test_declare_immediately_fails_unless_the_first_statement_is_vulnerable():
    env = CFGNavEnv([line_graph(3, vuln=3), line_graph(3, vuln=1)])
    miss = rollout(env, DeclareImmediatelyPolicy(), seed=0, options=pinned(0))
    hit = rollout(env, DeclareImmediatelyPolicy(), seed=0, options=pinned(1))
    assert not miss.success and miss.info["end_reason"] == "wrong_declare"
    assert len(miss.actions) == 2
    assert hit.success


def test_never_declare_always_times_out():
    env = CFGNavEnv(GRAPHS[:20])
    for index in range(20):
        result = rollout(env, NeverDeclarePolicy(), seed=index, options=pinned(index))
        assert result.info["end_reason"] == "timeout"
        assert len(result.actions) == env.max_steps
        assert env.declare_action not in result.actions


@pytest.mark.parametrize("policy", [RandomPolicy(), NeverDeclarePolicy(), OraclePolicy()])
def test_trajectories_are_deterministic(policy):
    """The same seed gives the same graph, actions, rewards and observations."""
    env_a, env_b = CFGNavEnv(GRAPHS), CFGNavEnv(GRAPHS)
    for seed in range(15):
        a = rollout(env_a, policy, seed=seed)
        b = rollout(env_b, policy, seed=seed)
        assert a.graph_id == b.graph_id
        assert a.actions == b.actions
        assert a.rewards == b.rewards
        assert a.nodes == b.nodes
    assert len({rollout(env_a, policy, seed=s).graph_id for s in range(15)}) > 1


def test_same_action_sequence_gives_identical_observations():
    env_a, env_b = CFGNavEnv([loop_graph()]), CFGNavEnv([loop_graph()])
    reference = rollout(env_a, RandomPolicy(), seed=4)
    obs_a, _ = env_a.reset(seed=4)
    obs_b, _ = env_b.reset(seed=4)
    np.testing.assert_array_equal(obs_a, obs_b)
    for action in reference.actions:
        step_a = env_a.step(action)
        step_b = env_b.step(action)
        np.testing.assert_array_equal(step_a[0], step_b[0])
        assert step_a[1:4] == step_b[1:4]


@pytest.mark.parametrize("shaping", [False, True])
def test_reward_sanity_oracle_beats_degenerate_policies(shaping):
    """Guards the reward scale: the intended behaviour must be the best-paid one."""
    env = CFGNavEnv(GRAPHS, reward=RewardConfig(shaping=shaping))
    for index in range(len(GRAPHS)):
        oracle = rollout(env, OraclePolicy(), seed=0, options=pinned(index))
        for policy in (DeclareImmediatelyPolicy(), NeverDeclarePolicy(), RandomPolicy()):
            other = rollout(env, policy, seed=index, options=pinned(index))
            assert oracle.episode_return >= other.episode_return
            if not other.success:
                assert oracle.episode_return > other.episode_return


def test_reward_sanity_holds_with_a_declare_budget():
    env = CFGNavEnv(GRAPHS[:50], EnvConfig(max_declares=3))
    for index in range(50):
        oracle = rollout(env, OraclePolicy(), seed=0, options=pinned(index))
        guesser = rollout(env, DeclareImmediatelyPolicy(), seed=0, options=pinned(index))
        assert oracle.episode_return >= guesser.episode_return


@pytest.mark.slow
def test_random_policy_stress():
    """10,000 random episodes across configurations without an exception."""
    graphs = generate_dataset(400, seed=33, cfg=SyntheticConfig(min_nodes=6, max_nodes=90))
    configs = [
        (EnvConfig(), RewardConfig()),
        (EnvConfig(max_declares=3), RewardConfig(shaping=True)),
        (EnvConfig(allow_backtrack=False), RewardConfig()),
        (EnvConfig(strict_masks=False), RewardConfig()),
    ]
    episodes = 0
    for env_cfg, reward_cfg in configs:
        env = CFGNavEnv(graphs, env_cfg, reward_cfg)
        for seed in range(2500):
            result = rollout(env, RandomPolicy(), seed=seed)
            assert len(result.actions) <= env.max_steps
            assert np.isfinite(result.rewards).all()
            assert result.info["end_reason"] in {"correct", "wrong_declare", "timeout", "dead_end"}
            episodes += 1
    assert episodes == 10_000
