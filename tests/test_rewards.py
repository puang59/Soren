import numpy as np
import pytest
from helpers import diamond_graph, line_graph, loop_graph

from soren.data.synthetic import generate_dataset
from soren.env.cfg_nav_env import CFGNavEnv, EnvConfig
from soren.env.rewards import (
    RewardConfig,
    RewardModel,
    StepEvent,
    base_reward,
    distance_to_vulnerable,
)

CFG = RewardConfig()
K = EnvConfig().k
BACKTRACK, DECLARE = K, K + 1


def make_env(graph, reward=None, **cfg) -> CFGNavEnv:
    env = CFGNavEnv([graph], EnvConfig(**cfg), reward)
    env.reset(seed=0)
    return env


def test_move_costs_a_step():
    env = make_env(diamond_graph())
    assert env.step(0)[1] == pytest.approx(CFG.step)


def test_backtrack_costs_a_step_without_a_revisit_penalty():
    env = make_env(diamond_graph())
    env.step(0)
    assert env.step(BACKTRACK)[1] == pytest.approx(CFG.step)


def test_moving_onto_a_visited_node_costs_extra():
    env = make_env(loop_graph())
    for action in (0, 0, 0):
        env.step(action)
    # Node 3 -> loop header (node 1), already visited.
    assert env.step(0)[1] == pytest.approx(CFG.step + CFG.revisit)


def test_correct_and_wrong_declarations():
    env = make_env(diamond_graph(vuln=3))
    env.step(0)
    env.step(1)
    assert env.step(DECLARE)[1] == pytest.approx(CFG.correct)

    env = make_env(diamond_graph(vuln=3))
    env.step(0)
    assert env.step(DECLARE)[1] == pytest.approx(CFG.wrong)


def test_timeout_penalty_is_added_to_the_last_step():
    env = make_env(line_graph(5), max_steps_cap=2)
    env.step(0)
    _, reward, terminated, truncated, info = env.step(0)
    assert truncated and not terminated
    assert reward == pytest.approx(CFG.step + CFG.timeout)
    assert info["episode_return"] == pytest.approx(2 * CFG.step + CFG.timeout)


def test_dead_end_is_penalised_like_a_timeout():
    env = make_env(line_graph(1), allow_backtrack=False)
    env.step(0)
    _, reward, terminated, _, _ = env.step(0)
    assert terminated
    assert reward == pytest.approx(CFG.step + CFG.timeout)


def test_wasted_action_costs_a_step_in_lenient_mode():
    env = make_env(diamond_graph(), strict_masks=False)
    assert env.step(BACKTRACK)[1] == pytest.approx(CFG.step)


def test_declare_budget():
    env = make_env(diamond_graph(vuln=3), max_declares=2)
    env.step(0)
    _, reward, terminated, _, _ = env.step(DECLARE)
    assert reward == pytest.approx(CFG.wrong) and not terminated
    env.step(0)
    _, reward, terminated, _, info = env.step(DECLARE)
    assert reward == pytest.approx(CFG.wrong) and terminated
    assert info["episode_return"] == pytest.approx(2 * CFG.step + 2 * CFG.wrong)


def test_custom_values_are_used():
    cfg = RewardConfig(step=-0.1, correct=5.0)
    env = make_env(diamond_graph(vuln=3), reward=cfg)
    assert env.step(0)[1] == pytest.approx(-0.1)
    env.step(1)
    assert env.step(DECLARE)[1] == pytest.approx(5.0)


def test_base_reward_event_table():
    g = line_graph()
    assert base_reward(StepEvent(g, "move", 0, 1), CFG) == CFG.step
    assert base_reward(StepEvent(g, "backtrack", 1, 0, revisit=True), CFG) == CFG.step
    assert base_reward(StepEvent(g, "declare", 1, 1, correct=False), CFG) == CFG.wrong
    timed_out = StepEvent(g, "move", 0, 1, revisit=True, timeout=True)
    assert base_reward(timed_out, CFG) == pytest.approx(CFG.step + CFG.revisit + CFG.timeout)


def test_distance_to_vulnerable_is_undirected():
    # 0 ENTRY -> 1 -> 2 -> 3 (vulnerable) -> 4 -> 5 EXIT
    dist = distance_to_vulnerable(line_graph(4, vuln=3))
    assert dist.tolist() == [3, 2, 1, 0, 1, 2]
    assert distance_to_vulnerable(diamond_graph(vuln=3)).tolist() == [2, 1, 2, 0, 1, 2]


def _rollout(env: CFGNavEnv, actions) -> tuple[list[float], dict]:
    env.reset(seed=0)
    rewards, info = [], {}
    for action in actions:
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    return rewards, info


def test_shaping_rewards_progress_and_leaves_the_unshaped_return_alone():
    graph = line_graph(4, vuln=3)
    actions = [0, 0, 0, DECLARE]
    plain_rewards, plain = _rollout(make_env(graph), actions)
    shaped_rewards, shaped = _rollout(make_env(graph, RewardConfig(shaping=True)), actions)

    assert shaped["episode_return"] == pytest.approx(plain["episode_return"])
    assert plain["shaped_return"] == pytest.approx(plain["episode_return"])
    assert shaped["shaped_return"] == pytest.approx(sum(shaped_rewards))
    # Each move towards the target earns a positive shaping bonus.
    for with_shaping, without in zip(shaped_rewards[:3], plain_rewards[:3], strict=True):
        assert with_shaping > without


def test_shaping_penalises_moving_away():
    graph = line_graph(4, vuln=1)
    plain, _ = _rollout(make_env(graph), [0, 0])
    shaped, _ = _rollout(make_env(graph, RewardConfig(shaping=True)), [0, 0])
    assert shaped[1] < plain[1]


def test_shaping_telescopes_to_the_start_potential():
    """With gamma = 1 the shaping terms of a finished episode sum to -phi(start)."""
    cfg = RewardConfig(shaping=True, gamma=1.0, shaping_beta=0.05)
    rng = np.random.default_rng(0)
    for graph in generate_dataset(25, seed=9):
        env = CFGNavEnv([graph], reward=cfg)
        env.reset(seed=0)
        start_potential = RewardModel(cfg).potential(graph, graph.entry)
        info, done = {}, False
        while not done:
            action = rng.choice(env.valid_actions())
            *_, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        bonus = info["shaped_return"] - info["episode_return"]
        assert bonus == pytest.approx(-start_potential)


def test_break_even_search_length():
    """A success found within the break-even number of steps beats a blind wrong guess.

    With the defaults the break-even is (correct - wrong) / |step| = 150 steps, so the whole
    budget is worth spending on graphs of up to 37 nodes. On larger graphs the tail of the
    200-step cap is worth less than guessing; the hyperparameter sweep revisits this.
    """
    break_even = (CFG.correct - CFG.wrong) / abs(CFG.step)
    assert break_even == pytest.approx(150)
    env_cfg = EnvConfig()
    for num_nodes in (5, 10, 20, 37):
        budget = env_cfg.max_steps_for(num_nodes)
        assert budget < break_even
        assert CFG.correct + budget * CFG.step > CFG.wrong
    assert env_cfg.max_steps_for(300) > break_even
