import itertools

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from helpers import diamond_graph, line_graph, loop_graph, make_graph

from soren.data.features import feature_dim
from soren.data.schema import NODE_KINDS, GraphRecord, Node
from soren.data.synthetic import SyntheticConfig, generate_dataset
from soren.env.cfg_nav_env import CFGNavEnv, EnvConfig, InvalidActionError
from soren.env.rewards import RewardConfig

K = EnvConfig().k
BACKTRACK, DECLARE = K, K + 1


def make_env(graphs, **cfg) -> CFGNavEnv:
    return CFGNavEnv(graphs, EnvConfig(**cfg))


def start(env: CFGNavEnv, index: int = 0, **options):
    return env.reset(seed=0, options={"graph_index": index, **options})


def test_check_env_passes():
    graphs = generate_dataset(20, seed=0)
    # check_env samples actions without consulting the mask, so run it in lenient mode.
    check_env(make_env(graphs, strict_masks=False), skip_render_check=True)


def test_spaces():
    env = make_env([diamond_graph()])
    f = feature_dim("S")
    assert env.action_space.n == K + 2
    assert env.observation_space.shape == (f + 2 + K * (f + 3) + 2 + len(NODE_KINDS) + 5,)
    assert env.action_name(0) == "MOVE_0"
    assert env.action_name(BACKTRACK) == "BACKTRACK"
    assert env.action_name(DECLARE) == "DECLARE"


def test_observation_shape_is_constant_across_graph_sizes():
    graphs = [line_graph(), diamond_graph(), loop_graph()]
    graphs += generate_dataset(10, seed=1, cfg=SyntheticConfig(min_nodes=8, max_nodes=80))
    env = make_env(graphs)
    for index in range(len(graphs)):
        obs, _ = start(env, index)
        while True:
            assert obs.shape == env.observation_space.shape
            assert obs.dtype == np.float32
            assert env.observation_space.contains(obs)
            valid = env.valid_actions()
            obs, _, terminated, truncated, _ = env.step(valid[index % len(valid)])
            if terminated or truncated:
                assert env.observation_space.contains(obs)
                break


def _expected_mask(env: CFGNavEnv) -> list[bool]:
    graph, node = env.graph, env.node
    out_degree = sum(1 for u, _ in graph.edges if u == node)
    mask = [i < out_degree for i in range(K)]
    mask.append(len(env.stack) > 0)
    mask.append(graph.nodes[node].kind not in ("ENTRY", "EXIT") and node not in env.declared)
    return mask


@pytest.mark.parametrize("graph", [line_graph(), diamond_graph(), loop_graph()])
def test_masks_match_valid_actions_in_every_reachable_state(graph):
    """Replay every valid action sequence up to a fixed length and check each mask."""
    env = make_env([graph], max_declares=2)
    frontier: list[tuple[int, ...]] = [()]
    checked = 0
    for _ in range(7):
        next_frontier = []
        for prefix in frontier:
            start(env)
            done = False
            for action in prefix:
                *_, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            if done:
                assert not env.action_masks().any()
                continue
            mask = env.action_masks()
            assert mask.tolist() == _expected_mask(env)
            checked += 1
            next_frontier += [(*prefix, a) for a in np.flatnonzero(mask)]
        frontier = next_frontier
    assert checked > 10


def test_masked_action_raises_in_strict_mode():
    env = make_env([diamond_graph()])
    start(env)
    for action in (1, BACKTRACK, DECLARE):  # ENTRY: one successor, empty stack, no declare
        with pytest.raises(InvalidActionError):
            env.step(action)
    with pytest.raises(InvalidActionError):
        env.step(K + 5)


def test_masked_action_is_a_wasted_step_in_lenient_mode():
    env = make_env([diamond_graph()], strict_masks=False)
    start(env)
    _, _, terminated, truncated, info = env.step(BACKTRACK)
    assert not terminated and not truncated
    assert info["current_node"] == 0 and info["steps"] == 1


def test_move_follows_canonical_successor_order():
    env = make_env([diamond_graph()])
    start(env)
    env.step(0)
    assert env.node == 1
    env.step(1)
    assert env.node == 3 and env.stack == [0, 1]


def test_move_then_backtrack_restores_node_and_stack():
    env = make_env([diamond_graph()])
    start(env)
    env.step(0)
    node, stack = env.node, list(env.stack)
    env.step(0)
    _, _, _, _, info = env.step(BACKTRACK)
    assert env.node == node and env.stack == stack
    assert info["action_name"] == "BACKTRACK"
    assert info["visited_unique"] == 3


def test_backtrack_leaves_a_dead_end():
    env = make_env([line_graph(2)])
    start(env)
    for _ in range(3):
        env.step(0)
    assert env.graph.nodes[env.node].kind == "EXIT"
    assert env.valid_actions() == [BACKTRACK]


def test_correct_declare_terminates_with_success():
    env = make_env([diamond_graph(vuln=3)])
    start(env)
    env.step(0)
    env.step(1)
    _, reward, terminated, truncated, info = env.step(DECLARE)
    assert terminated and not truncated
    assert reward == 1.0
    assert info["is_success"] and info["declared_node"] == 3
    assert info["end_reason"] == "correct"
    assert info["first_hit_step"] == 2
    with pytest.raises(RuntimeError):
        env.step(0)


def test_wrong_declare_ends_the_episode_when_the_budget_is_spent():
    env = make_env([diamond_graph(vuln=3)])
    start(env)
    env.step(0)
    _, _, terminated, _, info = env.step(DECLARE)
    assert terminated and not info["is_success"]
    assert info["end_reason"] == "wrong_declare"
    assert info["first_hit_step"] is None


def test_declare_budget_allows_further_attempts():
    env = make_env([diamond_graph(vuln=3)], max_declares=3)
    start(env)
    env.step(0)
    obs, _, terminated, _, _ = env.step(DECLARE)
    assert not terminated
    assert DECLARE not in env.valid_actions()  # cannot declare the same node twice
    assert obs[feature_dim("S") + 1] == 1.0  # declared-here flag
    assert obs[-1] == pytest.approx(2 / 3)  # declares remaining
    env.step(1)
    _, _, terminated, _, info = env.step(DECLARE)
    assert terminated and info["is_success"]
    assert info["first_declared_node"] == 1 and info["declared_node"] == 3


def test_step_limit_truncates():
    env = make_env([loop_graph()], max_steps_cap=5)
    start(env)
    for step in range(5):
        action = 0 if env.action_masks()[0] else BACKTRACK
        _, _, terminated, truncated, info = env.step(action)
        assert not terminated
        assert truncated == (step == 4)
    assert info["end_reason"] == "timeout"


def test_max_steps_scales_with_graph_size_and_can_be_overridden():
    env = make_env([line_graph(3)])
    start(env)
    assert env.max_steps == 4 * 5
    start(env, max_steps=7)
    assert env.max_steps == 7
    assert EnvConfig().max_steps_for(1000) == 200


def test_disabling_backtrack_makes_exit_a_dead_end():
    env = make_env([line_graph(2)], allow_backtrack=False)
    start(env)
    env.step(0)
    assert BACKTRACK not in env.valid_actions()
    env.step(0)
    _, _, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert info["end_reason"] == "dead_end"


def test_reset_options_select_the_graph():
    graphs = [line_graph(), diamond_graph(), loop_graph()]
    env = make_env(graphs)
    _, info = env.reset(options={"graph_id": "loop"})
    assert info["graph_id"] == "loop"
    _, info = env.reset(options={"graph_index": 1})
    assert info["graph_id"] == "diamond"
    with pytest.raises(ValueError):
        env.reset(options={"graph_id": "nope"})
    with pytest.raises(ValueError):
        env.reset(options={"graph_index": 9})


def test_sampler_hook_and_uniform_sampling():
    graphs = [line_graph(), diamond_graph(), loop_graph()]
    env = CFGNavEnv(graphs, sampler=lambda rng: 2)
    assert env.reset(seed=0)[1]["graph_id"] == "loop"
    uniform = make_env(graphs)
    uniform.reset(seed=0)
    seen = {uniform.reset()[1]["graph_id"] for _ in range(60)}
    assert seen == {"line", "diamond", "loop"}


def test_info_keys():
    env = make_env([diamond_graph()])
    _, info = start(env)
    expected = {
        "graph_id",
        "current_node",
        "action_name",
        "visited_unique",
        "is_success",
        "declared_node",
        "first_hit_step",
    }
    assert expected <= set(info)
    assert info["current_node"] == 0 and info["visited_unique"] == 1


def test_successor_slots_describe_the_successors():
    env = make_env([loop_graph()])
    f = feature_dim("S")
    slot = f + 3
    start(env)
    env.step(0)  # at the loop header: successors are node 2 (body) and node 4 (return)
    for action in (0, 0):
        obs, *_ = env.step(action)
    # Now at node 3, whose only successor is the header via a back edge, already visited.
    base = f + 2
    assert obs[base] == 1.0
    assert obs[base + 1 + f] == 1.0  # visited_before
    assert obs[base + 2 + f] == 1.0  # is_back_edge
    assert not obs[base + slot : base + K * slot].any()  # remaining slots are empty


def _permuted(graph: GraphRecord, perm: list[int]) -> GraphRecord:
    """Relabel node ``i`` as ``perm[i]`` without changing lines or structure."""
    nodes: list[Node | None] = [None] * graph.num_nodes
    for node in graph.nodes:
        nodes[perm[node.id]] = Node(**{**node.__dict__, "id": perm[node.id]})
    return GraphRecord(
        sample_id=graph.sample_id,
        nodes=nodes,
        edges=[(perm[u], perm[v]) for u, v in graph.edges],
        back_edges=[(perm[u], perm[v]) for u, v in graph.back_edges],
        entry=perm[graph.entry],
        exit=perm[graph.exit],
        vuln_nodes=[perm[v] for v in graph.vuln_nodes],
        source_lines=graph.source_lines,
    ).validate()


def test_observations_do_not_depend_on_node_ids():
    rng = np.random.default_rng(0)
    for graph in generate_dataset(15, seed=3):
        perm = [int(i) for i in rng.permutation(graph.num_nodes)]
        env_a, env_b = make_env([graph]), make_env([_permuted(graph, perm)])
        obs_a, _ = start(env_a)
        obs_b, _ = start(env_b)
        np.testing.assert_array_equal(obs_a, obs_b)
        for step in itertools.count():
            valid = env_a.valid_actions()
            assert valid == env_b.valid_actions()
            action = valid[step % len(valid)]
            obs_a, _, term_a, trunc_a, _ = env_a.step(action)
            obs_b, _, term_b, trunc_b, _ = env_b.step(action)
            np.testing.assert_array_equal(obs_a, obs_b)
            assert (term_a, trunc_a) == (term_b, trunc_b)
            if term_a or trunc_a:
                break


def test_rejects_graphs_wider_than_k():
    kinds = ["ENTRY", "SWITCH", *["ASSIGN"] * 3, "EXIT"]
    edges = [(0, 1), (1, 2), (1, 3), (1, 4), (2, 5), (3, 5), (4, 5)]
    wide = make_graph(kinds, edges, [2])
    with pytest.raises(ValueError, match="exceeds k=2"):
        make_env([wide], k=2)
    with pytest.raises(ValueError, match="at least one graph"):
        make_env([])


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError, match="reset"):
        make_env([line_graph()]).step(0)


def test_default_config_file_matches_the_dataclass_defaults():
    from pathlib import Path

    from soren.config import load_config

    path = Path(__file__).parent.parent / "configs" / "env.yaml"
    assert load_config(EnvConfig, path, section="env") == EnvConfig()
    assert load_config(RewardConfig, path, section="reward") == RewardConfig()
