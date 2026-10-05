import numpy as np
import pytest
from helpers import diamond_graph, line_graph, loop_graph, make_graph

from soren.baselines import BFS, DFS, EpisodeResult, LineOrder, RandomOrder, RandomWalk, Searcher
from soren.data.synthetic import SyntheticConfig, generate_dataset
from soren.env.cfg_nav_env import EnvConfig
from soren.env.rewards import RewardConfig

R = RewardConfig()
GRAPHS = generate_dataset(120, seed=5, cfg=SyntheticConfig(min_nodes=8, max_nodes=50))


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def tree_graph(vuln: int):
    """
    0 ENTRY -> 1 BRANCH -> 2 BRANCH -> 4 -> 7 EXIT
                        \\          \\-> 5 -> 7
                         -> 3 ------> 6 -> 7
    """
    kinds = ["ENTRY", "BRANCH", "BRANCH", "ASSIGN", "ASSIGN", "CALL", "ASSIGN", "EXIT"]
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
    return make_graph(kinds, edges, [vuln], sample_id="tree")


def test_searchers_satisfy_the_protocol():
    for searcher in (BFS(), DFS(), RandomWalk(), RandomOrder(), LineOrder()):
        assert isinstance(searcher.name, str)
        assert isinstance(searcher.stochastic, bool)
        run: Searcher = searcher
        assert isinstance(run.run(line_graph(), 50, rng()), EpisodeResult)


def test_bfs_visit_order():
    result = BFS().run(tree_graph(vuln=6), 100, rng())
    assert result.visit_order == [0, 1, 2, 3, 4, 5, 6]
    assert result.success and result.declared_node == 6
    assert result.nodes_inspected == 7
    assert result.first_hit_step == 6
    assert result.actions_taken == 7  # six inspections plus the declaration
    assert result.cumulative_reward == pytest.approx(6 * R.step + R.correct)


def test_dfs_visit_order_and_backtracking():
    result = DFS().run(tree_graph(vuln=6), 100, rng())
    # Down the first branch to EXIT, back up to the second leaf, then over to the other side.
    assert result.visit_order == [0, 1, 2, 4, 7, 5, 3, 6]
    assert result.success and result.declared_node == 6
    assert result.nodes_inspected == 8
    # Moves 0-1-2-4-7, back to 4 and 2, move to 5, back to 2 and 1, moves to 3 and 6: 11
    # actions, plus the declaration. Moving 5 -> 7 is skipped because EXIT is already visited.
    assert result.actions_taken == 12
    assert result.cumulative_reward == pytest.approx(11 * R.step + R.correct)


def test_dfs_and_bfs_differ_on_deep_versus_shallow_targets():
    deep, shallow = tree_graph(vuln=4), tree_graph(vuln=3)
    assert DFS().run(deep, 100, rng()).nodes_inspected == 4
    assert BFS().run(deep, 100, rng()).nodes_inspected == 5
    assert DFS().run(shallow, 100, rng()).nodes_inspected == 7
    assert BFS().run(shallow, 100, rng()).nodes_inspected == 4


def test_dfs_handles_loops():
    result = DFS().run(loop_graph(vuln=3), 100, rng())
    assert result.visit_order == [0, 1, 2, 3]
    assert result.success


def test_oracle_stop_declares_on_arrival():
    for searcher in (BFS(), DFS(), RandomWalk()):
        result = searcher.run(diamond_graph(vuln=3), 200, rng(1))
        assert result.success
        assert result.declared_node == 3
        assert result.visit_order[-1] == 3
        assert result.end_reason == "correct"


def test_all_baselines_solve_synthetic_graphs_given_enough_budget():
    for graph in GRAPHS:
        budget = 20 * graph.num_nodes
        for searcher in (BFS(), DFS()):
            result = searcher.run(graph, budget, rng())
            assert result.success, (searcher.name, graph.sample_id)
            assert result.visit_order[0] == graph.entry
            assert len(set(result.visit_order)) == len(result.visit_order)
            assert result.nodes_inspected <= graph.num_nodes
            assert result.actions_taken >= result.nodes_inspected - 1


def test_dfs_never_needs_more_than_two_actions_per_node():
    for graph in GRAPHS:
        result = DFS().run(graph, 20 * graph.num_nodes, rng())
        assert result.actions_taken <= 2 * graph.num_nodes


def test_budget_exhaustion_is_a_failure_with_the_timeout_penalty():
    graph = line_graph(6, vuln=6)
    for searcher in (BFS(), DFS(), RandomWalk()):
        result = searcher.run(graph, 3, rng())
        assert not result.success
        assert result.end_reason == "timeout"
        assert result.declared_node is None
        assert result.actions_taken == 3
        assert result.cumulative_reward <= 3 * R.step + R.timeout + 1e-9


def test_random_walk_is_reproducible_and_seed_dependent():
    graph = GRAPHS[0]
    walker = RandomWalk()
    a = walker.run(graph, 200, rng(3))
    b = walker.run(graph, 200, rng(3))
    assert a == b
    orders = {tuple(walker.run(graph, 200, rng(s)).visit_order) for s in range(20)}
    assert len(orders) > 1


def test_random_walk_usually_needs_more_actions_than_dfs():
    walk, dfs = [], []
    for i, graph in enumerate(GRAPHS):
        budget = EnvConfig().max_steps_for(graph.num_nodes)
        walk.append(RandomWalk().run(graph, budget, rng(i)))
        dfs.append(DFS().run(graph, budget, rng(i)))
    assert np.mean([r.success for r in dfs]) > np.mean([r.success for r in walk])
    assert all(r.end_reason in {"correct", "timeout"} for r in walk)


def test_custom_reward_config_is_used():
    reward = RewardConfig(step=-1.0, correct=10.0)
    result = DFS(reward_config=reward).run(line_graph(3, vuln=2), 50, rng())
    assert result.cumulative_reward == pytest.approx(2 * -1.0 + 10.0)
    result = BFS(reward_config=reward).run(line_graph(3, vuln=2), 50, rng())
    assert result.cumulative_reward == pytest.approx(2 * -1.0 + 10.0)


def test_line_order_reads_top_to_bottom():
    # Lines are out of step with node ids: node 3 sits above node 2.
    kinds = ["ENTRY", "BRANCH", "ASSIGN", "CALL", "ASSIGN", "EXIT"]
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]
    graph = make_graph(kinds, edges, [4], lines=[1, 2, 4, 3, 5, 6])
    result = LineOrder().run(graph, 100, rng())
    assert result.visit_order == [0, 1, 3, 2, 4]
    assert result.success and result.declared_node == 4
    assert result.nodes_inspected == 5
    assert result.actions_taken == 5  # four further inspections plus the declaration
    assert result.cumulative_reward == pytest.approx(4 * R.step + R.correct)


def test_random_order_matches_the_analytic_expectation():
    """One target among n nodes is found after (n + 1) / 2 inspections on average."""
    graph = line_graph(18, vuln=7)  # 20 nodes
    generator = rng(0)
    inspected = [RandomOrder().run(graph, 1000, generator).nodes_inspected for _ in range(4000)]
    assert np.mean(inspected) == pytest.approx((graph.num_nodes + 1) / 2, abs=0.3)
    assert min(inspected) == 1 and max(inspected) == graph.num_nodes


def test_random_order_is_a_seeded_permutation():
    graph = GRAPHS[0]
    a = RandomOrder().run(graph, 1000, rng(5))
    b = RandomOrder().run(graph, 1000, rng(5))
    assert a == b and a.success
    assert len(set(a.visit_order)) == len(a.visit_order)
    orders = {tuple(RandomOrder().run(graph, 1000, rng(s)).visit_order) for s in range(10)}
    assert len(orders) > 1


def test_order_searchers_respect_the_budget():
    graph = line_graph(10, vuln=10)
    for searcher in (LineOrder(), BFS()):
        result = searcher.run(graph, 4, rng())
        assert not result.success and result.end_reason == "timeout"
        assert result.actions_taken == 4
        assert result.cumulative_reward == pytest.approx(4 * R.step + R.timeout)


def test_only_seeded_methods_are_marked_stochastic():
    flags = {s.name: s.stochastic for s in (BFS(), DFS(), LineOrder(), RandomOrder(), RandomWalk())}
    assert flags == {
        "bfs": False,
        "dfs": False,
        "line_order": False,
        "random_order": True,
        "random_walk": True,
    }
