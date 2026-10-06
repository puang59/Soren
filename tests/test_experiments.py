import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from helpers import diamond_graph

from soren.data.features import TIER_L_NAMES, feature_dim
from soren.data.schema import write_jsonl
from soren.data.synthetic import SyntheticConfig, generate_dataset
from soren.data.transform import permute_node_ids, shuffle_labels
from soren.env.cfg_nav_env import CFGNavEnv, EnvConfig
from soren.env.policies import RandomPolicy, rollout

ROOT = Path(__file__).parent.parent
GRAPHS = generate_dataset(40, seed=3, cfg=SyntheticConfig(min_nodes=8, max_nodes=25))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_masked_features_are_zeroed_everywhere_in_the_observation():
    column = TIER_L_NAMES.index("rel_line")
    f = feature_dim("L")
    plain = CFGNavEnv([GRAPHS[0]], EnvConfig())
    masked = CFGNavEnv([GRAPHS[0]], EnvConfig(masked_features=["rel_line"]))
    a, _ = plain.reset(seed=0)
    b, _ = masked.reset(seed=0)
    a, *_ = plain.step(0)
    b, *_ = masked.step(0)
    assert a[column] > 0 and b[column] == 0.0
    slot = f + 2 + 1 + column  # the same feature of the first successor
    assert a[slot] > 0 and b[slot] == 0.0
    differing = np.flatnonzero(a != b)
    assert set(differing) <= {
        column + offset for offset in [0, *(f + 3 + k * (f + 3) for k in range(6))]
    }
    with pytest.raises(ValueError, match="cannot mask unknown features"):
        EnvConfig(masked_features=["nope"])


def test_shuffle_labels_keeps_structure_and_count():
    rng = np.random.default_rng(0)
    moved = 0
    for graph in GRAPHS:
        shuffled = shuffle_labels(graph, rng)
        assert shuffled.edges == graph.edges and shuffled.nodes == graph.nodes
        assert len(shuffled.vuln_nodes) == len(graph.vuln_nodes)
        assert all(shuffled.nodes[v].kind not in ("ENTRY", "EXIT") for v in shuffled.vuln_nodes)
        moved += shuffled.vuln_nodes != graph.vuln_nodes
    assert moved > len(GRAPHS) * 0.7
    assert (
        GRAPHS[0].vuln_nodes
        == generate_dataset(1, seed=3, cfg=SyntheticConfig(min_nodes=8, max_nodes=25))[0].vuln_nodes
    )


def test_permuting_node_ids_does_not_change_what_a_policy_sees():
    rng = np.random.default_rng(1)
    for graph in GRAPHS[:15]:
        permuted = permute_node_ids(graph, rng)
        assert sorted(n.line for n in permuted.nodes) == sorted(n.line for n in graph.nodes)
        env_a, env_b = CFGNavEnv([graph]), CFGNavEnv([permuted])
        obs_a, _ = env_a.reset(seed=0)
        obs_b, _ = env_b.reset(seed=0)
        np.testing.assert_array_equal(obs_a, obs_b)
        a = rollout(env_a, RandomPolicy(), seed=5)
        b = rollout(env_b, RandomPolicy(), seed=5)
        assert a.actions == b.actions and a.rewards == b.rewards and a.success == b.success
    assert permute_node_ids(diamond_graph(), np.random.default_rng(0)).num_nodes == 6


def test_run_arm_and_collect(tmp_path, capsys):
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    write_jsonl(GRAPHS[:30], train)
    write_jsonl(GRAPHS[30:], val)
    ppo = tmp_path / "ppo.yaml"
    ppo.write_text(
        "total_timesteps: 256\nn_envs: 2\nn_steps: 64\nbatch_size: 64\nn_epochs: 1\n"
        "net_arch: [16]\nvec_env: dummy\neval_freq: 256\n"
    )
    run_arm = _load("run_arm")
    common = ["--group", "g", "--train", str(train), "--val", str(val), "--out-dir", str(tmp_path),
              "--config", str(ppo), "--env-config", str(ROOT / "configs" / "env.yaml")]  # fmt: skip
    run_arm.main([*common, "--name", "base", "--seed", "0"])
    run_arm.main([*common, "--name", "base", "--seed", "1"])
    run_arm.main(
        [*common, "--name", "three", "--seed", "0", "--set", "env.max_declares=3",
         "--set", "reward.wrong=-1.0", "--set", "ppo.ent_coef=0.03", "--shuffle-labels"]
    )  # fmt: skip
    config = json.loads((tmp_path / "g" / "three" / "seed0" / "config.json").read_text())
    assert config["env"]["max_declares"] == 3
    assert config["reward"]["wrong"] == -1.0 and config["ppo"]["ent_coef"] == 0.03
    metrics = json.loads((tmp_path / "g" / "base" / "seed1" / "metrics.json").read_text())
    assert metrics["arm"] == "base" and metrics["seed"] == 1 and 0 <= metrics["success_rate"] <= 1

    table = _load("collect_runs").collect(tmp_path / "g").set_index("arm")
    assert table.loc["base", "seeds"] == 2 and table.loc["three", "seeds"] == 1
    with pytest.raises(SystemExit, match="bad override"):
        run_arm.main([*common, "--name", "x", "--set", "nonsense"])
    with pytest.raises(SystemExit, match="has no setting"):
        run_arm.main([*common, "--name", "x", "--set", "env.nope=1"])
    capsys.readouterr()
