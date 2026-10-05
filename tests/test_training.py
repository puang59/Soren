import json
from pathlib import Path

import numpy as np
import pytest

from soren.agents.ppo import PPOConfig, learning_rate_schedule
from soren.agents.searcher import PolicySearcher, evaluate_policy
from soren.agents.train import load_model, train
from soren.config import load_config
from soren.data.synthetic import SyntheticConfig, generate_dataset
from soren.env.cfg_nav_env import CFGNavEnv, EnvConfig
from soren.env.policies import RandomPolicy, rollout
from soren.env.rewards import RewardConfig
from soren.eval.metrics import summarize

EASY = SyntheticConfig(min_nodes=8, max_nodes=16, hit_rate=1.0, decoy_rate=0.0)
TINY = PPOConfig(
    total_timesteps=512,
    n_envs=2,
    n_steps=64,
    batch_size=64,
    n_epochs=2,
    net_arch=[32, 32],
    vec_env="dummy",
    eval_freq=256,
)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    graphs = generate_dataset(24, seed=0, cfg=EASY)
    run_dir = tmp_path_factory.mktemp("runs") / "tiny"
    return train(graphs[:16], graphs[16:], run_dir, TINY, seed=3), graphs


def test_train_writes_run_artifacts(tiny_run):
    result, _ = tiny_run
    for name in ("config.json", "best_model.zip", "final_model.zip", "val_history.json"):
        assert (result.run_dir / name).exists(), name
    assert any((result.run_dir / "tb").rglob("events.out.tfevents.*"))

    config = json.loads((result.run_dir / "config.json").read_text())
    assert config["seed"] == 3
    assert config["train_graphs"] == 16 and config["val_graphs"] == 8
    assert config["ppo"]["n_steps"] == 64
    assert config["env"]["feature_tier"] == "L"


def test_validation_history_and_best_checkpoint(tiny_run):
    result, _ = tiny_run
    assert [h["timesteps"] for h in result.history] == [256, 512]
    for entry in result.history:
        assert entry["episodes"] == 8
        assert 0.0 <= entry["success_rate"] <= 1.0
    best = max(result.history, key=lambda h: (h["success_rate"], -h["nodes_inspected"]))
    assert result.best_metrics == best
    assert json.loads((result.run_dir / "val_history.json").read_text()) == result.history


def test_checkpoint_round_trip_and_policy_searcher(tiny_run):
    result, graphs = tiny_run
    model = load_model(result.final_model_path)
    val = graphs[16:]
    direct = evaluate_policy(result.model, val)
    reloaded = evaluate_policy(model, val)
    assert [r.visit_order for r in direct] == [r.visit_order for r in reloaded]

    searcher = PolicySearcher(model, name="agent")
    single = searcher.run(val[0], max_steps=5, rng=np.random.default_rng(0))
    assert single.method == "agent" and single.graph_id == val[0].sample_id
    assert single.actions_taken <= 5
    assert single.nodes_inspected == len(single.visit_order)
    assert set(summarize(direct)) >= {"success_rate", "nodes_inspected", "cumulative_reward"}


def test_training_is_reproducible(tmp_path):
    graphs = generate_dataset(12, seed=1, cfg=EASY)
    a = train(graphs[:8], graphs[8:], tmp_path / "a", TINY, seed=7, tensorboard=False)
    b = train(graphs[:8], graphs[8:], tmp_path / "b", TINY, seed=7, tensorboard=False)
    assert a.history == b.history
    orders_a = [r.visit_order for r in evaluate_policy(a.model, graphs)]
    orders_b = [r.visit_order for r in evaluate_policy(b.model, graphs)]
    assert orders_a == orders_b


def test_reward_gamma_follows_the_learner(tmp_path):
    graphs = generate_dataset(6, seed=2, cfg=EASY)
    cfg = PPOConfig(**{**TINY.__dict__, "gamma": 0.9, "total_timesteps": 128, "eval_freq": 128})
    result = train(
        graphs, graphs, tmp_path / "g", cfg, reward_config=RewardConfig(shaping=True, gamma=0.5)
    )
    config = json.loads((result.run_dir / "config.json").read_text())
    assert config["reward"]["gamma"] == 0.9 and config["reward"]["shaping"] is True


def test_ppo_config():
    path = Path(__file__).parent.parent / "configs" / "ppo.yaml"
    assert load_config(PPOConfig, path) == PPOConfig()
    schedule = learning_rate_schedule(PPOConfig())
    assert schedule(1.0) == pytest.approx(3e-4) and schedule(0.0) == 0.0
    assert learning_rate_schedule(PPOConfig(lr_schedule="constant")) == 3e-4
    with pytest.raises(ValueError):
        PPOConfig(activation="gelu")
    with pytest.raises(ValueError):
        PPOConfig(vec_env="threads")


def test_summarize():
    graphs = generate_dataset(5, seed=0, cfg=EASY)

    class AlwaysFirstValid:
        def predict(self, obs, action_masks, deterministic=True):
            return int(np.flatnonzero(action_masks)[0]), None

    results = evaluate_policy(AlwaysFirstValid(), graphs)
    summary = summarize(results)
    assert summary["episodes"] == 5
    assert sum(v for k, v in summary.items() if k.startswith("end_")) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        summarize([])


@pytest.mark.slow
def test_smoke_training_beats_the_random_policy(tmp_path):
    """About 20k timesteps on 50 synthetic graphs is enough to beat acting at random."""
    graphs = generate_dataset(50, seed=0, cfg=EASY)
    cfg = PPOConfig(
        total_timesteps=20_480,
        n_envs=8,
        n_steps=256,
        batch_size=256,
        net_arch=[64, 64],
        vec_env="dummy",
        eval_freq=10_240,
    )
    result = train(graphs, graphs, tmp_path / "smoke", cfg, seed=0, tensorboard=False)
    trained = summarize(evaluate_policy(load_model(result.best_model_path), graphs))

    env = CFGNavEnv(graphs, EnvConfig())
    random_success = np.mean(
        [
            rollout(env, RandomPolicy(), seed=s, options={"graph_index": i}).success
            for i in range(len(graphs))
            for s in range(10)
        ]
    )
    assert trained["success_rate"] > random_success + 0.2
