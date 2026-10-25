"""The hand-written gradients of soren.nn and of the PPO loss, checked numerically.

A gradient check perturbs one parameter by ±h and compares (L(θ+h) − L(θ−h)) / 2h with the
analytic gradient that backpropagation produced.
"""

import numpy as np
import pytest

from soren.agents.buffer import compute_gae
from soren.agents.policy import ActorCritic
from soren.agents.ppo import ppo_loss
from soren.agents.vec_env import RunningMeanStd
from soren.baselines.node_classifier import weighted_bce
from soren.nn import MLP, Adam, clip_grad_norm, entropy, log_softmax, masked_logits, sample


def numerical_gradient(loss, param, index, h=1e-6):
    old = param[index]
    param[index] = old + h
    up = loss()
    param[index] = old - h
    down = loss()
    param[index] = old
    return (up - down) / (2 * h)


def check(loss, params, grads, rng, samples=6):
    for param, grad in zip(params, grads, strict=True):
        for _ in range(samples):
            index = tuple(rng.integers(s) for s in param.shape)
            assert grad[index] == pytest.approx(
                numerical_gradient(loss, param, index), rel=1e-4, abs=1e-7
            )


@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_mlp_backward_matches_numerical_gradient(activation):
    rng = np.random.default_rng(0)
    net = MLP([5, 7, 6, 3], activation, rng)
    x, target = rng.normal(size=(4, 5)), rng.normal(size=(4, 3))

    def loss():
        return float(0.5 * np.sum((net.forward(x) - target) ** 2))

    loss()
    grad_x = net.backward(net.forward(x) - target)
    check(loss, net.params, net.grads, rng)
    # The gradient with respect to the input as well.
    for index in [(0, 0), (3, 4)]:
        assert grad_x[index] == pytest.approx(numerical_gradient(loss, x, index), rel=1e-4)


def test_orthogonal_initialisation():
    net = MLP([8, 8, 2], "tanh", np.random.default_rng(1))
    w = net.linears[0].W
    np.testing.assert_allclose(w.T @ w, 2.0 * np.eye(8), atol=1e-10)  # gain √2, squared


def test_masked_softmax_and_entropy():
    z = np.array([[1.0, 2.0, 3.0, 4.0]])
    mask = np.array([[True, False, True, False]])
    log_p = log_softmax(masked_logits(z, mask))
    p = np.exp(log_p)
    assert p[0, 1] == 0.0 and p[0, 3] == 0.0
    assert p[0, 2] / p[0, 0] == pytest.approx(np.e**2)
    assert entropy(log_p)[0] == pytest.approx(-(p[0, 0] * log_p[0, 0] + p[0, 2] * log_p[0, 2]))


def test_sampling_follows_the_probabilities_and_never_picks_masked_actions():
    probs = np.tile([0.0, 0.2, 0.0, 0.8, 0.0], (20_000, 1))
    actions = sample(probs, np.random.default_rng(0))
    assert set(np.unique(actions)) == {1, 3}
    assert np.mean(actions == 3) == pytest.approx(0.8, abs=0.01)


def test_adam_matches_the_formula():
    p, g = np.array([1.0]), np.array([0.5])
    opt = Adam([p], [g], lr=0.1)
    opt.step()
    # After one step m̂ = g and v̂ = g², so the update is lr · g / (|g| + ε) ≈ lr.
    assert p[0] == pytest.approx(1.0 - 0.1 * 0.5 / (0.5 + 1e-8))
    before = p[0]
    opt.step()
    m = 0.9 * 0.05 + 0.1 * 0.5
    v = 0.999 * 0.00025 + 0.001 * 0.25
    assert opt.m[0][0] == pytest.approx(m) and opt.v[0][0] == pytest.approx(v)
    step = 0.1 * (m / (1 - 0.9**2)) / (np.sqrt(v / (1 - 0.999**2)) + 1e-8)
    assert p[0] == pytest.approx(before - step)


def test_clip_grad_norm():
    grads = [np.array([3.0]), np.array([4.0])]
    norm, scale = clip_grad_norm(grads, 1.0)
    assert norm == pytest.approx(5.0) and scale == pytest.approx(0.2, rel=1e-5)
    assert np.sqrt(grads[0][0] ** 2 + grads[1][0] ** 2) == pytest.approx(1.0, rel=1e-5)
    assert clip_grad_norm([np.array([0.1])], 1.0) == (pytest.approx(0.1), 1.0)


def test_ppo_loss_gradient_matches_numerical_gradient():
    rng = np.random.default_rng(3)
    policy = ActorCritic(obs_dim=6, n_actions=5, net_arch=(8, 8), seed=0)
    for p in policy.actor.params:  # larger output weights than the default, for a real signal
        p += rng.normal(scale=0.3, size=p.shape)
    size = 16
    obs = rng.random((size, 6))
    masks = rng.random((size, 5)) < 0.7
    masks[:, 0] = True
    log_p = policy.log_probs(obs, masks)
    actions = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
    batch = {
        "obs": obs,
        "masks": masks,
        "actions": actions,
        # Near the current policy, so no ratio sits exactly on a clipping boundary.
        "old_log_probs": log_p[np.arange(size), actions] + rng.normal(scale=0.05, size=size),
        "advantages": rng.normal(size=size),
        "returns": rng.normal(size=size),
    }

    def loss():
        return ppo_loss(policy, batch, 0.2, 0.01, 0.5)["loss"]

    loss()
    check(loss, policy.params, [g.copy() for g in policy.grads], rng, samples=4)


def test_gae_by_hand():
    rewards = np.array([[0.0], [0.0], [1.0]])
    values = np.array([[0.5], [0.6], [0.7]])
    dones = np.array([[False], [False], [True]])
    adv, returns = compute_gae(rewards, values, dones, np.array([9.9]), gamma=0.9, lam=0.5)
    d2 = 1.0 - 0.7  # done: the next value does not count
    d1 = 0.0 + 0.9 * 0.7 - 0.6
    d0 = 0.0 + 0.9 * 0.6 - 0.5
    a2 = d2
    a1 = d1 + 0.9 * 0.5 * a2
    a0 = d0 + 0.9 * 0.5 * a1
    np.testing.assert_allclose(adv[:, 0], [a0, a1, a2])
    np.testing.assert_allclose(returns, adv + values)


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, size=(50, 7))
    rms = RunningMeanStd()
    for row in data:
        rms.update(row)
    assert rms.mean == pytest.approx(data.mean(), rel=1e-4)
    assert rms.var == pytest.approx(data.var(), rel=1e-3)


def test_weighted_bce_gradient():
    rng = np.random.default_rng(0)
    logits, labels = rng.normal(size=10), (rng.random(10) < 0.3).astype(float)
    _, grad = weighted_bce(logits, labels, 4.0)
    for i in range(10):
        assert grad[i] == pytest.approx(
            numerical_gradient(lambda: weighted_bce(logits, labels, 4.0)[0], logits, i),
            rel=1e-5,
        )


def test_policy_save_and_load_round_trip(tmp_path):
    policy = ActorCritic(obs_dim=4, n_actions=3, net_arch=(5,), seed=2)
    path = policy.save(tmp_path / "p.npz", num_timesteps=7)
    loaded = ActorCritic.load(path)
    obs, mask = np.random.default_rng(0).random(4), np.array([True, False, True])
    np.testing.assert_allclose(loaded.action_probs(obs, mask), policy.action_probs(obs, mask))
    assert loaded.value(obs) == pytest.approx(policy.value(obs))
