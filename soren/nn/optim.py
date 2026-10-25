"""The Adam optimiser and gradient clipping.

Adam (Kingma and Ba, 2015) keeps two running averages per parameter: ``m`` of the gradient
and ``v`` of its square. Dividing by ``√v`` gives every parameter its own step size, large
where gradients are small and steady, small where they are large or noisy.

    m ← β₁·m + (1 − β₁)·g              running mean of the gradient
    v ← β₂·v + (1 − β₂)·g²             running mean of its square
    m̂ = m / (1 − β₁ᵗ),  v̂ = v / (1 − β₂ᵗ)   bias correction: m and v start at zero
    θ ← θ − lr · m̂ / (√v̂ + ε)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class Adam:
    """Updates ``params`` in place from the matching arrays in ``grads``.

    ``weight_decay`` adds ``weight_decay · θ`` to each gradient first (L2 regularisation).
    """

    def __init__(
        self,
        params: Sequence[np.ndarray],
        grads: Sequence[np.ndarray],
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if len(params) != len(grads):
            raise ValueError("params and grads must pair up")
        self.params = list(params)
        self.grads = list(grads)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, lr: float | None = None) -> None:
        lr = self.lr if lr is None else lr
        self.t += 1
        correction1 = 1.0 - self.beta1**self.t
        correction2 = 1.0 - self.beta2**self.t
        for p, g, m, v in zip(self.params, self.grads, self.m, self.v, strict=True):
            if self.weight_decay:
                g = g + self.weight_decay * p
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            p -= lr * (m / correction1) / (np.sqrt(v / correction2) + self.eps)


def global_norm(grads: Sequence[np.ndarray]) -> float:
    """``‖g‖ = √(Σ g²)`` over every gradient array together."""
    return float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))


def clip_grad_norm(grads: Sequence[np.ndarray], max_norm: float) -> tuple[float, float]:
    """Scale all gradients down together if their global norm exceeds ``max_norm``.

    Returns ``(norm before clipping, scale applied)``. Clipping bounds the size of one update,
    so a single unusual minibatch cannot throw the policy far.
    """
    norm = global_norm(grads)
    scale = max_norm / (norm + 1e-6)
    if scale < 1.0:
        for g in grads:
            g *= scale
        return norm, scale
    return norm, 1.0
