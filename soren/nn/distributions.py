"""A categorical distribution over actions, with invalid actions masked out.

The policy network outputs one number per action, a *logit* ``z``. Softmax turns logits into
probabilities. Actions that are impossible in the current state (moving to a successor that
does not exist, backtracking with an empty stack) get the logit −10⁸ first, so their
probability is exactly zero and the policy never picks them.

    p_i = exp(z_i) / Σ_j exp(z_j)                      softmax
    log p_i = z_i − max(z) − log Σ_j exp(z_j − max(z))  the same, computed without overflow
    H = −Σ_i p_i · log p_i                             entropy: how spread out the choice is
"""

from __future__ import annotations

import numpy as np

MASKED_LOGIT = -1e8


def masked_logits(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace the logits of invalid actions (``mask`` False) with −10⁸."""
    return np.where(mask, logits, MASKED_LOGIT)


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Log-probabilities over the last axis. Subtracting the max avoids ``exp`` overflowing."""
    shifted = z - z.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def entropy(log_p: np.ndarray) -> np.ndarray:
    """``H = −Σ p·log p`` per row. Masked actions have ``p = 0`` and add nothing."""
    p = np.exp(log_p)
    return -(p * np.where(p > 0, log_p, 0.0)).sum(axis=-1)


def sample(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One action per row, by inverse transform sampling.

    Draw ``u`` uniform in ``[0, 1)`` and pick the first action whose cumulative probability
    exceeds it: action ``i`` is chosen with probability ``p_i``.
    """
    cumulative = np.cumsum(probs, axis=-1)
    cumulative /= cumulative[..., -1:]  # exactly 1 at the end, despite rounding
    u = rng.random((*probs.shape[:-1], 1))
    return (cumulative <= u).sum(axis=-1)
