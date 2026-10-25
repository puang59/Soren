"""A small neural network library written with NumPy only.

It has what Soren needs and nothing more: fully connected layers, tanh and ReLU, backpropagation
by hand, the Adam optimiser, gradient clipping, and a categorical distribution with an action
mask. ``tests/test_nn.py`` checks every hand-written gradient against a numerical one.
"""

from soren.nn.distributions import entropy, log_softmax, masked_logits, sample
from soren.nn.layers import MLP, Linear, ReLU, Tanh, orthogonal
from soren.nn.optim import Adam, clip_grad_norm

__all__ = [
    "MLP",
    "Adam",
    "Linear",
    "ReLU",
    "Tanh",
    "clip_grad_norm",
    "entropy",
    "log_softmax",
    "masked_logits",
    "orthogonal",
    "sample",
]
