"""Layers with a forward pass and a hand-written backward pass.

Every layer remembers what it needs during ``forward``. Given ``g = dL/d(output)``,
``backward`` stores the gradients of the layer's own parameters and returns ``dL/d(input)``,
which becomes ``g`` for the layer before it. Running that from the last layer to the first is
backpropagation: the chain rule, one layer at a time.

    Linear   y = x·W + b         dL/dW = xᵀ·g    dL/db = Σ_rows g    dL/dx = g·Wᵀ
    Tanh     y = tanh(x)         dL/dx = g ⊙ (1 − y²)
    ReLU     y = max(x, 0)       dL/dx = g ⊙ [x > 0]

Inputs are batches: ``x`` has one row per example.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def orthogonal(shape: tuple[int, int], gain: float, rng: np.random.Generator) -> np.ndarray:
    """A random matrix with orthonormal rows or columns, times ``gain``.

    Orthogonal weights keep the scale of a signal steady from layer to layer at the start of
    training (Saxe et al., 2014). It is the initialisation PPO implementations commonly use.
    Built from the QR decomposition of a Gaussian matrix.
    """
    rows, cols = shape
    a = rng.standard_normal((max(rows, cols), min(rows, cols)))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))  # makes the result uniformly distributed over orthogonal matrices
    if rows < cols:
        q = q.T
    return gain * q[:rows, :cols]


class Linear:
    """Fully connected layer ``y = x·W + b``."""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, gain: float = 1.0):
        self.W = orthogonal((n_in, n_out), gain, rng)
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    @property
    def params(self) -> list[np.ndarray]:
        return [self.W, self.b]

    @property
    def grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, g: np.ndarray) -> np.ndarray:
        assert self._x is not None, "forward before backward"
        self.dW[...] = self._x.T @ g
        self.db[...] = g.sum(axis=0)
        return g @ self.W.T


class Tanh:
    params: list[np.ndarray] = []
    grads: list[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._y = np.tanh(x)
        return self._y

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g * (1.0 - self._y**2)


class ReLU:
    params: list[np.ndarray] = []
    grads: list[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._positive = x > 0
        return np.where(self._positive, x, 0.0)

    def backward(self, g: np.ndarray) -> np.ndarray:
        return g * self._positive


_ACTIVATIONS = {"tanh": Tanh, "relu": ReLU}


class MLP:
    """A multilayer perceptron: ``Linear → activation → … → Linear``.

    ``sizes`` lists the width of every layer, input first: ``[248, 256, 256, 8]`` is
    248 inputs, two hidden layers of 256 and 8 outputs. Hidden layers are initialised with gain
    √2 and the output layer with ``output_gain``.
    """

    def __init__(
        self,
        sizes: Sequence[int],
        activation: str = "tanh",
        rng: np.random.Generator | None = None,
        output_gain: float = 1.0,
    ) -> None:
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}")
        if len(sizes) < 2:
            raise ValueError("an MLP needs at least an input and an output size")
        rng = rng or np.random.default_rng()
        self.sizes = list(sizes)
        self.activation = activation
        self.layers: list[Linear | Tanh | ReLU] = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            last = i == len(sizes) - 2
            self.layers.append(Linear(n_in, n_out, rng, output_gain if last else np.sqrt(2)))
            if not last:
                self.layers.append(_ACTIVATIONS[activation]())
        self.activations: list[np.ndarray] = []
        """Output of every layer from the last forward pass, for explanations."""

    @property
    def linears(self) -> list[Linear]:
        return [layer for layer in self.layers if isinstance(layer, Linear)]

    @property
    def params(self) -> list[np.ndarray]:
        return [p for layer in self.layers for p in layer.params]

    @property
    def grads(self) -> list[np.ndarray]:
        return [g for layer in self.layers for g in layer.grads]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.activations = []
        for layer in self.layers:
            x = layer.forward(x)
            self.activations.append(x)
        return x

    def backward(self, g: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g

    def num_params(self) -> int:
        return sum(p.size for p in self.params)
