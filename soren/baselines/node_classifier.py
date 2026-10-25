"""Supervised node classifier: a reference that ranks statements without navigating.

It answers "what if you just classify every node?". A small MLP on the per-node features
predicts whether a statement is vulnerable; at test time all statements of a function are
ranked by that probability. It sees the whole function at once, so it measures how much the
features can say about which statement is vulnerable. It is not a traversal method.

The network, its loss and its training loop are written with ``soren.nn``. The loss is binary
cross-entropy on the logit ``x`` with label ``y``, weighting the rare positives by ``w``:

    ℓ = −[ w·y·log σ(x) + (1 − y)·log(1 − σ(x)) ]        σ(x) = 1 / (1 + e⁻ˣ)
    ∂ℓ/∂x = w·y·(σ(x) − 1) + (1 − y)·σ(x)
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from soren.data.features import feature_dim, featurize
from soren.data.schema import GraphRecord
from soren.nn import MLP, Adam


def _statements(graph: GraphRecord) -> np.ndarray:
    """Ids of the nodes that can be declared: everything but ENTRY and EXIT."""
    return np.array([n.id for n in graph.nodes if n.kind not in ("ENTRY", "EXIT")])


def top_k_accuracy(
    scores: Sequence[np.ndarray], graphs: Sequence[GraphRecord], ks: Sequence[int] = (1, 3, 5)
) -> dict[str, float]:
    """Share of graphs with a vulnerable statement among the ``k`` highest-scoring ones."""
    hits = {k: 0 for k in ks}
    for graph_scores, graph in zip(scores, graphs, strict=True):
        statements = _statements(graph)
        ranked = statements[np.argsort(-graph_scores[statements], kind="stable")]
        for k in ks:
            hits[k] += any(node in graph.vuln_set for node in ranked[:k])
    return {f"top{k}": hits[k] / len(graphs) for k in ks}


def chance_top_k(graphs: Sequence[GraphRecord], ks: Sequence[int] = (1, 3, 5)) -> dict[str, float]:
    """Expected top-k accuracy of a uniformly random ranking."""
    out = {}
    for k in ks:
        total = 0.0
        for graph in graphs:
            n, v = len(_statements(graph)), len(graph.vuln_nodes)
            miss = 1.0
            for i in range(min(k, n)):  # probability that none of the first k picks is vulnerable
                miss *= max(n - v - i, 0) / (n - i)
            total += 1.0 - miss
        out[f"top{k}"] = total / len(graphs)
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    """``1 / (1 + e⁻ˣ)``, written so that large ``|x|`` does not overflow."""
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def weighted_bce(logits: np.ndarray, labels: np.ndarray, pos_weight: float):
    """Mean weighted binary cross-entropy and its gradient with respect to the logits.

    ``log σ(x) = −softplus(−x)`` and ``log(1 − σ(x)) = −softplus(x)``, where
    ``softplus(x) = log(1 + eˣ)``; ``np.logaddexp(0, x)`` computes it without overflow.
    """
    loss = pos_weight * labels * np.logaddexp(0.0, -logits) + (1 - labels) * np.logaddexp(
        0.0, logits
    )
    s = sigmoid(logits)
    grad = (pos_weight * labels * (s - 1.0) + (1 - labels) * s) / len(labels)
    return float(loss.mean()), grad


class NodeClassifier:
    """MLP over node features, trained with class weighting for the heavy imbalance."""

    name = "classifier"

    def __init__(
        self,
        tier: str = "L",
        hidden: int = 64,
        epochs: int = 80,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> None:
        self.config = {
            "tier": tier,
            "hidden": hidden,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
        }
        rng = np.random.default_rng(seed)
        self.model = MLP([feature_dim(tier), hidden, hidden, 1], "relu", rng)
        self.history: list[dict[str, float]] = []
        # Keyed by object identity: different datasets may reuse sample ids.
        self._cache: dict[int, tuple[GraphRecord, np.ndarray]] = {}

    def _dataset(self, graphs: Sequence[GraphRecord]) -> tuple[np.ndarray, np.ndarray]:
        features, labels = [], []
        for graph in graphs:
            statements = _statements(graph)
            features.append(featurize(graph, self.config["tier"])[statements])
            labels.append(np.isin(statements, graph.vuln_nodes).astype(np.float64))
        return np.concatenate(features).astype(np.float64), np.concatenate(labels)

    def fit(self, train: Sequence[GraphRecord], val: Sequence[GraphRecord]) -> NodeClassifier:
        """Train on ``train`` and keep the weights with the best top-1 accuracy on ``val``."""
        cfg = self.config
        x, y = self._dataset(train)
        positives = float(y.sum())
        pos_weight = (len(y) - positives) / positives  # balances the two classes
        optimiser = Adam(
            self.model.params,
            self.model.grads,
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        rng = np.random.default_rng(cfg["seed"])
        best, best_state = -1.0, None
        for epoch in range(cfg["epochs"]):
            order = rng.permutation(len(y))
            for start in range(0, len(y), 256):
                batch = order[start : start + 256]
                logits = self.model.forward(x[batch])[:, 0]
                loss, grad = weighted_bce(logits, y[batch], pos_weight)
                self.model.backward(grad[:, None])
                optimiser.step()
            self._cache.clear()
            metrics = self.top_k(val)
            self.history.append({"epoch": epoch + 1, "loss": loss, **metrics})
            key = metrics["top1"] + 1e-3 * metrics["top3"]
            if key > best:
                best = key
                best_state = [p.copy() for p in self.model.params]
        if best_state is not None:
            for p, saved in zip(self.model.params, best_state, strict=True):
                p[...] = saved
        self._cache.clear()
        return self

    def scores(self, graph: GraphRecord) -> np.ndarray:
        """Predicted probability that each node is vulnerable; ENTRY and EXIT score 0."""
        entry = self._cache.get(id(graph))
        if entry is not None:
            return entry[1]
        features = featurize(graph, self.config["tier"]).astype(np.float64)
        scores = sigmoid(self.model.forward(features)[:, 0])
        scores[[graph.entry, graph.exit]] = 0.0
        self._cache[id(graph)] = (graph, scores)
        return scores

    def top_k(
        self, graphs: Sequence[GraphRecord], ks: Sequence[int] = (1, 3, 5)
    ) -> dict[str, float]:
        return top_k_accuracy([self.scores(g) for g in graphs], graphs, ks)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"param_{i}": p for i, p in enumerate(self.model.params)}
        with open(path, "wb") as fh:
            np.savez(fh, config=np.array(json.dumps(self.config)), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> NodeClassifier:
        with np.load(path, allow_pickle=False) as data:
            classifier = cls(**json.loads(str(data["config"])))
            for i, p in enumerate(classifier.model.params):
                p[...] = data[f"param_{i}"]
        return classifier
