"""Supervised node classifier: a reference that ranks statements without navigating.

It answers "what if you just classify every node?". A small MLP on the per-node features
predicts whether a statement is vulnerable; at test time all statements of a function are
ranked by that probability. It sees the whole function at once, so it measures how much the
features can say about which statement is vulnerable. It is not a traversal method.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from soren.data.features import feature_dim, featurize
from soren.data.schema import GraphRecord


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
        torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(feature_dim(tier), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.history: list[dict[str, float]] = []
        # Keyed by object identity: different datasets may reuse sample ids.
        self._cache: dict[int, tuple[GraphRecord, np.ndarray]] = {}

    def _dataset(self, graphs: Sequence[GraphRecord]) -> tuple[torch.Tensor, torch.Tensor]:
        features, labels = [], []
        for graph in graphs:
            statements = _statements(graph)
            features.append(featurize(graph, self.config["tier"])[statements])
            labels.append(np.isin(statements, graph.vuln_nodes).astype(np.float32))
        return torch.from_numpy(np.concatenate(features)), torch.from_numpy(np.concatenate(labels))

    def fit(self, train: Sequence[GraphRecord], val: Sequence[GraphRecord]) -> NodeClassifier:
        """Train on ``train`` and keep the weights with the best top-1 accuracy on ``val``."""
        cfg = self.config
        x, y = self._dataset(train)
        positives = float(y.sum())
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - positives) / positives))
        optimiser = torch.optim.Adam(
            self.model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )
        generator = torch.Generator().manual_seed(cfg["seed"])
        best, best_state = -1.0, None
        for epoch in range(cfg["epochs"]):
            self.model.train()
            order = torch.randperm(len(y), generator=generator)
            for start in range(0, len(y), 256):
                batch = order[start : start + 256]
                optimiser.zero_grad()
                loss = loss_fn(self.model(x[batch]).squeeze(1), y[batch])
                loss.backward()
                optimiser.step()
            self._cache.clear()
            metrics = self.top_k(val)
            self.history.append({"epoch": epoch + 1, "loss": loss.item(), **metrics})
            key = metrics["top1"] + 1e-3 * metrics["top3"]
            if key > best:
                best = key
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._cache.clear()
        return self

    def scores(self, graph: GraphRecord) -> np.ndarray:
        """Predicted probability that each node is vulnerable; ENTRY and EXIT score 0."""
        entry = self._cache.get(id(graph))
        if entry is not None:
            return entry[1]
        self.model.eval()
        with torch.no_grad():
            features = torch.from_numpy(featurize(graph, self.config["tier"]))
            scores = torch.sigmoid(self.model(features).squeeze(1)).numpy().astype(np.float64)
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
        torch.save({"config": self.config, "state": self.model.state_dict()}, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> NodeClassifier:
        payload: dict[str, Any] = torch.load(path, weights_only=True)
        classifier = cls(**payload["config"])
        classifier.model.load_state_dict(payload["state"])
        return classifier
