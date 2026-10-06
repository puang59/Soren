"""Statement embeddings for the Tier E features.

Each statement's text is encoded with a frozen pretrained code model; the ``[CLS]`` vectors
are reduced with PCA fitted on the training split only, squashed into ``[0, 1]`` and stored in
``GraphRecord.features["embed"]``. The model never sees labels, and nothing about validation
or test graphs enters the projection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soren.data.schema import GraphRecord

EMBED_KEY = "embed"
DEFAULT_MODEL = "microsoft/codebert-base"
DEFAULT_DIM = 32
MAX_TOKENS = 64


def statement_texts(graph: GraphRecord) -> list[str]:
    """The text to encode for each node; ENTRY and EXIT are encoded as their kind."""
    return [
        node.kind if node.kind in ("ENTRY", "EXIT") else (node.code or node.kind)
        for node in graph.nodes
    ]


def encode_texts(
    texts: Sequence[str], model_name_or_path: str = DEFAULT_MODEL, batch_size: int = 64
) -> np.ndarray:
    """Encode ``texts`` with a frozen transformer; returns the ``[CLS]`` vectors.

    Identical texts are encoded once. ``transformers`` is imported here so that the rest of
    the package does not need it.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).eval()
    unique = sorted(set(texts))
    vectors = []
    with torch.no_grad():
        for start in range(0, len(unique), batch_size):
            batch = tokenizer(
                unique[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
                return_tensors="pt",
            )
            vectors.append(model(**batch).last_hidden_state[:, 0].numpy())
    table = dict(zip(unique, np.concatenate(vectors), strict=True))
    return np.stack([table[text] for text in texts])


@dataclass
class Projection:
    """PCA followed by per-component scaling and a squash into ``[0, 1]``."""

    mean: np.ndarray
    components: np.ndarray  # [dim, input_dim]
    scale: np.ndarray  # standard deviation of each component on the fitting data

    @classmethod
    def fit(cls, vectors: np.ndarray, dim: int = DEFAULT_DIM) -> Projection:
        """Fit on training vectors only."""
        if dim > min(vectors.shape):
            raise ValueError(f"cannot fit {dim} components to data of shape {vectors.shape}")
        mean = vectors.mean(axis=0)
        _, singular, rows = np.linalg.svd(vectors - mean, full_matrices=False)
        scale = singular[:dim] / np.sqrt(max(len(vectors) - 1, 1))
        return cls(mean=mean, components=rows[:dim], scale=np.maximum(scale, 1e-8))

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Project, standardise and squash: 0.5 is the training mean of each component."""
        reduced = (vectors - self.mean) @ self.components.T / self.scale
        return (0.5 + 0.5 * np.tanh(reduced / 2.0)).astype(np.float32)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, components=self.components, scale=self.scale)
        return path

    @classmethod
    def load(cls, path: str | Path) -> Projection:
        data = np.load(path)
        return cls(mean=data["mean"], components=data["components"], scale=data["scale"])


def attach_embeddings(
    graphs: Sequence[GraphRecord], vectors: np.ndarray, projection: Projection
) -> None:
    """Store the projected vectors on each graph; ``vectors`` holds all nodes in graph order."""
    reduced = projection.transform(vectors)
    offset = 0
    for graph in graphs:
        rows = reduced[offset : offset + graph.num_nodes]
        graph.features[EMBED_KEY] = [[round(float(v), 4) for v in row] for row in rows]
        offset += graph.num_nodes
    if offset != len(vectors):
        raise ValueError(f"{len(vectors)} vectors for {offset} nodes")
