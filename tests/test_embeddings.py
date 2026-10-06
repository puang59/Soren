import numpy as np
import pytest
from helpers import diamond_graph, line_graph

from soren.data.embeddings import EMBED_KEY, Projection, attach_embeddings, statement_texts
from soren.data.features import EMBED_DIM, TIER_E_NAMES, feature_dim, featurize
from soren.data.schema import GraphRecord
from soren.env.cfg_nav_env import CFGNavEnv, EnvConfig


def vectors(n: int, seed: int = 0, width: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, width)) * np.linspace(3.0, 0.1, width)


def test_statement_texts():
    graph = diamond_graph()
    texts = statement_texts(graph)
    assert texts[0] == "ENTRY" and texts[-1] == "EXIT" and texts[1] == "s1;"
    assert len(texts) == graph.num_nodes


def test_projection_reduces_scales_and_squashes():
    train = vectors(500)
    projection = Projection.fit(train, dim=EMBED_DIM)
    out = projection.transform(train)
    assert out.shape == (500, EMBED_DIM) and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Centred on the training mean, with comparable spread in every component.
    np.testing.assert_allclose(out.mean(axis=0), 0.5, atol=0.03)
    assert out.std(axis=0).min() > 0.15
    # The first components carry the directions of largest variance.
    assert abs(projection.components[0]).argmax() == 0


def test_projection_is_fitted_on_training_data_only():
    train, other = vectors(400, seed=0), vectors(50, seed=1) + 5.0
    projection = Projection.fit(train)
    again = Projection.fit(train)
    np.testing.assert_allclose(projection.transform(other), again.transform(other))
    # Data from elsewhere is projected with the training statistics, not its own.
    assert abs(projection.transform(other).mean() - 0.5) > 0.02


def test_projection_round_trip_and_bad_dim(tmp_path):
    projection = Projection.fit(vectors(200))
    loaded = Projection.load(projection.save(tmp_path / "p" / "projection.npz"))
    sample = vectors(10, seed=4)
    np.testing.assert_allclose(loaded.transform(sample), projection.transform(sample))
    with pytest.raises(ValueError, match="cannot fit"):
        Projection.fit(vectors(5), dim=32)


def embedded_graphs():
    graphs = [line_graph(4), diamond_graph()]
    total = sum(g.num_nodes for g in graphs)
    projection = Projection.fit(vectors(300))
    attach_embeddings(graphs, vectors(total, seed=7), projection)
    return graphs


def test_attach_embeddings_and_tier_e_features():
    graphs = embedded_graphs()
    for graph in graphs:
        assert np.asarray(graph.features[EMBED_KEY]).shape == (graph.num_nodes, EMBED_DIM)
        graph.validate()
        assert GraphRecord.from_json(graph.to_json()) == graph  # survives the JSONL round trip
        tier_e = featurize(graph, "E")
        assert tier_e.shape == (graph.num_nodes, feature_dim("E"))
        assert feature_dim("E") == feature_dim("L") + EMBED_DIM == len(TIER_E_NAMES)
        np.testing.assert_array_equal(tier_e[:, : feature_dim("L")], featurize(graph, "L"))
        assert tier_e.min() >= 0.0 and tier_e.max() <= 1.0
    with pytest.raises(ValueError, match="vectors for"):
        attach_embeddings(graphs, vectors(3), Projection.fit(vectors(300)))


def test_tier_e_needs_stored_embeddings():
    with pytest.raises(ValueError, match="no statement embeddings"):
        featurize(diamond_graph(), "E")
    graph = diamond_graph()
    graph.features[EMBED_KEY] = [[0.5] * 8] * graph.num_nodes
    with pytest.raises(ValueError, match="embeddings have shape"):
        featurize(graph, "E")


def test_environment_runs_on_tier_e():
    env = CFGNavEnv(embedded_graphs(), EnvConfig(feature_tier="E"))
    obs, _ = env.reset(seed=0)
    f = feature_dim("E")
    assert obs.shape == (f + 2 + 6 * (f + 3) + 13 + 5,)
    assert env.observation_space.contains(obs)
    obs, *_ = env.step(0)
    assert env.observation_space.contains(obs)
