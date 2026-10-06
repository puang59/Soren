import importlib.util
from pathlib import Path

import numpy as np
import pytest
from helpers import diamond_graph, line_graph

from soren.baselines.heuristic import ThresholdRule, make_baseline
from soren.baselines.node_classifier import NodeClassifier, chance_top_k, top_k_accuracy
from soren.data.schema import write_jsonl
from soren.data.synthetic import SyntheticConfig, generate_dataset

ROOT = Path(__file__).parent.parent
CLEAN = SyntheticConfig(min_nodes=10, max_nodes=30, hit_rate=1.0, decoy_rate=0.0)
TRAIN = generate_dataset(150, seed=0, cfg=CLEAN)
VAL = generate_dataset(60, seed=1, cfg=CLEAN)


def test_top_k_accuracy_ranks_statements_only():
    graph = diamond_graph(vuln=3)
    scores = np.array([9.0, 0.1, 0.5, 0.4, 0.3, 9.0])  # ENTRY and EXIT score highest
    result = top_k_accuracy([scores], [graph])
    assert result == {"top1": 0.0, "top3": 1.0, "top5": 1.0}  # node 2 first, node 3 second
    assert top_k_accuracy([np.array([0, 0, 0, 1.0, 0, 0])], [graph])["top1"] == 1.0


def test_chance_top_k():
    graph = line_graph(4, vuln=2)  # four statements, one vulnerable
    chance = chance_top_k([graph], ks=(1, 2, 4, 9))
    assert chance == pytest.approx({"top1": 0.25, "top2": 0.5, "top4": 1.0, "top9": 1.0})


def test_classifier_learns_a_clean_signal():
    classifier = NodeClassifier(epochs=25, seed=0).fit(TRAIN, VAL)
    metrics = classifier.top_k(VAL)
    assert metrics["top1"] > 0.95
    assert metrics["top1"] > chance_top_k(VAL)["top1"] + 0.5
    assert len(classifier.history) == 25 and classifier.history[0]["epoch"] == 1


def test_scores_are_probabilities_and_ignore_entry_and_exit():
    classifier = NodeClassifier(epochs=3).fit(TRAIN[:40], VAL[:10])
    for graph in VAL[:10]:
        scores = classifier.scores(graph)
        assert scores.shape == (graph.num_nodes,)
        assert scores.min() >= 0.0 and scores.max() <= 1.0
        assert scores[graph.entry] == 0.0 and scores[graph.exit] == 0.0


def test_training_is_reproducible_and_round_trips(tmp_path):
    a = NodeClassifier(epochs=4, seed=3).fit(TRAIN[:50], VAL[:20])
    b = NodeClassifier(epochs=4, seed=3).fit(TRAIN[:50], VAL[:20])
    np.testing.assert_allclose(a.scores(VAL[0]), b.scores(VAL[0]))
    loaded = NodeClassifier.load(a.save(tmp_path / "m" / "model.pt"))
    np.testing.assert_allclose(loaded.scores(VAL[0]), a.scores(VAL[0]))
    assert loaded.config == a.config


def test_classifier_plugs_into_protocol_b():
    classifier = NodeClassifier(epochs=25, seed=0).fit(TRAIN, VAL)
    assert isinstance(ThresholdRule(classifier, 0.5)(VAL[0], VAL[0].vuln_nodes[0]), bool)
    assert make_baseline("dfs", scorer=classifier, threshold=0.5).name == "dfs+classifier"
    # Class weighting pushes every probability up, so a fixed 0.5 does not separate the
    # classes; take the threshold from the scores, as tuning does on real data.
    flagged = [classifier.scores(g)[g.vuln_nodes[0]] for g in VAL]
    others = [
        np.delete(classifier.scores(g), [g.vuln_nodes[0], g.entry, g.exit]).max() for g in VAL
    ]
    threshold = (np.quantile(flagged, 0.05) + np.quantile(others, 0.95)) / 2
    searcher = make_baseline("dfs", scorer=classifier, threshold=float(threshold))
    results = [searcher.run(g, 200, np.random.default_rng(0)) for g in VAL]
    assert np.mean([r.success for r in results]) > 0.9


def test_script(tmp_path, capsys):
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    write_jsonl(TRAIN[:60], train)
    write_jsonl(VAL[:20], val)
    spec = importlib.util.spec_from_file_location("tc", ROOT / "scripts" / "train_classifier.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "clf" / "model.pt"
    module.main(["--train", str(train), "--val", str(val), "--out", str(out), "--epochs", "5"])
    assert out.exists() and out.with_suffix(".json").exists()
    printed = capsys.readouterr().out
    assert "classifier_val" in printed and "chance_val" in printed
