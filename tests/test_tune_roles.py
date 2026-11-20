import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from legal_lexrank import role_features, tune_roles
from legal_lexrank.contextual_labeller import load_contextual_labeller
from legal_lexrank.data import ROLES, split_case_ids


@pytest.fixture
def feature_bundle(tmp_path, monkeypatch):
    cases = {
        f"case{i}": SimpleNamespace(
            labelled_sentences=[f"case{i} sentence {j}" for j in range(8)],
            labels=[*ROLES[:3], None, *ROLES[3:]],
        )
        for i in range(6)
    }
    monkeypatch.setattr(role_features, "load_cases", lambda root: cases)
    monkeypatch.setattr(tune_roles, "load_cases", lambda root: cases)
    monkeypatch.setattr(role_features.encoders, "sentence_embeddings",
                        lambda *args: np.eye(8, dtype=np.float32), raising=False)
    features = role_features.build(tmp_path, tmp_path / "features", "sbert", tmp_path / "cache")
    return cases, features


def test_manifest_rejects_test_examples_and_changed_labels(feature_bundle, tmp_path):
    cases, features = feature_bundle
    manifest, vectors, labels = tune_roles.load_features(tmp_path, features)
    split = split_case_ids(cases)
    assert set(vectors) == set(labels) == set(split["train"])
    assert set(vectors).isdisjoint(split["test"])
    path = features / "manifest.json"
    manifest["included_case_ids"].append(split["test"][0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="included_case_ids"):
        tune_roles.load_features(tmp_path, features)
    manifest["included_case_ids"] = split["train"]
    manifest["cases"][split["train"][0]]["labels"][0] = 6
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="annotations mismatch"):
        tune_roles.load_features(tmp_path, features)


def test_feature_checksum_and_missing_label_spans(feature_bundle, tmp_path):
    _, features = feature_bundle
    manifest, vectors, labels = tune_roles.load_features(tmp_path, features)
    ids = manifest["case_ids"]["fit"]
    sequences = tune_roles.training_sequences(vectors, labels, ids)
    assert len(sequences) == 2 * len(ids)
    assert [gold.tolist() for _, gold in sequences] == [[0, 1, 2], [3, 4, 5, 6]] * len(ids)
    assert all(len(values) == len(gold) for values, gold in sequences)
    feature_file = features / "features.npz"
    feature_file.write_bytes(feature_file.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="file checksum mismatch"):
        tune_roles.load_features(tmp_path, features)


def test_selection_uses_validation_and_keeps_best_epoch(feature_bundle, tmp_path, monkeypatch):
    cases, features = feature_bundle
    split = split_case_ids(cases)
    configs = [dict(tune_roles.CONFIGURATIONS[0], hidden_size=3),
               dict(tune_roles.CONFIGURATIONS[1], hidden_size=3)]
    monkeypatch.setattr(tune_roles, "CONFIGURATIONS", configs)
    scores = iter([0.8, 0.6, 0.7, 0.7])
    evaluations, states = [], []

    def controlled_evaluation(model, vectors, labels, ids):
        evaluations.append(list(ids))
        states.append({key: value.clone() for key, value in model.state_dict().items()})
        score = next(scores)
        return {"accuracy": score, "macro avg": {"f1-score": score / 2},
                "Ruling by Lower Court": {"recall": 0.5}}

    monkeypatch.setattr(tune_roles, "evaluate", controlled_evaluation)
    output = tmp_path / "selection"
    chosen = tune_roles.run(tmp_path, features, output, max_epochs=2, patience=2)
    assert evaluations == [split["validation"]] * 4
    assert chosen["selected"]["candidate"] == 1
    assert chosen["selected"]["best_epoch"] == 1
    assert chosen["selected"]["validation"]["accuracy"] == 0.8
    assert chosen["candidates"][1]["best_epoch"] == 1
    restored = load_contextual_labeller(output / "validation_labeller.pt")
    assert restored.metadata["fit_case_ids"] == split["fit"]
    assert restored.metadata["validation_case_ids"] == split["validation"]
    assert restored.metadata["epoch"] == 1
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, states[0][key])
    assert hashlib.sha256((output / "validation_labeller.pt").read_bytes()).hexdigest() == chosen["selected"]["checkpoint_sha256"]
    assert (output / "selection.json").exists()
    assert not (output / "role_labeller.pt").exists()
    with pytest.raises(ValueError, match="empty tuning directory"):
        tune_roles.run(tmp_path, features, output, max_epochs=1)
