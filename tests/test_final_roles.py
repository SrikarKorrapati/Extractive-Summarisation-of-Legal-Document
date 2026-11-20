import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from legal_lexrank import final_roles, role_features, tune_roles
from legal_lexrank.contextual_labeller import load_contextual_labeller
from legal_lexrank.data import ROLES, split_case_ids


@pytest.fixture
def selected_model(tmp_path, monkeypatch):
    cases = {
        f"case{i}": SimpleNamespace(
            labelled_sentences=[f"case{i} sentence {j}" for j in range(8)],
            labels=[*ROLES[:3], None, *ROLES[3:]],
        )
        for i in range(6)
    }
    for module in (role_features, tune_roles, final_roles):
        monkeypatch.setattr(module, "load_cases", lambda root: cases)
    monkeypatch.setattr(role_features.encoders, "sentence_embeddings",
                        lambda *args: np.eye(8, dtype=np.float32), raising=False)
    monkeypatch.setattr(tune_roles, "CONFIGURATIONS", [dict(tune_roles.CONFIGURATIONS[0], hidden_size=3)])
    features = role_features.build(tmp_path, tmp_path / "features", "sbert", tmp_path / "cache")
    tuning = tmp_path / "tuning"
    tune_roles.run(tmp_path, features, tuning, max_epochs=1, patience=1)
    return cases, features, tuning / "selection.json"


def test_final_fit_uses_all_training_cases_and_fixed_selected_epochs(selected_model, tmp_path, monkeypatch):
    cases, features, selection_path = selected_model
    split = split_case_ids(cases)
    seen = []
    training_sequences = tune_roles.training_sequences

    def observed_sequences(vectors, labels, ids):
        seen.extend(ids)
        assert set(vectors) == set(labels) == set(split["train"])
        return training_sequences(vectors, labels, ids)

    def forbidden_evaluation(*args):
        raise AssertionError("Final fitting must not use evaluation to change its settings")

    monkeypatch.setattr(tune_roles, "training_sequences", observed_sequences)
    monkeypatch.setattr(tune_roles, "evaluate", forbidden_evaluation)
    output = tmp_path / "final"
    protocol = final_roles.run(tmp_path, features, selection_path, output)
    assert seen == split["train"]
    assert set(seen).isdisjoint(split["test"])
    assert protocol["format"] == "legal-lexrank-role-study-v2"
    assert protocol["epochs"] == 1
    assert protocol["case_ids"] == split
    assert protocol["training_sha256"]["test"] != protocol["training_sha256"]["validation"]
    assert len(json.loads((output / "training_history.json").read_text())) == 1
    assert (output / "validation_labeller.pt").read_bytes() == (selection_path.parent / "validation_labeller.pt").read_bytes()
    restored = load_contextual_labeller(output / "role_labeller.pt")
    assert restored.metadata["train_case_ids"] == split["train"]
    assert restored.metadata["epoch"] == 1
    assert protocol["checkpoint_sha256"]["test"] == hashlib.sha256((output / "role_labeller.pt").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="empty final-training directory"):
        final_roles.run(tmp_path, features, selection_path, output)


def test_final_fit_rejects_changed_selected_checkpoint(selected_model, tmp_path):
    _, features, selection = selected_model
    checkpoint = selection.parent / "validation_labeller.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"altered")
    output = tmp_path / "final"
    with pytest.raises(ValueError, match="checkpoint checksum mismatch"):
        final_roles.run(tmp_path, features, selection, output)
    assert not output.exists()
