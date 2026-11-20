import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from legal_lexrank import role_features
from legal_lexrank.data import ROLES, split_case_ids


def synthetic_cases():
    return {
        f"case{i}": SimpleNamespace(
            labelled_sentences=[f"case{i} sentence {j}" for j in range(3)],
            labels=["Facts", None, "Ruling by Lower Court"],
        )
        for i in range(10)
    }


def test_feature_export_excludes_test_and_keeps_missing_labels_aligned(tmp_path, monkeypatch):
    cases = synthetic_cases()
    split = split_case_ids(cases)
    seen = []
    monkeypatch.setattr(role_features, "load_cases", lambda root: cases)

    def embed(backend, sentences, cache):
        seen.append(sentences[0].split()[0])
        return np.array([[3., 4.], [0., 10.], [10., 0.]], dtype=np.float64)

    monkeypatch.setattr(role_features.encoders, "sentence_embeddings", embed, raising=False)
    output = role_features.build(tmp_path, tmp_path / "features", "sbert", tmp_path / "cache")
    manifest = json.loads((output / "manifest.json").read_text())
    assert seen == split["train"]
    assert set(seen).isdisjoint(split["test"])
    assert manifest["case_ids"] == split
    assert manifest["included_case_ids"] == seen
    assert manifest["features_sha256"] == hashlib.sha256((output / "features.npz").read_bytes()).hexdigest()
    with np.load(output / "features.npz", allow_pickle=False) as saved:
        assert set(saved.files) == set(split["train"])
        for key in saved.files:
            assert saved[key].dtype == np.float32
            np.testing.assert_allclose(saved[key], [[.6, .8], [0., 1.], [1., 0.]], atol=1e-7)
            assert manifest["cases"][key]["labels"] == [ROLES.index("Facts"), -1, ROLES.index("Ruling by Lower Court")]

    before = len(seen)
    with pytest.raises(ValueError, match="empty feature directory"):
        role_features.build(tmp_path, output, "sbert", tmp_path / "cache")
    assert len(seen) == before

    full = role_features.build(tmp_path, tmp_path / "with-test", "sbert", tmp_path / "cache", include_test=True)
    full_manifest = json.loads((full / "manifest.json").read_text())
    assert full_manifest["included_case_ids"] == sorted(cases)
    assert full_manifest["training_sha256"] == manifest["training_sha256"]
    assert full_manifest["corpus_sha256"] != manifest["corpus_sha256"]


@pytest.mark.parametrize("values", [np.zeros((3, 2)), np.ones((2, 2)), np.full((3, 2), np.nan)])
def test_invalid_feature_output_cannot_publish_a_bundle(tmp_path, monkeypatch, values):
    monkeypatch.setattr(role_features, "load_cases", lambda root: synthetic_cases())
    monkeypatch.setattr(role_features.encoders, "sentence_embeddings", lambda *args: values, raising=False)
    output = tmp_path / "features"
    with pytest.raises(ValueError, match="sentence embedding"):
        role_features.build(tmp_path, output, "legalbert", tmp_path / "cache")
    assert not output.exists()
