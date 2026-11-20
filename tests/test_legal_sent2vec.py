"""Synthetic adapter checks; no native model or download is needed."""

import hashlib
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from legal_lexrank import legal_sent2vec as encoder


@pytest.fixture(autouse=True)
def clear_native_caches():
    encoder._verify_binary.cache_clear()
    encoder._native_model.cache_clear()
    yield
    encoder._verify_binary.cache_clear()
    encoder._native_model.cache_clear()


def fake_binary(tmp_path, monkeypatch):
    path = tmp_path / "model.bin"
    content = b"small synthetic model"
    path.write_bytes(content)
    monkeypatch.setattr(encoder, "BINARY_BYTES", len(content))
    monkeypatch.setattr(encoder, "BINARY_SHA256", hashlib.sha256(content).hexdigest())
    return path


def test_static_identity_empty_input_and_raw_ordered_cache(tmp_path, monkeypatch):
    path = fake_binary(tmp_path, monkeypatch)
    calls = []

    class Native:
        def load_model(self, model_path, *, inference_mode):
            calls.append((model_path, inference_mode))

        def get_emb_size(self):
            return 200

        def embed_sentences(self, sentences, *, num_threads):
            calls.append((sentences, num_threads))
            values = np.zeros((len(sentences), 200), dtype=np.float32)
            values[:, 0] = [7 if text == "Court." else 0 for text in sentences]
            return values

    monkeypatch.setitem(sys.modules, "sent2vec", SimpleNamespace(Sent2vecModel=Native))
    identity = encoder.encoder_identity()
    assert identity["normalization"] == "none"
    assert identity["dimensions"] == 200
    assert not calls
    assert encoder.sentence_vectors([], tmp_path / "empty").shape == (0, 200)
    assert not (tmp_path / "empty").exists()
    monkeypatch.setenv("LEGAL_SENT2VEC_MODEL", str(path))
    cache = tmp_path / "cache"
    actual = encoder.sentence_vectors(["  Court. ", "unknown", "Court."], cache)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual[:, 0], [7, 0, 7])
    assert calls == [(str(path.resolve()), True), (["Court.", "unknown", "Court."], 1)]

    # Cached vectors remain usable when the optional binary/binding are absent.
    path.unlink()
    monkeypatch.setitem(sys.modules, "sent2vec", None)
    cached = encoder.sentence_vectors(["Court.", "unknown", "Court."], cache)
    np.testing.assert_array_equal(actual, cached)
    assert len(calls) == 2
    with pytest.raises(FileNotFoundError):
        encoder.sentence_vectors(["unknown", "Court.", "Court."], cache)
    monkeypatch.setattr(encoder, "SOURCE_REVISION", "another-revision")
    with pytest.raises(FileNotFoundError):
        encoder.sentence_vectors(["Court.", "unknown", "Court."], cache)


def test_binary_hash_mismatch_prevents_native_load(tmp_path, monkeypatch):
    path = fake_binary(tmp_path, monkeypatch)
    path.write_bytes(b"x" * path.stat().st_size)
    monkeypatch.setitem(sys.modules, "sent2vec", None)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        encoder.sentence_vectors(["Court."], tmp_path / "cache", model_path=path)
    assert not list((tmp_path / "cache").rglob("*.npy"))


@pytest.mark.parametrize("values", [
    np.zeros((1, 199), dtype=np.float32),
    np.zeros((2, 200), dtype=np.float32),
    np.zeros((1, 200), dtype=np.float64),
    np.full((1, 200), np.nan, dtype=np.float32),
    np.full((1, 200), np.inf, dtype=np.float32),
])
def test_invalid_native_vectors_are_not_cached(tmp_path, monkeypatch, values):
    path = fake_binary(tmp_path, monkeypatch)
    model = SimpleNamespace(embed_sentences=lambda sentences, num_threads: values)
    monkeypatch.setattr(encoder, "_native_model", lambda state: model)
    with pytest.raises(ValueError, match="finite float32"):
        encoder.sentence_vectors(["Court."], tmp_path / "cache", model_path=path)
    assert not list((tmp_path / "cache").rglob("*.npy"))


def test_cache_corruption_and_multiline_input_fail_explicitly(tmp_path, monkeypatch):
    path = fake_binary(tmp_path, monkeypatch)
    model = SimpleNamespace(embed_sentences=lambda sentences, num_threads: np.zeros((1, 200), dtype=np.float32))
    monkeypatch.setattr(encoder, "_native_model", lambda state: model)
    cache = tmp_path / "cache"
    encoder.sentence_vectors(["unknown"], cache, model_path=path)
    saved = next(cache.rglob("*.npy"))
    np.save(saved, np.full((1, 200), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="finite float32"):
        encoder.sentence_vectors(["unknown"], cache, model_path=path)
    with pytest.raises(ValueError, match="one line"):
        encoder.sentence_vectors(["First\nSecond"], cache, model_path=path)
    with pytest.raises(TypeError, match="sequence"):
        encoder.sentence_vectors("single string", cache, model_path=path)
