import numpy as np
import pytest
import torch
from torch.nn import functional as F

from legal_lexrank import encoders
from legal_lexrank.contextual_labeller import ContextualRoleLabeller, load_contextual_labeller


def test_sequence_loss_keeps_crf_and_weighted_sentence_gradients():
    model = ContextualRoleLabeller(4, ["fact", "lower court", "decision"], "sbert",
                                   hidden_size=5, dropout=0, use_position=False)
    vectors = F.normalize(torch.tensor([[1., 2., 3., 4.], [4., 2., 3., 1.], [1., 4., 3., 2.]]), dim=1)
    labels = torch.tensor([0, 1, 2])
    class_weights = torch.tensor([1., 3., 1.])
    emissions = model(vectors)
    sequence_loss = model.crf.neg_log_likelihood(emissions, labels) / len(labels)
    expected = sequence_loss + 0.3 * F.cross_entropy(emissions, labels, weight=class_weights)
    actual = model.loss(vectors, labels, class_weights=class_weights)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model.loss(vectors, labels, auxiliary_strength=0), sequence_loss)
    actual.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("use_position", [False, True])
def test_checkpoint_and_sentence_inference_roundtrip(tmp_path, monkeypatch, use_position):
    model = ContextualRoleLabeller(4, ["fact", "decision"], "legalbert", hidden_size=6,
                                   dropout=0.5, use_position=use_position)
    vectors = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]], dtype=np.float32)
    before = model.predict_vectors(vectors)
    assert model.training
    assert model.predict_vectors(vectors) == before
    checkpoint = tmp_path / "nested" / "model.pt"
    model.save(checkpoint, {"seed": 42, "epoch": 7})
    rng = torch.get_rng_state()
    restored = load_contextual_labeller(checkpoint, cache_dir=tmp_path / "vectors")
    assert torch.equal(rng, torch.get_rng_state())
    assert restored.metadata == {"seed": 42, "epoch": 7}
    assert restored.predict_vectors(vectors) == before
    assert not restored.training
    calls = []

    def sentence_embeddings(backend, sentences, cache_dir):
        calls.append((backend, sentences, cache_dir))
        return vectors

    monkeypatch.setattr(encoders, "sentence_embeddings", sentence_embeddings, raising=False)
    assert restored.predict(["first sentence", "second sentence"]) == before
    assert calls == [("legalbert", ["first sentence", "second sentence"], tmp_path / "vectors")]
    assert restored.predict([]) == []
    assert restored.predict_vectors(np.empty((0, 4))) == []
    assert len(calls) == 1


def test_single_sentence_loss_and_invalid_inputs():
    model = ContextualRoleLabeller(3, ["fact", "decision"], "openai", hidden_size=4)
    assert torch.isfinite(model.loss([[1., 0., 0.]], [1]))
    assert len(model.predict_vectors([[1., 0., 0.]])) == 1
    with pytest.raises(ValueError, match="shape"):
        model([[1., 0.]])
    with pytest.raises(ValueError, match="finite"):
        model([[1., float("nan"), 0.]])
    with pytest.raises(ValueError, match="per sentence"):
        model.loss([[1., 0., 0.]], [])
    with pytest.raises(ValueError, match="configured roles"):
        model.loss([[1., 0., 0.]], [2])
    with pytest.raises(ValueError, match="positive"):
        model.loss([[1., 0., 0.]], [0], class_weights=[0, 1])


def test_failed_save_preserves_existing_checkpoint(tmp_path, monkeypatch):
    model = ContextualRoleLabeller(3, ["fact", "decision"], "sbert", hidden_size=4)
    checkpoint = tmp_path / "model.pt"
    model.save(checkpoint)
    previous = checkpoint.read_bytes()

    def interrupted_save(payload, path):
        path.write_bytes(b"partial")
        raise OSError("interrupted write")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="interrupted"):
        model.save(checkpoint)
    assert checkpoint.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_centering_roundtrip_preserves_legacy_state_and_input_transform(tmp_path):
    legacy = ContextualRoleLabeller(3, ["fact", "decision"], "sbert", hidden_size=4)
    assert "center_inputs" not in legacy.config
    assert "feature_mean" not in legacy.state_dict()
    centered = ContextualRoleLabeller(3, ["fact", "decision"], "sbert", hidden_size=4,
                                      dropout=0, center_inputs=True)
    centered.feature_mean.copy_(torch.tensor([0.2, 0.3, 0.4]))
    vectors = torch.tensor([[0.2, 0.3, 0.4], [0.6, 0.2, 0.1]])
    observed = []
    hook = centered.projection.register_forward_pre_hook(lambda module, args: observed.append(args[0].clone()))
    expected = centered.predict_vectors(vectors)
    hook.remove()
    torch.testing.assert_close(observed[0], F.normalize(vectors - centered.feature_mean, dim=1, eps=1e-12))
    assert torch.isfinite(observed[0]).all()
    path = tmp_path / "centered.pt"
    centered.save(path)
    restored = load_contextual_labeller(path)
    assert restored.config["center_inputs"] is True
    torch.testing.assert_close(restored.feature_mean, centered.feature_mean)
    assert restored.predict_vectors(vectors) == expected
