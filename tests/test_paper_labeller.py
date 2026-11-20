"""Synthetic checks only; these tests never load the legal embedding binary."""

import numpy as np
import pytest
import torch
from torch import nn

from legal_lexrank.data import ROLES
from legal_lexrank.paper_labeller import PaperRoleLabeller, load_paper_labeller


@pytest.fixture(autouse=True)
def limited_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_paper_architecture_receives_raw_vectors_and_has_only_document_layers():
    model = PaperRoleLabeller()
    assert model.input_size == 200
    assert model.config == {"hidden_size": 100, "dropout": 0.5}
    assert model.roles == ROLES
    assert model.document_lstm.input_size == 200
    assert model.document_lstm.hidden_size == 100
    assert model.document_lstm.bidirectional
    assert model.document_lstm.num_layers == 1
    assert model.to_roles.in_features == 200
    assert model.to_roles.out_features == 7
    assert set(dict(model.named_children())) == {"document_lstm", "dropout", "to_roles", "crf"}
    assert not any(isinstance(layer, (nn.LayerNorm, nn.Embedding, nn.GELU)) for layer in model.modules())
    assert all(parameter.dtype == torch.float32 and parameter.device.type == "cpu"
               for parameter in model.parameters())

    # Deliberately nonunit inputs and a true zero sentence must reach the LSTM
    # unchanged; sent2vec magnitudes are part of the pretrained representation.
    vectors = torch.arange(600, dtype=torch.float32).reshape(3, 200) / 100
    vectors[1].zero_()
    received = []
    hook = model.document_lstm.register_forward_pre_hook(
        lambda module, args: received.append(args[0].detach().clone())
    )
    model.eval()
    actual = model(vectors)
    hook.remove()
    torch.testing.assert_close(received[0].squeeze(0), vectors)
    contextual, _ = model.document_lstm(vectors.unsqueeze(0))
    expected = model.to_roles(model.dropout(contextual.squeeze(0)))
    torch.testing.assert_close(actual, expected)


def test_full_sequence_loss_matches_crf_without_sentence_reduction():
    model = PaperRoleLabeller(hidden_size=4, dropout=0)
    vectors = torch.linspace(-2, 3, 600).reshape(3, 200)
    labels = torch.tensor([0, 6, 5])
    expected = model.crf.neg_log_likelihood(model(vectors), labels)
    actual = model.loss(vectors, labels)
    torch.testing.assert_close(actual, expected)
    assert not torch.isclose(actual, expected / len(labels))
    actual.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert model.document_lstm.weight_ih_l0.grad.abs().sum() > 0
    assert model.crf.transitions.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_packed_batch_loss_and_gradients_match_independent_documents(dtype):
    model = PaperRoleLabeller(hidden_size=4, dropout=0).to(dtype=dtype)
    with torch.no_grad():
        model.crf.start.copy_(torch.linspace(-0.2, 0.3, 7))
        model.crf.end.copy_(torch.linspace(0.4, -0.1, 7))
        model.crf.transitions.copy_(torch.linspace(-0.3, 0.2, 49).reshape(7, 7))
    # Unsorted lengths exercise packing's order restoration. A zero vector is a
    # real sentence, including the complete one-sentence second document.
    vectors = [torch.linspace(-2, 1, 600).reshape(3, 200).to(dtype),
               torch.zeros(1, 200, dtype=dtype),
               torch.linspace(0.5, 3, 1000).reshape(5, 200).to(dtype)]
    vectors[2][2].zero_()
    labels = [[0, 6, 5], [6], [2, 3, 4, 1, 5]]
    independent = sum(model.loss(values, gold) for values, gold in zip(vectors, labels))
    independent.backward()
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    batched = model.loss_batch(vectors, labels)
    torch.testing.assert_close(batched, independent, atol=3e-6, rtol=1e-5)
    batched.backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name], atol=3e-6, rtol=1e-4)
    single = model.loss_batch([vectors[1]], [labels[1]])
    torch.testing.assert_close(single, model.loss(vectors[1], labels[1]), atol=3e-6, rtol=1e-5)


def test_batch_rejects_empty_or_misaligned_documents():
    model = PaperRoleLabeller(hidden_size=4)
    with pytest.raises(ValueError, match="document lists"):
        model.loss_batch([], [])
    with pytest.raises(ValueError, match="document lists"):
        model.loss_batch([torch.zeros(1, 200)], [])
    with pytest.raises(ValueError, match="per sentence"):
        model.loss_batch([torch.zeros(0, 200)], [[]])
    with pytest.raises(ValueError, match="per sentence"):
        model.loss_batch([torch.zeros(2, 200)], [[0]])


def test_prediction_and_safe_checkpoint_preserve_rng_and_modes(tmp_path, monkeypatch):
    from legal_lexrank import legal_sent2vec

    model = PaperRoleLabeller(hidden_size=6)
    model.document_lstm.eval()  # Preserve mixed module modes, too.
    modes = [module.training for module in model.modules()]
    vectors = np.arange(400, dtype=np.float32).reshape(2, 200) / 80
    rng = torch.get_rng_state().clone()
    expected = model.predict_vectors(vectors)
    assert torch.equal(rng, torch.get_rng_state())
    assert [module.training for module in model.modules()] == modes
    assert model.predict_vectors(vectors) == expected

    checkpoint = tmp_path / "nested" / "paper.pt"
    model.save(checkpoint, {"seed": 42, "epochs": 7, "source_hashes": {"model": "abc"}})
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["format"] == "paper-role-v1"
    assert payload["encoder_identity"] == legal_sent2vec.encoder_identity()
    assert all(value.device.type == "cpu" for value in payload["state"].values())
    restored = load_paper_labeller(checkpoint, cache_dir=tmp_path / "cache",
                                   model_path=tmp_path / "external.bin")
    assert torch.equal(rng, torch.get_rng_state())
    assert not restored.training
    assert restored.metadata == {"seed": 42, "epochs": 7, "source_hashes": {"model": "abc"}}
    assert restored.predict_vectors(vectors) == expected
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)

    calls = []

    def sentence_vectors(sentences, *, cache_dir, model_path):
        calls.append((sentences, cache_dir, model_path))
        return vectors

    monkeypatch.setattr(legal_sent2vec, "sentence_vectors", sentence_vectors)
    assert restored.predict(["First sentence.", "Second sentence."]) == expected
    assert calls == [(["First sentence.", "Second sentence."], tmp_path / "cache",
                      tmp_path / "external.bin")]
    assert restored.predict([]) == []
    assert restored.predict_vectors(np.empty((0, 200), dtype=np.float32)) == []
    assert len(calls) == 1


@pytest.mark.parametrize("field,value,message", [
    ("format", "different-format", "format"),
    ("input_size", 300, "200-dimensional"),
    ("roles", list(reversed(ROLES)), "canonical"),
    ("encoder_identity", {"binary_sha256": "0" * 64}, "encoder identity"),
    ("encoder_identity", None, "missing its encoder identity"),
    ("config", {"hidden_size": 100, "dropout": 0.5, "positions": True}, "configuration"),
])
def test_checkpoint_rejects_incompatible_identity_or_architecture(tmp_path, field, value, message):
    checkpoint = tmp_path / "paper.pt"
    PaperRoleLabeller(hidden_size=4).save(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    payload[field] = value
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match=message):
        load_paper_labeller(checkpoint)


def test_failed_atomic_save_preserves_existing_checkpoint(tmp_path, monkeypatch):
    model = PaperRoleLabeller(hidden_size=4)
    checkpoint = tmp_path / "paper.pt"
    model.save(checkpoint)
    original = checkpoint.read_bytes()

    def interrupted_save(payload, path):
        path.write_bytes(b"partial")
        raise OSError("interrupted save")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="interrupted"):
        model.save(checkpoint)
    assert checkpoint.read_bytes() == original
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_input_validation_and_mode_restoration_after_failure():
    model = PaperRoleLabeller(hidden_size=4)
    vectors = torch.zeros(1, 200)
    assert torch.isfinite(model.loss(vectors, [6]))
    assert len(model.predict_vectors(vectors)) == 1
    for bad, message in [(torch.zeros(1, 199), "shape"),
                         (torch.full((1, 200), float("nan")), "finite")]:
        with pytest.raises(ValueError, match=message):
            model.predict_vectors(bad)
        assert model.training
    for labels, message in [([], "per sentence"), ([7], "configured roles"),
                            ([-1], "configured roles"), ([1.5], "integer role indices")]:
        with pytest.raises(ValueError, match=message):
            model.loss(vectors, labels)
    with pytest.raises(ValueError, match="positive integer"):
        PaperRoleLabeller(hidden_size=0)
    with pytest.raises(ValueError, match="Dropout"):
        PaperRoleLabeller(dropout=float("nan"))


def test_vector_inference_follows_model_dtype():
    model = PaperRoleLabeller(hidden_size=4).double()
    vectors = np.zeros((1, 200), dtype=np.float32)
    assert model(vectors).dtype == torch.float64
    assert torch.isfinite(model.loss(vectors, [6]))
    assert len(model.predict_vectors(vectors)) == 1
