from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel, BertTokenizer, DebertaV2Config, DebertaV2Model

from legal_lexrank.data import ROLES
from legal_lexrank import transformer_labeller as module
from legal_lexrank.transformer_labeller import TransformerRoleLabeller, load_transformer_labeller


@pytest.fixture
def tokenizer(tmp_path):
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"t{i}" for i in range(1300)]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocabulary), encoding="utf-8")
    return BertTokenizer(vocab_file=str(path), do_lower_case=False)


def tiny_backbone(encoder="inlegalbert"):
    config = dict(vocab_size=1305, hidden_size=8, num_hidden_layers=1,
                  num_attention_heads=2, intermediate_size=12,
                  hidden_dropout_prob=0, attention_probs_dropout_prob=0,
                  max_position_embeddings=512)
    if encoder == "deberta":
        return DebertaV2Model(DebertaV2Config(**config, type_vocab_size=0))
    return BertModel(BertConfig(**config))


def make_model(tokenizer, **kwargs):
    encoder = kwargs.pop("encoder", "inlegalbert")
    kwargs.setdefault("dropout", 0)
    return TransformerRoleLabeller(encoder, tokenizer=tokenizer,
                                  backbone=tiny_backbone(encoder), **kwargs)


def target_tokens(example):
    return [token for token, target in zip(example["input_ids"], example["target_mask"]) if target]


def test_long_target_preserves_every_token_and_weights_with_bounded_same_case_context(tokenizer):
    model = make_model(tokenizer)
    sentences = [" ".join(f"t{i}" for i in range(150)),
                 " ".join(f"t{i}" for i in range(150, 1151)),
                 " ".join(f"t{i}" for i in range(1151, 1300))]
    examples = model.make_examples(sentences, [ROLES[0], ROLES[6], ROLES[5]])
    chunks = [row for row in examples if row["sentence_index"] == 1]
    assert [sum(row["target_mask"]) for row in chunks] == [384, 384, 233]
    assert [token for row in chunks for token in target_tokens(row)] == tokenizer.encode(
        sentences[1], add_special_tokens=False)
    assert sum(row["chunk_weight"] for row in chunks) == pytest.approx(1)
    assert [row["chunk_weight"] for row in chunks] == pytest.approx([384 / 1001, 384 / 1001, 233 / 1001])
    assert all(len(row["input_ids"]) <= 512 for row in examples)
    assert all(row["label"] == 6 for row in chunks)
    # Immediate context is the previous tail and following head; distant tokens
    # and text from a separate call cannot enter this case.
    context = [token for token, target in zip(chunks[0]["input_ids"], chunks[0]["target_mask"]) if not target]
    assert tokenizer.convert_tokens_to_ids("t149") in context
    assert tokenizer.convert_tokens_to_ids("t1151") in context
    assert tokenizer.convert_tokens_to_ids("t0") not in context
    assert tokenizer.convert_tokens_to_ids("t1299") not in context
    isolated = model.make_examples(["t0"])[0]
    assert isolated["input_ids"] == [tokenizer.cls_token_id, tokenizer.convert_tokens_to_ids("t0"),
                                     tokenizer.sep_token_id, tokenizer.sep_token_id]


def test_literal_special_tokens_are_targets_and_padding_never_is(tokenizer):
    model = make_model(tokenizer, max_length=12, target_chunk_size=5)
    examples = model.make_examples(["t0 [SEP] t1 [CLS] t2", "t3"])
    assert target_tokens(examples[0]) == tokenizer.encode("t0 [SEP] t1 [CLS] t2", add_special_tokens=False)
    assert sum(examples[0]["target_mask"]) == 5
    batch = model.collate(examples)
    assert batch["input_ids"].shape[1] == max(len(row["input_ids"]) for row in examples)
    assert not batch["target_mask"][batch["attention_mask"] == 0].any()
    assert batch["input_ids"][batch["attention_mask"] == 0].eq(tokenizer.pad_token_id).all()


def test_missing_labels_remain_context_and_empty_sentences_are_aligned(tokenizer):
    model = make_model(tokenizer, max_length=12, target_chunk_size=4)
    examples = model.make_examples(["t0", "t1", ""], [None, ROLES[1], ROLES[0]])
    assert [row["sentence_index"] for row in examples] == [1, 2]
    assert [row["label"] for row in examples] == [1, 0]
    assert tokenizer.convert_tokens_to_ids("t0") in examples[0]["input_ids"]
    assert target_tokens(examples[1]) == [tokenizer.unk_token_id]
    assert model.make_examples([]) == []


@pytest.mark.parametrize("encoder", ["inlegalbert", "deberta"])
def test_encoder_and_new_head_receive_gradients_without_recurrent_layers(tokenizer, encoder):
    if encoder == "deberta":
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
    model = make_model(tokenizer, encoder=encoder, max_length=12, target_chunk_size=4)
    examples = model.make_examples(["t0 t1", "t2"], [ROLES[0], ROLES[6]])
    batch = model.collate(examples)
    if encoder == "deberta":
        assert "token_type_ids" not in batch
    assert model.roles == ROLES
    assert model(batch).shape == (2, 7)
    loss = (F.cross_entropy(model(**batch), batch["labels"], reduction="none") * batch["chunk_weight"]).sum()
    loss.backward()
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())
    assert model.backbone.get_input_embeddings().weight.grad.abs().sum() > 0
    assert model.to_roles.weight.grad.abs().sum() > 0
    attention = [parameter for name, parameter in model.backbone.named_parameters() if "attention" in name]
    assert attention and any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in attention)
    assert not any(isinstance(layer, (nn.LSTM, nn.GRU, nn.RNN)) for layer in model.modules())


def test_forward_pools_targets_only(tokenizer, monkeypatch):
    model = make_model(tokenizer, max_length=12, target_chunk_size=4)

    def uncontextualized(**inputs):
        return SimpleNamespace(last_hidden_state=inputs["input_ids"].float()[..., None].expand(-1, -1, 8))

    monkeypatch.setattr(model.backbone, "forward", uncontextualized)
    with torch.no_grad():
        model.to_roles.weight.fill_(1 / 8)
        model.to_roles.bias.zero_()
    examples = model.make_examples(["t1 t3", "t100 t110"])
    actual = model(model.collate(examples))
    expected = torch.tensor([sum(target_tokens(row)) / len(target_tokens(row)) for row in examples])
    torch.testing.assert_close(actual, expected[:, None].expand(-1, 7))


def test_prediction_combines_logits_by_token_fraction_before_argmax(tokenizer, monkeypatch):
    model = make_model(tokenizer, max_length=12, target_chunk_size=3)
    batch_sizes = []

    def chunk_logits(batch):
        batch_sizes.append(len(batch["input_ids"]))
        logits = torch.zeros((len(batch["input_ids"]), 7))
        for index, (ids, mask) in enumerate(zip(batch["input_ids"], batch["target_mask"])):
            first = ids[mask][0].item()
            if first == tokenizer.convert_tokens_to_ids("t6"):
                logits[index, 1] = 5
            elif first == tokenizer.convert_tokens_to_ids("t9"):
                logits[index, 2] = 7
            else:
                logits[index, 0] = 1
        return logits

    monkeypatch.setattr(model, "forward", chunk_logits)
    sentences = ["t0 t1 t2 t3 t4 t5 t6", "t9"]
    # Equal averaging would incorrectly select Argument (5/3 > 2/3).
    assert model.predict(sentences, batch_size=2) == [ROLES[0], ROLES[2]]
    assert batch_sizes == [2, 2]
    assert model.training
    model.eval()
    assert model.predict(sentences, batch_size=8) == [ROLES[0], ROLES[2]]
    assert not model.training
    assert model.predict([]) == []


def test_checkpoint_roundtrip_uses_full_weights_and_preserves_rng(tokenizer, tmp_path, monkeypatch):
    model = make_model(tokenizer, max_length=12, target_chunk_size=4, dropout=0.5)
    sentences = ["t1 t2", "t3 t4 t5 t6 t7"]
    expected = model.predict(sentences)
    path = tmp_path / "nested" / "model.pt"
    model.save(path, {"seed": 42, "roles": list(ROLES), "epoch": 3})

    def no_pretrained_weights(*args, **kwargs):
        pytest.fail("Loading trained weights must not download the pretrained weights")

    monkeypatch.setattr(module.AutoModel, "from_pretrained", no_pretrained_weights)
    before = torch.get_rng_state().clone()
    restored = load_transformer_labeller(path, tokenizer=tokenizer)
    assert torch.equal(torch.get_rng_state(), before)
    assert restored.predict(sentences) == expected
    assert restored.metadata == {"seed": 42, "roles": list(ROLES), "epoch": 3}
    assert not restored.training
    assert restored.config == model.config
    assert restored.encoder_revision == module.MODEL_REVISIONS["inlegalbert"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


@pytest.mark.parametrize("field,value", [
    ("roles", list(reversed(ROLES))), ("encoder_revision", "main"),
    ("encoder_model", "an/unpinned-model"), ("format", "unknown"),
])
def test_checkpoint_rejects_changed_identity_before_loading(tokenizer, tmp_path, field, value):
    model = make_model(tokenizer, max_length=12, target_chunk_size=4)
    path = tmp_path / "model.pt"
    model.save(path)
    payload = torch.load(path, weights_only=True)
    payload[field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_transformer_labeller(path, tokenizer=tokenizer)


def test_failed_save_keeps_existing_checkpoint(tokenizer, tmp_path, monkeypatch):
    model = make_model(tokenizer, max_length=12, target_chunk_size=4)
    path = tmp_path / "model.pt"
    model.save(path)
    previous = path.read_bytes()

    def interrupted(payload, temporary):
        temporary.write_bytes(b"partial")
        raise OSError("interrupted")

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        model.save(path)
    assert path.read_bytes() == previous
    assert not [candidate for candidate in tmp_path.iterdir() if candidate.name not in ("model.pt", "vocab.txt")]


@pytest.mark.parametrize("encoder", ["inlegalbert", "deberta"])
def test_production_tokenizer_uses_pinned_slow_tokenizer(tokenizer, monkeypatch, encoder):
    observed = []

    def from_pretrained(name, **kwargs):
        observed.append((name, kwargs))
        return tokenizer

    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", from_pretrained)
    model = TransformerRoleLabeller(encoder, backbone=tiny_backbone(encoder))
    assert model.tokenizer is tokenizer
    assert observed == [(module.MODEL_IDS[encoder], {
        "revision": module.MODEL_REVISIONS[encoder], "cache_dir": None,
        "use_fast": False, "trust_remote_code": False,
    })]


def test_invalid_configuration_and_masks_fail_clearly(tokenizer):
    for kwargs in ({"max_length": 513}, {"target_chunk_size": 385},
                   {"max_length": 10, "target_chunk_size": 8},
                   {"batch_size": 0}, {"dropout": 1}):
        with pytest.raises(ValueError):
            make_model(tokenizer, **kwargs)
    model = make_model(tokenizer, max_length=12, target_chunk_size=4)
    with pytest.raises(ValueError, match="align"):
        model.make_examples(["t1"], [])
    with pytest.raises(ValueError, match="canonical"):
        model.make_examples(["t1"], ["fact"])
    with pytest.raises(ValueError, match="empty"):
        model.collate([])
    batch = model.collate(model.make_examples(["t1"]))
    batch["target_mask"].zero_()
    with pytest.raises(ValueError, match="target mask"):
        model(batch)
