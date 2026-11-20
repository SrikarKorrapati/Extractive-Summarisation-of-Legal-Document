import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from legal_lexrank import ablate_roles, final_roles, role_features, role_study, tune_roles
from legal_lexrank.contextual_labeller import ContextualRoleLabeller, load_contextual_labeller
from legal_lexrank.data import ROLES, split_case_ids


def test_mean_uses_fitting_cases_only():
    vectors = {"fit-a": torch.tensor([[1., 0.], [0., 1.]]),
               "fit-b": torch.tensor([[0., 1.]]), "validation": torch.tensor([[-1., 0.]])}
    config = {"hidden_size": 3, "dropout": 0.3, "center_inputs": True}
    model = ablate_roles.initialize_model({"dimensions": 2, "backend": "sbert"}, config,
                                          vectors, ["fit-a", "fit-b"])
    torch.testing.assert_close(model.feature_mean, torch.tensor([1 / 3, 2 / 3]))


def test_four_document_batch_matches_sentence_weighted_gradient_with_annotation_gap():
    model = ContextualRoleLabeller(3, ["fact", "decision"], "sbert", hidden_size=3, dropout=0)
    expected = copy.deepcopy(model)
    vectors = {"a": torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
               "b": torch.tensor([[0., 1., 0.]]), "c": torch.tensor([[0., 0., 1.]]),
               "d": torch.tensor([[1., 0., 0.], [0., 0., 1.]])}
    labels = {"a": torch.tensor([0, -1, 1]), "b": torch.tensor([1]),
              "c": torch.tensor([0]), "d": torch.tensor([0, 1])}
    order = ["a", "b", "c", "d"]
    sequences = tune_roles.training_sequences(vectors, labels, order)
    assert len(sequences) == 5
    weights = torch.tensor([1., 2.])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(expected.parameters(), lr=0.1)
    total = sum(len(gold) for _, gold in sequences)
    reference_loss = sum(expected.loss(values, gold, weights) * len(gold) / total
                         for values, gold in sequences)
    reference_loss.backward()
    torch.nn.utils.clip_grad_norm_(expected.parameters(), 5.0)
    reference_optimizer.step()
    actual_loss = ablate_roles.fit_epoch(model, vectors, labels, order, optimizer, weights,
                                         {"batch_documents": 4, "auxiliary_strength": 0.3})
    assert actual_loss == pytest.approx(float(reference_loss.detach()))
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, expected.state_dict()[name], atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("scores, selected", [([0.2, 0.2, 0.2], 1), ([0.9, 0.3, 0.4], 2)])
def test_ablation_preserves_baseline_and_refits_selected_center_from_all_training(tmp_path, monkeypatch, scores, selected):
    cases = {
        f"case{i}": SimpleNamespace(
            labelled_sentences=[f"case{i} sentence {j}" for j in range(8)],
            labels=[*ROLES[:3], None, *ROLES[3:]],
        )
        for i in range(6)
    }
    split = split_case_ids(cases)
    for module in (role_features, tune_roles, final_roles):
        monkeypatch.setattr(module, "load_cases", lambda root: cases)

    def embed(backend, sentences, cache):
        seed = int(sentences[0].split()[0].removeprefix("case"))
        return np.random.default_rng(seed).normal(size=(8, 4)).astype(np.float32)

    monkeypatch.setattr(role_features.encoders, "sentence_embeddings", embed)
    monkeypatch.setattr(tune_roles, "CONFIGURATIONS", [dict(tune_roles.CONFIGURATIONS[0], hidden_size=3)])
    features = role_features.build(tmp_path, tmp_path / "features", "sbert", tmp_path / "cache")
    metrics = iter([0.2, *scores])
    evaluated = []

    def evaluate(model, vectors, labels, ids):
        evaluated.append(list(ids))
        value = next(metrics)
        return {"accuracy": value, "macro avg": {"f1-score": value / 2},
                "Ruling by Lower Court": {"recall": 0.1}}

    monkeypatch.setattr(tune_roles, "evaluate", evaluate)
    baseline = tmp_path / "baseline"
    original = tune_roles.run(tmp_path, features, baseline, max_epochs=1, patience=1)
    output = tmp_path / "ablation"
    result = ablate_roles.run(tmp_path, features, baseline / "selection.json", output, max_epochs=1, patience=1)
    assert evaluated == [split["validation"]] * 4
    assert len(result["candidates"]) == 4
    assert result["selected"]["candidate"] == selected
    assert result["candidates"][0]["validation"] == original["selected"]["validation"]
    assert (output / "candidate-1/checkpoint.pt").read_bytes() == (baseline / "validation_labeller.pt").read_bytes()
    manifest, vectors, _ = tune_roles.load_features(tmp_path, features)
    final_roles.validate_selection(manifest, output / "selection.json")
    final = tmp_path / "final"
    protocol = final_roles.run(tmp_path, features, output / "selection.json", final)
    assert len(evaluated) == 4
    assert protocol["selection_runner"] == "ablate_roles"
    assert role_study.verified_checkpoints(final, cases, split, 42, 8) == protocol["checkpoint_sha256"]
    restored = load_contextual_labeller(final / "role_labeller.pt")
    if selected == 2:
        fit_mean = torch.cat([vectors[key] for key in split["fit"]]).mean(0)
        train_mean = torch.cat([vectors[key] for key in split["train"]]).mean(0)
        validation_model = load_contextual_labeller(final / "validation_labeller.pt")
        torch.testing.assert_close(validation_model.feature_mean, fit_mean)
        torch.testing.assert_close(restored.feature_mean, train_mean)
        assert not torch.allclose(fit_mean, train_mean)
    else:
        assert "feature_mean" not in restored.state_dict()
    with pytest.raises(ValueError, match="empty ablation directory"):
        ablate_roles.run(tmp_path, features, baseline / "selection.json", output)
