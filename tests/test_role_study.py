import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from legal_lexrank import experiment, role_study
from legal_lexrank.data import ROLES, load_cases, split_case_ids


def test_unlabelled_span_breaks_crf_transitions_and_does_not_hide_accuracy():
    case = SimpleNamespace(labelled_sentences=["a", "b", "c", "d"], labels=["Facts", None, "Argument", "Facts"])
    assert role_study.training_pairs({"x": case}, ["x"]) == [
        (("a",), ("Facts",)), (("c", "d"), ("Argument", "Facts")),
    ]
    model = SimpleNamespace(predict=lambda s: ["Facts"] * len(s))
    report = role_study.role_accuracy(model, {"x": case}, ["x"])
    assert report["accuracy"] == pytest.approx(2 / 3)


def test_colab_bundle_roundtrip_and_rejection(tmp_path, monkeypatch):
    root = tmp_path / "corpus"
    for folder in ("Judgement", "Labelled_Data", "Summary", "Summary_2"):
        (root / folder).mkdir(parents=True)
        for i in range(10):
            lines = [f"Case {i} sentence {j}" for j in range(7)]
            if folder == "Labelled_Data":
                lines = [f"{s}\t{label}" for s, label in zip(lines, ROLES, strict=True)]
            (root / folder / f"case{i}.txt").write_text("\n".join(lines), encoding="utf-8")

    output = tmp_path / "role-run"
    output.mkdir()
    role_study.run(root, output, seed=42, epochs=1)
    with pytest.raises(ValueError, match="empty directory"):
        role_study.run(root, output, seed=42, epochs=1)
    manifest = json.loads((output / "protocol.json").read_text())
    cases = load_cases(root)
    split = split_case_ids(cases, 42)
    assert manifest["case_ids"] == split
    assert role_study.verified_checkpoints(output, cases, split, 42, 1) == manifest["checkpoint_sha256"]
    assert manifest["checkpoint_sha256"]["validation"] != manifest["checkpoint_sha256"]["test"]

    monkeypatch.setattr(experiment, "EPOCHS", 1)
    def reject_local_fit(*args, **kwargs):
        pytest.fail("Experiment retrained despite an imported checked role bundle")
    monkeypatch.setattr(experiment, "trained_model", reject_local_fit)
    loaded = []
    load = experiment.load_labeller
    def observe_load(path, **kwargs):
        loaded.append(path.name)
        return load(path, **kwargs)
    monkeypatch.setattr(experiment, "load_labeller", observe_load)
    benchmark = tmp_path / "benchmark"
    experiment.run(root, benchmark, tmp_path / "cache", ["tfidf"], role_run=output)
    assert loaded == ["validation_labeller.pt", "role_labeller.pt"]
    assert (benchmark / "role_metrics.json").read_bytes() == (output / "role_metrics.json").read_bytes()
    assert (benchmark / "role_labeller.pt").read_bytes() == (output / "role_labeller.pt").read_bytes()
    assert json.loads((benchmark / "protocol.json").read_text())["role_checkpoint_sha256"] == manifest["checkpoint_sha256"]

    # Altering annotations on a held-out case cannot invalidate the fitted weights.
    cases[split["test"][0]].labels[0] = "Statute"
    role_study.verified_checkpoints(output, cases, split, 42, 1)
    cases[split["fit"][0]].labels[0] = "Statute"
    with pytest.raises(ValueError, match="training_sha256"):
        role_study.verified_checkpoints(output, cases, split, 42, 1)
    cases = load_cases(root)
    with pytest.raises(ValueError, match="case_ids"):
        role_study.verified_checkpoints(output, cases, split_case_ids(cases, 9), 9, 1)
    path = output / "validation_labeller.pt"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["checkpoint_sha256"]["validation"]
    path.write_bytes((output / "role_labeller.pt").read_bytes())
    with pytest.raises(ValueError, match="validation_labeller.pt"):
        role_study.verified_checkpoints(output, cases, split, 42, 1)


def test_contextual_bundle_uses_selected_epochs_and_shared_prediction_cache(tmp_path, monkeypatch):
    from legal_lexrank import encoders
    from legal_lexrank.contextual_labeller import ContextualRoleLabeller
    from legal_lexrank.labeller import load_labeller

    root = tmp_path / "corpus"
    for folder in ("Judgement", "Labelled_Data", "Summary", "Summary_2"):
        (root / folder).mkdir(parents=True)
        for i in range(10):
            lines = [f"Case {i} sentence {j}" for j in range(7)]
            if folder == "Labelled_Data":
                lines = [f"{text}\t{role}" for text, role in zip(lines, ROLES, strict=True)]
            (root / folder / f"case{i}.txt").write_text("\n".join(lines))
    cases = load_cases(root)
    split = split_case_ids(cases)
    output = tmp_path / "contextual-roles"
    output.mkdir()
    config = {"hidden_size": 4, "dropout": 0.0, "learning_rate": 0.001, "auxiliary_strength": 0.3}
    manifest = role_study.study_settings(cases, split, 42, 12)
    manifest.update({
        "format": "legal-lexrank-role-study-v2", "config": config,
        "features_sha256": "f" * 64, "selection_sha256": "a" * 64,
        "feature_encoder": {"backend": "sbert", "model": encoders.MODEL_IDS["sbert"],
                            "revision": encoders.MODEL_REVISIONS["sbert"],
                            "dimensions": 4, "normalization": "l2"},
        "validation_metrics": {"accuracy": 0.0},
    })
    for field, module in (
        ("contextual_labeller_sha256", "contextual_labeller"), ("encoder_sha256", "encoders"),
        ("tuning_sha256", "tune_roles"), ("final_training_sha256", "final_roles"),
    ):
        manifest[field] = hashlib.sha256(Path(role_study.__file__).with_name(module + ".py").read_bytes()).hexdigest()
    hashes = {}
    for phase, (_, filename) in role_study.STAGES.items():
        model = ContextualRoleLabeller(4, ROLES, "sbert", hidden_size=4, dropout=0.0)
        metadata = {"seed": 42, "epoch": 12, "config": config, "features_sha256": "f" * 64}
        if phase == "validation":
            metadata.update({"fit_case_ids": split["fit"], "validation_case_ids": split["validation"],
                             "validation": manifest["validation_metrics"]})
        else:
            metadata.update({"train_case_ids": split["train"], "selection_sha256": "a" * 64})
        model.save(output / filename, metadata=metadata)
        hashes[phase] = hashlib.sha256((output / filename).read_bytes()).hexdigest()
    manifest["checkpoint_sha256"] = hashes
    manifest_path = output / "protocol.json"
    manifest_path.write_text(json.dumps(manifest))
    assert role_study.verified_checkpoints(output, cases, split, 42, 8) == hashes
    cache = tmp_path / "cache"
    restored = load_labeller(output / "role_labeller.pt", cache_dir=cache / "similarities")
    assert isinstance(restored, ContextualRoleLabeller)
    assert restored.cache_dir == cache / "similarities"
    requested_caches = []

    def embed(backend, sentences, cache_dir):
        requested_caches.append(cache_dir)
        return np.tile(np.array([[1., 0., 0., 0.]], dtype=np.float32), (len(sentences), 1))

    monkeypatch.setattr(encoders, "sentence_embeddings", embed)
    benchmark = tmp_path / "benchmark"
    experiment.run(root, benchmark, cache, ["tfidf"], role_run=output)
    settings = json.loads((benchmark / "protocol.json").read_text())
    assert settings["role_epochs"] == 12
    assert settings["role_encoder"] == manifest["feature_encoder"]
    assert requested_caches and set(requested_caches) == {cache / "similarities"}
    for field, value, message in (
        ("epochs", 13, "checkpoint metadata"),
        ("contextual_labeller_sha256", "changed", "contextual_labeller_sha256"),
        ("training_sha256", {}, "training_sha256"),
    ):
        changed = dict(manifest, **{field: value})
        manifest_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=message):
            role_study.verified_checkpoints(output, cases, split, 42, 8)
    changed = dict(manifest, feature_encoder={**manifest["feature_encoder"], "model": "changed"})
    manifest_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="feature encoder"):
        role_study.verified_checkpoints(output, cases, split, 42, 8)
