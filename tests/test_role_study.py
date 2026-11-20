import hashlib
import json
from types import SimpleNamespace

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
    def observe_load(path):
        loaded.append(path.name)
        return load(path)
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
