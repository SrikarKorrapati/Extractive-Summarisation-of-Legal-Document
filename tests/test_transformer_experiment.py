import csv
from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest

from legal_lexrank import experiment
from legal_lexrank.data import Case, ROLES, split_case_ids
from legal_lexrank.ranking import DEFAULT_ROLE_WEIGHTS


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    cases = {}
    for index in range(50):
        case_id = f"case-{index:02}"
        sentences = [f"{case_id} judgment sentence {number}" for number in range(7)]
        annotations = [f"{case_id} annotation sentence {number}" for number in range(7)]
        cases[case_id] = Case(case_id, sentences, annotations, list(ROLES), [sentences[0], sentences[-1]])
    split = split_case_ids(cases)
    role_run = tmp_path / "deployment"
    role_run.mkdir()
    checkpoint = role_run / "role_labeller.pt"
    checkpoint.write_bytes(b"synthetic final model")
    config = {
        "encoder": "inlegalbert", "encoder_model": "law-ai/InLegalBERT",
        "encoder_revision": "b5ecfed8ed6cf9d25a3cb8225a8c52f161f7401a",
        "max_length": 512, "target_chunk_size": 384, "dropout": 0.1,
        "microbatch_size": 4, "gradient_accumulation_steps": 8,
        "encoder_learning_rate": 2e-5, "head_learning_rate": 1e-4,
    }
    selected_metrics = {"accuracy": 0.875, "macro avg": {"f1-score": 0.71},
                        ROLES[6]: {"precision": 0.6, "recall": 0.5, "f1-score": 0.54545, "support": 8}}
    protocol = {
        "format": "legal-lexrank-role-study-v4", "seed": 42, "epochs": 3,
        "architecture": "contextual-transformer-role-classifier", "config": config,
        "case_ids": split, "validation_metrics": selected_metrics,
        "checkpoint_sha256": {"test": "verified-final-checkpoint"},
    }
    (role_run / "protocol.json").write_text(json.dumps(protocol))
    calls = {"verify": [], "load": [], "predict": [], "similarity": [], "device": []}

    class FinalModel:
        def to(self, device):
            calls["device"].append(device)
            return self

        def predict(self, sentences):
            assert calls["device"], "Move the classifier before inference"
            calls["predict"].append(list(sentences))
            return [ROLES[int(sentence.rsplit(" ", 1)[1])] for sentence in sentences]

    def verify(directory, provided_cases, provided_split, seed, epochs):
        calls["verify"].append((directory, provided_split, seed, epochs))
        assert provided_cases is cases
        return protocol["checkpoint_sha256"]

    def load(path, cache_dir=None):
        calls["load"].append((path, cache_dir))
        assert path == checkpoint
        return FinalModel()

    def similarities(backend, sentences, cache):
        calls["similarity"].append((backend, list(sentences)))
        return np.eye(len(sentences), dtype=np.float32)

    def no_training(*args, **kwargs):
        pytest.fail("A deployment benchmark must never fit a model")

    original_read_bytes = Path.read_bytes

    def no_checkpoint_copy(path):
        if path == checkpoint:
            pytest.fail("The benchmark must not read/copy the large checkpoint bytes")
        return original_read_bytes(path)

    monkeypatch.setattr(experiment, "load_cases", lambda root: cases)
    monkeypatch.setattr(experiment, "verified_checkpoints", verify)
    monkeypatch.setattr(experiment, "load_labeller", load)
    monkeypatch.setattr(experiment, "similarity_matrix", similarities)
    monkeypatch.setattr(experiment, "trained_model", no_training)
    monkeypatch.setattr(Path, "read_bytes", no_checkpoint_copy)
    return role_run, cases, split, protocol, calls


def test_v4_scores_only_reserved_cases_for_all_encoders_and_uses_saved_validation_metrics(tmp_path, deployment):
    role_run, cases, split, role_protocol, calls = deployment
    output, cache = tmp_path / "benchmark", tmp_path / "cache"
    experiment.run(tmp_path / "corpus", output, cache, list(experiment.BACKENDS), role_run=role_run)

    assert not (role_run / "validation_labeller.pt").exists()
    assert not (output / "role_labeller.pt").exists()
    assert calls["verify"] == [(role_run, split, 42, experiment.EPOCHS)]
    assert calls["load"] == [(role_run / "role_labeller.pt", cache / "similarities")]
    assert calls["device"] == ["cpu"]
    expected_prediction_calls = ([cases[key].sentences for key in split["test"]]
                                 + [cases[key].labelled_sentences for key in split["test"]])
    assert calls["predict"] == expected_prediction_calls
    assert calls["similarity"] == [(backend, cases[key].sentences)
                                    for backend in experiment.BACKENDS for key in split["test"]]

    for backend in experiment.BACKENDS:
        report = json.loads((output / f"{backend}.json").read_text())
        assert report["validation"] == []
        assert report["model"] == experiment.MODEL_IDS[backend]
        assert sorted((row["case"], row["weighting"], row["ratio"]) for row in report["test"]) == sorted(
            product(split["test"], ("fixed", "unweighted"), (0.3, 0.4, 0.5)))
        assert len(report["test"]) == 60
    with (output / "metrics.csv").open() as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == len(experiment.BACKENDS) * 6
    assert {row["phase"] for row in table} == {"test"}
    assert {int(row["cases"]) for row in table} == {10}
    metrics = json.loads((output / "role_metrics.json").read_text())
    assert metrics["validation"] == role_protocol["validation_metrics"]
    assert metrics["test"]["accuracy"] == 1
    assert metrics["test"]["macro avg"]["support"] == 70
    settings = json.loads((output / "protocol.json").read_text())
    assert settings["role_checkpoint_sha256"] == {"test": "verified-final-checkpoint"}
    assert settings["role_config"] == role_protocol["config"]
    assert settings["role_encoder"] == {
        "backend": "inlegalbert", "model": role_protocol["config"]["encoder_model"],
        "revision": role_protocol["config"]["encoder_revision"],
    }
    assert settings["role_architecture"] == role_protocol["architecture"]
    assert settings["role_epochs"] == 3
    assert settings["role_weights"] == DEFAULT_ROLE_WEIGHTS
    assert settings["evaluation_phases"] == ["test"]
    assert "32-case" in settings["role_validation_metrics_source"]


def test_v4_rejects_unverified_deployment_before_inference(tmp_path, deployment, monkeypatch):
    role_run, _, _, _, calls = deployment

    def reject(*args):
        raise ValueError("Transformer checkpoint metadata mismatch")

    monkeypatch.setattr(experiment, "verified_checkpoints", reject)
    output = tmp_path / "benchmark"
    with pytest.raises(ValueError, match="metadata mismatch"):
        experiment.run(tmp_path, output, tmp_path / "cache", ["tfidf"], role_run=role_run)
    assert calls["load"] == calls["predict"] == calls["similarity"] == []
    assert not (output / "protocol.json").exists()


def test_v4_existing_benchmark_rejects_changed_role_configuration(tmp_path, deployment):
    role_run, _, _, role_protocol, calls = deployment
    output, cache = tmp_path / "benchmark", tmp_path / "cache"
    experiment.run(tmp_path, output, cache, ["tfidf"], role_run=role_run)
    old_report = (output / "tfidf.json").read_bytes()
    old_load_count = len(calls["load"])
    role_protocol["config"]["encoder_learning_rate"] = 1e-5
    (role_run / "protocol.json").write_text(json.dumps(role_protocol))
    with pytest.raises(ValueError, match="different protocol"):
        experiment.run(tmp_path, output, cache, ["tfidf"], role_run=role_run)
    assert len(calls["load"]) == old_load_count
    assert (output / "tfidf.json").read_bytes() == old_report


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_v4_classifier_moves_to_requested_device_before_inference(tmp_path, deployment, device):
    role_run, _, _, _, calls = deployment
    experiment.run(tmp_path, tmp_path / "benchmark", tmp_path / "cache", ["tfidf"],
                   role_run=role_run, device=device)
    assert calls["device"] == [device]
    assert len(calls["predict"]) == 20


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_legacy_bundle_rejects_non_cpu_device_before_loading(tmp_path, deployment, version, device):
    role_run, _, _, protocol, calls = deployment
    protocol["format"] = f"legal-lexrank-role-study-v{version}"
    (role_run / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="legacy models support CPU only"):
        experiment.run(tmp_path, tmp_path / "benchmark", tmp_path / "cache", ["tfidf"],
                       role_run=role_run, device=device)
    assert calls["verify"] == calls["load"] == calls["device"] == calls["predict"] == []


def test_implicit_legacy_training_rejects_non_cpu_device(tmp_path, deployment):
    _, _, _, _, calls = deployment
    with pytest.raises(ValueError, match="requires a transformer v4"):
        experiment.run(tmp_path, tmp_path / "benchmark", tmp_path / "cache", ["tfidf"], device="cuda")
    assert calls["load"] == calls["predict"] == []


def test_cli_passes_selected_device(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(experiment, "run", lambda *args: calls.append(args))
    monkeypatch.setattr("sys.argv", ["experiment", "--role-run", str(tmp_path), "--device", "cuda"])
    experiment.main()
    assert calls[0][-2:] == (tmp_path, "cuda")
