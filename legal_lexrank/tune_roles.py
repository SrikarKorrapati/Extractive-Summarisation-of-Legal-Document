"""Select a pretrained-vector role labeller using the eight validation judgments."""

import argparse
from collections import Counter
import hashlib
from itertools import groupby
import json
from pathlib import Path
import random

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import torch

from . import encoders
from .contextual_labeller import ContextualRoleLabeller
from .data import ROLES, load_cases, split_case_ids
from .storage import atomic_write, json_bytes


SEED = 42
CONFIGURATIONS = (
    {"hidden_size": 64, "dropout": 0.3, "learning_rate": 0.001, "auxiliary_strength": 0.3},
    {"hidden_size": 128, "dropout": 0.5, "learning_rate": 0.001, "auxiliary_strength": 0.3},
    {"hidden_size": 64, "dropout": 0.5, "learning_rate": 0.003, "auxiliary_strength": 0.3},
    {"hidden_size": 128, "dropout": 0.3, "learning_rate": 0.001, "auxiliary_strength": 0.0},
)


def load_features(data, features):
    """Verify a training-only feature bundle against the local annotated corpus."""
    features = Path(features)
    manifest = json.loads((features / "manifest.json").read_text(encoding="utf-8"))
    cases = load_cases(Path(data))
    split = split_case_ids(cases, SEED)
    expected = {
        "format": "legal-lexrank-role-features-v1", "seed": SEED,
        "case_ids": split, "included_case_ids": split["train"],
        "include_test": False, "roles": list(ROLES), "dtype": "float32", "normalization": "l2",
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"Training feature manifest mismatch: {field}")
    backend = manifest["backend"]
    if (backend not in ("sbert", "legalbert", "openai")
            or manifest["model"] != encoders.MODEL_IDS[backend]
            or manifest["revision"] != encoders.MODEL_REVISIONS.get(backend)):
        raise ValueError("Training feature encoder identity mismatch")
    feature_file = features / "features.npz"
    if hashlib.sha256(feature_file.read_bytes()).hexdigest() != manifest["features_sha256"]:
        raise ValueError("Training feature file checksum mismatch")
    if set(manifest["cases"]) != set(split["train"]):
        raise ValueError("Training feature metadata must contain exactly the training cases")
    vectors, labels = {}, {}
    digest = hashlib.sha256()
    with np.load(feature_file, allow_pickle=False) as arrays:
        if len(arrays.files) != len(split["train"]) or set(arrays.files) != set(split["train"]):
            raise ValueError("Training feature arrays must contain exactly the training cases")
        for case_id in split["train"]:
            case = cases[case_id]
            annotations = json_bytes([case_id, case.labelled_sentences, case.labels])
            digest.update(annotations)
            metadata = manifest["cases"][case_id]
            gold = [ROLES.index(role) if role is not None else -1 for role in case.labels]
            if (metadata["annotation_sha256"] != hashlib.sha256(annotations).hexdigest()
                    or metadata["text_sha256"] != hashlib.sha256(json_bytes(case.labelled_sentences)).hexdigest()
                    or metadata["sentence_count"] != len(gold) or metadata["labels"] != gold):
                raise ValueError(f"Training feature annotations mismatch: {case_id}")
            values = arrays[case_id]
            if (values.dtype != np.float32 or values.shape != (len(gold), manifest["dimensions"])
                    or values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all()
                    or not np.allclose(np.linalg.norm(values, axis=1), 1, rtol=1e-4, atol=1e-6)):
                raise ValueError(f"Invalid normalized training features: {case_id}")
            vectors[case_id] = torch.from_numpy(values.copy())
            labels[case_id] = torch.tensor(gold, dtype=torch.long)
    if (digest.hexdigest() != manifest["training_sha256"]
            or digest.hexdigest() != manifest["corpus_sha256"]):
        raise ValueError("Training corpus checksum mismatch")
    return manifest, vectors, labels


def training_sequences(vectors, labels, ids):
    """Split sequences at annotation gaps without joining unrelated sentences."""
    sequences = []
    for case_id in ids:
        before = len(sequences)
        indices = range(len(labels[case_id]))
        for annotated, span in groupby(indices, key=lambda i: int(labels[case_id][i]) >= 0):
            if annotated:
                positions = list(span)
                start, stop = positions[0], positions[-1] + 1
                sequences.append((vectors[case_id][start:stop], labels[case_id][start:stop]))
        if len(sequences) == before:
            raise ValueError(f"Training case has no role annotations: {case_id}")
    return sequences


def class_weights(labels, ids):
    """Bound inverse-square-root class frequencies learned from the fitting cases."""
    counts = Counter(int(role) for case_id in ids for role in labels[case_id] if int(role) >= 0)
    frequencies = torch.tensor([counts[index] for index in range(len(ROLES))], dtype=torch.float32)
    weights = torch.sqrt(frequencies.sum() / (len(ROLES) * frequencies.clamp_min(1)))
    weights.clamp_(max=4)
    return weights / weights.mean()


def evaluate(model, vectors, labels, ids):
    gold, predicted = [], []
    for case_id in ids:
        guesses = model.predict_vectors(vectors[case_id])
        for expected, guess in zip(labels[case_id].tolist(), guesses, strict=True):
            if expected >= 0:
                gold.append(ROLES[expected])
                predicted.append(guess)
    if not gold:
        raise ValueError("Validation needs annotated sentences")
    report = classification_report(gold, predicted, labels=ROLES, output_dict=True, zero_division=0)
    report["accuracy"] = float(accuracy_score(gold, predicted))
    return report


def score_key(report):
    return report["accuracy"], report["macro avg"]["f1-score"]


def run(data, features, output, max_epochs=100, patience=20):
    if max_epochs < 1 or patience < 1:
        raise ValueError("Maximum epochs and patience must be positive")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Choose a new or empty tuning directory: {output}")
    manifest, vectors, labels = load_features(data, features)
    fit_ids = manifest["case_ids"]["fit"]
    validation_ids = manifest["case_ids"]["validation"]
    sequences = training_sequences(vectors, labels, fit_ids)
    weights = class_weights(labels, fit_ids)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    for number, config in enumerate(CONFIGURATIONS, 1):
        candidate = output / f"candidate-{number}"
        candidate.mkdir()
        checkpoint = candidate / "checkpoint.pt"
        history, best, best_epoch = [], None, 0
        rng = random.Random(SEED)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(SEED)
            model = ContextualRoleLabeller(
                manifest["dimensions"], ROLES, manifest["backend"],
                hidden_size=config["hidden_size"], dropout=config["dropout"],
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
            order = list(range(len(sequences)))
            for epoch in range(1, max_epochs + 1):
                model.train()
                rng.shuffle(order)
                losses = []
                for index in order:
                    values, gold = sequences[index]
                    optimizer.zero_grad(set_to_none=True)
                    loss = model.loss(values, gold, weights, config["auxiliary_strength"])
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite training loss in candidate {number}, epoch {epoch}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                report = evaluate(model, vectors, labels, validation_ids)
                if best is None or score_key(report) > score_key(best):
                    best, best_epoch = report, epoch
                    model.save(checkpoint, metadata={
                        "seed": SEED, "epoch": epoch, "config": config,
                        "fit_case_ids": fit_ids, "validation_case_ids": validation_ids,
                        "features_sha256": manifest["features_sha256"], "validation": report,
                    })
                history.append({"epoch": epoch, "training_loss": sum(losses) / len(losses),
                                "validation": report, "best_epoch": best_epoch})
                atomic_write(candidate / "history.json", json_bytes(history))
                if epoch == 1 or epoch % 5 == 0:
                    print(f"candidate {number}, epoch {epoch}: accuracy {report['accuracy']:.4f}, "
                          f"macro F1 {report['macro avg']['f1-score']:.4f}, "
                          f"lower-court recall {report['Ruling by Lower Court']['recall']:.4f}, "
                          f"best epoch {best_epoch}", flush=True)
                if epoch - best_epoch >= patience:
                    break
        candidates.append({
            "candidate": number, "config": config, "best_epoch": best_epoch,
            "epochs_completed": len(history), "validation": best,
            "checkpoint": str(checkpoint.relative_to(output)),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        })
    chosen = max(candidates, key=lambda item: score_key(item["validation"]))
    atomic_write(output / "validation_labeller.pt", (output / chosen["checkpoint"]).read_bytes())
    selection = {
        "format": "contextual-role-selection-v1", "seed": SEED,
        "selection_metric": "validation accuracy, then macro F1; earliest epoch and candidate break ties",
        "max_epochs": max_epochs, "patience": patience,
        "feature_manifest": manifest, "class_weights": weights.tolist(),
        "candidates": candidates, "selected": chosen,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    atomic_write(output / "selection.json", json_bytes(selection))
    print(f"Selected candidate {chosen['candidate']} at epoch {chosen['best_epoch']}: "
          f"validation accuracy {chosen['validation']['accuracy']:.4f}", flush=True)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("."))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    run(args.data, args.features, args.output, args.max_epochs, args.patience)


if __name__ == "__main__":
    main()
