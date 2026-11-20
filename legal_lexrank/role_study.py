"""Train and validate independent rhetorical-role checkpoints on an 80/20 case split."""

import argparse
import hashlib
import json
import os
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from sklearn.metrics import accuracy_score, classification_report

from .data import ROLES, load_cases, split_case_ids
from .labeller import fit_labeller, load_labeller

EPOCHS = 8
STAGES = {
    "validation": ("fit", "validation_labeller.pt"),
    "test": ("train", "role_labeller.pt"),
}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def training_pairs(cases, ids):
    """Keep each CRF sequence within one judgment and one contiguous labelled span."""
    documents = []
    for case_id in ids:
        case = cases[case_id]
        rows = zip(case.labelled_sentences, case.labels, strict=True)
        start_count = len(documents)
        for labelled, span in groupby(rows, key=lambda row: row[1] is not None):
            if labelled:
                sentences, labels = zip(*span)
                documents.append((sentences, labels))
        if len(documents) == start_count:
            raise ValueError(f"No labelled sentences in {case_id}")
    return documents


def role_accuracy(model, cases, ids):
    gold, predicted = [], []
    for case_id in ids:
        case = cases[case_id]
        guesses = model.predict(case.labelled_sentences)
        for expected, guess in zip(case.labels, guesses, strict=True):
            if expected is not None:
                gold.append(expected)
                predicted.append(guess)
    if not gold:
        raise ValueError("Role evaluation needs at least one gold annotation")
    report = classification_report(gold, predicted, labels=ROLES, output_dict=True, zero_division=0)
    # Some sklearn versions omit accuracy when the split lacks one of the seven roles.
    report["accuracy"] = float(accuracy_score(gold, predicted))
    return report


def study_settings(cases, split, seed, epochs):
    training_hashes = {}
    for phase, (train_key, _) in STAGES.items():
        digest = hashlib.sha256()
        for case_id in split[train_key]:
            case = cases[case_id]
            digest.update(json_bytes([case_id, case.labelled_sentences, case.labels]))
        training_hashes[phase] = digest.hexdigest()
    return {
        "format": "legal-lexrank-role-study-v1",
        "case_ids": split, "seed": seed, "epochs": epochs, "roles": list(ROLES),
        "training_sha256": training_hashes,
        "labeller_sha256": hashlib.sha256(Path(__file__).with_name("labeller.py").read_bytes()).hexdigest(),
    }


def verified_checkpoints(directory, cases, split, seed, epochs):
    """Reject a mismatched split, training corpus, model implementation, or checkpoint."""
    manifest = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    for field, expected in study_settings(cases, split, seed, epochs).items():
        if manifest.get(field) != expected:
            raise ValueError(f"Role run is incompatible with this experiment: {field}")
    hashes = manifest.get("checkpoint_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(STAGES):
        raise ValueError("Role run is missing its validation or final checkpoint checksum")
    for phase, (_, filename) in STAGES.items():
        if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != hashes[phase]:
            raise ValueError(f"Role checkpoint checksum mismatch: {filename}")
    return hashes


def run(data, output, seed=42, epochs=EPOCHS):
    """Fit both stages from scratch and publish the complete bundle together."""
    cases = load_cases(Path(data))
    split = split_case_ids(cases, seed)
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Choose an empty directory for a fresh role study: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    manifest = study_settings(cases, split, seed, epochs)
    reports, hashes = {}, {}
    with TemporaryDirectory(dir=output.parent) as temp:
        bundle = Path(temp) / "bundle"
        bundle.mkdir()
        for phase, (train_key, filename) in STAGES.items():
            ids = split[train_key]
            checkpoint = bundle / filename
            print(f"Training {phase} labeller from scratch on {len(ids)} judgments", flush=True)
            fit_labeller(training_pairs(cases, ids), ROLES, seed, checkpoint, epochs=epochs)
            reports[phase] = role_accuracy(load_labeller(checkpoint), cases, split[phase])
            hashes[phase] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            print(f"{phase}: accuracy {reports[phase]['accuracy']:.4f}, "
                  f"macro F1 {reports[phase]['macro avg']['f1-score']:.4f}", flush=True)
        manifest["checkpoint_sha256"] = hashes
        (bundle / "protocol.json").write_bytes(json_bytes(manifest))
        (bundle / "role_metrics.json").write_bytes(json_bytes(reports))
        os.replace(bundle, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("."), help="Repository or corpus root")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    run(args.data, args.output, args.seed, args.epochs)


if __name__ == "__main__":
    main()
