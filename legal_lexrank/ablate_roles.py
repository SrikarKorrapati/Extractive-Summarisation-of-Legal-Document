"""Compare centering and four-document batches using training and validation cases."""

import argparse
import hashlib
from pathlib import Path
import random

import torch

from .contextual_labeller import ContextualRoleLabeller
from .data import ROLES
from .storage import atomic_write, json_bytes
from . import tune_roles


def initialize_model(manifest, config, vectors, fit_ids):
    """Build a fresh model and estimate optional centering from fitting cases only."""
    model = ContextualRoleLabeller(
        manifest["dimensions"], ROLES, manifest["backend"],
        hidden_size=config["hidden_size"], dropout=config["dropout"],
        center_inputs=config.get("center_inputs", False),
    )
    if config.get("center_inputs", False):
        count = sum(len(vectors[case_id]) for case_id in fit_ids)
        if count == 0:
            raise ValueError("Centering requires sentences in the fitting cases")
        total = sum(vectors[case_id].to(torch.float64).sum(dim=0) for case_id in fit_ids)
        model.feature_mean.copy_((total / count).to(torch.float32))
    return model


def fit_epoch(model, vectors, labels, order, optimizer, weights, config):
    """Keep each judgment's sequences intact while accumulating sentence-weighted loss."""
    batch_size = config.get("batch_documents", 1)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_documents must be a positive integer")
    model.train()
    total_loss, total_sentences = 0.0, 0
    for start in range(0, len(order), batch_size):
        sequences = tune_roles.training_sequences(vectors, labels, order[start:start + batch_size])
        batch_sentences = sum(len(gold) for _, gold in sequences)
        optimizer.zero_grad(set_to_none=True)
        for values, gold in sequences:
            loss = model.loss(values, gold, weights, config["auxiliary_strength"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            (loss * (len(gold) / batch_sentences)).backward()
            total_loss += float(loss.detach()) * len(gold)
            total_sentences += len(gold)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    if total_sentences == 0:
        raise ValueError("Training requires annotated sentences")
    return total_loss / total_sentences


def run(data, features, baseline_selection, output, max_epochs=100, patience=20):
    """Retain the completed baseline and train exactly three predefined ablations."""
    from .final_roles import validate_selection

    if max_epochs < 1 or patience < 1:
        raise ValueError("Maximum epochs and patience must be positive")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Choose a new or empty ablation directory: {output}")
    manifest, vectors, labels = tune_roles.load_features(data, features)
    baseline_path = Path(baseline_selection)
    baseline, checkpoint = validate_selection(manifest, baseline_path)
    if baseline.get("runner", "tune_roles") != "tune_roles":
        raise ValueError("Ablations require a completed original tuning selection")
    original = baseline["selected"]
    if original["config"].get("center_inputs", False) or original["config"].get("batch_documents", 1) != 1:
        raise ValueError("The baseline already uses an ablation setting")
    fit_ids = manifest["case_ids"]["fit"]
    validation_ids = manifest["case_ids"]["validation"]
    weights = tune_roles.class_weights(labels, fit_ids)
    output.mkdir(parents=True, exist_ok=True)
    baseline_checkpoint = output / "candidate-1" / "checkpoint.pt"
    atomic_write(baseline_checkpoint, checkpoint.read_bytes())
    candidates = [{**original, "candidate": 1, "name": "baseline",
                   "source_candidate": original["candidate"],
                   "checkpoint": str(baseline_checkpoint.relative_to(output))}]
    variants = (
        ("centered", {"center_inputs": True}),
        ("four-document batches", {"batch_documents": 4}),
        ("centered four-document batches", {"center_inputs": True, "batch_documents": 4}),
    )
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    for number, (name, changes) in enumerate(variants, 2):
        config = {**original["config"], **changes}
        directory = output / f"candidate-{number}"
        directory.mkdir()
        checkpoint = directory / "checkpoint.pt"
        history, best, best_epoch = [], None, 0
        rng = random.Random(tune_roles.SEED)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(tune_roles.SEED)
            model = initialize_model(manifest, config, vectors, fit_ids)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
            order = list(fit_ids)
            for epoch in range(1, max_epochs + 1):
                rng.shuffle(order)
                loss = fit_epoch(model, vectors, labels, order, optimizer, weights, config)
                report = tune_roles.evaluate(model, vectors, labels, validation_ids)
                if best is None or tune_roles.score_key(report) > tune_roles.score_key(best):
                    best, best_epoch = report, epoch
                    model.save(checkpoint, metadata={
                        "seed": tune_roles.SEED, "epoch": epoch, "config": config,
                        "fit_case_ids": fit_ids, "validation_case_ids": validation_ids,
                        "features_sha256": manifest["features_sha256"], "validation": report,
                    })
                history.append({"epoch": epoch, "training_loss": loss,
                                "validation": report, "best_epoch": best_epoch})
                atomic_write(directory / "history.json", json_bytes(history))
                if epoch == 1 or epoch % 5 == 0:
                    print(f"{name}, epoch {epoch}: accuracy {report['accuracy']:.4f}, "
                          f"macro F1 {report['macro avg']['f1-score']:.4f}, "
                          f"lower-court recall {report['Ruling by Lower Court']['recall']:.4f}, "
                          f"best epoch {best_epoch}", flush=True)
                if epoch - best_epoch >= patience:
                    break
        candidates.append({
            "candidate": number, "name": name, "config": config, "best_epoch": best_epoch,
            "epochs_completed": len(history), "validation": best,
            "checkpoint": str(checkpoint.relative_to(output)),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        })
    selected = max(candidates, key=lambda candidate: tune_roles.score_key(candidate["validation"]))
    atomic_write(output / "validation_labeller.pt", (output / selected["checkpoint"]).read_bytes())
    selection = {
        "format": "contextual-role-selection-v1", "runner": "ablate_roles", "seed": tune_roles.SEED,
        "selection_metric": "validation accuracy, then macro F1; earliest epoch and candidate break ties",
        "max_epochs": max(max_epochs, baseline["max_epochs"]), "ablation_max_epochs": max_epochs,
        "patience": patience, "feature_manifest": manifest, "class_weights": weights.tolist(),
        "baseline_selection_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "candidates": candidates, "selected": selected,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    atomic_write(output / "selection.json", json_bytes(selection))
    print(f"Selected {selected['name']} at epoch {selected['best_epoch']}: "
          f"validation accuracy {selected['validation']['accuracy']:.4f}", flush=True)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("."))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--baseline-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    run(args.data, args.features, args.baseline_selection, args.output, args.max_epochs, args.patience)


if __name__ == "__main__":
    main()
