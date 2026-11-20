"""Refit a validation-selected contextual role labeller on all training judgments."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from tempfile import TemporaryDirectory

import torch

from . import contextual_labeller, encoders, tune_roles
from .ablate_roles import initialize_model, fit_epoch
from .contextual_labeller import load_contextual_labeller
from .data import ROLES, load_cases
from .role_study import study_settings
from .storage import atomic_write, json_bytes


def validate_selection(features_manifest, selection_path):
    """Verify the published choice and its best validation checkpoint together."""
    selection_path = Path(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection.get("format") != "contextual-role-selection-v1"
            or selection.get("seed") != tune_roles.SEED
            or selection.get("feature_manifest") != features_manifest):
        raise ValueError("Selection is incompatible with the training feature bundle")
    runner = selection.get("runner", "tune_roles")
    if runner not in ("tune_roles", "ablate_roles"):
        raise ValueError("Unsupported role selection runner")
    runner_source = Path(__file__).with_name(runner + ".py")
    if selection["implementation_sha256"] != hashlib.sha256(runner_source.read_bytes()).hexdigest():
        raise ValueError("The tuning implementation changed after model selection")
    chosen = selection["selected"]
    winner = max(selection["candidates"], key=lambda item: tune_roles.score_key(item["validation"]))
    if chosen != winner:
        raise ValueError("Selected model differs from the recorded validation winner")
    if type(chosen["best_epoch"]) is not int or not 1 <= chosen["best_epoch"] <= selection["max_epochs"]:
        raise ValueError("Selected epoch is outside the recorded tuning schedule")
    checkpoint = selection_path.parent / "validation_labeller.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != chosen["checkpoint_sha256"]:
        raise ValueError("Selected validation checkpoint checksum mismatch")
    model = load_contextual_labeller(checkpoint)
    expected_metadata = {
        "seed": tune_roles.SEED, "epoch": chosen["best_epoch"], "config": chosen["config"],
        "fit_case_ids": features_manifest["case_ids"]["fit"],
        "validation_case_ids": features_manifest["case_ids"]["validation"],
        "features_sha256": features_manifest["features_sha256"], "validation": chosen["validation"],
    }
    expected_config = {"hidden_size": chosen["config"]["hidden_size"],
                       "dropout": chosen["config"]["dropout"], "use_position": True}
    if chosen["config"].get("center_inputs", False):
        expected_config["center_inputs"] = True
    if (model.metadata != expected_metadata or model.roles != ROLES
            or model.encoder != features_manifest["backend"]
            or model.input_size != features_manifest["dimensions"]
            or model.config != expected_config):
        raise ValueError("Selected validation checkpoint metadata mismatch")
    return selection, checkpoint


def run(data, features, selection, output):
    """Fit the selected configuration for its fixed epoch count, then publish it."""
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Choose a new or empty final-training directory: {output}")
    manifest, vectors, labels = tune_roles.load_features(data, features)
    selection_path = Path(selection)
    selection, validation_checkpoint = validate_selection(manifest, selection_path)
    chosen = selection["selected"]
    config = chosen["config"]
    epochs = chosen["best_epoch"]
    split = manifest["case_ids"]
    weights = tune_roles.class_weights(labels, split["train"])
    cases = load_cases(Path(data))
    protocol = study_settings(cases, split, tune_roles.SEED, epochs)
    protocol.update({
        "format": "legal-lexrank-role-study-v2",
        "config": config, "selected_candidate": chosen["candidate"],
        "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "features_sha256": manifest["features_sha256"],
        "feature_encoder": {
            "backend": manifest["backend"], "model": manifest["model"],
            "revision": manifest["revision"], "dimensions": manifest["dimensions"],
            "normalization": manifest["normalization"],
        },
        "contextual_labeller_sha256": hashlib.sha256(Path(contextual_labeller.__file__).read_bytes()).hexdigest(),
        "encoder_sha256": hashlib.sha256(Path(encoders.__file__).read_bytes()).hexdigest(),
        "tuning_sha256": hashlib.sha256(Path(tune_roles.__file__).read_bytes()).hexdigest(),
        "ablation_sha256": hashlib.sha256(Path(__file__).with_name("ablate_roles.py").read_bytes()).hexdigest(),
        "selection_runner": selection.get("runner", "tune_roles"),
        "selection_runner_sha256": selection["implementation_sha256"],
        "final_training_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "optimizer": {"name": "AdamW", "weight_decay": 0.01, "gradient_clip_norm": 5.0},
        "training_class_weights": weights.tolist(),
        "validation_metrics": chosen["validation"],
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    history = []
    with TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        rng = random.Random(tune_roles.SEED)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(tune_roles.SEED)
            model = initialize_model(manifest, config, vectors, split["train"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
            order = list(split["train"])
            print(f"Refitting on {len(split['train'])} judgments for {epochs} selected epochs", flush=True)
            for epoch in range(1, epochs + 1):
                rng.shuffle(order)
                loss = fit_epoch(model, vectors, labels, order, optimizer, weights, config)
                history.append({"epoch": epoch, "training_loss": loss})
                atomic_write(bundle / "training_history.json", json_bytes(history))
                if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                    print(f"final epoch {epoch}/{epochs}: training loss {history[-1]['training_loss']:.4f}", flush=True)
            model.save(bundle / "role_labeller.pt", metadata={
                "seed": tune_roles.SEED, "epoch": epochs, "config": config,
                "train_case_ids": split["train"], "features_sha256": manifest["features_sha256"],
                "selection_sha256": protocol["selection_sha256"],
            })
        atomic_write(bundle / "validation_labeller.pt", validation_checkpoint.read_bytes())
        protocol["checkpoint_sha256"] = {
            stage: hashlib.sha256((bundle / filename).read_bytes()).hexdigest()
            for stage, filename in (("validation", "validation_labeller.pt"), ("test", "role_labeller.pt"))
        }
        atomic_write(bundle / "protocol.json", json_bytes(protocol))
        os.replace(bundle, output)
    print(f"Final model saved to {output / 'role_labeller.pt'}", flush=True)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("."))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True, help="Selected backend's selection.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.features, args.selection, args.output)


if __name__ == "__main__":
    main()
