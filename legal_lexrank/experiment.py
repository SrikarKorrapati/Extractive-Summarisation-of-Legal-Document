"""Evaluate LexRank with fixed rhetorical-role weights and a held-out role labeller."""

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import torch
from filelock import FileLock

from .data import ROLES, load_cases, split_case_ids
from .encoders import BACKENDS, MODEL_IDS, MODEL_REVISIONS, similarity_matrix
from .labeller import fit_labeller, load_labeller
from .metrics import score_summaries
from .ranking import DEFAULT_ROLE_WEIGHTS, rank_sentences, select_indices
from .role_study import EPOCHS, STAGES, json_bytes, role_accuracy, training_pairs, verified_checkpoints
from .storage import atomic_write

RATIOS = (0.3, 0.4, 0.5)
METRICS = [f"{name}_{stat}" for name in ("rouge1", "rouge2", "rougeL")
           for stat in ("precision", "recall", "f1")]


def save_csv(path, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, buffer.getvalue().encode())


def trained_model(cases, ids, seed, cache):
    docs = training_pairs(cases, ids)
    signature = hashlib.sha256(Path(__file__).with_name("labeller.py").read_bytes())
    signature.update(json_bytes([ROLES, seed, EPOCHS, torch.__version__, ids, docs]))
    checkpoint = cache / "labellers" / (signature.hexdigest() + ".pt")
    if checkpoint.is_file():
        return load_labeller(checkpoint), checkpoint
    print(f"Training role labeller on {len(ids)} cases ({EPOCHS} epochs)", flush=True)
    return fit_labeller(docs, ROLES, seed, checkpoint, epochs=EPOCHS), checkpoint


def evaluate(backend, cases, ids, predictions, cache):
    rows = []
    for key in ids:
        case = cases[key]
        similarities = similarity_matrix(backend, case.sentences, cache / "similarities")
        for weighting, weights in (("fixed", DEFAULT_ROLE_WEIGHTS), ("unweighted", None)):
            ranking = rank_sentences(similarities, predictions[key], weights)
            for ratio in RATIOS:
                indices = select_indices(ranking, ratio)
                summary = "\n".join(case.sentences[index] for index in indices)
                rows.append({
                    "case": key, "weighting": weighting, "ratio": ratio,
                    "source_sentences": len(case.sentences), "reference_count": len(case.references),
                    "selected_indices": indices, **score_summaries(summary, case.references),
                })
    return rows


def macro_rows(backend, phase, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["weighting"], row["ratio"]].append(row)
    return [
        {"encoder": backend, "phase": phase, "weighting": weighting, "ratio": ratio,
         "cases": len(group),
         **{metric: fmean(row[metric] for row in group) for metric in METRICS}}
        for (weighting, ratio), group in groups.items()
    ]


def protocol(cases, split, seed):
    corpus = hashlib.sha256()
    for key, case in sorted(cases.items()):
        corpus.update(json_bytes([key, case.sentences, case.labelled_sentences, case.labels, case.references]))
    implementation = hashlib.sha256()
    for module in ("data", "encoders", "experiment", "labeller", "contextual_labeller", "metrics", "ranking", "role_study"):
        implementation.update(Path(__file__).with_name(module + ".py").read_bytes())
    return {
        "implementation_sha256": implementation.hexdigest(),
        "version": 3, "seed": seed, "case_ids": split, "corpus_sha256": corpus.hexdigest(),
        "roles": list(ROLES), "ratios": list(RATIOS), "role_epochs": EPOCHS,
        "role_weights": DEFAULT_ROLE_WEIGHTS,
        "references": "Mean across distinct A1 and A2 references per case, then macro-average cases.",
        "models": MODEL_IDS, "revisions": MODEL_REVISIONS,
    }


def record_protocol(out, settings):
    path = out / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != settings:
            raise ValueError(f"{out} contains a different protocol or corpus; choose another --output")
        return

    if any(item.name != ".benchmark.lock" for item in out.iterdir()):
        raise ValueError(f"{out} contains benchmark artifacts without a protocol; choose another --output")
    atomic_write(path, json_bytes(settings))


def rebuild_tables(out):
    totals = []
    for backend in BACKENDS:
        path = out / (backend + ".json")
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        for phase in ("validation", "test"):
            totals.extend(macro_rows(backend, phase, report[phase]))
    save_csv(out / "metrics.csv", totals)


def run(root, output, cache, backends, seed=42, role_run=None, device="cpu"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Protect both protocol validation and every shared table/checkpoint update.
    # FileLock releases the underlying OS lock if the writer is interrupted.
    with FileLock(output / ".benchmark.lock"):
        _run(root, output, cache, backends, seed, role_run, device)


def _run(root, output, cache, backends, seed, role_run, device="cpu"):
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be cpu, cuda, or mps")
    if device != "cpu" and role_run is None:
        raise ValueError("Non-CPU role inference requires a transformer v4 --role-run; legacy models support CPU only")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cases = load_cases(root)
    split = split_case_ids(cases, seed)
    settings = protocol(cases, split, seed)
    transformer = False
    if role_run is not None:
        role_run = Path(role_run)
        role_protocol = json.loads((role_run / "protocol.json").read_text(encoding="utf-8"))
        if device != "cpu" and role_protocol.get("format") != "legal-lexrank-role-study-v4":
            raise ValueError("Non-CPU role inference requires a transformer v4 --role-run; legacy models support CPU only")
        settings["role_checkpoint_sha256"] = verified_checkpoints(role_run, cases, split, seed, EPOCHS)
        settings["role_epochs"] = role_protocol["epochs"]
        if role_protocol["format"] == "legal-lexrank-role-study-v2":
            settings["role_encoder"] = role_protocol["feature_encoder"]
        elif role_protocol["format"] == "legal-lexrank-role-study-v3":
            settings["role_encoder"] = role_protocol["encoder_identity"]
            settings["role_architecture"] = role_protocol["architecture"]
        elif role_protocol["format"] == "legal-lexrank-role-study-v4":
            transformer = True
            config = role_protocol["config"]
            settings["role_encoder"] = {
                "backend": config["encoder"], "model": config["encoder_model"],
                "revision": config["encoder_revision"],
            }
            settings["role_architecture"] = role_protocol["architecture"]
            settings["role_config"] = config
            settings["role_inference_device"] = device
            settings["evaluation_phases"] = ["test"]
            settings["role_validation_metrics_source"] = (
                "Saved selection report from the 32-case fit model on eight validation cases"
            )
    record_protocol(output, settings)

    def fit_stage(phase):
        train_key, filename = STAGES[phase]
        ids = split[train_key]
        if role_run is None:
            model, checkpoint = trained_model(cases, ids, seed, cache)
        else:
            checkpoint = role_run / filename
            model = load_labeller(checkpoint, cache_dir=cache / "similarities")
        if transformer:
            model = model.to(device)
        return model, checkpoint

    validation_reports = {}
    if transformer:
        # Deployment carries only the model refitted on all 40 training cases.
        # Reusing it on validation would evaluate cases it has already learned.
        validation_reports = {
            backend: {"encoder": backend, "model": MODEL_IDS[backend], "validation": []}
            for backend in backends
        }
        validation_metrics = role_protocol["validation_metrics"]
    else:
        dev_model, _ = fit_stage("validation")
        dev_ids = split["validation"]
        dev_roles = {key: dev_model.predict(cases[key].sentences) for key in dev_ids}
        for backend in backends:
            print(f"{backend}: scoring {len(dev_ids)} validation cases", flush=True)
            rows = evaluate(backend, cases, dev_ids, dev_roles, cache)
            validation_reports[backend] = {"encoder": backend, "model": MODEL_IDS[backend],
                                           "validation": rows}
        validation_metrics = role_accuracy(dev_model, cases, dev_ids)

    final_model, checkpoint = fit_stage("test")
    if not transformer:
        atomic_write(output / "role_labeller.pt", checkpoint.read_bytes())
    test_ids = split["test"]
    test_roles = {key: final_model.predict(cases[key].sentences) for key in test_ids}
    for backend, report in validation_reports.items():
        print(f"{backend}: scoring {len(test_ids)} test cases", flush=True)
        report["test"] = evaluate(backend, cases, test_ids, test_roles, cache)
        atomic_write(output / (backend + ".json"), json_bytes(report))
        rebuild_tables(output)
    diagnostics = {"validation": validation_metrics,
                   "test": role_accuracy(final_model, cases, test_ids)}
    atomic_write(output / "role_metrics.json", json_bytes(diagnostics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("."), help="Repository or corpus root")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/legal_lexrank"))
    parser.add_argument("--backends", choices=BACKENDS, nargs="+", default=list(BACKENDS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--role-run", type=Path,
                        help="Verified role-study bundle (transformer v4 needs only the final checkpoint)")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu",
                        help="Role classifier inference device; non-CPU requires a transformer v4 bundle")
    args = parser.parse_args()
    run(args.data, args.output, args.cache, list(dict.fromkeys(args.backends)), args.seed, args.role_run, args.device)
