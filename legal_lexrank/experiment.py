"""Evaluate LexRank using corpus-wide human role weights and a held-out role labeller."""

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
from .metrics import SELECTION_METRIC, score_summaries
from .ranking import rank_sentences, select_indices
from .role_study import EPOCHS, STAGES, json_bytes, role_accuracy, training_pairs, verified_checkpoints
from .role_weights import fit_corpus_weights
from .storage import atomic_write

RATIOS = (0.3, 0.4, 0.5)
METHODS = ("mean", "chebyshev", "geometric_median")
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


def evaluate(backend, cases, ids, predictions, profiles, cache):
    rows = []
    for key in ids:
        case = cases[key]
        similarities = similarity_matrix(backend, case.sentences, cache / "similarities")
        for method, weights in profiles.items():
            ranking = rank_sentences(similarities, predictions[key], weights)
            for ratio in RATIOS:
                indices = select_indices(ranking, ratio)
                summary = "\n".join(case.sentences[index] for index in indices)
                rows.append({
                    "case": key, "method": method, "ratio": ratio,
                    "source_sentences": len(case.sentences), "reference_count": len(case.references),
                    "selected_indices": indices, **score_summaries(summary, case.references),
                })
    return rows


def choose_method(validation_rows):
    """Choose among fitted methods; ties resolve in the declared method order."""
    scores = {name: fmean(row[SELECTION_METRIC] for row in validation_rows
                          if row["method"] == name) for name in METHODS}
    return max(METHODS, key=scores.get), scores


def macro_rows(backend, phase, rows, selected):
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"], row["ratio"]].append(row)
    return [
        {"encoder": backend, "phase": phase, "method": method, "ratio": ratio,
         "validation_winner": method == selected, "cases": len(group),
         **{metric: fmean(row[metric] for row in group) for metric in METRICS}}
        for (method, ratio), group in groups.items()
    ]


def protocol(cases, split, seed, weight_metadata=None):
    corpus = hashlib.sha256()
    for key, case in sorted(cases.items()):
        corpus.update(json_bytes([key, case.sentences, case.labelled_sentences, case.labels, case.references]))
    implementation = hashlib.sha256()
    for module in ("data", "encoders", "experiment", "labeller", "contextual_labeller", "metrics", "ranking", "role_study", "role_weights"):
        implementation.update(Path(__file__).with_name(module + ".py").read_bytes())
    return {
        "implementation_sha256": implementation.hexdigest(),
        "version": 2, "seed": seed, "case_ids": split, "corpus_sha256": corpus.hexdigest(),
        "roles": list(ROLES), "ratios": list(RATIOS), "role_epochs": EPOCHS, "methods": list(METHODS),
        "role_weights": fit_corpus_weights(cases)["metadata"] if weight_metadata is None else weight_metadata,
        "metric": SELECTION_METRIC,
        "selection": "Highest validation mean across cases and all three ratios; ties in method order.",
        "references": "Mean across distinct A1 and A2 references per case, then macro-average cases.",
        "models": MODEL_IDS, "revisions": MODEL_REVISIONS,
    }


def record_protocol(out, settings):
    path = out / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != settings:
            raise ValueError(f"{out} contains a different protocol or corpus; choose another --output")
        return

    artifacts = [f"{name}.json" for name in BACKENDS]
    artifacts.extend(["selection.csv", "metrics.csv", "weights.json", "role_metrics.json", "role_labeller.pt"])
    if any((out / name).exists() for name in artifacts):
        raise ValueError(f"{out} contains benchmark artifacts without a protocol; choose another --output")
    atomic_write(path, json_bytes(settings))


def rebuild_tables(out):
    totals, selection = [], []
    for backend in BACKENDS:
        path = out / (backend + ".json")
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        winner = report["selected_method"]
        for name, row in report["selection_scores"].items():
            selection.append({"encoder": backend, "method": name, "selected": name == winner,
                              "validation_mean_rougeL_f1": row})
        for phase in ("validation", "test"):
            totals.extend(macro_rows(backend, phase, report[phase], winner))
    save_csv(out / "metrics.csv", totals)
    save_csv(out / "selection.csv", selection)


def run(root, output, cache, backends, seed=42, role_run=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Protect both protocol validation and every shared table/checkpoint update.
    # FileLock releases the underlying OS lock if the writer is interrupted.
    with FileLock(output / ".benchmark.lock"):
        _run(root, output, cache, backends, seed, role_run)


def _run(root, output, cache, backends, seed, role_run):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cases = load_cases(root)
    split = split_case_ids(cases, seed)
    weight_bundle = fit_corpus_weights(cases)
    profiles = weight_bundle["profiles"]
    settings = protocol(cases, split, seed, weight_bundle["metadata"])
    if role_run is not None:
        role_run = Path(role_run)
        settings["role_checkpoint_sha256"] = verified_checkpoints(role_run, cases, split, seed, EPOCHS)
        role_protocol = json.loads((role_run / "protocol.json").read_text(encoding="utf-8"))
        settings["role_epochs"] = role_protocol["epochs"]
        if role_protocol["format"] == "legal-lexrank-role-study-v2":
            settings["role_encoder"] = role_protocol["feature_encoder"]
    record_protocol(output, settings)

    def fit_stage(phase):
        train_key, filename = STAGES[phase]
        ids = split[train_key]
        if role_run is None:
            model, checkpoint = trained_model(cases, ids, seed, cache)
        else:
            checkpoint = role_run / filename
            model = load_labeller(checkpoint, cache_dir=cache / "similarities")
        return model, checkpoint

    dev_model, _ = fit_stage("validation")
    dev_ids = split["validation"]
    dev_roles = {key: dev_model.predict(cases[key].sentences) for key in dev_ids}
    validation_reports = {}
    for backend in backends:
        print(f"{backend}: comparing weights on {len(dev_ids)} validation cases", flush=True)
        rows = evaluate(backend, cases, dev_ids, dev_roles, {"unweighted": None, **profiles}, cache)
        winner, scores = choose_method(rows)
        validation_reports[backend] = {"encoder": backend, "model": MODEL_IDS[backend],
                                       "selected_method": winner, "selection_scores": scores,
                                       "validation": rows}

    final_model, checkpoint = fit_stage("test")
    atomic_write(output / "role_labeller.pt", checkpoint.read_bytes())
    atomic_write(output / "weights.json", json_bytes({
        "metadata": settings["role_weights"], "validation": profiles, "test": profiles,
    }))
    test_ids = split["test"]
    test_roles = {key: final_model.predict(cases[key].sentences) for key in test_ids}
    for backend, report in validation_reports.items():
        print(f"{backend}: selected {report['selected_method']}; scoring {len(test_ids)} test cases", flush=True)
        report["test"] = evaluate(backend, cases, test_ids, test_roles,
                                  {"unweighted": None, **profiles}, cache)
        atomic_write(output / (backend + ".json"), json_bytes(report))
        rebuild_tables(output)
    diagnostics = {"validation": role_accuracy(dev_model, cases, dev_ids),
                   "test": role_accuracy(final_model, cases, test_ids)}
    atomic_write(output / "role_metrics.json", json_bytes(diagnostics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("."), help="Repository or corpus root")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/legal_lexrank"))
    parser.add_argument("--backends", choices=BACKENDS, nargs="+", default=list(BACKENDS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--role-run", type=Path, help="Extracted role-study bundle with both fitted checkpoints")
    args = parser.parse_args()
    run(args.data, args.output, args.cache, list(dict.fromkeys(args.backends)), args.seed, args.role_run)
