"""Train and evaluate role-weighted LexRank by judgment, with a held-out test set."""

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import torch
from sklearn.metrics import classification_report

from .data import ROLES, load_cases, split_case_ids
from .encoders import BACKENDS, MODEL_IDS, MODEL_REVISIONS, similarity_matrix
from .labeller import fit_labeller, load_labeller
from .metrics import SELECTION_METRIC, score_summaries
from .ranking import fit_weights, rank_sentences, select_indices

RATIOS = (0.3, 0.4, 0.5)
METHODS = ("mean", "chebyshev", "geometric_median")
METRICS = [f"{name}_{stat}" for name in ("rouge1", "rouge2", "rougeL")
           for stat in ("precision", "recall", "f1")]
EPOCHS = 8


def atomic_write(path, data):
    """Replace one result file only after the new contents have been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def save_csv(path, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, buffer.getvalue().encode())


def training_pairs(cases, ids):
    documents = []
    for key in ids:
        case = cases[key]
        pairs = [(s, role) for s, role in zip(case.labelled_sentences, case.labels, strict=True)
                 if role is not None]
        if not pairs:
            raise ValueError(f"No labelled sentences in {key}")
        documents.append(tuple(zip(*pairs)))
    return documents


def trained_model(cases, ids, seed, cache):
    docs = training_pairs(cases, ids)
    signature = hashlib.sha256(Path(__file__).with_name("labeller.py").read_bytes())
    signature.update(json_bytes([ROLES, seed, EPOCHS, torch.__version__, ids, docs]))
    checkpoint = cache / "labellers" / (signature.hexdigest() + ".pt")
    if checkpoint.is_file():
        return load_labeller(checkpoint), checkpoint
    print(f"Training role labeller on {len(ids)} cases ({EPOCHS} epochs)", flush=True)
    return fit_labeller(docs, ROLES, seed, checkpoint, epochs=EPOCHS), checkpoint


def role_accuracy(model, cases, ids):
    gold, predicted = [], []
    for key in ids:
        case = cases[key]
        guesses = model.predict(case.labelled_sentences)
        for expected, guess in zip(case.labels, guesses, strict=True):
            if expected is not None:
                gold.append(expected)
                predicted.append(guess)
    return classification_report(gold, predicted, labels=ROLES, output_dict=True, zero_division=0)


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


def protocol(cases, split, seed):
    corpus = hashlib.sha256()
    for key, case in sorted(cases.items()):
        corpus.update(json_bytes([key, case.sentences, case.labelled_sentences, case.labels, case.references]))
    implementation = hashlib.sha256()
    for module in ("data", "encoders", "experiment", "labeller", "metrics", "ranking"):
        implementation.update(Path(__file__).with_name(module + ".py").read_bytes())
    return {
        "implementation_sha256": implementation.hexdigest(),
        "version": 1, "seed": seed, "case_ids": split, "corpus_sha256": corpus.hexdigest(),
        "roles": list(ROLES), "ratios": list(RATIOS), "role_epochs": EPOCHS, "methods": list(METHODS),
        "metric": SELECTION_METRIC,
        "selection": "Highest validation mean across cases and all three ratios; ties in method order.",
        "references": "Mean across distinct A1 and A2 references per case, then macro-average cases.",
        "models": MODEL_IDS, "revisions": MODEL_REVISIONS,
    }


def record_protocol(out, settings):
    path = out / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError(f"{out} contains a different protocol or corpus; choose another --output")
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


def run(root, output, cache, backends, seed=42):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cases = load_cases(root)
    split = split_case_ids(cases, seed)
    record_protocol(output, protocol(cases, split, seed))
    def fit_stage(train_key):
        ids = split[train_key]
        model, checkpoint = trained_model(cases, ids, seed, cache)
        profiles = fit_weights([labels for _, labels in training_pairs(cases, ids)], ROLES)
        return model, checkpoint, profiles

    dev_model, _, dev_profiles = fit_stage("fit")
    dev_ids = split["validation"]
    dev_roles = {key: dev_model.predict(cases[key].sentences) for key in dev_ids}
    validation_reports = {}
    for backend in backends:
        print(f"{backend}: comparing weights on {len(dev_ids)} validation cases", flush=True)
        rows = evaluate(backend, cases, dev_ids, dev_roles, {"unweighted": None, **dev_profiles}, cache)
        winner, scores = choose_method(rows)
        validation_reports[backend] = {"encoder": backend, "model": MODEL_IDS[backend],
                                       "selected_method": winner, "selection_scores": scores,
                                       "validation": rows}

    final_model, checkpoint, final_profiles = fit_stage("train")
    atomic_write(output / "role_labeller.pt", checkpoint.read_bytes())
    atomic_write(output / "weights.json", json_bytes({"validation": dev_profiles, "test": final_profiles}))
    test_ids = split["test"]
    test_roles = {key: final_model.predict(cases[key].sentences) for key in test_ids}
    for backend, report in validation_reports.items():
        print(f"{backend}: selected {report['selected_method']}; scoring {len(test_ids)} test cases", flush=True)
        report["test"] = evaluate(backend, cases, test_ids, test_roles,
                                  {"unweighted": None, **final_profiles}, cache)
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
    args = parser.parse_args()
    run(args.data, args.output, args.cache, list(dict.fromkeys(args.backends)), args.seed)
