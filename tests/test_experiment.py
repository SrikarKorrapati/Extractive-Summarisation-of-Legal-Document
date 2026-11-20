import csv
import json
import math
from itertools import product
from statistics import fmean

import pytest
from rouge_score.rouge_scorer import RougeScorer

from legal_lexrank import experiment
from legal_lexrank.data import ROLES, load_cases
from legal_lexrank.ranking import fit_weights


def write_corpus(root):
    for folder in ("Judgement", "Labelled_Data", "Summary", "Summary_2"):
        (root / folder).mkdir(parents=True)
    sentences = [
        "The appeal challenges the final order.",
        "Counsel proposes a different interpretation.",
        "The statute protects the tenant.",
        "The tenant relied on an earlier judgment.",
        "The court applies the statute to this tenant.",
        "The court grants the appeal.",
        "The lower court previously dismissed the claim.",
    ]
    for i in range(50):
        name = f"case-{i:02}.txt"
        # Distinct representations ensure the experiment predicts source sentence
        # roles even when the labelled corpus uses a different segmentation.
        raw = sentences[:5 + i % 3]
        tags = [*ROLES, ROLES[i % len(ROLES)], ROLES[i % 3]]
        annotated = [f"{name} annotation {j}.\t{role}" for j, role in enumerate(tags)]
        (root / "Judgement" / name).write_text("\n".join(raw))
        (root / "Labelled_Data" / name).write_text("\n".join(annotated))
        (root / "Summary" / name).write_text(raw[0])
        (root / "Summary_2" / name).write_text(raw[0] if i % 2 else raw[-1])


def read_json(directory, name):
    return json.loads((directory / name).read_text())


def test_document_protocol_end_to_end(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    output = tmp_path / "first"
    cache = tmp_path / "cache"
    write_corpus(corpus)
    monkeypatch.setattr(experiment, "EPOCHS", 1)
    # Record every case contributing to method selection. Identical ROUGE across
    # two methods could otherwise hide an accidental test-set selection.
    selection_cases = []
    choose_method = experiment.choose_method

    def observe_selection(rows):
        selection_cases.append({row["case"] for row in rows})
        return choose_method(rows)

    monkeypatch.setattr(experiment, "choose_method", observe_selection)
    experiment.run(corpus, output, cache, ["tfidf"], seed=42)
    cases = load_cases(corpus)
    split = read_json(output, "protocol.json")["case_ids"]
    assert {k: len(v) for k, v in split.items()} == {
        "fit": 32, "validation": 8, "train": 40, "test": 10,
    }
    assert set(split["fit"]).isdisjoint(split["validation"])
    assert set(split["fit"]) | set(split["validation"]) == set(split["train"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["train"]) | set(split["test"]) == cases.keys()
    assert selection_cases == [set(split["validation"])]
    weights = read_json(output, "weights.json")
    report = read_json(output, "tfidf.json")
    with (output / "metrics.csv").open() as handle:
        macro = list(csv.DictReader(handle))
    scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    for phase, training in [("validation", "fit"), ("test", "train")]:
        expected = fit_weights([cases[key].labels for key in split[training]], ROLES)
        for method in experiment.METHODS:
            assert weights[phase][method] == pytest.approx(expected[method])
        rows = report[phase]
        combinations = [(row["case"], row["method"], row["ratio"]) for row in rows]
        assert sorted(combinations) == sorted(product(
            split[phase], ["unweighted", *experiment.METHODS], (0.3, 0.4, 0.5)
        ))
        for row in rows:
            case = cases[row["case"]]
            indices = row["selected_indices"]
            assert indices == sorted(set(indices))
            assert len(indices) == math.ceil(len(case.sentences) * row["ratio"])
            assert row["source_sentences"] == len(case.sentences)
            assert row["reference_count"] == len(case.references)
            summary = "\n".join(case.sentences[j] for j in indices)
            reference_scores = [scorer.score(reference, summary) for reference in case.references]
            for metric in ("rouge1", "rouge2", "rougeL"):
                for stat, attribute in [("precision", "precision"), ("recall", "recall"), ("f1", "fmeasure")]:
                    assert row[f"{metric}_{stat}"] == pytest.approx(
                        fmean(getattr(values[metric], attribute) for values in reference_scores)
                    )
        for aggregate in [record for record in macro if record["phase"] == phase]:
            group = [row for row in rows if row["method"] == aggregate["method"]
                     and row["ratio"] == float(aggregate["ratio"])]
            assert int(aggregate["cases"]) == len(split[phase]) == len(group)
            assert float(aggregate["rougeL_f1"]) == pytest.approx(fmean(row["rougeL_f1"] for row in group))
    validation_means = {method: fmean(row["rougeL_f1"] for row in report["validation"]
                                      if row["method"] == method) for method in experiment.METHODS}
    assert report["selection_scores"] == pytest.approx(validation_means)
    assert report["selected_method"] == max(experiment.METHODS, key=validation_means.get)

    # Rewrite every gold test label. This must change the diagnostic report but
    # leave fitted weights, chosen method, rankings, and summarization scores intact.
    for case_id in split["test"]:
        path = corpus / "Labelled_Data" / f"{case_id}.txt"
        path.write_text("\n".join(f"{s}\tFacts" for s in cases[case_id].labelled_sentences))
    changed = tmp_path / "altered-test-labels"
    experiment.run(corpus, changed, cache, ["tfidf"], seed=42)
    assert selection_cases[-1] == set(split["validation"])
    assert read_json(changed, "role_metrics.json")["test"] != read_json(output, "role_metrics.json")["test"]
    assert read_json(changed, "weights.json") == weights
    assert read_json(changed, "tfidf.json") == report


def test_rejects_unmanaged_output_with_missing_protocol(tmp_path):
    existing = tmp_path / "tfidf.json"
    existing.write_text("{}")
    with pytest.raises(ValueError, match="without a protocol"):
        experiment.record_protocol(tmp_path, {"seed": 42})
    assert existing.read_text() == "{}"
    assert not (tmp_path / "protocol.json").exists()


def test_overlapping_benchmark_writers_share_an_output_lock(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from filelock import FileLock, Timeout

    entered, release = Event(), Event()

    def paused_run(*args):
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(experiment, "_run", paused_run)
    out = tmp_path / "output"
    with ThreadPoolExecutor() as executor:
        job = executor.submit(experiment.run, tmp_path, out, tmp_path, ["tfidf"])
        try:
            assert entered.wait(5)
            with pytest.raises(Timeout):
                FileLock(out / ".benchmark.lock").acquire(timeout=0)
        finally:
            release.set()
        job.result()
    with FileLock(out / ".benchmark.lock", timeout=0):
        assert out.is_dir()
