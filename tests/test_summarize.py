"""Verify inference uses the same fitted checkpoint and role weights as evaluation."""

import json
from pathlib import Path

import pytest

from legal_lexrank.summarize import main


@pytest.mark.parametrize("method", [None, "unweighted"])
def test_reproduce_published_tfidf_summary(tmp_path, capsys, method):
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    report = json.loads((results / "tfidf.json").read_text())
    row = next(row for row in report["test"]
               if row["method"] == (method or report["selected_method"]) and row["ratio"] == 0.3)
    judgment = root / "Judgement" / (row["case"] + ".txt")
    args = [str(judgment), "--backend", "tfidf", "--results", str(results),
            "--cache", str(tmp_path), "--ratio", "0.3"]
    if method:
        args.extend(["--method", method])
    main(args)
    sentences = [line.strip() for line in judgment.read_text().splitlines() if line.strip()]
    assert capsys.readouterr().out.splitlines() == [sentences[i] for i in row["selected_indices"]]
