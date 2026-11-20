import json
import subprocess
import sys

import pytest

from legal_lexrank.data import ROLES


def test_role_weights_cli_uses_human_annotations_without_a_labeller(tmp_path):
    corpus = tmp_path / "corpus"
    for folder in ("Judgement", "Labelled_Data", "Summary", "Summary_2"):
        directory = corpus / folder
        directory.mkdir(parents=True)
        text = "A judgment sentence."
        if folder == "Labelled_Data":
            text = "\n".join(f"Sentence {i}.\t{role}" for i, role in enumerate(ROLES))
            text += "\nAn unlabelled sentence.\t"
        (directory / "one-case.txt").write_text(text)

    output = tmp_path / "results" / "weights.json"
    # A single case suffices for corpus weights, with no train/test partition
    # or checkpoint available. Invoke the real CLI in a fresh process.
    result = subprocess.run(
        [sys.executable, "-m", "legal_lexrank.role_weights", "--data", str(corpus), "--output", str(output)],
        capture_output=True, text=True, check=True,
    )
    assert "1 judgments" in result.stdout
    bundle = json.loads(output.read_text())
    assert bundle["metadata"]["source"] == "human_annotations"
    assert bundle["metadata"]["case_ids"] == ["one-case"]
    assert bundle["metadata"]["labelled_sentence_count"] == 7
    assert len(bundle["metadata"]["source_annotations_sha256"]) == 64
    assert set(bundle["profiles"]) == {"mean", "chebyshev", "geometric_median"}
    for profile in bundle["profiles"].values():
        assert profile == pytest.approx({role: 1 / 7 for role in ROLES})
    assert list(output.parent.iterdir()) == [output]
