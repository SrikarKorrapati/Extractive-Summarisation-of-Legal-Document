from pathlib import Path

import pytest

from legal_lexrank.data import ROLES, load_cases, split_case_ids


@pytest.fixture
def corpus(tmp_path):
    source = {
        "Judgement": "\n  Raw   text!  \n next line\n",
        "Labelled_Data": "Original  spacing    Facts\nSecond.\tArgument\nThird.\t\nMore  text.    \n",
        "Summary": "First reference.\nSecond line.\n",
        "Summary_2": "Another reference.",
    }
    for name, text in source.items():
        directory = tmp_path / name
        directory.mkdir()
        (directory / "case.txt").write_text(text, encoding="utf-8")
    return tmp_path


def test_keeps_independent_sentence_sources_and_reference_order(corpus):
    case = load_cases(corpus)["case"]
    assert case.id == "case"
    assert case.sentences == ["Raw   text!", "next line"]
    assert case.labelled_sentences == ["Original  spacing", "Second.", "Third.", "More  text."]
    assert case.labels == ["Facts", "Argument", None, None]
    assert case.references == ["First reference.\nSecond line.", "Another reference."]


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        ("Text.\tUnknown", "unknown role"),
        ("Text.    Unknown", "unknown role"),
        ("Missing delimiter", "missing role delimiter"),
        ("\tFacts", "empty annotated sentence"),
    ],
)
def test_malformed_annotation_includes_file_and_line_number(corpus, row, reason):
    path = corpus / "Labelled_Data" / "case.txt"
    path.write_text(f"Valid.\tFacts\n{row}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"case\.txt:2: {reason}"):
        load_cases(corpus)


def test_reports_each_unmatched_case_id(corpus):
    directory = corpus / "Summary_2"
    (directory / "case.txt").rename(directory / "other.txt")
    with pytest.raises(ValueError) as error:
        load_cases(corpus)
    assert "Summary_2" in str(error.value)
    assert "missing=['case'], extra=['other']" in str(error.value)


@pytest.mark.parametrize("directory", ["Judgement", "Labelled_Data", "Summary", "Summary_2"])
def test_rejects_empty_sources(corpus, directory):
    path = corpus / directory / "case.txt"
    path.write_text(" \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty") as error:
        load_cases(corpus)
    assert str(path) in str(error.value)


def test_bundled_corpus():
    root = Path(__file__).resolve().parents[1]
    assert ROLES == tuple((root / "RR.txt").read_text(encoding="utf-8").splitlines())
    cases = load_cases(root)
    assert len(cases) == 50
    assert sum(len(c.sentences) for c in cases.values()) == 8656
    assert sum(len(c.labels) for c in cases.values()) == 9380
    assert all(len(c.labels) == len(c.labelled_sentences) for c in cases.values())
    assert sum(len(c.labels) != len(c.sentences) for c in cases.values()) == 49
    assert {role for c in cases.values() for role in c.labels} == set(ROLES)
    assert cases["1953_L_1"].labels[0] == "Facts"  # space-delimited in the corpus
    assert [c.id for c in cases.values() if len(c.references) == 1] == ["2010_J_55"]
    assert sum(len(c.references) for c in cases.values()) == 99


def test_case_split_is_repeatable_and_disjoint():
    ids = [f"case-{i}" for i in range(50)]
    split = split_case_ids(ids)
    assert split == split_case_ids(reversed(ids), seed=42)
    assert split["test"] != split_case_ids(ids, seed=7)["test"]
    assert {key: len(value) for key, value in split.items()} == {
        "fit": 32, "validation": 8, "train": 40, "test": 10,
    }
    fit, validation, train, test = (set(split[k]) for k in ("fit", "validation", "train", "test"))
    assert fit.isdisjoint(validation)
    assert fit | validation == train
    assert train.isdisjoint(test)
    assert train | test == set(ids)


def test_case_split_rejects_duplicate_ids_and_empty_partitions():
    with pytest.raises(ValueError, match="exactly once"):
        split_case_ids(["a", "b", "c", "a"])
    with pytest.raises(ValueError, match="three cases"):
        split_case_ids(["a", "b"])
    split = split_case_ids(["a", "b", "c"])
    assert all(len(split[k]) == 1 for k in ("fit", "validation", "test"))
