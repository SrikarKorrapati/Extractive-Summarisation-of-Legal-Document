"""Case-level access to the legal summarisation corpus and its train/test split."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from random import Random


# Keep the order used in the corpus's RR.txt and role-weight vectors.
ROLES = (
    "Facts",
    "Argument",
    "Statute",
    "Precedent",
    "Ratio of the decision",
    "Ruling by Present Court",
    "Ruling by Lower Court",
)


@dataclass
class Case:
    """Labels index labelled_sentences. Judgements have independent segmentation."""

    id: str
    sentences: list[str]
    labelled_sentences: list[str]
    labels: list[str | None]
    references: list[str]


def _read_annotations(path: Path) -> tuple[list[str], list[str | None]]:
    sentences: list[str] = []
    labels: list[str | None] = []
    for number, row in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not row.strip():
            continue
        if "\t" in row:
            sentence, _, label = row.partition("\t")
        else:
            # A space-delimited sentence can itself contain repeated spaces.
            sentence, separator, label = row.rstrip().rpartition("  ")
            if row.endswith("  ") and label not in ROLES:
                sentence, label = row, ""
            elif not separator:
                raise ValueError(f"{path}:{number}: missing role delimiter")
        sentence, label = sentence.strip(), label.strip()
        if not sentence:
            raise ValueError(f"{path}:{number}: empty annotated sentence")
        if label and label not in ROLES:
            raise ValueError(f"{path}:{number}: unknown role {label!r}")
        sentences.append(sentence)
        # Keep explicitly empty role fields aligned with their source sentences.
        labels.append(label or None)
    if not sentences:
        raise ValueError(f"{path}: empty annotation file")
    return sentences, labels


def load_cases(root: Path) -> dict[str, Case]:
    """Load each case from its four matching UTF-8 text files."""
    sources = {}
    for name in ("Judgement", "Labelled_Data", "Summary", "Summary_2"):
        directory = root / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing corpus directory: {directory}")
        sources[name] = {p.stem: p for p in directory.glob("*.txt") if p.is_file()}

    judgements = sources["Judgement"]
    if not judgements:
        raise ValueError(f"{root / 'Judgement'}: no judgement files")
    for name, files in sources.items():
        missing = sorted(judgements.keys() - files.keys())
        extra = sorted(files.keys() - judgements.keys())
        if missing or extra:
            raise ValueError(f"{root / name}: case ID mismatch (missing={missing}, extra={extra})")

    cases = {}
    for case_id, path in sorted(judgements.items()):
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        sentences = [line.strip() for line in lines if line.strip()]
        if not sentences:
            raise ValueError(f"{path}: empty judgement")
        labelled_sentences, labels = _read_annotations(sources["Labelled_Data"][case_id])
        references = []
        for name in ("Summary", "Summary_2"):
            reference_path = sources[name][case_id]
            reference = reference_path.read_text(encoding="utf-8-sig").strip()
            if not reference:
                raise ValueError(f"{reference_path}: empty reference")
            if reference not in references:
                references.append(reference)
        cases[case_id] = Case(case_id, sentences, labelled_sentences, labels, references)
    return cases


def split_case_ids(ids: Iterable[str], seed: int = 42) -> dict[str, list[str]]:
    """Hold out 20% for test, then 20% of training for validation, rounding up."""
    shuffled = sorted(ids)
    if len(set(shuffled)) != len(shuffled):
        raise ValueError("Each case ID must occur exactly once")
    if len(shuffled) < 3:
        raise ValueError("At least three cases are needed for fit, validation and test")
    Random(seed).shuffle(shuffled)
    n_test = (len(shuffled) + 4) // 5
    train = shuffled[n_test:]
    n_validation = (len(train) + 4) // 5
    return {
        "fit": sorted(train[n_validation:]),
        "validation": sorted(train[:n_validation]),
        "train": sorted(train),
        "test": sorted(shuffled[:n_test]),
    }
