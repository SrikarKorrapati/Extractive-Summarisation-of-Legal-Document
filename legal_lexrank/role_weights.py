"""Estimate rhetorical-role weights directly from all human-annotated judgments."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .data import ROLES, load_cases
from .ranking import fit_weights


def fit_corpus_weights(cases):
    """Give each annotated judgment one vote, independently of the labeller split."""
    case_ids = sorted(cases)
    documents = []
    annotations = hashlib.sha256()
    for case_id in case_ids:
        case = cases[case_id]
        documents.append([label for label in case.labels if label is not None])
        source = json.dumps(
            [case_id, case.labelled_sentences, case.labels],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        annotations.update(source + b"\n")
    return {
        "format": "legal-lexrank-role-weights-v1",
        "roles": list(ROLES),
        "metadata": {
            "source": "human_annotations",
            "scope": "all_loaded_cases",
            "case_ids": case_ids,
            "case_count": len(case_ids),
            "labelled_sentence_count": sum(map(len, documents)),
            "source_annotations_sha256": annotations.hexdigest(),
            "aggregation_unit": "one role-frequency vector per judgment",
            "missing_labels": "excluded from each judgment's role counts",
            "split_usage": "The case split controls role-labeller training and evaluation.",
        },
        "profiles": fit_weights(documents, ROLES),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("."), help="Corpus root")
    parser.add_argument("--output", type=Path, required=True, help="Destination weights.json file")
    args = parser.parse_args(argv)
    bundle = fit_corpus_weights(load_cases(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        temporary_path.write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        os.replace(temporary_path, args.output)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Saved three RR profiles from {len(bundle['metadata']['case_ids'])} judgments to {args.output}")


if __name__ == "__main__":
    main()
