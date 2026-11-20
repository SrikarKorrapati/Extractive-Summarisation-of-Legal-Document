"""Export sentence embeddings and aligned role labels for hosted training."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from . import encoders
from .data import ROLES, load_cases, split_case_ids


SEED = 42
BACKENDS = ("sbert", "legalbert", "openai")


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def build(root, output, backend, cache, include_test=False):
    """Publish a complete feature bundle, excluding held-out test examples by default.

    Arrays in features.npz are keyed by case ID and preserve annotation sentence
    order. manifest.json carries aligned integer labels, with -1 for missing labels.
    Only the included cases contribute text or labels to the exported bundle.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown sentence embedding backend: {backend}")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Choose a new or empty feature directory: {output}")

    cases = load_cases(Path(root))
    split = split_case_ids(cases, SEED)
    included = sorted(cases) if include_test else split["train"]
    role_indices = {role: index for index, role in enumerate(ROLES)}
    arrays, metadata = {}, {}
    dimensions = None
    training_hash = hashlib.sha256()
    corpus_hash = hashlib.sha256()
    for case_id in included:
        case = cases[case_id]
        print(f"{backend}: embedding annotated sentences in {case_id}", flush=True)
        values = np.asarray(
            encoders.sentence_embeddings(backend, case.labelled_sentences, Path(cache)),
            dtype=np.float32,
        )
        if (values.ndim != 2 or values.shape[0] != len(case.labelled_sentences)
                or values.shape[1] == 0 or not np.isfinite(values).all()):
            raise ValueError(f"Invalid sentence embeddings for {case_id}")
        if dimensions is None:
            dimensions = values.shape[1]
        if values.shape[1] != dimensions:
            raise ValueError(f"Embedding dimension changed in {case_id}")
        norms = np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError(f"Zero sentence embedding in {case_id}")
        arrays[case_id] = np.asarray(values / norms, dtype=np.float32)

        text = _json_bytes(case.labelled_sentences)
        annotations = _json_bytes([case_id, case.labelled_sentences, case.labels])
        corpus_hash.update(annotations)
        if case_id in split["train"]:
            training_hash.update(annotations)
        metadata[case_id] = {
            "sentence_count": len(case.labelled_sentences),
            "text_sha256": hashlib.sha256(text).hexdigest(),
            "annotation_sha256": hashlib.sha256(annotations).hexdigest(),
            "labels": [role_indices[label] if label is not None else -1 for label in case.labels],
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        features = bundle / "features.npz"
        np.savez_compressed(features, **arrays)
        manifest = {
            "format": "legal-lexrank-role-features-v1",
            "backend": backend,
            "model": encoders.MODEL_IDS[backend],
            "revision": encoders.MODEL_REVISIONS.get(backend),
            "encoder_sha256": hashlib.sha256(Path(encoders.__file__).read_bytes()).hexdigest(),
            "seed": SEED,
            "case_ids": split,
            "included_case_ids": included,
            "include_test": include_test,
            "roles": list(ROLES),
            "dimensions": dimensions,
            "dtype": "float32",
            "normalization": "l2",
            "corpus_sha256": corpus_hash.hexdigest(),
            "training_sha256": training_hash.hexdigest(),
            "features_sha256": hashlib.sha256(features.read_bytes()).hexdigest(),
            "cases": metadata,
        }
        (bundle / "manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(bundle, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True, help="New or empty bundle directory")
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/legal_lexrank"))
    parser.add_argument("--include-test", action="store_true", help="Also export the held-out case features and labels")
    args = parser.parse_args()
    build(args.data, args.output, args.backend, args.cache, args.include_test)


if __name__ == "__main__":
    main()
