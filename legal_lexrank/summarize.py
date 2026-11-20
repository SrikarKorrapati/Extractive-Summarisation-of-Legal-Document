"""Extract a summary of a new, sentence-segmented judgment using a fitted experiment."""

import argparse
import json
from pathlib import Path

from .encoders import BACKENDS, similarity_matrix
from .experiment import METHODS, RATIOS
from .labeller import load_labeller
from .ranking import rank_sentences, select_indices


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 judgment with one sentence per nonempty line")
    parser.add_argument("--backend", choices=BACKENDS, default="tfidf")
    parser.add_argument("--ratio", type=float, choices=RATIOS, default=0.3,
                        help="fraction of input sentences to extract (default 0.3)")
    parser.add_argument("--results", type=Path, default=Path("results"),
                        help="directory containing the fitted model and experiment reports")
    parser.add_argument("--cache", type=Path, default=Path(".cache/legal_lexrank"))
    parser.add_argument("--method", choices=("unweighted", *METHODS),
                        help="override the backend's validation-selected weighting method")
    args = parser.parse_args(argv)
    try:
        lines = args.input.read_text(encoding="utf-8-sig").splitlines()
        sentences = [line.strip() for line in lines if line.strip()]
        if not sentences:
            parser.error(f"{args.input} contains no sentences")
        method = args.method
        if method is None:
            report = json.loads((args.results / f"{args.backend}.json").read_text(encoding="utf-8"))
            method = report["selected_method"]
        weights = None
        labels = None
        if method != "unweighted":
            profiles = json.loads((args.results / "weights.json").read_text(encoding="utf-8"))
            weights = profiles["test"][method]
            labeller = load_labeller(args.results / "role_labeller.pt", cache_dir=args.cache / "similarities")
            labels = labeller.predict(sentences)
        cosine = similarity_matrix(args.backend, sentences, args.cache / "similarities")
        ranking = rank_sentences(cosine, labels, weights)
        for index in select_indices(ranking, args.ratio):
            print(sentences[index])
    except (FileNotFoundError, UnicodeError, ValueError, KeyError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
