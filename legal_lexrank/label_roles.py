"""Assign rhetorical roles to the sentences of a new legal judgment."""

import argparse
from pathlib import Path

from .labeller import load_labeller


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 judgment, one sentence per nonempty line")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained role-labeller checkpoint")
    parser.add_argument("--cache", type=Path, default=Path(".cache/legal_lexrank/similarities"))
    args = parser.parse_args(argv)
    sentences = [line.strip() for line in args.input.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip()]
    if not sentences:
        parser.error("The judgment contains no sentences")
    model = load_labeller(args.checkpoint)
    if hasattr(model, "cache_dir"):
        model.cache_dir = args.cache
    labels = model.predict(sentences)
    for sentence, label in zip(sentences, labels, strict=True):
        print(f"{sentence}\t{label}")


if __name__ == "__main__":
    main()
