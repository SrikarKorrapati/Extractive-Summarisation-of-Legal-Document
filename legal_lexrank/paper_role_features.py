"""Export and verify raw legal sent2vec features for the paper's role classifier."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from . import legal_sent2vec
from .data import ROLES, load_cases, split_case_ids
from .storage import json_bytes

SEED = 42
FORMAT = 'legal-sent2vec-role-features-v1'


def annotation_metadata(case_id, case):
    return {
        'sentence_count': len(case.labels),
        'annotation_sha256': hashlib.sha256(json_bytes([case_id, case.labelled_sentences, case.labels])).hexdigest(),
        'labels': [ROLES.index(role) if role is not None else -1 for role in case.labels],
    }


def build(data, output, cache):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'Choose a new or empty feature directory: {output}')
    cases = load_cases(Path(data))
    split = split_case_ids(cases, SEED)
    arrays, metadata = {}, {}
    for case_id in split['train']:
        case = cases[case_id]
        vectors = legal_sent2vec.sentence_vectors(case.labelled_sentences, Path(cache))
        values = np.asarray(vectors, dtype=np.float32)
        if values.shape != (len(case.labels), 200) or not np.isfinite(values).all():
            raise ValueError(f'Invalid legal sent2vec features: {case_id}')
        # Unknown/very short fragments may have a zero vector. Preserve their
        # sequence position and label, as in the original pretrained model.
        arrays[case_id] = values
        metadata[case_id] = annotation_metadata(case_id, case)
        print(f'Exported {case_id}: {len(values)} sentences', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / 'bundle'
        bundle.mkdir()
        np.savez_compressed(bundle / 'features.npz', **arrays)
        manifest = {
            'format': FORMAT, 'seed': SEED, 'case_ids': split,
            'included_case_ids': split['train'], 'roles': list(ROLES),
            'dimensions': 200, 'normalization': 'none', 'dtype': 'float32',
            'encoder_identity': legal_sent2vec.encoder_identity(),
            'cases': metadata,
            'features_sha256': hashlib.sha256((bundle / 'features.npz').read_bytes()).hexdigest(),
        }
        (bundle / 'manifest.json').write_bytes(json_bytes(manifest))
        os.replace(bundle, output)
    return manifest


def load(data, features):
    """Accept exactly the 40 training judgments, including their validation subset."""
    import torch

    features = Path(features)
    manifest = json.loads((features / 'manifest.json').read_text(encoding='utf-8'))
    cases = load_cases(Path(data))
    split = split_case_ids(cases, SEED)
    expected = {
        'format': FORMAT, 'seed': SEED, 'case_ids': split,
        'included_case_ids': split['train'], 'roles': list(ROLES),
        'dimensions': 200, 'normalization': 'none', 'dtype': 'float32',
        'encoder_identity': legal_sent2vec.encoder_identity(),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f'Paper feature manifest mismatch: {key}')
    if set(manifest['cases']) != set(split['train']):
        raise ValueError('Feature annotations must contain exactly the training judgments')
    file = features / 'features.npz'
    if hashlib.sha256(file.read_bytes()).hexdigest() != manifest['features_sha256']:
        raise ValueError('Paper feature checksum mismatch')
    vectors, labels = {}, {}
    with np.load(file, allow_pickle=False) as arrays:
        if len(arrays.files) != len(split['train']) or set(arrays.files) != set(split['train']):
            raise ValueError('Feature arrays must contain exactly the training judgments')
        for case_id in split['train']:
            metadata = annotation_metadata(case_id, cases[case_id])
            if manifest['cases'][case_id] != metadata:
                raise ValueError(f'Feature annotation mismatch: {case_id}')
            values = arrays[case_id]
            if (values.shape != (metadata['sentence_count'], 200)
                    or values.dtype != np.float32 or not np.isfinite(values).all()):
                raise ValueError(f'Invalid raw paper features: {case_id}')
            vectors[case_id] = torch.from_numpy(values.copy())
            labels[case_id] = torch.tensor(metadata['labels'], dtype=torch.long)
    return manifest, vectors, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('.cache/legal_sent2vec'))
    args = parser.parse_args()
    build(args.data, args.output, args.cache)


if __name__ == '__main__':
    main()
