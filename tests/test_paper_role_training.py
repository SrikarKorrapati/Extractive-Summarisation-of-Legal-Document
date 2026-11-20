import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from legal_lexrank import paper_role_features as features
from legal_lexrank import paper_role_training as training
from legal_lexrank import role_study
from legal_lexrank.data import ROLES, split_case_ids
from legal_lexrank.labeller import load_labeller


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    cases = {f'case{i}': SimpleNamespace(
        labelled_sentences=[f'case{i} sentence{j}' for j in range(8)],
        labels=[*ROLES[:3], None, *ROLES[3:]]) for i in range(6)}
    monkeypatch.setattr(features, 'load_cases', lambda root: cases)
    monkeypatch.setattr(training, 'load_cases', lambda root: cases)
    values = np.zeros((8, 200), dtype=np.float32)
    values[1:, :8] = 3 * np.eye(8, dtype=np.float32)[1:]
    monkeypatch.setattr(features.legal_sent2vec, 'sentence_vectors', lambda *args: values.copy())
    bundle = tmp_path / 'features'
    manifest = features.build(tmp_path, bundle, tmp_path / 'cache')
    return cases, bundle, manifest


def test_raw_features_preserve_zero_and_magnitude_and_reject_tampering(corpus, tmp_path):
    cases, bundle, manifest = corpus
    _, vectors, _ = features.load(tmp_path, bundle)
    split = split_case_ids(cases)
    assert set(vectors) == set(split['train'])
    assert set(vectors).isdisjoint(split['test'])
    matrix = vectors[split['fit'][0]]
    assert float(matrix[0].sum()) == 0
    assert float(matrix[1].norm()) == 3
    manifest['cases'][split['fit'][0]]['labels'][0] = 2
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='annotation mismatch'):
        features.load(tmp_path, bundle)


def test_paper_tuning_and_final_refit_keep_test_cases_reserved(corpus, tmp_path, monkeypatch):
    cases, bundle, manifest = corpus
    split = split_case_ids(cases)
    monkeypatch.setattr(training, 'CONFIGURATIONS', [dict(training.CONFIGURATIONS[0], hidden_size=3)])
    seen_fit, seen_validation = [], []
    real_fit, real_evaluate = training.fit_epoch, training.evaluate
    def fit(model, vectors, labels, order, optimizer, config):
        seen_fit.append(set(order))
        return real_fit(model, vectors, labels, order, optimizer, config)
    def evaluate(model, vectors, labels, ids):
        seen_validation.append(ids)
        return real_evaluate(model, vectors, labels, ids)
    monkeypatch.setattr(training, 'fit_epoch', fit)
    monkeypatch.setattr(training, 'evaluate', evaluate)
    tuning = tmp_path / 'tuning'
    selection = training.tune(tmp_path, bundle, tuning, max_epochs=1, patience=1)
    assert seen_fit == [set(split['fit'])]
    assert seen_validation == [split['validation']]
    assert selection['selected']['best_epoch'] == 1
    seen_fit.clear()
    def forbidden(*args):
        raise AssertionError('Final refitting must not evaluate to alter its settings')
    monkeypatch.setattr(training, 'evaluate', forbidden)
    final = tmp_path / 'final'
    protocol = training.refit(tmp_path, bundle, tuning / 'selection.json', final)
    assert seen_fit == [set(split['train'])]
    assert protocol['epochs'] == 1
    hashes = role_study.verified_checkpoints(final, cases, split, 42, 8)
    assert hashes['test'] == hashlib.sha256((final / 'role_labeller.pt').read_bytes()).hexdigest()
    model = load_labeller(final / 'role_labeller.pt')
    assert model.metadata['train_case_ids'] == split['train']
    assert model.config['hidden_size'] == 3
    assert (final / 'validation_labeller.pt').read_bytes() == (tuning / 'validation_labeller.pt').read_bytes()
    # A modified selected checkpoint is never silently accepted for refitting.
    checkpoint = tuning / 'validation_labeller.pt'
    checkpoint.write_bytes(checkpoint.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        training.refit(tmp_path, bundle, tuning / 'selection.json', tmp_path / 'bad-final')
