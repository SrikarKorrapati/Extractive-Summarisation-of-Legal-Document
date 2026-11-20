import json
from types import SimpleNamespace

import pytest

from legal_lexrank import evaluate_roles
from legal_lexrank.data import ROLES


def test_frozen_evaluation_uses_only_reserved_cases_and_ignores_missing_gold(tmp_path, monkeypatch):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'protocol.json').write_text(json.dumps({'seed': 42}))
    cases = {
        'fit': SimpleNamespace(labelled_sentences=['never predict me'], labels=[ROLES[0]]),
        'heldout': SimpleNamespace(labelled_sentences=['fact', 'missing', 'lower'], labels=[ROLES[0], None, ROLES[-1]]),
    }
    monkeypatch.setattr(evaluate_roles, 'load_cases', lambda root: cases)
    monkeypatch.setattr(evaluate_roles, 'split_case_ids', lambda cases, seed: {'train': ['fit'], 'test': ['heldout']})
    monkeypatch.setattr(evaluate_roles, 'verified_checkpoints', lambda *args: {'test': 'frozen-checkpoint'})
    seen = []
    def predict(sentences):
        seen.append(sentences)
        return [ROLES[0], ROLES[1], ROLES[0]]
    monkeypatch.setattr(evaluate_roles, 'load_labeller', lambda *args, **kwargs: SimpleNamespace(predict=predict))
    output = tmp_path / 'evaluation'
    report = evaluate_roles.evaluate(tmp_path, bundle, output)
    assert seen == [cases['heldout'].labelled_sentences]
    assert report['accuracy'] == .5
    assert report[ROLES[-1]]['recall'] == 0
    provenance = json.loads((output / 'evaluation.json').read_text())
    assert provenance['sentence_count'] == 2
    assert provenance['confusion_matrix'][0][0] == 1
    assert provenance['confusion_matrix'][-1][0] == 1
    assert provenance['checkpoint_sha256'] == 'frozen-checkpoint'
    rows = json.loads((output / 'test_predictions.json').read_text())[0]['sentences']
    assert rows[1]['gold'] is None
    with pytest.raises(ValueError, match='empty evaluation directory'):
        evaluate_roles.evaluate(tmp_path, bundle, output)
