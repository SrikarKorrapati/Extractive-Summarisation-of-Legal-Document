"""Verify synthetic v4 artifacts without fitting or downloading any model."""

import json
from types import SimpleNamespace

import pytest
import torch

from legal_lexrank import role_study
from legal_lexrank import transformer_role_training as training
from legal_lexrank.data import ROLES, split_case_ids
from legal_lexrank.storage import json_bytes


class ReservedTestCase:
    def __getattr__(self, name):
        raise AssertionError(f'Verification accessed held-out test annotations: {name}')


def write_json(path, value):
    path.write_bytes(json_bytes(value))


@pytest.fixture
def bundle(tmp_path):
    split = split_case_ids([f'case{i:02}' for i in range(50)])
    cases = {}
    for case_id in split['train']:
        labels = list(ROLES)
        if case_id in split['validation']:
            labels += [ROLES[-1]] * 3
        cases[case_id] = SimpleNamespace(labels=labels, labelled_sentences=[f'{case_id} sentence {i}' for i in range(len(labels))])
    cases.update({case_id: ReservedTestCase() for case_id in split['test']})
    corpus = training.corpus_manifest(cases, split)
    fit_weights = training.class_weights(cases, split['fit']).tolist()
    refit_weights = training.class_weights(cases, split['train']).tolist()
    candidates = []
    for number, config in enumerate(training.CONFIGURATIONS, 1):
        f1 = .7 if number == 1 else .6
        report = {role: {'precision': f1, 'recall': f1, 'f1-score': f1, 'support': 8} for role in ROLES}
        report['macro avg'] = {'precision': f1, 'recall': f1, 'f1-score': f1, 'support': 56}
        report['accuracy'] = .8 if number == 1 else .9
        candidates.append({'candidate': number, 'config': dict(config), 'best_epoch': 1,
                           'epochs_completed': 3, 'validation': report, 'class_weights': fit_weights,
                           'checkpoint': f'candidate-{number}/checkpoint.pt',
                           'checkpoint_sha256': str(number) * 64})
    selected = candidates[0]
    selection = {
        **corpus, 'format': training.SELECTION_FORMAT, 'architecture': training.ARCHITECTURE,
        'max_epochs': 6, 'patience': 2, 'source_sha256': training.source_hashes(),
        'optimizer': training.OPTIMIZER_SETTINGS,
        'selection_metric': 'validation macro F1 over canonical seven roles, then accuracy; earliest epoch and candidate break ties',
        'candidates': candidates, 'selected': selected,
    }
    directory = tmp_path / 'deployment'
    directory.mkdir()
    write_json(directory / 'selection.json', selection)
    selection_hash = training.file_sha256(directory / 'selection.json')
    config = selected['config']
    model_config = {key: config[key] for key in ('max_length', 'target_chunk_size', 'dropout')}
    model_config['batch_size'] = config['microbatch_size']
    # Primitive metadata and one tiny synthetic tensor suffice for verification;
    # no encoder is instantiated or fitted.
    torch.save({
        'format': 'transformer-role-v1', 'roles': list(ROLES), 'encoder': config['encoder'],
        'encoder_model': config['encoder_model'], 'encoder_revision': config['encoder_revision'],
        'config': model_config, 'state': {'synthetic': torch.zeros(7, 1)},
        'metadata': training.final_metadata(corpus, selection, selection_hash, refit_weights),
    }, directory / 'role_labeller.pt')
    protocol = role_study.study_settings(cases, split, 42, selected['best_epoch'])
    protocol.update({
        **corpus, 'format': training.PROTOCOL_FORMAT, 'architecture': training.ARCHITECTURE,
        'config': config, 'schedule_epochs': 6, 'source_sha256': training.source_hashes(),
        'selection_sha256': selection_hash, 'optimizer': training.OPTIMIZER_SETTINGS,
        'validation_metrics': selected['validation'], 'selected_candidate': selected['candidate'],
        'fit_class_weights': fit_weights, 'refit_class_weights': refit_weights,
        'validation_checkpoint_sha256': selected['checkpoint_sha256'],
        'selection_metric': selection['selection_metric'],
        'refit_policy': 'fresh original pretrained encoder and new head; fixed selected configuration and epoch count; no evaluation',
        'checkpoint_sha256': {'test': training.file_sha256(directory / 'role_labeller.pt')},
    })
    write_json(directory / 'protocol.json', protocol)
    write_json(directory / 'training_history.json', [{'epoch': 1, 'training_loss': 1.}])
    return directory, cases, split, protocol


def verify(bundle):
    directory, cases, split, _ = bundle
    return role_study.verified_checkpoints(directory, cases, split, 42, 8)


def test_deployment_verifies_without_validation_weights_model_loading_or_evaluation(bundle, monkeypatch):
    directory, _, _, protocol = bundle

    def forbidden(*args, **kwargs):
        raise AssertionError('Verification must not construct, fit, or evaluate a model')

    for name in ('fresh_model', 'evaluate', 'tune', 'refit'):
        monkeypatch.setattr(training, name, forbidden)
    monkeypatch.setattr(role_study, 'load_labeller', forbidden)
    assert not (directory / 'validation_labeller.pt').exists()
    assert verify(bundle) == protocol['checkpoint_sha256']
    assert set(verify(bundle)) == {'test'}
    assert protocol['epochs'] == 1  # The unrelated legacy default of 8 is ignored.
    assert protocol['schedule_epochs'] == 6
    assert protocol['fit_class_weights'] != protocol['refit_class_weights']


@pytest.mark.parametrize('field,value', [
    ('epochs', 2), ('schedule_epochs', 1), ('architecture', 'another-model'),
    ('config', {}), ('source_sha256', {}), ('corpus_sha256', {}), ('split_sha256', 'changed'),
    ('case_ids', {}), ('seed', 9), ('roles', list(reversed(ROLES))),
    ('training_sha256', {}), ('labeller_sha256', 'changed'), ('selection_sha256', 'changed'),
    ('selected_candidate', 2), ('fit_class_weights', [1.] * 6), ('refit_class_weights', [1.] * 7),
    ('validation_metrics', {'accuracy': 1.}), ('validation_checkpoint_sha256', 'changed'),
    ('optimizer', {}), ('selection_metric', 'accuracy'), ('refit_policy', 'continue validation model'),
])
def test_protocol_requires_exact_selection_corpus_and_training_provenance(bundle, field, value):
    directory, _, _, protocol = bundle
    protocol[field] = value
    write_json(directory / 'protocol.json', protocol)
    with pytest.raises(ValueError, match=field):
        verify(bundle)


@pytest.mark.parametrize('hashes', [{}, {'validation': 'a'}, {'test': 'a', 'validation': 'b'}])
def test_deployment_requires_exactly_one_final_checkpoint_hash(bundle, hashes):
    directory, _, _, protocol = bundle
    protocol['checkpoint_sha256'] = hashes
    write_json(directory / 'protocol.json', protocol)
    with pytest.raises(ValueError, match='exactly the final test checkpoint'):
        verify(bundle)


def test_final_checkpoint_byte_changes_are_rejected(bundle):
    directory, _, _, _ = bundle
    checkpoint = directory / 'role_labeller.pt'
    checkpoint.write_bytes(checkpoint.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='checkpoint checksum mismatch'):
        verify(bundle)


@pytest.mark.parametrize('field', ['train_case_ids', 'epoch', 'selection_sha256', 'class_weights'])
def test_rehashed_final_checkpoint_still_requires_refit_metadata(bundle, field):
    directory, _, split, protocol = bundle
    checkpoint = directory / 'role_labeller.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    payload['metadata'][field] = split['fit'] if field == 'train_case_ids' else 'changed'
    torch.save(payload, checkpoint)
    protocol['checkpoint_sha256']['test'] = training.file_sha256(checkpoint)
    write_json(directory / 'protocol.json', protocol)
    with pytest.raises(ValueError, match='checkpoint metadata mismatch'):
        verify(bundle)


@pytest.mark.parametrize('phase', ['fit', 'validation'])
def test_changed_training_or_validation_annotations_are_rejected(bundle, phase):
    _, cases, split, _ = bundle
    cases[split[phase][0]].labels[0] = ROLES[-1]
    with pytest.raises(ValueError, match='corpus'):
        verify(bundle)


def test_noncanonical_split_or_seed_is_rejected(bundle):
    directory, cases, split, _ = bundle
    with pytest.raises(ValueError, match='seed-42'):
        role_study.verified_checkpoints(directory, cases, split, 9, 8)
    with pytest.raises(ValueError, match='seed-42'):
        role_study.verified_checkpoints(directory, cases, split_case_ids(cases, 9), 42, 8)


@pytest.mark.parametrize('change', ['source', 'winner', 'class_weights'])
def test_frozen_selection_is_verified_even_if_its_protocol_digest_is_updated(bundle, change):
    directory, _, _, protocol = bundle
    path = directory / 'selection.json'
    selection = json.loads(path.read_text())
    if change == 'source':
        selection['source_sha256']['data'] = 'changed'
        message = 'training source'
    elif change == 'winner':
        selection['selected'] = selection['candidates'][1]
        message = 'validation winner'
    else:
        selection['candidates'][0]['class_weights'] = [2.] * 7
        selection['selected']['class_weights'] = [2.] * 7
        message = 'class weights'
    write_json(path, selection)
    protocol['selection_sha256'] = training.file_sha256(path)
    write_json(directory / 'protocol.json', protocol)
    with pytest.raises(ValueError, match=message):
        verify(bundle)


def test_current_source_change_invalidates_frozen_selection(bundle, monkeypatch):
    original = training.source_hashes()
    monkeypatch.setattr(training, 'source_hashes', lambda: {**original, 'data': 'changed'})
    with pytest.raises(ValueError, match='training source'):
        verify(bundle)
