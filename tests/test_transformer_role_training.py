"""Offline protocol tests; these never fit the repository's legal corpus."""

import copy
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from legal_lexrank.data import ROLES, split_case_ids
from legal_lexrank import transformer_role_training as training


class ForbiddenCase:
    def __getattr__(self, name):
        raise AssertionError(f'Held-out test data was accessed: {name}')


class TinyRoleLabeller(nn.Module):
    """A tiny injected encoder/head, with two chunks per synthetic sentence."""
    def __init__(self, config):
        super().__init__()
        self.config = {key: config[key] for key in ('max_length', 'target_chunk_size', 'dropout')}
        self.config['batch_size'] = config['microbatch_size']
        self.encoder = config['encoder']
        self.identity = config
        self.backbone = nn.Embedding(8, 5)
        self.to_roles = nn.Linear(5, len(ROLES))

    def make_examples(self, sentences, labels=None):
        examples = []
        for index, sentence in enumerate(sentences):
            label = None if labels is None else labels[index]
            if labels is not None and label is None:
                continue
            token = int(sentence.rsplit(':', 1)[1])
            for fraction in (.25, .75):
                examples.append({'input_ids': [token], 'sentence_index': index,
                                 'chunk_weight': fraction,
                                 'label': -1 if label is None else ROLES.index(label)})
        return examples

    def collate(self, examples):
        return {'input_ids': torch.tensor([row['input_ids'] for row in examples]),
                'chunk_weight': torch.tensor([row['chunk_weight'] for row in examples]),
                'labels': torch.tensor([row['label'] for row in examples])}

    def forward(self, batch):
        return self.to_roles(self.backbone(batch['input_ids']).mean(dim=1))

    def save(self, path, metadata=None):
        torch.save({'format': 'transformer-role-v1', 'roles': list(ROLES),
                    'encoder': self.encoder, 'encoder_model': self.identity['encoder_model'],
                    'encoder_revision': self.identity['encoder_revision'],
                    'config': self.config, 'metadata': metadata, 'state': self.state_dict()}, path)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    split = split_case_ids([f'case{i:02}' for i in range(50)])
    cases = {}
    for case_id in split['train']:
        labels = list(ROLES) + [None]
        if case_id in split['validation']:
            labels += [ROLES[-1]] * 3
        sentences = [f'{case_id}:{ROLES.index(label) if label else 7}' for label in labels]
        cases[case_id] = SimpleNamespace(labelled_sentences=sentences, labels=labels)
    cases.update({case_id: ForbiddenCase() for case_id in split['test']})
    monkeypatch.setattr(training, 'load_cases', lambda path: cases)
    monkeypatch.setattr(training, 'fresh_model', lambda config, cache_dir=None: TinyRoleLabeller(config))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield cases, split
    torch.set_num_threads(threads)


def test_canonical_split_and_class_weights_use_only_fitting_labels(synthetic):
    cases, split = synthetic
    assert {key: len(value) for key, value in split.items()} == {'fit': 32, 'validation': 8, 'train': 40, 'test': 10}
    manifest = training.corpus_manifest(cases)
    assert manifest['case_ids'] == split
    assert set(manifest['corpus_sha256']) == {'fit', 'validation', 'train'}
    fit_weights = training.class_weights(cases, split['fit'])
    assert torch.equal(fit_weights, torch.ones(7))
    expected = torch.tensor([40.] * 6 + [64.]).rsqrt()
    expected /= expected.mean()
    assert torch.allclose(training.class_weights(cases, split['train']), expected)
    cases[split['validation'][0]].labels = [ROLES[-1]] * len(cases[split['validation'][0]].labels)
    assert torch.equal(training.class_weights(cases, split['fit']), fit_weights)
    assert training.corpus_manifest(cases)['corpus_sha256']['fit'] == manifest['corpus_sha256']['fit']
    assert training.corpus_manifest(cases)['corpus_sha256']['validation'] != manifest['corpus_sha256']['validation']


def test_fractional_chunk_loss_and_prediction_aggregation_preserve_sentence_unit():
    weights = torch.arange(1., 8.)
    logits = torch.tensor([[2., 1., 0., 0., 0., 0., 0.], [0., 1., 2., 0., 0., 0., 0.]], requires_grad=True)
    expected = training.weighted_chunk_loss(logits, torch.tensor([0, 2]), torch.ones(2), weights)
    chunks = torch.stack([logits[0], logits[0], logits[1]])
    actual = training.weighted_chunk_loss(chunks, torch.tensor([0, 0, 2]), torch.tensor([.25, .75, 1.]), weights)
    assert torch.allclose(actual, expected)
    gradient, = torch.autograd.grad(actual, logits, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, logits)
    assert torch.allclose(gradient, expected_gradient)
    examples = [{'sentence_index': 0, 'chunk_weight': .25},
                {'sentence_index': 0, 'chunk_weight': .75},
                {'sentence_index': 1, 'chunk_weight': 1.}]
    values = torch.tensor([[9., 0., 0., 0., 0., 0., 0.],
                           [0., 4., 0., 0., 0., 0., 0.],
                           [0., 0., 3., 0., 0., 0., 0.]])
    aggregated = training.aggregate_sentence_logits(examples, values, 2)
    assert aggregated.argmax(dim=1).tolist() == [1, 2]
    assert aggregated.shape == (2, 7)
    with pytest.raises(ValueError, match='cover every sentence'):
        training.aggregate_sentence_logits(examples[:-1], values[:-1], 2)


def test_evaluation_counts_one_prediction_per_labelled_sentence(synthetic):
    cases, split = synthetic
    config = training.CONFIGURATIONS[0]
    model = TinyRoleLabeller(config)
    model.forward = lambda batch: torch.nn.functional.one_hot(batch['input_ids'][:, 0].clamp_max(6), 7).float()
    ids = split['validation'][:1]
    report = training.evaluate(model, cases, ids, config, torch.device('cpu'))
    assert report['accuracy'] == 1
    assert report['macro avg']['f1-score'] == 1
    assert sum(report[role]['support'] for role in ROLES) == sum(label is not None for label in cases[ids[0]].labels)
    assert report[ROLES[-1]]['support'] == 4


def test_macro_f1_has_priority_and_accuracy_breaks_ties():
    lower_accuracy = {'accuracy': .7, 'macro avg': {'f1-score': .6}}
    higher_accuracy = {'accuracy': .9, 'macro avg': {'f1-score': .5}}
    assert training.score(lower_accuracy) > training.score(higher_accuracy)
    assert training.score({'accuracy': .8, 'macro avg': {'f1-score': .6}}) > training.score(lower_accuracy)


def test_tuning_and_fresh_refit_are_isolated_and_final_refit_never_evaluates(synthetic, tmp_path, monkeypatch):
    cases, split = synthetic
    fit_ids, validation_ids, weight_ids, initial_states = [], [], [], []
    real_examples, real_evaluate, real_weights = training.training_examples, training.evaluate, training.class_weights

    def examples(model, cases_arg, ids):
        fit_ids.append(list(ids))
        return real_examples(model, cases_arg, ids)

    def evaluate(model, cases_arg, ids, config, device):
        validation_ids.append(list(ids))
        return real_evaluate(model, cases_arg, ids, config, device)

    def weights(cases_arg, ids):
        weight_ids.append(list(ids))
        return real_weights(cases_arg, ids)

    def fresh(config, cache_dir=None):
        model = TinyRoleLabeller(config)
        initial_states.append(copy.deepcopy(model.state_dict()))
        return model

    monkeypatch.setattr(training, 'training_examples', examples)
    monkeypatch.setattr(training, 'evaluate', evaluate)
    monkeypatch.setattr(training, 'class_weights', weights)
    monkeypatch.setattr(training, 'fresh_model', fresh)
    tuning = tmp_path / 'tuning'
    selection = training.tune(tmp_path, tuning, max_epochs=1, patience=1, device='cpu')
    assert fit_ids == [split['fit'], split['fit']]
    assert validation_ids == [split['validation'], split['validation']]
    assert weight_ids == [split['fit']]
    assert len(selection['candidates']) == 2
    for number in (1, 2):
        assert (tuning / f'candidate-{number}' / 'checkpoint.pt').is_file()
        history = json.loads((tuning / f'candidate-{number}' / 'history.json').read_text())
        assert len(history) == 1
        assert history[0]['best_epoch'] == 1
    assert selection['selected'] == max(selection['candidates'], key=lambda item: training.score(item['validation']))
    assert all(torch.equal(initial_states[0][name], initial_states[1][name]) for name in initial_states[0])

    def forbidden(*args, **kwargs):
        raise AssertionError('Final refitting must never evaluate to alter the fixed settings')

    monkeypatch.setattr(training, 'evaluate', forbidden)
    final = tmp_path / 'final'
    protocol = training.refit(tmp_path, tuning / 'selection.json', final, device='cpu')
    assert fit_ids[-1] == split['train']
    assert weight_ids[-1] == split['train']
    assert len(initial_states) == 3
    assert all(torch.equal(initial_states[0][name], initial_states[2][name]) for name in initial_states[0])
    assert protocol['format'] == 'legal-lexrank-role-study-v4'
    assert protocol['epochs'] == 1
    assert protocol['validation_metrics'] == selection['selected']['validation']
    assert protocol['fit_class_weights'] != protocol['refit_class_weights']
    assert protocol['checkpoint_sha256'] == {'test': training.file_sha256(final / 'role_labeller.pt')}
    assert protocol['validation_checkpoint_sha256'] == training.file_sha256(tuning / 'validation_labeller.pt')
    assert set(path.name for path in final.iterdir()) == {'role_labeller.pt', 'selection.json', 'protocol.json', 'training_history.json'}
    assert (final / 'selection.json').read_bytes() == (tuning / 'selection.json').read_bytes()
    history = json.loads((final / 'training_history.json').read_text())
    assert len(history) == protocol['epochs']
    assert all(set(row) == {'epoch', 'training_loss'} for row in history)
    expected = training.final_metadata(training.corpus_manifest(cases), selection,
                                       protocol['selection_sha256'], protocol['refit_class_weights'])
    training.verify_checkpoint_metadata(final / 'role_labeller.pt', expected, selection['selected']['config'])
    assert expected['train_case_ids'] == split['train']
    assert not set(expected['train_case_ids']) & set(split['test'])


def test_selection_rejects_source_corpus_checkpoint_and_metadata_tampering(synthetic, tmp_path, monkeypatch):
    cases, split = synthetic
    tuning = tmp_path / 'tuning'
    training.tune(tmp_path, tuning, max_epochs=1, patience=1, device='cpu')
    selection_path = tuning / 'selection.json'
    manifest = training.corpus_manifest(cases)
    training.validate_selection(manifest, selection_path)
    original_sources = training.source_hashes
    with monkeypatch.context() as patch:
        patch.setattr(training, 'source_hashes', lambda: {**original_sources(), 'data': 'changed'})
        with pytest.raises(ValueError, match='training source'):
            training.validate_selection(manifest, selection_path)
    changed = copy.deepcopy(manifest)
    changed['corpus_sha256']['fit'] = 'changed'
    with pytest.raises(ValueError, match='corpus'):
        training.validate_selection(changed, selection_path)
    checkpoint = tuning / 'validation_labeller.pt'
    original_bytes = checkpoint.read_bytes()
    checkpoint.unlink()
    # A deployed bundle retains frozen provenance, but omits the large tuning
    # checkpoint. Only the explicit metadata-only verification may skip it.
    checked, _ = training.validate_selection(manifest, selection_path, verify_checkpoint=False)
    assert checked['format'] == training.SELECTION_FORMAT
    with pytest.raises(FileNotFoundError):
        training.validate_selection(manifest, selection_path)
    checkpoint.write_bytes(original_bytes + b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        training.validate_selection(manifest, selection_path)
    checkpoint.write_bytes(original_bytes)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    payload['metadata']['fit_case_ids'] = split['train']
    torch.save(payload, checkpoint)
    selection = json.loads(selection_path.read_text())
    selection['selected']['checkpoint_sha256'] = training.file_sha256(checkpoint)
    for candidate in selection['candidates']:
        if candidate['candidate'] == selection['selected']['candidate']:
            candidate['checkpoint_sha256'] = selection['selected']['checkpoint_sha256']
    selection_path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match='metadata mismatch'):
        training.validate_selection(manifest, selection_path)


def test_refit_preserves_selection_schedule_horizon_and_early_epoch_tie(synthetic, tmp_path, monkeypatch):
    cases, split = synthetic
    report = training.evaluate(TinyRoleLabeller(training.CONFIGURATIONS[0]), cases,
                               split['validation'], training.CONFIGURATIONS[0], torch.device('cpu'))
    monkeypatch.setattr(training, 'evaluate', lambda *args: copy.deepcopy(report))
    horizons = []
    real_optimizer = training.make_optimizer

    def optimizer(model, config, steps):
        horizons.append(steps)
        return real_optimizer(model, config, steps)

    monkeypatch.setattr(training, 'make_optimizer', optimizer)
    tuning = tmp_path / 'tuning'
    selection = training.tune(tmp_path, tuning, max_epochs=6, patience=2, device='cpu')
    assert selection['selected']['candidate'] == 1
    assert selection['selected']['best_epoch'] == 1
    assert all(candidate['epochs_completed'] == 3 for candidate in selection['candidates'])
    final = tmp_path / 'final'
    protocol = training.refit(tmp_path, tuning / 'selection.json', final, device='cpu')
    assert protocol['epochs'] == 1
    assert protocol['schedule_epochs'] == 6
    # 32 * 7 * 2 fitting chunks; refit adds 8 * (7 + 3) * 2 chunks.
    assert horizons == [14 * 6, 14 * 6, 19 * 6]


def test_real_rng_state_is_seeded_and_restored():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    previous_workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    with training.deterministic_run(42):
        assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
        assert torch.are_deterministic_algorithms_enabled()
        first = (random.random(), np.random.random(), torch.rand(2))
    assert random.getstate() == python_state
    assert np.random.get_state()[0] == numpy_state[0]
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == previous_workspace
    with training.deterministic_run(42):
        second = (random.random(), np.random.random(), torch.rand(2))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_optimizer_uses_encoder_and_head_rates_with_linear_warmup():
    config = training.CONFIGURATIONS[0]
    model = TinyRoleLabeller(config)
    optimizer, scheduler = training.make_optimizer(model, config, steps=20)
    assert optimizer.defaults['weight_decay'] == .01
    assert scheduler.base_lrs == [2e-5, 1e-4]
    assert {id(p) for p in optimizer.param_groups[0]['params']} == {id(p) for p in model.backbone.parameters()}
    assert {id(p) for p in optimizer.param_groups[1]['params']} == {id(p) for p in model.to_roles.parameters()}
    rates = []
    for _ in range(20):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]['lr'])
    assert rates[0] == pytest.approx(1e-5)
    assert rates[1] == pytest.approx(2e-5)
    assert rates[-1] == 0


@pytest.mark.parametrize('encoder', ['inlegalbert', 'deberta'])
def test_training_loop_updates_injected_tiny_transformer_and_head(tmp_path, monkeypatch, encoder):
    from transformers import BertConfig, BertModel, BertTokenizer, DebertaV2Config, DebertaV2Model
    from legal_lexrank.transformer_labeller import TransformerRoleLabeller

    vocabulary = tmp_path / 'vocab.txt'
    vocabulary.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'a', 'b', 'c', 'd']))
    tokenizer = BertTokenizer(vocab_file=str(vocabulary))
    config = dict(next(row for row in training.CONFIGURATIONS if row['encoder'] == encoder),
                  max_length=16, target_chunk_size=8, microbatch_size=1, gradient_accumulation_steps=2)
    architecture = dict(vocab_size=9, hidden_size=8, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=12, max_position_embeddings=16)
    if encoder == 'deberta':
        # Two layers reuse the relative-position graph; this reproduces the
        # checkpointing dependency present in the real pretrained backbone.
        architecture['num_hidden_layers'] = 2
        backbone = DebertaV2Model(DebertaV2Config(
            **architecture, type_vocab_size=0, relative_attention=True,
            position_buckets=16, pos_att_type=['p2c', 'c2p']))
        tokenizer.model_input_names = ['input_ids', 'attention_mask']
    else:
        backbone = BertModel(BertConfig(**architecture))
    monkeypatch.setattr(training, 'TransformerRoleLabeller',
                        lambda **kwargs: TransformerRoleLabeller(**kwargs, tokenizer=tokenizer, backbone=backbone))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with training.deterministic_run():
            model = training.fresh_model(config)
            assert model.backbone.is_gradient_checkpointing
            assert training.OPTIMIZER_SETTINGS['gradient_checkpointing_use_reentrant'] is False
            assert model.backbone.encoder._gradient_checkpointing_func.keywords['use_reentrant'] is False
            cases = {'synthetic': SimpleNamespace(labelled_sentences=['a b', 'b c', 'c d', 'd a'], labels=list(ROLES[:4]))}
            examples = training.training_examples(model, cases, ['synthetic'])
            optimizer, scheduler = training.make_optimizer(model, config, steps=3)
            before_backbone = [value.detach().clone() for value in model.backbone.parameters()]
            before_head = model.to_roles.weight.detach().clone()
            loss = training.fit_epoch(model, examples, optimizer, scheduler,
                                      torch.cuda.amp.GradScaler(enabled=False), torch.ones(7),
                                      config, random.Random(42), torch.device('cpu'))
            assert np.isfinite(loss)
            assert any(not torch.equal(before, after) for before, after in zip(before_backbone, model.backbone.parameters()))
            assert not torch.equal(before_head, model.to_roles.weight)
            if encoder == 'deberta':
                relative_embeddings = model.backbone.encoder.rel_embeddings.weight
                assert relative_embeddings.grad is not None
                assert torch.isfinite(relative_embeddings.grad).all()
                assert torch.count_nonzero(relative_embeddings.grad) > 0
            report = training.evaluate(model, cases, ['synthetic'], config, torch.device('cpu'))
            assert sum(report[role]['support'] for role in ROLES) == 4
            path = tmp_path / 'checkpoint.pt'
            model.save(path, metadata={'synthetic': True})
            training.verify_checkpoint_metadata(path, {'synthetic': True}, config)
    finally:
        torch.set_num_threads(threads)
