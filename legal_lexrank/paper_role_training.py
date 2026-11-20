"""Tune and refit the original pretrained Hier-BiLSTM-CRF on judgment-level splits."""

import argparse
import hashlib
from itertools import groupby
import json
import os
from pathlib import Path
import random
from tempfile import TemporaryDirectory

from sklearn.metrics import accuracy_score, classification_report
import torch

from .data import ROLES, load_cases
from .paper_labeller import PaperRoleLabeller, load_paper_labeller
from .paper_role_features import SEED, load
from .role_study import study_settings
from .storage import atomic_write, json_bytes

# Architecture stays fixed. Only the optimization schedule is compared.
CONFIGURATIONS = (
    {'hidden_size': 100, 'dropout': .5, 'learning_rate': .01, 'batch_documents': 32},
    {'hidden_size': 100, 'dropout': .5, 'learning_rate': .003, 'batch_documents': 8},
    {'hidden_size': 100, 'dropout': .5, 'learning_rate': .001, 'batch_documents': 4},
)


def source_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name + '.py').read_bytes()).hexdigest()
            for name in ('paper_role_training', 'paper_role_features', 'paper_labeller',
                         'legal_sent2vec', 'labeller', 'data')}


def score(report):
    return report['accuracy'], report['macro avg']['f1-score']


def sequences(vectors, labels, ids):
    for case_id in ids:
        count = 0
        for present, indices in groupby(range(len(labels[case_id])), key=lambda i: int(labels[case_id][i]) >= 0):
            if present:
                positions = list(indices)
                start, stop = positions[0], positions[-1] + 1
                count += stop - start
                yield vectors[case_id][start:stop], labels[case_id][start:stop]
        if count == 0:
            raise ValueError(f'No annotated sentences in fitting judgment {case_id}')


def fit_epoch(model, vectors, labels, order, optimizer, config):
    model.train()
    total_loss, count = 0., 0
    batch_size = config['batch_documents']
    for start in range(0, len(order), batch_size):
        batch = list(sequences(vectors, labels, order[start:start + batch_size]))
        n = sum(len(gold) for _, gold in batch)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss_batch([values for values, _ in batch], [gold for _, gold in batch])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite CRF training loss')
        (loss / n).backward()
        total_loss += float(loss.detach())
        count += n
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
    return total_loss / count


def evaluate(model, vectors, labels, ids):
    gold, predicted = [], []
    for case_id in ids:
        guesses = model.predict_vectors(vectors[case_id])
        for expected, guess in zip(labels[case_id].tolist(), guesses, strict=True):
            if expected >= 0:
                gold.append(ROLES[expected])
                predicted.append(guess)
    if not gold:
        raise ValueError('Validation requires annotated sentences')
    result = classification_report(gold, predicted, labels=ROLES, output_dict=True, zero_division=0)
    result['accuracy'] = float(accuracy_score(gold, predicted))
    return result


def fresh_model(manifest, config):
    return PaperRoleLabeller(ROLES, manifest['encoder_identity'],
                            hidden_size=config['hidden_size'], dropout=config['dropout'])


def new_output(output):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'Choose a new or empty training directory: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def tune(data, features, output, max_epochs=300, patience=40):
    if type(max_epochs) is not int or type(patience) is not int or min(max_epochs, patience) < 1:
        raise ValueError('Epoch limit and patience must be positive integers')
    output = new_output(output)
    manifest, vectors, labels = load(data, features)
    split = manifest['case_ids']
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    output.mkdir(exist_ok=True)
    candidates = []
    for number, config in enumerate(CONFIGURATIONS, 1):
        directory = output / f'candidate-{number}'
        directory.mkdir()
        checkpoint = directory / 'checkpoint.pt'
        best, best_epoch, history = None, 0, []
        rng = random.Random(SEED)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(SEED)
            model = fresh_model(manifest, config)
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            order = list(split['fit'])
            for epoch in range(1, max_epochs + 1):
                rng.shuffle(order)
                loss = fit_epoch(model, vectors, labels, order, optimizer, config)
                report = evaluate(model, vectors, labels, split['validation'])
                if best is None or score(report) > score(best):
                    best, best_epoch = report, epoch
                    model.save(checkpoint, metadata={
                        'seed': SEED, 'epoch': epoch, 'config': config,
                        'fit_case_ids': split['fit'], 'validation_case_ids': split['validation'],
                        'features_sha256': manifest['features_sha256'], 'validation': report,
                    })
                history.append({'epoch': epoch, 'training_loss': loss, 'validation': report, 'best_epoch': best_epoch})
                atomic_write(directory / 'history.json', json_bytes(history))
                if epoch == 1 or epoch % 5 == 0:
                    print(f"candidate {number}, epoch {epoch}: accuracy {report['accuracy']:.4f}, "
                          f"macro F1 {report['macro avg']['f1-score']:.4f}, "
                          f"lower-court recall {report['Ruling by Lower Court']['recall']:.4f}, best epoch {best_epoch}", flush=True)
                if epoch - best_epoch >= patience:
                    break
        candidates.append({'candidate': number, 'config': config, 'best_epoch': best_epoch,
                           'epochs_completed': len(history), 'validation': best,
                           'checkpoint': str(checkpoint.relative_to(output)),
                           'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    chosen = max(candidates, key=lambda item: score(item['validation']))
    atomic_write(output / 'validation_labeller.pt', (output / chosen['checkpoint']).read_bytes())
    selection = {
        'format': 'paper-role-selection-v1', 'seed': SEED, 'feature_manifest': manifest,
        'max_epochs': max_epochs, 'patience': patience, 'source_sha256': source_hashes(),
        'selection_metric': 'validation accuracy, then macro F1; earliest epoch and candidate break ties',
        'candidates': candidates, 'selected': chosen,
    }
    atomic_write(output / 'selection.json', json_bytes(selection))
    print(f"Selected candidate {chosen['candidate']}, epoch {chosen['best_epoch']}: {chosen['validation']['accuracy']:.4%}", flush=True)
    return selection


def validate_selection(manifest, selection_path):
    selection_path = Path(selection_path)
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if (selection.get('format') != 'paper-role-selection-v1'
            or selection.get('seed') != SEED or selection.get('feature_manifest') != manifest
            or selection.get('source_sha256') != source_hashes()):
        raise ValueError('Paper selection differs from the feature bundle or training source')
    chosen = selection['selected']
    if chosen != max(selection['candidates'], key=lambda item: score(item['validation'])):
        raise ValueError('Selection is not the recorded validation winner')
    if type(chosen['best_epoch']) is not int or not 1 <= chosen['best_epoch'] <= selection['max_epochs']:
        raise ValueError('Invalid selected epoch')
    checkpoint = selection_path.parent / 'validation_labeller.pt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != chosen['checkpoint_sha256']:
        raise ValueError('Selected validation checkpoint checksum mismatch')
    model = load_paper_labeller(checkpoint)
    expected = {
        'seed': SEED, 'epoch': chosen['best_epoch'], 'config': chosen['config'],
        'fit_case_ids': manifest['case_ids']['fit'], 'validation_case_ids': manifest['case_ids']['validation'],
        'features_sha256': manifest['features_sha256'], 'validation': chosen['validation'],
    }
    if (model.metadata != expected or model.encoder_identity != manifest['encoder_identity']
            or model.roles != ROLES or model.config != {key: chosen['config'][key] for key in ('hidden_size', 'dropout')}):
        raise ValueError('Selected checkpoint metadata mismatch')
    return selection, checkpoint


def refit(data, features, selection_path, output):
    output = new_output(output)
    manifest, vectors, labels = load(data, features)
    selection, checkpoint = validate_selection(manifest, selection_path)
    chosen, split = selection['selected'], manifest['case_ids']
    config, epochs = chosen['config'], chosen['best_epoch']
    protocol = study_settings(load_cases(Path(data)), split, SEED, epochs)
    protocol.update({
        'format': 'legal-lexrank-role-study-v3', 'architecture': 'paper-pretrained-Hier-BiLSTM-CRF',
        'encoder_identity': manifest['encoder_identity'], 'config': config,
        'features_sha256': manifest['features_sha256'], 'source_sha256': source_hashes(),
        'selection_sha256': hashlib.sha256(Path(selection_path).read_bytes()).hexdigest(),
        'validation_metrics': chosen['validation'], 'selected_candidate': chosen['candidate'],
        'optimizer': {'name': 'Adam', 'gradient_clip_norm': 5., 'loss_reduction': 'mean per labelled sentence in batch'},
    })
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / 'bundle'
        bundle.mkdir()
        rng = random.Random(SEED)
        history = []
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(SEED)
            model = fresh_model(manifest, config)
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            order = list(split['train'])
            print(f'Refitting on {len(order)} judgments for {epochs} fixed epochs', flush=True)
            for epoch in range(1, epochs + 1):
                rng.shuffle(order)
                loss = fit_epoch(model, vectors, labels, order, optimizer, config)
                history.append({'epoch': epoch, 'training_loss': loss})
                if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                    print(f'final epoch {epoch}/{epochs}: loss {loss:.4f}', flush=True)
            model.save(bundle / 'role_labeller.pt', metadata={
                'seed': SEED, 'epoch': epochs, 'config': config, 'train_case_ids': split['train'],
                'features_sha256': manifest['features_sha256'], 'selection_sha256': protocol['selection_sha256'],
            })
        atomic_write(bundle / 'validation_labeller.pt', checkpoint.read_bytes())
        protocol['checkpoint_sha256'] = {
            phase: hashlib.sha256((bundle / filename).read_bytes()).hexdigest()
            for phase, filename in (('validation', 'validation_labeller.pt'), ('test', 'role_labeller.pt'))}
        atomic_write(bundle / 'training_history.json', json_bytes(history))
        atomic_write(bundle / 'protocol.json', json_bytes(protocol))
        os.replace(bundle, output)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('tune', 'refit'))
    parser.add_argument('--data', type=Path, default=Path('.'))
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--max-epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)
    args = parser.parse_args()
    if args.stage == 'tune':
        tune(args.data, args.features, args.output, args.max_epochs, args.patience)
    elif args.selection is None:
        parser.error('--selection is required for refitting')
    else:
        refit(args.data, args.features, args.selection, args.output)


if __name__ == '__main__':
    main()
