"""Select and refit pretrained, non-generative encoders on judgment-level splits.

The production study runs in hosted Colab. CPU execution exists for synthetic
unit tests. Neither selection nor final refitting evaluates the held-out test set.
"""

import argparse
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.nn import functional as F

from .data import ROLES, load_cases, split_case_ids
from .storage import atomic_write, json_bytes
from .transformer_labeller import TransformerRoleLabeller

SEED = 42
ARCHITECTURE = 'contextual-transformer-role-classifier'
SELECTION_FORMAT = 'transformer-role-selection-v1'
PROTOCOL_FORMAT = 'legal-lexrank-role-study-v4'
CONFIGURATIONS = (
    {
        'encoder': 'inlegalbert', 'encoder_model': 'law-ai/InLegalBERT',
        'encoder_revision': 'b5ecfed8ed6cf9d25a3cb8225a8c52f161f7401a',
        'encoder_learning_rate': 2e-5, 'head_learning_rate': 1e-4,
        'microbatch_size': 4, 'gradient_accumulation_steps': 8,
        'max_length': 512, 'target_chunk_size': 384, 'dropout': .1,
    },
    {
        'encoder': 'deberta', 'encoder_model': 'microsoft/deberta-v3-base',
        'encoder_revision': '8ccc9b6f36199bec6961081d44eb72fb3f7353f3',
        'encoder_learning_rate': 1e-5, 'head_learning_rate': 1e-4,
        'microbatch_size': 2, 'gradient_accumulation_steps': 16,
        'max_length': 512, 'target_chunk_size': 384, 'dropout': .1,
    },
)
OPTIMIZER_SETTINGS = {
    'name': 'AdamW', 'weight_decay': .01, 'warmup_fraction': .1,
    'schedule': 'linear', 'gradient_clip_norm': 1.,
    'gradient_checkpointing': True, 'cuda_precision': 'float16 with GradScaler',
    'class_weights': 'sqrt inverse fit frequency, normalized to arithmetic mean one',
    'loss_reduction': 'sum(weighted chunk CE * target-token fraction) / sum(target-token fractions) per accumulation group',
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    return {name: file_sha256(Path(__file__).with_name(name + '.py'))
            for name in ('transformer_role_training', 'transformer_labeller', 'data', 'storage')}


def _annotation_hash(cases, ids):
    digest = hashlib.sha256()
    for case_id in ids:
        case = cases[case_id]
        digest.update(json_bytes([case_id, case.labelled_sentences, case.labels]))
    return digest.hexdigest()


def corpus_manifest(cases, split=None):
    """Fingerprint annotations used by this study, without reading test labels."""
    split = split_case_ids(cases, SEED) if split is None else split
    expected = split_case_ids(cases, SEED)
    if split != expected:
        raise ValueError('Transformer study requires the canonical seed-42 judgment split')
    return {
        'case_ids': split, 'seed': SEED, 'roles': list(ROLES),
        'split_sha256': hashlib.sha256(json_bytes(split)).hexdigest(),
        'corpus_sha256': {key: _annotation_hash(cases, split[key])
                          for key in ('fit', 'validation', 'train')},
    }


def class_weights(cases, ids):
    """Use only the supplied fitting labels, never validation or test labels."""
    counts = torch.zeros(len(ROLES), dtype=torch.float64)
    for case_id in ids:
        labels = cases[case_id].labels
        if len(labels) != len(cases[case_id].labelled_sentences):
            raise ValueError(f'Annotation alignment mismatch in {case_id}')
        if not any(label is not None for label in labels):
            raise ValueError(f'No annotated sentences in fitting judgment {case_id}')
        for label in labels:
            if label is not None:
                if label not in ROLES:
                    raise ValueError(f'Unknown role in {case_id}: {label!r}')
                counts[ROLES.index(label)] += 1
    if not ids or not counts.sum():
        raise ValueError('Class weights require fitting annotations')
    # A missing class has no fitting examples; a unit pseudocount keeps its
    # otherwise unused weight finite without consulting any other split.
    weights = counts.clamp_min(1).rsqrt()
    return (weights / weights.mean()).to(torch.float32)


def score(report):
    """Macro F1 over all seven roles is primary; accuracy breaks ties."""
    return report['macro avg']['f1-score'], report['accuracy']


@contextmanager
def deterministic_run(seed=SEED):
    """Set real CPU/CUDA RNGs, restoring the caller's RNG state afterwards."""
    previous_workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_benchmark, cudnn_deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = cudnn_benchmark, cudnn_deterministic
        if previous_workspace is None:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        else:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = previous_workspace


def fresh_model(config, cache_dir=None):
    model = TransformerRoleLabeller(
        encoder=config['encoder'], max_length=config['max_length'],
        target_chunk_size=config['target_chunk_size'], dropout=config['dropout'],
        batch_size=config['microbatch_size'], cache_dir=cache_dir)
    model.backbone.gradient_checkpointing_enable()
    if hasattr(model.backbone.config, 'use_cache'):
        model.backbone.config.use_cache = False
    return model


def training_examples(model, cases, ids):
    examples = []
    for case_id in ids:
        case = cases[case_id]
        rows = model.make_examples(case.labelled_sentences, case.labels)
        if not rows:
            raise ValueError(f'No annotated sentences in fitting judgment {case_id}')
        examples.extend(rows)
    return examples


def weighted_chunk_loss(logits, labels, chunk_weights, weights, *, normalizer=None):
    """One complete sentence has total chunk weight one, regardless of length."""
    values = F.cross_entropy(logits.float(), labels, weight=weights.float(), reduction='none')
    fractions = chunk_weights.to(values)
    denominator = fractions.sum() if normalizer is None else normalizer
    return (values * fractions).sum() / denominator


def aggregate_sentence_logits(examples, logits, sentence_count):
    """Return one weighted logit vector per original sentence, not per chunk."""
    if len(examples) != len(logits):
        raise ValueError('Chunk logits and examples are misaligned')
    result = logits.new_zeros((sentence_count, len(ROLES)))
    totals = logits.new_zeros(sentence_count)
    for row, values in zip(examples, logits, strict=True):
        index, fraction = row['sentence_index'], row['chunk_weight']
        if not 0 <= index < sentence_count or not 0 < fraction <= 1:
            raise ValueError('Invalid sentence chunk metadata')
        result[index] += values * fraction
        totals[index] += fraction
    if not torch.allclose(totals, torch.ones_like(totals), atol=1e-5):
        raise ValueError('Chunk fractions must cover every sentence exactly once')
    return result


def _device(device):
    value = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if value.type not in ('cpu', 'cuda'):
        raise ValueError('Use CUDA for the hosted study, or CPU for synthetic tests')
    return value


def _move(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def _autocast(device):
    return torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()


def make_optimizer(model, config, steps):
    backbone = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    head = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    optimizer = torch.optim.AdamW([
        {'params': backbone, 'lr': config['encoder_learning_rate']},
        {'params': head, 'lr': config['head_learning_rate']},
    ], weight_decay=OPTIMIZER_SETTINGS['weight_decay'])
    warmup = math.ceil(steps * OPTIMIZER_SETTINGS['warmup_fraction'])

    def multiplier(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0., (steps - step) / max(1, steps - warmup))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def fit_epoch(model, examples, optimizer, scheduler, scaler, weights, config, rng, device):
    model.train()
    order = list(range(len(examples)))
    rng.shuffle(order)
    microbatch = config['microbatch_size']
    group_size = microbatch * config['gradient_accumulation_steps']
    total_loss, total_fraction = 0., 0.
    for start in range(0, len(order), group_size):
        group = [examples[index] for index in order[start:start + group_size]]
        denominator = sum(row['chunk_weight'] for row in group)
        optimizer.zero_grad(set_to_none=True)
        for offset in range(0, len(group), microbatch):
            batch = _move(model.collate(group[offset:offset + microbatch]), device)
            with _autocast(device):
                logits = model(batch)
                loss = weighted_chunk_loss(logits, batch['labels'], batch['chunk_weight'], weights,
                                           normalizer=denominator)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite transformer training loss')
            scaler.scale(loss).backward()
            total_loss += float(loss.detach()) * denominator
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), OPTIMIZER_SETTINGS['gradient_clip_norm'])
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale:
            scheduler.step()
        total_fraction += denominator
    if not total_fraction:
        raise ValueError('No training chunks')
    return total_loss / total_fraction


@torch.inference_mode()
def evaluate(model, cases, ids, config, device):
    model.eval()
    gold, predicted = [], []
    for case_id in ids:
        case = cases[case_id]
        # Gold labels never affect tokenization or chunk construction at evaluation.
        examples = model.make_examples(case.labelled_sentences)
        values = []
        for start in range(0, len(examples), config['microbatch_size']):
            batch = _move(model.collate(examples[start:start + config['microbatch_size']]), device)
            with _autocast(device):
                values.append(model(batch).float().cpu())
        logits = aggregate_sentence_logits(examples, torch.cat(values), len(case.labelled_sentences))
        for expected, guess in zip(case.labels, logits.argmax(dim=1).tolist(), strict=True):
            if expected is not None:
                gold.append(expected)
                predicted.append(ROLES[guess])
    if not gold:
        raise ValueError('Validation requires annotated sentences')
    report = classification_report(gold, predicted, labels=list(ROLES), output_dict=True, zero_division=0)
    report['accuracy'] = float(accuracy_score(gold, predicted))
    return report


def new_output(output):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'Choose a new or empty training directory: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _copy_atomic(source, destination):
    destination = Path(destination)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def validation_metadata(manifest, chosen):
    return {
        'seed': SEED, 'epoch': chosen['best_epoch'], 'config': chosen['config'],
        'fit_case_ids': manifest['case_ids']['fit'],
        'validation_case_ids': manifest['case_ids']['validation'],
        'split_sha256': manifest['split_sha256'], 'corpus_sha256': manifest['corpus_sha256'],
        'source_sha256': source_hashes(), 'class_weights': chosen['class_weights'],
        'validation': chosen['validation'],
    }


def final_metadata(manifest, selection, selection_sha256, weights):
    chosen = selection['selected']
    return {
        'seed': SEED, 'epoch': chosen['best_epoch'], 'config': chosen['config'],
        'train_case_ids': manifest['case_ids']['train'],
        'split_sha256': manifest['split_sha256'], 'corpus_sha256': manifest['corpus_sha256'],
        'source_sha256': source_hashes(), 'class_weights': list(weights),
        'selection_sha256': selection_sha256,
    }


def verify_checkpoint_metadata(path, expected, config):
    """Validate a local checkpoint without allocating/downloading an encoder."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    expected_config = {key: config[key] for key in ('max_length', 'target_chunk_size', 'dropout')}
    # The inference microbatch is stored with the architecture settings.
    expected_config['batch_size'] = config['microbatch_size']
    if (payload.get('format') != 'transformer-role-v1' or tuple(payload.get('roles', ())) != ROLES
            or payload.get('encoder') != config['encoder']
            or payload.get('encoder_model') != config['encoder_model']
            or payload.get('encoder_revision') != config['encoder_revision']
            or payload.get('config') != expected_config or payload.get('metadata') != expected):
        raise ValueError('Transformer checkpoint metadata mismatch')


def tune(data, output, max_epochs=6, patience=2, *, device=None, cache_dir=None):
    if (type(max_epochs) is not int or not 1 <= max_epochs <= 6
            or type(patience) is not int or not 1 <= patience <= 2):
        raise ValueError('The transformer study permits 1-6 epochs and patience 1-2')
    output = new_output(output)
    cases = load_cases(Path(data))
    manifest = corpus_manifest(cases)
    split = manifest['case_ids']
    weights = class_weights(cases, split['fit'])
    output.mkdir(exist_ok=True)
    candidates = []
    for number, config in enumerate(CONFIGURATIONS, 1):
        config = dict(config)
        directory = output / f'candidate-{number}'
        directory.mkdir()
        checkpoint = directory / 'checkpoint.pt'
        history, best, best_epoch = [], None, 0
        with deterministic_run():
            selected_device = _device(device)
            model = fresh_model(config, cache_dir).to(selected_device)
            examples = training_examples(model, cases, split['fit'])
            steps = math.ceil(len(examples) / (config['microbatch_size'] * config['gradient_accumulation_steps'])) * max_epochs
            optimizer, scheduler = make_optimizer(model, config, steps)
            scaler = torch.cuda.amp.GradScaler(enabled=selected_device.type == 'cuda')
            rng = random.Random(SEED)
            for epoch in range(1, max_epochs + 1):
                loss = fit_epoch(model, examples, optimizer, scheduler, scaler, weights.to(selected_device), config, rng, selected_device)
                report = evaluate(model, cases, split['validation'], config, selected_device)
                if best is None or score(report) > score(best):
                    best, best_epoch = report, epoch
                    current = {'best_epoch': epoch, 'config': config,
                               'class_weights': weights.tolist(), 'validation': report}
                    model.save(checkpoint, metadata=validation_metadata(manifest, current))
                history.append({'epoch': epoch, 'training_loss': loss, 'validation': report,
                                'best_epoch': best_epoch})
                atomic_write(directory / 'history.json', json_bytes(history))
                print(f"candidate {number}, epoch {epoch}: macro F1 {report['macro avg']['f1-score']:.4f}, "
                      f"accuracy {report['accuracy']:.4f}, lower-court precision/recall/F1 "
                      f"{report['Ruling by Lower Court']['precision']:.4f}/"
                      f"{report['Ruling by Lower Court']['recall']:.4f}/"
                      f"{report['Ruling by Lower Court']['f1-score']:.4f} "
                      f"(support {report['Ruling by Lower Court']['support']}), best epoch {best_epoch}", flush=True)
                if epoch - best_epoch >= patience:
                    break
            candidates.append({
                'candidate': number, 'config': config, 'best_epoch': best_epoch,
                'epochs_completed': len(history), 'validation': best, 'class_weights': weights.tolist(),
                'checkpoint': str(checkpoint.relative_to(output)), 'checkpoint_sha256': file_sha256(checkpoint),
            })
            del model, optimizer, scheduler, scaler, examples
            gc.collect()
            if selected_device.type == 'cuda':
                torch.cuda.empty_cache()
    chosen = max(candidates, key=lambda item: score(item['validation']))
    _copy_atomic(output / chosen['checkpoint'], output / 'validation_labeller.pt')
    selection = {
        'format': SELECTION_FORMAT, **manifest, 'architecture': ARCHITECTURE,
        'max_epochs': max_epochs, 'patience': patience, 'source_sha256': source_hashes(),
        'optimizer': OPTIMIZER_SETTINGS,
        'selection_metric': 'validation macro F1 over canonical seven roles, then accuracy; earliest epoch and candidate break ties',
        'candidates': candidates, 'selected': chosen,
    }
    atomic_write(output / 'selection.json', json_bytes(selection))
    print(f"Selected candidate {chosen['candidate']}, epoch {chosen['best_epoch']}: macro F1 {score(chosen['validation'])[0]:.4f}", flush=True)
    return selection


def validate_selection(manifest, selection_path, *, verify_checkpoint=True):
    selection_path = Path(selection_path)
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if (selection.get('format') != SELECTION_FORMAT or selection.get('architecture') != ARCHITECTURE
            or selection.get('source_sha256') != source_hashes()
            or selection.get('optimizer') != OPTIMIZER_SETTINGS
            or any(selection.get(key) != value for key, value in manifest.items())):
        raise ValueError('Transformer selection differs from corpus, split, or training source')
    candidates, chosen = selection.get('candidates', []), selection.get('selected')
    if (not candidates or chosen != max(candidates, key=lambda item: score(item['validation']))
            or len(candidates) != len(CONFIGURATIONS)
            or [item['config'] for item in candidates] != list(CONFIGURATIONS)):
        raise ValueError('Selection is not the recorded validation winner/configuration comparison')
    if (type(selection.get('max_epochs')) is not int or not 1 <= selection['max_epochs'] <= 6
            or type(selection.get('patience')) is not int or not 1 <= selection['patience'] <= 2
            or type(chosen.get('best_epoch')) is not int
            or not 1 <= chosen['best_epoch'] <= selection['max_epochs']):
        raise ValueError('Invalid selected epoch or stopping settings')
    checkpoint = selection_path.parent / 'validation_labeller.pt'
    if verify_checkpoint:
        if file_sha256(checkpoint) != chosen['checkpoint_sha256']:
            raise ValueError('Selected validation checkpoint checksum mismatch')
        verify_checkpoint_metadata(checkpoint, validation_metadata(manifest, chosen), chosen['config'])
    return selection, checkpoint


def refit(data, selection_path, output, *, device=None, cache_dir=None):
    output = new_output(output)
    cases = load_cases(Path(data))
    manifest = corpus_manifest(cases)
    selection, validation_checkpoint = validate_selection(manifest, selection_path)
    chosen, split = selection['selected'], manifest['case_ids']
    config, epochs = chosen['config'], chosen['best_epoch']
    weights = class_weights(cases, split['train'])
    selection_hash = file_sha256(selection_path)
    protocol = {
        'format': PROTOCOL_FORMAT, **manifest, 'architecture': ARCHITECTURE,
        'epochs': epochs, 'schedule_epochs': selection['max_epochs'],
        'config': config, 'source_sha256': source_hashes(),
        'training_sha256': {'validation': manifest['corpus_sha256']['fit'],
                            'test': manifest['corpus_sha256']['train']},
        'labeller_sha256': file_sha256(Path(__file__).with_name('labeller.py')),
        'selection_sha256': selection_hash, 'validation_metrics': chosen['validation'],
        'selected_candidate': chosen['candidate'], 'optimizer': OPTIMIZER_SETTINGS,
        'fit_class_weights': chosen['class_weights'], 'refit_class_weights': weights.tolist(),
        'validation_checkpoint_sha256': file_sha256(validation_checkpoint),
        'selection_metric': selection['selection_metric'],
        'refit_policy': 'fresh original pretrained encoder and new head; fixed selected configuration and epoch count; no evaluation',
    }
    with TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / 'bundle'
        bundle.mkdir()
        history = []
        with deterministic_run():
            selected_device = _device(device)
            model = fresh_model(config, cache_dir).to(selected_device)
            examples = training_examples(model, cases, split['train'])
            # Keep the selected candidate's epoch-scaled LR schedule. Stopping at
            # the chosen epoch must not accelerate its decay during refitting.
            steps = math.ceil(len(examples) / (config['microbatch_size'] * config['gradient_accumulation_steps'])) * selection['max_epochs']
            optimizer, scheduler = make_optimizer(model, config, steps)
            scaler = torch.cuda.amp.GradScaler(enabled=selected_device.type == 'cuda')
            rng = random.Random(SEED)
            print(f'Refitting from pretrained weights on {len(split["train"])} judgments for {epochs} fixed epochs', flush=True)
            for epoch in range(1, epochs + 1):
                loss = fit_epoch(model, examples, optimizer, scheduler, scaler, weights.to(selected_device), config, rng, selected_device)
                history.append({'epoch': epoch, 'training_loss': loss})
                atomic_write(bundle / 'training_history.json', json_bytes(history))
                print(f'final epoch {epoch}/{epochs}: loss {loss:.4f}', flush=True)
            model.save(bundle / 'role_labeller.pt', metadata=final_metadata(manifest, selection, selection_hash, weights.tolist()))
        protocol['checkpoint_sha256'] = {'test': file_sha256(bundle / 'role_labeller.pt')}
        _copy_atomic(selection_path, bundle / 'selection.json')
        atomic_write(bundle / 'protocol.json', json_bytes(protocol))
        # The selected validation checkpoint remains in the tuning folder. The
        # deployment bundle needs only the refitted model and frozen provenance.
        os.replace(bundle, output)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('tune', 'refit'))
    parser.add_argument('--data', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--max-epochs', type=int, default=6)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--device', choices=('cuda', 'cpu'))
    parser.add_argument('--cache-dir', type=Path)
    args = parser.parse_args()
    if args.stage == 'tune':
        tune(args.data, args.output, args.max_epochs, args.patience, device=args.device, cache_dir=args.cache_dir)
    elif args.selection is None:
        parser.error('--selection is required for refitting')
    else:
        refit(args.data, args.selection, args.output, device=args.device, cache_dir=args.cache_dir)


if __name__ == '__main__':
    main()
