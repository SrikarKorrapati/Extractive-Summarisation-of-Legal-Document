"""Evaluate a frozen classifier on the reserved judgments without refitting it."""

import argparse
import hashlib
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .data import ROLES, load_cases, split_case_ids
from .labeller import load_labeller
from .role_study import EPOCHS, verified_checkpoints
from .storage import atomic_write, json_bytes


def evaluate(data, role_run, output, cache=Path('.cache/legal_lexrank/similarities'), device='cpu'):
    role_run, output = Path(role_run), Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'Choose a new or empty evaluation directory: {output}')
    protocol = json.loads((role_run / 'protocol.json').read_text(encoding='utf-8'))
    cases = load_cases(Path(data))
    split = split_case_ids(cases, protocol['seed'])
    hashes = verified_checkpoints(role_run, cases, split, protocol['seed'], EPOCHS)
    model = load_labeller(role_run / 'role_labeller.pt', cache_dir=Path(cache))
    if device != 'cpu':
        model.to(device)
    gold, predictions, records = [], [], []
    for case_id in split['test']:
        case = cases[case_id]
        guesses = model.predict(case.labelled_sentences)
        rows = []
        for index, (text, expected, guess) in enumerate(zip(
                case.labelled_sentences, case.labels, guesses, strict=True)):
            if guess not in ROLES:
                raise ValueError(f'Unknown predicted role for {case_id}, sentence {index}')
            rows.append({'sentence_index': index, 'text': text, 'gold': expected, 'predicted': guess})
            if expected is not None:
                gold.append(expected)
                predictions.append(guess)
        records.append({'case_id': case_id, 'sentences': rows})
        print(f'Evaluated reserved judgment {case_id}', flush=True)
    if not gold:
        raise ValueError('No gold annotations in the reserved judgments')
    report = classification_report(gold, predictions, labels=ROLES, output_dict=True, zero_division=0)
    report['accuracy'] = float(accuracy_score(gold, predictions))
    matrix = confusion_matrix(gold, predictions, labels=ROLES).tolist()
    provenance = {
        'format': 'legal-lexrank-role-evaluation-v1',
        'checkpoint_sha256': hashes['test'],
        'protocol_sha256': hashlib.sha256((role_run / 'protocol.json').read_bytes()).hexdigest(),
        'case_ids': split['test'], 'sentence_count': len(gold), 'roles': list(ROLES),
        'confusion_matrix': matrix, 'confusion_matrix_axes': {'rows': 'gold', 'columns': 'predicted'},
        'selection_metric': protocol.get('selection_metric', 'validation accuracy, then macro F1'),
        'refitting_during_evaluation': False,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(output / 'test_predictions.json', json_bytes(records))
    atomic_write(output / 'role_metrics.json', json_bytes({'test': report}))
    atomic_write(output / 'evaluation.json', json_bytes(provenance))
    print(f"Test accuracy: {report['accuracy']:.4%}; macro F1: {report['macro avg']['f1-score']:.4f}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('.'))
    parser.add_argument('--role-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('.cache/legal_lexrank/similarities'))
    parser.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    args = parser.parse_args()
    evaluate(args.data, args.role_run, args.output, args.cache, args.device)


if __name__ == '__main__':
    main()
