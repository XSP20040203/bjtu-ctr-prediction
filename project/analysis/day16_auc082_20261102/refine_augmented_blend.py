"""Refine fixed blends of the verified augmented-row saved predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from fit_stacker import ROOT, HERE, digest, write
from verify_stacker import rank_sum_auc
from augmented_row_catboost import BASES

WEIGHTS = [.1, .2, .3, .4, .5]


def locations(day):
    return HERE/f'runs/augmented_rows_v1_day{day}', HERE/f'runs/augmented_blend_v1_day{day}'


def evaluate(day, verifying):
    source, out = locations(day)
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf8'))
    for path, h in protocol['sources_sha256'].items():
        assert digest(path) == h, path
    for name, h in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == h, name
    verification = json.loads((source/'verification.json').read_text(encoding='utf8'))
    assert verification['status'] == 'passed' and not verification[f'evaluation{day}_labels_fit']
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
    y = raw.label.to_numpy()[rows]
    assert len(y) == {15: 424780, 16: 421983}[day] and int(y.sum()) == {15: 3523, 16: 3724}[day]
    np.testing.assert_array_equal(rows, np.load(source/'valid_indices.npy'))
    base = np.load(BASES[day])
    a = logit(base.clip(1e-6, 1-1e-6))
    forecasts = {}
    locks = json.loads((source/'models_locked.json').read_text(encoding='utf8'))
    for model in locks['models']:
        p = np.load(source/(model['name']+'_prediction.npy'))
        reference = next(r['auc'] for r in verification['records'] if r['name'] == model['name'] and r['recipe'] == 'single')
        assert abs(rank_sum_auc(y, p)-reference) < 1e-12
        z = logit(p.clip(1e-6, 1-1e-6))
        forecasts[model['name']] = (z-z.mean())*a.std()/max(z.std(), 1e-6)+a.mean()
    last = [z for name, z in forecasts.items() if name.endswith('_trees1200')]
    assert len(last) == 2
    forecasts['mean_last_two'] = np.mean(last, axis=0)
    forecasts['mean_all_six'] = np.mean(list(forecasts.values())[:-1], axis=0)
    winner = np.load(source/'prediction.npy').copy()
    best = {'source': 'original_augmented_winner', 'weight': 0., 'auc': rank_sum_auc(y, winner)}
    assert abs(best['auc']-verification['auc']) < 1e-12
    records = [best]
    for name, z in forecasts.items():
        for weight in WEIGHTS:
            p = expit((1-weight)*a+weight*z)
            record = {'source': name, 'weight': weight, 'auc': rank_sum_auc(y, p)}
            records.append(record)
            if record['auc'] > best['auc']:
                best = record
                winner = p.copy()
    error = 0.
    if verifying:
        np.testing.assert_array_equal(rows, np.load(out/'valid_indices.npy'))
        error = float(np.max(np.abs(winner-np.load(out/'prediction.npy'))))
        assert error < 1e-12
    else:
        np.save(out/'valid_indices.npy', rows)
        np.save(out/'prediction.npy', winner)
    report = {'status': 'passed' if verifying else 'evaluated', 'auc': best['auc'], 'best': best,
              'records': records, 'rows': len(y), 'clicks': int(y.sum()), 'maximum_replay_error': error,
              f'evaluation{day}_labels_fit': False, 'development_day_reused': True,
              'original_datasets_unchanged': True, 'target_met': best['auc'] >= {15: .815, 16: .82}[day]}
    write(out/('verification.json' if verifying else 'result.json'), report)
    print(json.dumps({'day': day, 'status': report['status'], 'best': best}), flush=True)


def freeze(day):
    source, out = locations(day)
    assert not out.exists(), 'Preserving augmented blend experiment'
    verification = json.loads((source/'verification.json').read_text(encoding='utf8'))
    assert verification['status'] == 'passed'
    source_contract = json.loads((source/'input_contract.json').read_text(encoding='utf8'))
    assert source_contract['training_max_date'] < 20191100+day
    paths = [Path(__file__), HERE/'augmented_row_catboost.py', HERE/'fit_stacker.py', HERE/'verify_stacker.py',
             source/'verification.json', source/'models_locked.json', source/'input_contract.json',
             source/'prediction.npy', source/'valid_indices.npy', BASES[day], BASES[day].parent/'verification.json']
    models = json.loads((source/'models_locked.json').read_text(encoding='utf8'))
    paths += [source/(model['name']+'_prediction.npy') for model in models['models']]
    raw_hashes = json.loads((source/'protocol.json').read_text(encoding='utf8'))['raw_datasets_sha256']
    out.mkdir(parents=True)
    write(out/'protocol.json', {'day': 20191100+day, 'weights': WEIGHTS,
          'forecasts': 'Each six normalized row logits, mean of last two, mean of all six',
          'formula': 'sigmoid((1-weight)*original_base_logit+weight*normalized_forecast)',
          'sources_sha256': {str(path): digest(path) for path in paths}, 'raw_datasets_sha256': raw_hashes,
          f'evaluation{day}_labels_fit': False, 'inferred17_labels_read': False,
          'selection': 'Fixed refinement candidates on repeatedly reused development date; no new predictive model fitting'})
    evaluate(day, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=int, choices=[15, 16], required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    evaluate(args.day, True) if args.verify else freeze(args.day)
