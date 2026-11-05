"""Combine independently verified binary/count response corrections."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from fit_stacker import ROOT, HERE, digest, write
from verify_stacker import rank_sum_auc

OUT = HERE/'runs/compose_rich_response_v1'
WEIGHTS = [.25, .5, .75, 1.]
DEFINITIONS = {
    15: {'binary': ('rich_binary15_completed_v1', 'prediction.npy', 'valid_indices.npy', False),
         'count': ('rolling_rich_count_v1', 'day15_prediction.npy', 'day15_indices.npy', True),
         'anchor': ('catboost_global_rows15_v1', 'prediction.npy', 'valid_indices.npy', False)},
    16: {'binary': ('rich_binary16_cpu_v1', 'prediction.npy', 'valid_indices.npy', False),
         'count': ('rolling_rich_count_v1', 'day16_prediction.npy', 'day16_indices.npy', True),
         'anchor': ('rolling_previous_app_v1', 'day16_prediction.npy', 'day16_indices.npy', True)},
}


def evaluate(verifying):
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    results = {}
    for day, definitions in DEFINITIONS.items():
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
        y = raw.label.to_numpy()[rows]
        assert len(y) == {15: 424780, 16: 421983}[day]
        assert int(y.sum()) == {15: 3523, 16: 3724}[day]
        values = {}
        for title, (directory, filename, indexfile, rolling) in definitions.items():
            target = HERE/'runs'/directory
            verified = json.loads((target/'verification.json').read_text(encoding='utf8'))
            assert verified['status'] == 'passed'
            entry = verified['dates'][str(20191100+day)] if rolling else verified
            if rolling:
                assert not entry.get('evaluation_day_labels_fit', entry.get('own_held_day_ad_labels_fit', True))
            else:
                assert not entry[f'evaluation{day}_labels_fit']
            p = np.load(target/filename)
            np.testing.assert_array_equal(rows, np.load(target/indexfile))
            assert np.isfinite(p).all() and p.shape == y.shape
            assert abs(rank_sum_auc(y, p)-entry['auc']) < 1e-12
            values[title] = p
        best = {'formula': 'binary', 'weight': 0., 'auc': rank_sum_auc(y, values['binary'])}
        best_p = values['binary'].copy()
        records = [best]
        transformed = {name: logit(p.clip(1e-6, 1-1e-6)) for name, p in values.items()}
        for weight in WEIGHTS:
            candidates = {
                'count_delta': expit(transformed['binary']+weight*(transformed['count']-transformed['anchor'])),
                'logit_blend': expit((1-weight)*transformed['binary']+weight*transformed['count'])}
            for formula, p in candidates.items():
                record = {'formula': formula, 'weight': weight, 'auc': rank_sum_auc(y, p)}
                records.append(record)
                if record['auc'] > best['auc']:
                    best = record
                    best_p = p.copy()
        error = 0.
        if verifying:
            np.testing.assert_array_equal(rows, np.load(OUT/f'day{day}_indices.npy'))
            error = float(np.max(np.abs(best_p-np.load(OUT/f'day{day}_prediction.npy'))))
            assert error < 1e-12
        else:
            np.save(OUT/f'day{day}_indices.npy', rows)
            np.save(OUT/f'day{day}_prediction.npy', best_p)
        results[str(20191100+day)] = {'auc': best['auc'], 'best': best, 'records': records,
            'rows': len(y), 'clicks': int(y.sum()), 'maximum_replay_error': error,
            'evaluation_day_labels_fit': False, 'own_held_day_ad_labels_fit': False,
            'target_met': best['auc'] >= {15: .815, 16: .82}[day]}
        print(json.dumps({'day': day, 'best': best, 'verifying': verifying}), flush=True)
    report = {'status': 'passed' if verifying else 'evaluated', 'dates': results,
              'original_datasets_unchanged': True,
              'both_targets_met': all(record['target_met'] for record in results.values())}
    write(OUT/('verification.json' if verifying else 'result.json'), report)


def freeze():
    assert not OUT.exists(), 'Preserving composition experiment'
    expected = json.loads((HERE/'runs/weighted_v1_day16/protocol.json').read_text(encoding='utf8'))['raw_datasets_sha256']
    paths = [Path(__file__), HERE/'fit_stacker.py', HERE/'verify_stacker.py']
    for definitions in DEFINITIONS.values():
        for directory, filename, indexfile, rolling in definitions.values():
            target = HERE/'runs'/directory
            assert json.loads((target/'verification.json').read_text(encoding='utf8'))['status'] == 'passed'
            paths += [target/'verification.json', target/filename, target/indexfile]
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json', {'sources_sha256': {str(path): digest(path) for path in paths},
          'raw_datasets_sha256': expected, 'definitions': DEFINITIONS, 'weights': WEIGHTS,
          'formulas': ['binary', 'binary_logit + weight*(count_logit-anchor_logit)',
                       '(1-weight)*binary_logit+weight*count_logit'],
          'selection': 'Fixed arithmetic declared before own evaluation; both dates repeatedly reused for development',
          'own_evaluation_day_labels_fit': False, 'inferred17_labels_read': False})
    evaluate(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    evaluate(True) if parser.parse_args().verify else freeze()
