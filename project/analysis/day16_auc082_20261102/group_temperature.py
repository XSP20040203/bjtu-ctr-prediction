"""Fixed label-free group transforms, locked before reused-day recipe selection."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from fit_stacker import HERE, ROOT, write, digest
from rich_userday_activity import array_hash
from verify_stacker import rank_sum_auc

OUT = HERE/'runs/group_temperature_v1'
BASE = HERE/'runs/news_quality_daily_v1'
FIELDS = ['user_id', 'task_id', 'slot_id', 'spread_app_id']
GROUPS = {'user': ['user_id'], 'user_app': ['user_id', 'spread_app_id'],
          'user_slot': ['user_id', 'slot_id'], 'task': ['task_id']}
TEMPERATURES = [.5, .75, 1., 1.25, 1.5, 2.]
MODES = ['logit_mean', 'preserve_hazard_mass']


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def group_codes(frame, fields):
    if set(frame.columns) != set(FIELDS):
        raise ValueError('Group transforms accept only the four declared unlabeled IDs')
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(frame[fields]), sort=True)
    assert np.all(codes >= 0)
    return codes


def transform(p, groups, temperature, mode):
    if temperature <= 0 or mode not in MODES:
        raise ValueError('Positive temperature and declared normalization are required')
    p = np.asarray(p, dtype='float64')
    if len(p) != len(groups) or not np.isfinite(p).all() or not np.all((p > 0)&(p < 1)):
        raise ValueError('Aligned finite interior probabilities required')
    if temperature == 1.:
        return p.copy()
    z = logit(p.clip(1e-10, 1-1e-10))
    size = np.bincount(groups)
    mean = np.bincount(groups, weights=z)/size.clip(1)
    q = expit(mean[groups]+temperature*(z-mean[groups]))
    if mode == 'preserve_hazard_mass':
        old_hazard = -np.log1p(-p)
        new_hazard = -np.log1p(-q.clip(1e-12, 1-1e-12))
        ratio = np.bincount(groups, weights=old_hazard)/np.bincount(groups, weights=new_hazard).clip(1e-15)
        q = -np.expm1(-new_hazard*ratio[groups])
    assert np.isfinite(q).all() and np.all((q > 0)&(q < 1))
    return q


def recipes():
    yield {'name': 'base', 'group': 'user', 'temperature': 1., 'mode': 'logit_mean'}
    for group in GROUPS:
        for temperature in TEMPERATURES:
            if temperature == 1.:
                continue
            for mode in MODES:
                yield {'name': f'{group}_{mode}_t{temperature:g}', 'group': group,
                       'temperature': temperature, 'mode': mode}


def prepare():
    assert not OUT.exists(), 'Preserve all earlier and failed group transformations'
    verified = read(BASE/'verification.json')
    assert verified['status'] == 'passed'
    raw_hashes = read(BASE/'protocol.json')['raw_datasets_sha256']
    for name, expected in raw_hashes.items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    paths = [Path(__file__), HERE/'test_group_temperature.py', BASE/'verification.json',
             BASE/'protocol.json', HERE/'fit_stacker.py', HERE/'verify_stacker.py',
             HERE/'rich_userday_activity.py']
    for day in [15, 16]:
        paths += [BASE/f'day{day}_prediction.npy', BASE/f'day{day}_indices.npy']
        assert not verified['dates'][str(20191100+day)]['evaluation_day_labels_fit']
    OUT.mkdir(parents=True)
    protocol = {'dates': [20191115, 20191116], 'groups': GROUPS, 'temperatures': TEMPERATURES,
        'modes': MODES, 'recipes': list(recipes()), 'new_model_fitting': False,
        'transform_inputs': 'Frozen probabilities plus supplied same-date unlabeled ID groups',
        'selection': 'All transformed predictions for both dates locked before held label scoring; repeatedly reused development dates',
        'held_labels_used_to_select_fixed_recipe': True,
        'inferred17_labels_used': False, 'raw_datasets_sha256': raw_hashes,
        'sources_sha256': {str(p): digest(p) for p in paths}}
    write(OUT/'protocol.json', protocol)
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=FIELDS+['pt_d'], dtype='int32')
    files = {}
    for day in [15, 16]:
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
        np.testing.assert_array_equal(rows, np.load(BASE/f'day{day}_indices.npy'))
        np.save(OUT/f'day{day}_indices.npy', rows)
        frame = raw.iloc[rows][FIELDS]
        groups = {name: group_codes(frame, fields) for name, fields in GROUPS.items()}
        base = np.load(BASE/f'day{day}_prediction.npy')
        assert len(base) == len(rows)
        for record in protocol['recipes']:
            p = transform(base, groups[record['group']], record['temperature'], record['mode'])
            path = OUT/f'day{day}_{record["name"]}.npy'
            np.save(path, p)
            files[path.name] = digest(path)
        emit('group_transforms_frozen', day=day, rows=len(rows), recipes=len(protocol['recipes']))
    write(OUT/'predictions_locked.json', {'protocol_sha256': digest(OUT/'protocol.json'),
        'files_sha256': files, 'ad_outcomes_opened_for_transform_generation': False})


def verify():
    assert not (OUT/'verification.json').exists(), 'Preserve a completed verification'
    protocol = read(OUT/'protocol.json')
    locked = read(OUT/'predictions_locked.json')
    assert locked['protocol_sha256'] == digest(OUT/'protocol.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    for name, expected in locked['files_sha256'].items():
        assert digest(OUT/name) == expected, name
    # Fresh process, raw identifier reconstruction and complete candidate replay precede scoring.
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=FIELDS+['pt_d'], dtype='int32')
    errors, inputs = {}, {}
    for day in [15, 16]:
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
        np.testing.assert_array_equal(rows, np.load(OUT/f'day{day}_indices.npy'))
        frame = raw.iloc[rows][FIELDS]
        groups = {name: group_codes(frame, fields) for name, fields in GROUPS.items()}
        base = np.load(BASE/f'day{day}_prediction.npy')
        error = 0.
        for record in protocol['recipes']:
            replay = transform(base, groups[record['group']], record['temperature'], record['mode'])
            p = np.load(OUT/f'day{day}_{record["name"]}.npy')
            error = max(error, float(np.max(np.abs(p-replay))))
        assert error == 0.
        errors[day] = error
        inputs[day] = rows
        emit('group_transforms_replayed', day=day, maximum_error=error)
    labels = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['label'], dtype='int32').label.to_numpy()
    results = {}
    for day in [15, 16]:
        y = labels[inputs[day]]
        assert len(y) == {15: 424780, 16: 421983}[day] and int(y.sum()) == {15: 3523, 16: 3724}[day]
        scores = []
        for record in protocol['recipes']:
            p = np.load(OUT/f'day{day}_{record["name"]}.npy')
            score = rank_sum_auc(y, p)
            scores.append(dict(record, auc=score))
        best = max(scores, key=lambda r: r['auc'])
        p = np.load(OUT/f'day{day}_{best["name"]}.npy')
        np.save(OUT/f'day{day}_prediction.npy', p)
        np.testing.assert_array_equal(np.load(OUT/f'day{day}_prediction.npy'), p)
        assert abs(float(roc_auc_score(y, p))-best['auc']) < 1e-12
        results[str(20191100+day)] = {'auc': best['auc'], 'best': best, 'records': scores,
            'rows': len(y), 'clicks': int(y.sum()), 'maximum_replay_error': errors[day],
            'evaluation_day_labels_fit': False, 'held_labels_used_to_select_fixed_recipe': True,
            'target_met': best['auc'] >= {15: .815, 16: .82}[day]}
        emit('group_temperature_scored', day=day, best=best)
    write(OUT/'verification.json', {'status': 'passed', 'dates': results,
        'original_datasets_unchanged': True,
        'both_targets_met': all(r['target_met'] for r in results.values())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    verify() if parser.parse_args().verify else prepare()
