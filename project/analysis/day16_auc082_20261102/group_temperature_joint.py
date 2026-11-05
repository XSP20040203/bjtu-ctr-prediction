"""Lock fixed two-stage unlabeled group transforms before reused-day scoring."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from group_temperature import FIELDS, GROUPS, transform, group_codes, read, emit
from fit_stacker import ROOT, HERE, digest, write
from rich_userday_activity import array_hash
from verify_stacker import rank_sum_auc

OUT = HERE/'runs/group_temperature_joint_v1'
BASE = HERE/'runs/news_quality_daily_v1'
TASK_TEMPERATURES = [.75, 1., 1.25, 1.5]
USER_TEMPERATURES = [.5, .75, 1., 1.25]


def recipes():
    for group in ['user', 'user_app', 'user_slot']:
        for task_temperature in TASK_TEMPERATURES:
            for user_temperature in USER_TEMPERATURES:
                for order in ['task_then_user', 'user_then_task']:
                    yield {'name': f'{order}_{group}_task{task_temperature:g}_user{user_temperature:g}',
                           'group': group, 'task_temperature': task_temperature,
                           'user_temperature': user_temperature, 'order': order}


def predict(base, groups, recipe):
    operations = [
        (groups['task'], recipe['task_temperature'], 'logit_mean'),
        (groups[recipe['group']], recipe['user_temperature'], 'preserve_hazard_mass'),
    ]
    if recipe['order'] == 'user_then_task':
        operations.reverse()
    result = base.copy()
    for codes, temperature, mode in operations:
        result = transform(result, codes, temperature, mode)
    return result


def inputs():
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=FIELDS+['pt_d'], dtype='int32')
    result = {}
    for day in [15, 16]:
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
        np.testing.assert_array_equal(rows, np.load(BASE/f'day{day}_indices.npy'))
        frame = raw.iloc[rows][FIELDS]
        result[day] = {'rows': rows, 'base': np.load(BASE/f'day{day}_prediction.npy'),
                       'groups': {name: group_codes(frame, fields) for name, fields in GROUPS.items()}}
    return result


def prepare():
    assert not OUT.exists(), 'Preserve previous and failed joint group experiments'
    assert read(BASE/'verification.json')['status'] == 'passed'
    raw_hashes = read(BASE/'protocol.json')['raw_datasets_sha256']
    for name, expected in raw_hashes.items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    paths = [Path(__file__), HERE/'group_temperature.py', HERE/'test_group_temperature.py',
             HERE/'run_group_temperature_joint.ps1', HERE/'fit_stacker.py',
             HERE/'rich_userday_activity.py', HERE/'verify_stacker.py',
             BASE/'verification.json', BASE/'protocol.json']
    for day in [15, 16]:
        paths += [BASE/f'day{day}_indices.npy', BASE/f'day{day}_prediction.npy']
    protocol = {
        'dates': [20191115, 20191116], 'recipes': list(recipes()),
        'new_model_training': False, 'inferred17_labels_read': False,
        'selection': 'All two-date candidate hashes locked before any held outcomes opened; reused development dates',
        'transform_inputs': 'Verified predictions and four supplied unlabeled IDs only',
        'raw_datasets_sha256': raw_hashes, 'sources_sha256': {str(p): digest(p) for p in paths},
    }
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json', protocol)
    data = inputs()
    locks = {}
    for day in [15, 16]:
        np.save(OUT/f'day{day}_indices.npy', data[day]['rows'])
        locks[str(day)] = {recipe['name']: array_hash(predict(data[day]['base'], data[day]['groups'], recipe))
                           for recipe in protocol['recipes']}
        emit('joint_group_candidates_locked', day=day, candidates=len(locks[str(day)]))
    write(OUT/'predictions_locked.json', {'protocol_sha256': digest(OUT/'protocol.json'),
                                         'predictions_sha256': locks, 'held_labels_opened': False})


def verify():
    assert not (OUT/'verification.json').exists(), 'Preserve completed joint verification'
    protocol = read(OUT/'protocol.json')
    locked = read(OUT/'predictions_locked.json')
    assert locked['protocol_sha256'] == digest(OUT/'protocol.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    data = inputs()
    # Both cohorts are fully replayed in this fresh process before held-label scoring.
    for day in [15, 16]:
        np.testing.assert_array_equal(data[day]['rows'], np.load(OUT/f'day{day}_indices.npy'))
        for recipe in protocol['recipes']:
            replay = predict(data[day]['base'], data[day]['groups'], recipe)
            assert array_hash(replay) == locked['predictions_sha256'][str(day)][recipe['name']]
        emit('joint_group_candidates_replayed', day=day)
    labels = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['label'], dtype='int32').label.to_numpy()
    results = {}
    for day in [15, 16]:
        y = labels[data[day]['rows']]
        assert len(y) == {15: 424780, 16: 421983}[day] and int(y.sum()) == {15: 3523, 16: 3724}[day]
        records = []
        for recipe in protocol['recipes']:
            prediction = predict(data[day]['base'], data[day]['groups'], recipe)
            assert array_hash(prediction) == locked['predictions_sha256'][str(day)][recipe['name']]
            records.append(dict(recipe, auc=rank_sum_auc(y, prediction)))
        best = max(records, key=lambda row: row['auc'])
        prediction = predict(data[day]['base'], data[day]['groups'], best)
        np.save(OUT/f'day{day}_prediction.npy', prediction)
        results[str(20191100+day)] = {
            'auc': best['auc'], 'best': best, 'records': records, 'rows': len(y), 'clicks': int(y.sum()),
            'candidate_hashes_replayed_exactly': True, 'evaluation_day_labels_fit': False,
            'held_labels_used_to_select_fixed_recipe': True,
            'target_met': best['auc'] >= {15: .815, 16: .82}[day],
        }
        emit('joint_group_scored', day=day, best=best)
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    write(OUT/'verification.json', {'status': 'passed', 'dates': results,
        'original_datasets_unchanged': True, 'both_targets_met': all(r['target_met'] for r in results.values())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    verify() if parser.parse_args().verify else prepare()
