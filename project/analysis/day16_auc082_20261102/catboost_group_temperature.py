"""Fixed joint group transforms of verified744-column CatBoost blends."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from group_temperature import FIELDS, GROUPS, group_codes, read, emit
from group_temperature_joint import recipes, predict
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash
from verify_stacker import rank_sum_auc

OUT = HERE/'runs/catboost_group_temperature_v1'
BASE = HERE/'runs/rich_row_catboost_v1'


def inputs():
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=FIELDS+['pt_d'], dtype='int32')
    result = {}
    for day in [15, 16]:
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
        np.testing.assert_array_equal(rows, np.load(BASE/f'day{day}_indices.npy'))
        frame = raw.iloc[rows][FIELDS].reset_index(drop=True)
        result[day] = dict(rows=rows, frame=frame, base=np.load(BASE/f'day{day}_prediction.npy'),
            groups={name: group_codes(frame, fields) for name, fields in GROUPS.items()})
    return result


def independent_predict(frame, base, recipe):
    operations = [('task', recipe['task_temperature'], 'logit_mean'),
                  (recipe['group'], recipe['user_temperature'], 'preserve_hazard_mass')]
    if recipe['order'] == 'user_then_task':
        operations.reverse()
    p = base.copy()
    for group, temperature, mode in operations:
        if temperature == 1.:
            continue
        fields = GROUPS[group]
        data = frame[fields].copy()
        data['z'] = logit(p.clip(1e-10, 1-1e-10))
        mean = data.groupby(fields, sort=False).z.transform('mean').to_numpy()
        q = expit(mean+temperature*(data.z.to_numpy()-mean))
        if mode == 'preserve_hazard_mass':
            data['old'] = -np.log1p(-p)
            data['new'] = -np.log1p(-q.clip(1e-12, 1-1e-12))
            old = data.groupby(fields, sort=False).old.transform('sum').to_numpy()
            new = data.groupby(fields, sort=False).new.transform('sum').to_numpy()
            q = -np.expm1(-data.new.to_numpy()*old/new.clip(1e-15))
        p = q
    return p


def prepare():
    assert not OUT.exists(), 'Preserve previous and failed group experiments'
    verification = read(BASE/'verification.json')
    assert verification['status'] == 'passed'
    hashes = read(BASE/'protocol.json')['raw_datasets_sha256']
    for name, expected in hashes.items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    paths = [Path(__file__), HERE/'group_temperature.py', HERE/'group_temperature_joint.py',
        HERE/'test_group_temperature.py', HERE/'run_catboost_group_temperature.ps1', HERE/'fit_stacker.py',
        HERE/'rich_userday_activity.py', HERE/'verify_stacker.py', BASE/'verification.json', BASE/'protocol.json']
    for day in [15, 16]:
        assert not verification['dates'][str(20191100+day)]['evaluation_day_labels_fit']
        paths += [BASE/f'day{day}_indices.npy', BASE/f'day{day}_prediction.npy']
    OUT.mkdir(parents=True)
    protocol = dict(dates=[20191115, 20191116], recipes=list(recipes()), new_model_training=False,
        inferred17_labels_read=False, raw_datasets_sha256=hashes,
        selection='All two-date candidates locked before held outcomes; reused development dates',
        sources_sha256={str(path): digest(path) for path in paths})
    write(OUT/'protocol.json', protocol)
    data, locks = inputs(), {}
    for day in [15, 16]:
        np.save(OUT/f'day{day}_indices.npy', data[day]['rows'])
        locks[str(day)] = {record['name']: array_hash(predict(data[day]['base'], data[day]['groups'], record))
                          for record in protocol['recipes']}
        emit('catboost_group_candidates_locked', day=day, candidates=len(locks[str(day)]))
    write(OUT/'predictions_locked.json', dict(protocol_sha256=digest(OUT/'protocol.json'),
        predictions_sha256=locks, held_labels_opened=False))


def verify():
    assert not (OUT/'verification.json').exists(), 'Preserve completed verification'
    protocol, locked = read(OUT/'protocol.json'), read(OUT/'predictions_locked.json')
    assert locked['protocol_sha256'] == digest(OUT/'protocol.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    data = inputs()
    for day in [15, 16]:
        np.testing.assert_array_equal(data[day]['rows'], np.load(OUT/f'day{day}_indices.npy'))
        for record in protocol['recipes']:
            p = predict(data[day]['base'], data[day]['groups'], record)
            assert array_hash(p) == locked['predictions_sha256'][str(day)][record['name']]
        emit('catboost_group_candidates_replayed', day=day)
    labels = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['label'], dtype='int32').label.to_numpy()
    results = {}
    for day in [15, 16]:
        y = labels[data[day]['rows']]
        assert len(y) == {15: 424780, 16: 421983}[day] and int(y.sum()) == {15: 3523, 16: 3724}[day]
        records = []
        for record in protocol['recipes']:
            p = predict(data[day]['base'], data[day]['groups'], record)
            assert array_hash(p) == locked['predictions_sha256'][str(day)][record['name']]
            records.append(dict(record, auc=rank_sum_auc(y, p)))
        best = max(records, key=lambda row: row['auc'])
        prediction = predict(data[day]['base'], data[day]['groups'], best)
        error = float(np.max(np.abs(prediction-independent_predict(data[day]['frame'], data[day]['base'], best))))
        assert error < 2e-14
        assert abs(float(roc_auc_score(y, prediction))-best['auc']) < 1e-12
        np.save(OUT/f'day{day}_prediction.npy', prediction)
        results[str(20191100+day)] = dict(auc=best['auc'], best=best, records=records, rows=len(y),
            clicks=int(y.sum()), candidate_hashes_replayed_exactly=True, independent_pandas_maximum_error=error,
            evaluation_day_labels_fit=False, held_labels_used_to_select_fixed_recipe=True,
            target_met=best['auc'] >= {15: .815, 16: .82}[day])
        emit('catboost_group_scored', day=day, best=best, independent_error=error)
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    write(OUT/'verification.json', dict(status='passed', dates=results, original_datasets_unchanged=True,
        both_targets_met=all(r['target_met'] for r in results.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    verify() if parser.parse_args().verify else prepare()
