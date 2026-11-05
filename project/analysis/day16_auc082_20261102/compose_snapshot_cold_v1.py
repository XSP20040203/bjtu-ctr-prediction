"""Combine two verified corrections without using inferred17 outcomes."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from fit_stacker import HERE, ROOT, digest, write
from conditional_daily import read, emit
from rich_userday_activity import array_hash
from verify_stacker import rank_sum_auc

OUT = HERE / 'runs/compose_snapshot_cold_v1'
SOURCES = dict(base=HERE/'runs/cross_user_pair_ranker_v1',
               snapshot=HERE/'runs/snapshot_pair_ranker_v1',
               cold=HERE/'runs/cold_task_ranker_v1')
WEIGHTS = [0., .5, 1., 1.5]


def recipes():
    return [dict(name=f'snapshot{a:g}_cold{b:g}', snapshot_weight=a, cold_weight=b)
            for a in WEIGHTS for b in WEIGHTS]


def predict(values, record, independent=False):
    base, snapshot, cold = (values[key] for key in ['base', 'snapshot', 'cold'])
    a, b = record['snapshot_weight'], record['cold_weight']
    if a == 0 and b == 0:
        return base.copy()
    if a == 1 and b == 0:
        return snapshot.copy()
    if a == 0 and b == 1:
        return cold.copy()
    if independent:
        odds0, odds1, odds2 = (p/(1-p) for p in [base, snapshot, cold])
        odds = odds0*(odds1/odds0)**a*(odds2/odds0)**b
        return odds/(1+odds)
    z = logit(base)
    return expit(z+a*(logit(snapshot)-z)+b*(logit(cold)-z))


def inputs():
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d'], dtype='int32')
    result = {}
    for day in [15, 16]:
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
        values = {}
        for key, folder in SOURCES.items():
            report = read(folder/'verification.json')
            assert report['status'] == 'passed'
            assert not report['dates'][str(20191100+day)]['evaluation_day_labels_fit']
            np.testing.assert_array_equal(rows, np.load(folder/f'day{day}_indices.npy'))
            p = np.load(folder/f'day{day}_prediction.npy')
            assert p.shape == rows.shape and np.isfinite(p).all()
            assert np.all((p > 0) & (p < 1))
            values[key] = p
        result[day] = dict(rows=rows, values=values)
    return result


def check_sources():
    protocol = read(OUT/'protocol.json')
    for filename, expected in protocol['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/filename) == expected, filename
    return protocol


def prepare():
    assert not OUT.exists(), 'Preserve previous compositions'
    paths = [Path(__file__), HERE/'fit_stacker.py', HERE/'conditional_daily.py',
             HERE/'rich_userday_activity.py', HERE/'verify_stacker.py']
    hashes = read(SOURCES['base']/'protocol.json')['raw_datasets_sha256']
    for folder in SOURCES.values():
        assert read(folder/'verification.json')['status'] == 'passed'
        parent = read(folder/'protocol.json')
        assert parent['raw_datasets_sha256'] == hashes
        for filename, expected in parent['sources_sha256'].items():
            assert digest(filename) == expected, filename
        paths += [folder/'protocol.json', folder/'verification.json']
        paths += [folder/f'day{day}_{kind}.npy'
                  for day in [15, 16] for kind in ['prediction', 'indices']]
    for filename, expected in hashes.items():
        assert digest(ROOT/'2026ml'/filename) == expected, filename
    OUT.mkdir()
    write(OUT/'protocol.json', dict(dates=[20191115, 20191116], recipes=recipes(),
        raw_datasets_sha256=hashes,
        sources_sha256={str(path): digest(path) for path in paths},
        formula='Base log odds plus fixed weights times snapshot and cold-task log-odds increments',
        new_model_training=False, inferred17_labels_read=False,
        selection='All both-date candidates locked before scoring; development dates repeatedly reused'))
    data = inputs()
    locked = {}
    for day in [15, 16]:
        np.save(OUT/f'day{day}_indices.npy', data[day]['rows'])
        locked[str(day)] = {record['name']: array_hash(predict(data[day]['values'], record))
                            for record in recipes()}
    write(OUT/'predictions_locked.json', dict(protocol_sha256=digest(OUT/'protocol.json'),
        candidates_sha256=locked, held_labels_opened_for_scoring=False))
    emit('snapshot_cold_candidates_locked', candidates_per_day=len(recipes()))


def verify():
    protocol = check_sources()
    assert not (OUT/'verification.json').exists()
    locked = read(OUT/'predictions_locked.json')
    assert locked['protocol_sha256'] == digest(OUT/'protocol.json')
    data = inputs()
    for day in [15, 16]:
        np.testing.assert_array_equal(data[day]['rows'], np.load(OUT/f'day{day}_indices.npy'))
        for record in protocol['recipes']:
            assert array_hash(predict(data[day]['values'], record)) == locked['candidates_sha256'][str(day)][record['name']]
    labels = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['label'], dtype='int32').label.to_numpy()
    reports = {}
    for day in [15, 16]:
        y = labels[data[day]['rows']]
        assert len(y) == {15:424780, 16:421983}[day]
        assert y.sum() == {15:3523, 16:3724}[day]
        records, maximum = [], 0.
        for record in protocol['recipes']:
            p = predict(data[day]['values'], record)
            q = predict(data[day]['values'], record, True)
            assert np.isfinite(p).all() and np.all((p > 0) & (p < 1))
            error = float(np.max(np.abs(p-q)))
            assert error < 2e-14
            maximum = max(maximum, error)
            records.append(dict(record, auc=rank_sum_auc(y, p)))
        best = max(records, key=lambda record: record['auc'])
        p = predict(data[day]['values'], best)
        assert abs(roc_auc_score(y, p)-best['auc']) < 1e-12
        np.save(OUT/f'day{day}_prediction.npy', p)
        reports[str(20191100+day)] = dict(auc=best['auc'], best=best, records=records,
            rows=len(y), clicks=int(y.sum()), evaluation_day_labels_fit=False,
            held_labels_used_to_select_fixed_recipe=True,
            all_both_date_candidate_hashes_locked_before_scoring=True,
            maximum_independent_odds_error=maximum, candidate_hashes_replayed_exactly=True,
            target_met=best['auc'] >= {15:.815, 16:.82}[day])
        emit('snapshot_cold_scored', day=day, best=best)
    check_sources()
    write(OUT/'verification.json', dict(status='passed', dates=reports,
        original_datasets_unchanged=True,
        both_targets_met=all(report['target_met'] for report in reports.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['prepare', 'verify'])
    {'prepare':prepare, 'verify':verify}[parser.parse_args().phase]()
