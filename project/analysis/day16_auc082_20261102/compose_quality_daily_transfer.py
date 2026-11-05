"""Combine independently verified daily and row quality corrections on full cohorts."""
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
from group_temperature import FIELDS, GROUPS, group_codes

OUT = HERE/'runs/compose_quality_daily_transfer_v1'
SOURCES = dict(current=HERE/'runs/compose_full_quality_transfer_v1',
               daily=HERE/'runs/quality_daily_catboost_v1',
               anchor=HERE/'runs/grouped_catboost_residual_v1')
WEIGHTS = [.25, .5, .75, 1., 1.5, 2.]
GROUP_WEIGHTS = [0., .5, 1., 1.5]


def recipes():
    yield dict(name='base', formula='base', weight=0.)
    for weight in WEIGHTS:
        for formula in ['transfer_increment', 'logit_blend']:
            yield dict(name=f'{formula}_{weight:g}', formula=formula, weight=weight)
    for group in GROUPS:
        for between in GROUP_WEIGHTS:
            for within in GROUP_WEIGHTS:
                if between == within == 0:
                    continue
                yield dict(name=f'{group}_between{between:g}_within{within:g}', formula='group_transfer',
                           group=group, between=between, within=within)


def predict(data, record, independent=False):
    values, z = data['values'], data['logits']
    if record['formula'] == 'base':
        return values['current'].copy()
    if record['formula'] == 'logit_blend':
        return expit((1-record['weight'])*z['current']+record['weight']*z['daily'])
    delta = z['daily']-z['anchor']
    if record['formula'] == 'transfer_increment':
        return expit(z['current']+record['weight']*delta)
    if record['formula'] != 'group_transfer':
        raise ValueError('Undeclared composition formula')
    if independent:
        table = data['frame'][GROUPS[record['group']]].copy()
        table['delta'] = delta
        mean = table.groupby(GROUPS[record['group']], sort=False).delta.transform('mean').to_numpy()
    else:
        groups = data['groups'][record['group']]
        mean = (np.bincount(groups, weights=delta)/np.bincount(groups).clip(1))[groups]
    return expit(z['current']+record['between']*mean+record['within']*(delta-mean))


def inputs():
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=FIELDS+['pt_d'], dtype='int32')
    result = {}
    for day in [15, 16]:
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
        frame = raw.iloc[rows][FIELDS].reset_index(drop=True)
        values = {}
        for name, folder in SOURCES.items():
            report = read(folder/'verification.json')
            assert report['status'] == 'passed'
            assert not report['dates'][str(20191100+day)]['evaluation_day_labels_fit']
            np.testing.assert_array_equal(rows, np.load(folder/f'day{day}_indices.npy'))
            p = np.load(folder/f'day{day}_prediction.npy')
            assert p.shape == rows.shape and np.isfinite(p).all() and np.all((p > 0)&(p < 1))
            values[name] = p
        result[day] = dict(rows=rows, frame=frame, values=values,
            logits={name:logit(p) for name,p in values.items()},
            groups={name:group_codes(frame, fields) for name,fields in GROUPS.items()})
    return result


def prepare():
    assert not OUT.exists(), 'Preserve all previous compositions and failures'
    hashes = read(SOURCES['current']/'protocol.json')['raw_datasets_sha256']
    paths = [Path(__file__), HERE/'test_compose_quality_daily_transfer.py',
        HERE/'run_compose_quality_daily_transfer.ps1', HERE/'fit_stacker.py',
        HERE/'conditional_daily.py', HERE/'rich_userday_activity.py', HERE/'verify_stacker.py', HERE/'group_temperature.py']
    for folder in SOURCES.values():
        assert read(folder/'verification.json')['status'] == 'passed'
        assert read(folder/'protocol.json')['raw_datasets_sha256'] == hashes
        paths += [folder/'protocol.json', folder/'verification.json']
        for day in [15,16]:
            paths += [folder/f'day{day}_indices.npy', folder/f'day{day}_prediction.npy']
    for name,expected in hashes.items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    OUT.mkdir()
    protocol = dict(dates=[20191115,20191116], recipes=list(recipes()),
        sources={name:str(folder) for name,folder in SOURCES.items()},
        new_model_training=False, inferred17_labels_read=False,
        selection='All both-date candidate hashes lock before scoring; repeatedly reused development dates',
        raw_datasets_sha256=hashes, sources_sha256={str(p):digest(p) for p in paths})
    write(OUT/'protocol.json', protocol)
    data, locks = inputs(), {}
    for day in [15,16]:
        np.save(OUT/f'day{day}_indices.npy', data[day]['rows'])
        locks[str(day)] = {record['name']:array_hash(predict(data[day], record)) for record in protocol['recipes']}
        emit('quality_daily_transfer_candidates_locked', day=day, candidates=len(locks[str(day)]))
    write(OUT/'predictions_locked.json', dict(protocol_sha256=digest(OUT/'protocol.json'),
        predictions_sha256=locks, held_labels_opened=False))


def verify():
    assert not (OUT/'verification.json').exists(), 'Preserve completed verification'
    protocol, locked = read(OUT/'protocol.json'), read(OUT/'predictions_locked.json')
    assert locked['protocol_sha256'] == digest(OUT/'protocol.json')
    for path,expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name,expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    data = inputs()
    for day in [15,16]:
        np.testing.assert_array_equal(data[day]['rows'], np.load(OUT/f'day{day}_indices.npy'))
        for record in protocol['recipes']:
            assert array_hash(predict(data[day], record)) == locked['predictions_sha256'][str(day)][record['name']]
    y_all = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['label'], dtype='int32').label.to_numpy()
    results = {}
    for day in [15,16]:
        y = y_all[data[day]['rows']]
        assert len(y) == {15:424780,16:421983}[day] and int(y.sum()) == {15:3523,16:3724}[day]
        records = [dict(record, auc=rank_sum_auc(y,predict(data[day],record))) for record in protocol['recipes']]
        best = max(records,key=lambda r:r['auc'])
        p = predict(data[day],best)
        independent_error = float(np.max(np.abs(p-predict(data[day],best,independent=True))))
        assert independent_error < 2e-14 and abs(roc_auc_score(y,p)-best['auc']) < 1e-12
        np.save(OUT/f'day{day}_prediction.npy',p)
        results[str(20191100+day)] = dict(auc=best['auc'],best=best,records=records,rows=len(y),clicks=int(y.sum()),
            independent_pandas_maximum_error=independent_error, candidate_hashes_replayed_exactly=True,
            evaluation_day_labels_fit=False, held_labels_used_to_select_fixed_recipe=True,
            target_met=best['auc'] >= {15:.815,16:.82}[day])
        emit('quality_daily_transfer_scored',day=day,best=best)
    for name,expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected,name
    write(OUT/'verification.json',dict(status='passed',dates=results,original_datasets_unchanged=True,
        both_targets_met=all(r['target_met'] for r in results.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    verify() if parser.parse_args().verify else prepare()
