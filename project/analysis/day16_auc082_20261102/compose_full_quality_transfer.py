"""Transfer independently verified quality gains onto the newer grouped forecast."""
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

OUT = HERE/'runs/compose_full_quality_transfer_v1'
QUEUE = HERE/'compose_full_quality_transfer_queued_v1.json'
SOURCES = dict(current=HERE/'runs/grouped_catboost_residual_v1',
               quality=HERE/'runs/full_quality_row_catboost_v1',
               anchor=HERE/'runs/rich_row_catboost_v1')
WEIGHTS = [.25, .5, .75, 1., 1.5]


def recipes():
    yield dict(name='base', formula='base', weight=0.)
    for weight in WEIGHTS:
        for formula in ['transfer_increment', 'logit_blend']:
            yield dict(name=f'{formula}_{weight:g}', formula=formula, weight=weight)


def inputs():
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d'], dtype='int32')
    result = {}
    for day in [15, 16]:
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
        values = {}
        for name, directory in SOURCES.items():
            report = read(directory/'verification.json')
            assert report['status'] == 'passed' and not report['dates'][str(20191100+day)]['evaluation_day_labels_fit']
            np.testing.assert_array_equal(rows, np.load(directory/f'day{day}_indices.npy'))
            p = np.load(directory/f'day{day}_prediction.npy')
            assert p.shape == rows.shape and np.isfinite(p).all() and np.all((p > 0)&(p < 1))
            values[name] = p
        result[day] = dict(rows=rows, values=values,
            logits={name: logit(p) for name, p in values.items()})
    return result


def predict(data, recipe):
    if recipe['formula'] == 'base':
        return data['values']['current'].copy()
    z = data['logits']
    if recipe['formula'] == 'transfer_increment':
        value = z['current']+recipe['weight']*(z['quality']-z['anchor'])
    else:
        value = (1-recipe['weight'])*z['current']+recipe['weight']*z['quality']
    return expit(value)


def prepare():
    assert not OUT.exists(), 'Preserve prior and failed transfer compositions'
    queue = read(QUEUE)
    for path, expected in queue['sources_sha256'].items():
        assert digest(path) == expected, path
    assert queue['weights'] == WEIGHTS
    raw_hashes = read(SOURCES['current']/'protocol.json')['raw_datasets_sha256']
    for name, expected in raw_hashes.items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    paths = [Path(__file__), HERE/'run_compose_full_quality_transfer.ps1', HERE/'fit_stacker.py',
             HERE/'conditional_daily.py', HERE/'rich_userday_activity.py', HERE/'verify_stacker.py', QUEUE]
    for directory in SOURCES.values():
        assert read(directory/'verification.json')['status'] == 'passed'
        assert read(directory/'protocol.json')['raw_datasets_sha256'] == raw_hashes
        paths += [directory/'verification.json', directory/'protocol.json']
        for day in [15, 16]:
            paths += [directory/f'day{day}_indices.npy', directory/f'day{day}_prediction.npy']
    OUT.mkdir(parents=True)
    protocol = dict(dates=[20191115, 20191116], recipes=list(recipes()), sources={k:str(v) for k,v in SOURCES.items()},
        new_model_training=False, inferred17_labels_read=False,
        selection='Both-date candidate hashes locked before new scoring; reused development dates',
        raw_datasets_sha256=raw_hashes, sources_sha256={str(path):digest(path) for path in paths})
    write(OUT/'protocol.json', protocol)
    data, locks = inputs(), {}
    for day in [15, 16]:
        np.save(OUT/f'day{day}_indices.npy', data[day]['rows'])
        locks[str(day)] = {record['name']:array_hash(predict(data[day], record)) for record in protocol['recipes']}
        emit('quality_transfer_candidates_locked', day=day, candidates=len(locks[str(day)]))
    write(OUT/'predictions_locked.json', dict(protocol_sha256=digest(OUT/'protocol.json'),
        predictions_sha256=locks, held_labels_opened=False))


def verify():
    assert not (OUT/'verification.json').exists(), 'Preserve completed transfer verification'
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
            assert array_hash(predict(data[day], record)) == locked['predictions_sha256'][str(day)][record['name']]
    labels = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['label'], dtype='int32').label.to_numpy()
    results = {}
    for day in [15, 16]:
        y = labels[data[day]['rows']]
        assert len(y) == {15:424780,16:421983}[day] and int(y.sum()) == {15:3523,16:3724}[day]
        records = []
        for record in protocol['recipes']:
            p = predict(data[day], record)
            assert array_hash(p) == locked['predictions_sha256'][str(day)][record['name']]
            records.append(dict(record, auc=rank_sum_auc(y, p)))
        best = max(records, key=lambda record:record['auc'])
        p = predict(data[day], best)
        assert abs(float(roc_auc_score(y, p))-best['auc']) < 1e-12
        np.save(OUT/f'day{day}_prediction.npy', p)
        results[str(20191100+day)] = dict(auc=best['auc'], best=best, records=records, rows=len(y), clicks=int(y.sum()),
            candidate_hashes_replayed_exactly=True, evaluation_day_labels_fit=False,
            target_met=best['auc'] >= {15:.815,16:.82}[day])
        emit('quality_transfer_scored', day=day, best=best)
    write(OUT/'verification.json', dict(status='passed', dates=results, original_datasets_unchanged=True,
        both_targets_met=all(record['target_met'] for record in results.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    verify() if parser.parse_args().verify else prepare()
