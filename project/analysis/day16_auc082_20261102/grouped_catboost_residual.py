"""Fixed group-mean and within-group blending of independently verified forecasts."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from group_temperature import FIELDS, GROUPS, group_codes, read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash
from verify_stacker import rank_sum_auc

BASE = HERE/'runs/rich_row_catboost_v1'
OUT = HERE/'runs/grouped_catboost_residual_v1'
SOURCES = ['depth6_weight5_trees2400', 'depth8_weight5_trees2400', 'mean_last_two']
WEIGHTS = [-.1, 0., .1, .25, .5]


def components(base, forecast, groups):
    a = logit(np.asarray(base).clip(1e-6, 1-1e-6))
    b = logit(np.asarray(forecast).clip(1e-6, 1-1e-6))
    b = (b-b.mean())*a.std()/max(b.std(), 1e-6)+a.mean()
    delta = b-a
    size = np.bincount(groups)
    mean = (np.bincount(groups, weights=delta)/size.clip(1))[groups]
    return a, mean, delta-mean


def predict(parts, between, within):
    a, mean, centered = parts
    return expit(a+between*mean+within*centered)


def recipes():
    yield dict(name='base', source=None, group=None, between=0., within=0.)
    for source in SOURCES:
        for group in GROUPS:
            for between in WEIGHTS:
                for within in WEIGHTS:
                    if between == within == 0.:
                        continue
                    yield dict(name=f'{source}_{group}_between{between:g}_within{within:g}',
                               source=source, group=group, between=between, within=within)


def inputs():
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=FIELDS+['pt_d'], dtype='int32')
    result = {}
    for day in [15, 16]:
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
        np.testing.assert_array_equal(rows, np.load(BASE/f'day{day}_indices.npy'))
        frame = raw.iloc[rows][FIELDS].reset_index(drop=True)
        base = np.load(BASE/f'day{day}_prediction.npy')
        forecasts = {name: np.load(BASE/f'day{day}_{name}_prediction.npy') for name in SOURCES[:-1]}
        forecasts['mean_last_two'] = np.mean(list(forecasts.values()), axis=0)
        groups = {name: group_codes(frame, fields) for name, fields in GROUPS.items()}
        parts = {(source, group): components(base, forecasts[source], codes)
                 for source in SOURCES for group, codes in groups.items()}
        result[day] = dict(rows=rows, frame=frame, base=base, forecasts=forecasts, parts=parts)
    return result


def candidate(data, recipe):
    if recipe['source'] is None:
        return data['base'].copy()
    return predict(data['parts'][(recipe['source'], recipe['group'])], recipe['between'], recipe['within'])


def independent(frame, base, forecast, recipe):
    a = logit(base.clip(1e-6, 1-1e-6))
    b = logit(forecast.clip(1e-6, 1-1e-6))
    b = (b-b.mean())*a.std()/max(b.std(), 1e-6)+a.mean()
    values = frame[GROUPS[recipe['group']]].copy()
    values['delta'] = b-a
    mean = values.groupby(GROUPS[recipe['group']], sort=False).delta.transform('mean').to_numpy()
    return expit(a+recipe['between']*mean+recipe['within']*(b-a-mean))


def prepare():
    assert not OUT.exists(), 'Preserve prior blending experiments'
    verified = read(BASE/'verification.json')
    assert verified['status'] == 'passed'
    hashes = read(BASE/'protocol.json')['raw_datasets_sha256']
    for name, expected in hashes.items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    paths = [Path(__file__), HERE/'test_grouped_catboost_residual.py',
        HERE/'run_grouped_catboost_residual.ps1', HERE/'group_temperature.py', HERE/'fit_stacker.py',
        HERE/'rich_userday_activity.py', HERE/'verify_stacker.py', BASE/'verification.json', BASE/'protocol.json']
    for day in [15, 16]:
        report = verified['dates'][str(20191100+day)]
        assert not report['evaluation_day_labels_fit'] and report['independent_source_matrix_used']
        paths += [BASE/f'day{day}_indices.npy', BASE/f'day{day}_prediction.npy']
        paths += [BASE/f'day{day}_{source}_prediction.npy' for source in SOURCES[:-1]]
    OUT.mkdir(parents=True)
    protocol = dict(dates=[20191115, 20191116], sources=SOURCES, groups=GROUPS, weights=WEIGHTS,
        recipes=list(recipes()), formula='logit(base)+between*group_mean(delta)+within*(delta-group_mean(delta))',
        delta='Mean/standard-deviation normalized source logits minus base logits',
        new_model_training=False, inferred17_labels_read=False,
        selection='Both-date candidate hashes locked before held labels; reused development dates',
        raw_datasets_sha256=hashes, sources_sha256={str(path): digest(path) for path in paths})
    write(OUT/'protocol.json', protocol)
    data, locks = inputs(), {}
    for day in [15, 16]:
        np.save(OUT/f'day{day}_indices.npy', data[day]['rows'])
        locks[str(day)] = {recipe['name']: array_hash(candidate(data[day], recipe)) for recipe in protocol['recipes']}
        emit('grouped_catboost_candidates_locked', day=day, candidates=len(locks[str(day)]))
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
        for recipe in protocol['recipes']:
            assert array_hash(candidate(data[day], recipe)) == locked['predictions_sha256'][str(day)][recipe['name']]
        emit('grouped_catboost_candidates_replayed', day=day)
    labels = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['label'], dtype='int32').label.to_numpy()
    results = {}
    for day in [15, 16]:
        y = labels[data[day]['rows']]
        assert len(y) == {15: 424780, 16: 421983}[day] and int(y.sum()) == {15: 3523, 16: 3724}[day]
        records = []
        for recipe in protocol['recipes']:
            p = candidate(data[day], recipe)
            assert array_hash(p) == locked['predictions_sha256'][str(day)][recipe['name']]
            records.append(dict(recipe, auc=rank_sum_auc(y, p)))
        best = max(records, key=lambda record: record['auc'])
        p = candidate(data[day], best)
        replay = (data[day]['base'] if best['source'] is None else independent(data[day]['frame'],
                  data[day]['base'], data[day]['forecasts'][best['source']], best))
        error = float(np.max(np.abs(p-replay)))
        assert error < 2e-14 and abs(float(roc_auc_score(y, p))-best['auc']) < 1e-12
        np.save(OUT/f'day{day}_prediction.npy', p)
        results[str(20191100+day)] = dict(auc=best['auc'], best=best, records=records, rows=len(y),
            clicks=int(y.sum()), candidate_hashes_replayed_exactly=True, independent_pandas_maximum_error=error,
            evaluation_day_labels_fit=False, held_labels_used_to_select_fixed_recipe=True,
            target_met=best['auc'] >= {15: .815, 16: .82}[day])
        emit('grouped_catboost_scored', day=day, best=best, independent_error=error)
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    write(OUT/'verification.json', dict(status='passed', dates=results, original_datasets_unchanged=True,
        both_targets_met=all(record['target_met'] for record in results.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    verify() if parser.parse_args().verify else prepare()
