"""Fitted group-event probabilities with fully locked both-date validation."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash
from verify_stacker import rank_sum_auc
from group_event_posterior_core_v1 import FIELDS, events, fit, deltas, predict

OUT = HERE / 'runs/group_event_posterior_v1'
BASE = HERE / 'runs/compose_family_quantiles_v1'
STRENGTHS = [5., 20.]
WEIGHTS = [.25, .5, .75, 1.]


def check():
    protocol = read(OUT / 'protocol.json')
    for name, expected in protocol['sources_sha256'].items():
        assert digest(name) == expected, name
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / name) == expected, name


def raw_data():
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=FIELDS + ['label'],
                      dtype={name: str for name in FIELDS if 'list' in name})
    return raw


def hash_groups(grouped):
    return {channel: {key: array_hash(value) for key, value in groups.items()} for channel, groups in grouped.items()}


def fit_models():
    assert not OUT.exists(), 'Preserve previous fits'
    assert read(BASE / 'verification.json')['status'] == 'passed'
    paths = [Path(__file__), HERE / 'group_event_posterior_core_v1.py',
             HERE / 'test_group_event_posterior_core_v1.py', HERE / 'run_group_event_posterior_v1.ps1',
             HERE / 'transition_posterior_core_v1.py', BASE / 'protocol.json', BASE / 'verification.json']
    for day in (15, 16):
        paths += [BASE / f'day{day}_indices.npy', BASE / f'day{day}_prediction.npy']
    OUT.mkdir()
    write(OUT / 'protocol.json', dict(strengths=STRENGTHS, weights=WEIGHTS,
        objective='Historical probability of any click per unordered user/candidate/history-state exposure group',
        transfer='Calibrate baseline prior, condition row marginals on event probability, restore score scale; largest channel correction',
        channels=['task', 'advertiser', 'app', 'maximum'], recent_windows=[False, True],
        selection='Both-date candidate hashes locked before labels are used to score; dates repeatedly reused',
        evaluation_day_labels_fit=False, inferred17_labels_read=False,
        raw_datasets_sha256=read(BASE / 'protocol.json')['raw_datasets_sha256'],
        sources_sha256={str(path): digest(path) for path in paths}))
    raw = raw_data()
    grouped = events(raw[FIELDS])
    emit('group_events_built', groups={k: len(v['dates']) for k, v in grouped.items()})
    locks, contracts = {}, {}
    for day in (15, 16):
        cutoff = 20191100 + day
        labels = raw.label.to_numpy(dtype='int32', copy=True)
        labels[raw.pt_d.to_numpy() >= cutoff] = -1
        models = {}
        for recent in (False, True):
            for strength in STRENGTHS:
                name = f'{"recent" if recent else "all"}_s{strength:g}'
                models[name] = fit(grouped, raw.pt_d.to_numpy(), labels, cutoff, strength, recent)
        locks[str(day)] = models
        rows = np.flatnonzero(raw.pt_d.to_numpy() == cutoff)
        np.testing.assert_array_equal(rows, np.load(BASE / f'day{day}_indices.npy'))
        np.save(OUT / f'day{day}_indices.npy', rows)
        local = events(raw.loc[raw.pt_d.eq(cutoff), FIELDS].reset_index(drop=True))
        contracts[str(day)] = hash_groups(local)
        emit('group_event_rates_fitted', day=day, max_training_date=models['all_s5']['max_training_date'])
    write(OUT / 'models_locked.json', dict(protocol_sha256=digest(OUT / 'protocol.json'), rates=locks,
                                          all_groups_sha256=hash_groups(grouped), query_groups_sha256=contracts))
    check()


def candidates(day, grouped, rates, independent=False):
    base = np.load(BASE / f'day{day}_prediction.npy')
    yield dict(name='base', channel='base', weight=0.), base.copy()
    for name, record in rates[str(day)].items():
        changes = deltas(base, grouped, record, independent)
        for channel, delta in changes.items():
            for weight in WEIGHTS:
                yield dict(name=name, channel=channel, weight=weight), predict(base, delta, weight, independent)


def key(record):
    return f'{record["name"]}__{record["channel"]}__{record["weight"]:g}'


def evaluate():
    check()
    assert not (OUT / 'predictions_locked.json').exists()
    raw = raw_data()
    models = read(OUT / 'models_locked.json')
    assert models['protocol_sha256'] == digest(OUT / 'protocol.json')
    locks = {}
    for day in (15, 16):
        local = events(raw.loc[raw.pt_d.eq(20191100 + day), FIELDS].reset_index(drop=True))
        assert hash_groups(local) == models['query_groups_sha256'][str(day)]
        locks[str(day)] = {key(r): array_hash(p) for r, p in candidates(day, local, models['rates'])}
    write(OUT / 'predictions_locked.json', dict(models_locked_sha256=digest(OUT / 'models_locked.json'),
                                               dates=locks, all_both_date_candidates_locked_before_scoring=True))
    emit('group_event_candidates_locked', counts={day: len(v) for day, v in locks.items()})


def verify():
    check()
    assert not (OUT / 'verification.json').exists()
    raw = raw_data()
    models = read(OUT / 'models_locked.json')
    locked = read(OUT / 'predictions_locked.json')
    assert models['protocol_sha256'] == digest(OUT / 'protocol.json')
    assert locked['models_locked_sha256'] == digest(OUT / 'models_locked.json')
    grouped = events(raw[FIELDS], True)
    assert hash_groups(grouped) == models['all_groups_sha256']
    checked = {}
    for day in (15, 16):
        cutoff = 20191100 + day
        hidden = raw.label.to_numpy(dtype='int32', copy=True)
        hidden[raw.pt_d.to_numpy() >= cutoff] = -1
        for recent in (False, True):
            for strength in STRENGTHS:
                name = f'{"recent" if recent else "all"}_s{strength:g}'
                assert fit(grouped, raw.pt_d.to_numpy(), hidden, cutoff, strength, recent, True) == models['rates'][str(day)][name]
        query = raw.loc[raw.pt_d.eq(cutoff), FIELDS].reset_index(drop=True)
        primary, independent = events(query), events(query, True)
        assert hash_groups(primary) == hash_groups(independent) == models['query_groups_sha256'][str(day)]
        maximum = 0.
        for (record, p), (other, q) in zip(candidates(day, primary, models['rates']), candidates(day, independent, models['rates'], True)):
            assert record == other and array_hash(p) == locked['dates'][str(day)][key(record)]
            error = float(np.max(np.abs(p - q)))
            assert error < 2e-12
            maximum = max(maximum, error)
        checked[day] = primary, maximum
        emit('group_event_independent_replay', day=day, maximum_error=maximum)
    reports = {}
    for day in (15, 16):
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day)
        np.testing.assert_array_equal(rows, np.load(OUT / f'day{day}_indices.npy'))
        y = raw.label.to_numpy()[rows]
        assert len(y) == {15: 424780, 16: 421983}[day] and y.sum() == {15: 3523, 16: 3724}[day]
        records, best_value = [], -1.
        for record, prediction in candidates(day, checked[day][0], models['rates']):
            assert np.isfinite(prediction).all() and np.all((prediction > 0) & (prediction < 1))
            value = rank_sum_auc(y, prediction)
            record = dict(record, auc=value)
            records.append(record)
            if value > best_value:
                best_value, best, best_prediction = value, record, prediction.copy()
        assert abs(roc_auc_score(y, best_prediction) - best_value) < 1e-12
        np.save(OUT / f'day{day}_prediction.npy', best_prediction)
        reports[str(20191100 + day)] = dict(auc=best_value, best=best, records=records,
            rows=len(y), clicks=int(y.sum()), evaluation_day_labels_fit=False, inferred17_labels_read=False,
            held_labels_used_to_select_fixed_recipe=True, all_both_date_candidates_locked_before_scoring=True,
            independent_all_event_groups_rates_and_conditionals_replayed=True,
            maximum_independent_prediction_error=checked[day][1], target_met=best_value >= {15: .815, 16: .82}[day])
        emit('group_event_scored', day=day, best=best)
    check()
    write(OUT / 'verification.json', dict(status='passed', dates=reports, original_datasets_unchanged=True,
                                          both_targets_met=all(v['target_met'] for v in reports.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['fit', 'evaluate', 'verify'])
    {'fit': fit_models, 'evaluate': evaluate, 'verify': verify}[parser.parse_args().phase]()
