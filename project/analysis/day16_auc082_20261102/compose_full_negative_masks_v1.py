"""Cohort-conditioned transfer of the independently verified all-negative models."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash
from cross_user_pair_ranker_v1 import prediction as row_prediction
from quantile_blend_core_v1 import matched_logit, blend
from verify_stacker import rank_sum_auc
from full_negative_mask_core_v1 import masks as build_masks, NAMES

OUT = HERE / 'runs/compose_full_negative_masks_v1'
BASE = HERE / 'runs/full_negative_context_v2'
ANCHOR = HERE / 'runs/group_event_posterior_v1'
INPUT = HERE / 'runs/full_negative_context_v1'
WEIGHTS = [0., .025, .05, .1, .2, .35]
MODES = ['replace_global', 'rank_global']
ARMS = ['all_negative_d6', 'all_negative_d8_recent2', 'mean_last']


def check():
    protocol = read(OUT / 'protocol.json')
    for filename, expected in protocol['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    return protocol


def inputs(day, independent=False):
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'user_id', 'task_id'], dtype='int32')
    folder = INPUT / f'input_day{day}'
    global_rows = np.load(folder / 'rows.npy')
    training = global_rows[np.load(folder / 'train.npy')]
    query = global_rows[np.load(folder / 'held.npy')]
    np.testing.assert_array_equal(query, np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day))
    np.testing.assert_array_equal(query, np.load(BASE / f'day{day}_indices.npy'))
    np.testing.assert_array_equal(query, np.load(ANCHOR / f'day{day}_indices.npy'))
    masks = build_masks(raw.pt_d.to_numpy(), raw.user_id.to_numpy(), raw.task_id.to_numpy(),
        training, np.load(folder / 'labels.npy'), query, 20191100 + day, independent)
    groups, _ = pd.factorize(raw.user_id.to_numpy()[query], sort=True)
    margins = {arm: np.load(BASE / f'day{day}_{arm}_trees2400_held_margin.npy') for arm in ARMS[:2]}
    margins['mean_last'] = np.mean(list(margins.values()), axis=0)
    current = np.load(BASE / f'day{day}_prediction.npy')
    anchor = np.load(ANCHOR / f'day{day}_prediction.npy')
    assert all(value.shape == current.shape for value in [anchor, *margins.values(), *masks.values()])
    contract = dict(rows_sha256=array_hash(query), groups_sha256=array_hash(groups),
        masks_sha256={name: array_hash(value) for name, value in masks.items()},
        mask_rows={name: int(value.sum()) for name, value in masks.items()},
        margins_sha256={name: array_hash(value) for name, value in margins.items()},
        current_sha256=array_hash(current), anchor_sha256=array_hash(anchor),
        training_rows_sha256=array_hash(training), max_mask_label_date=20191100 + day - 1,
        inferred17_labels_read=False)
    return dict(current=current, anchor=anchor, groups=groups, margins=margins, masks=masks, rows=query), contract


def candidates(data, independent=False):
    yield dict(name='base', mask='all', mode='base', weight=0.), data['current'].copy()
    for name, margin in data['margins'].items():
        target = matched_logit(data['anchor'], margin, independent)
        for mode in MODES:
            for weight in WEIGHTS:
                adjusted = (blend(data['anchor'], target, np.ones(len(margin), dtype=bool), weight, independent)
                            if mode == 'rank_global' else
                            row_prediction(data['anchor'], margin, data['groups'], weight, mode, independent))
                for mask in NAMES:
                    yield dict(name=name, mask=mask, mode=mode, weight=weight), np.where(data['masks'][mask], adjusted, data['current'])


def key(record):
    return f'{record["name"]}__{record["mask"]}__{record["mode"]}__{record["weight"]:g}'


def prepare():
    assert not OUT.exists(), 'Preserve every earlier composition'
    paths = [Path(__file__)] + [HERE / name for name in ('full_negative_mask_core_v1.py',
        'test_full_negative_mask_core_v1.py', 'run_compose_full_negative_masks_v1.ps1',
        'conditional_daily.py', 'fit_stacker.py', 'rich_userday_activity.py',
        'cross_user_pair_ranker_v1.py', 'quantile_blend_core_v1.py', 'verify_stacker.py')]
    hashes = read(BASE / 'protocol.json')['raw_datasets_sha256']
    for run in (BASE, ANCHOR):
        report = read(run / 'verification.json')
        assert report['status'] == 'passed'
        assert read(run / 'protocol.json')['raw_datasets_sha256'] == hashes
        paths += [run / 'protocol.json', run / 'verification.json']
        for day in (15, 16):
            assert report['dates'][str(20191100 + day)]['evaluation_day_labels_fit'] is False
            paths += [run / f'day{day}_indices.npy', run / f'day{day}_prediction.npy']
    assert read(BASE / 'verification.json')['dates']['20191116']['independent_full_population_features_and_models_replayed']
    for day in (15, 16):
        paths += [BASE / f'day{day}_{name}_trees2400_held_margin.npy' for name in ARMS[:2]]
        paths += [INPUT / f'input_day{day}/{name}.npy' for name in ('rows', 'train', 'held', 'labels')]
    for filename, expected in hashes.items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    OUT.mkdir()
    write(OUT / 'protocol.json', dict(dates=[20191115, 20191116], arms=ARMS, modes=MODES,
        weights=WEIGHTS, masks=NAMES, new_model_training=False, inferred17_labels_read=False,
        mask_training_labels='Only complete strictly earlier populations; query advertisement labels unavailable to mask builder',
        selection='Both-date complete candidate prediction hashes frozen before held scoring; repeatedly reused development dates',
        raw_datasets_sha256=hashes, sources_sha256={str(path): digest(path) for path in paths}))
    contracts, predictions = {}, {}
    for day in (15, 16):
        data, contract = inputs(day)
        contracts[str(day)] = contract
        predictions[str(day)] = {key(record): array_hash(prediction) for record, prediction in candidates(data)}
        np.save(OUT / f'day{day}_indices.npy', data['rows'])
        emit('full_negative_mask_candidates_locked', day=day, candidates=len(predictions[str(day)]))
    write(OUT / 'inputs_locked.json', contracts)
    write(OUT / 'predictions_locked.json', dict(protocol_sha256=digest(OUT / 'protocol.json'),
        dates=predictions, all_both_date_candidates_locked_before_scoring=True))
    check()


def verify():
    check()
    assert not (OUT / 'verification.json').exists()
    locked = read(OUT / 'predictions_locked.json')
    assert locked['protocol_sha256'] == digest(OUT / 'protocol.json')
    contracts = read(OUT / 'inputs_locked.json')
    checked = {}
    for day in (15, 16):
        primary, contract = inputs(day)
        independent, other = inputs(day, True)
        assert contract == other == contracts[str(day)]
        checked[day] = (primary, independent)
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    reports = {}
    for day in (15, 16):
        primary, independent = checked[day]
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day)
        np.testing.assert_array_equal(rows, primary['rows'])
        np.testing.assert_array_equal(rows, np.load(OUT / f'day{day}_indices.npy'))
        labels = raw.label.to_numpy()[rows]
        assert len(labels) == {15: 424780, 16: 421983}[day] and labels.sum() == {15: 3523, 16: 3724}[day]
        records, best_value, maximum = [], -1., 0.
        for (record, prediction), (other, replay) in zip(candidates(primary), candidates(independent, True)):
            assert record == other and array_hash(prediction) == locked['dates'][str(day)][key(record)]
            error = float(np.max(np.abs(prediction - replay)))
            assert error < 2e-13
            maximum = max(maximum, error)
            assert np.isfinite(prediction).all() and np.all((prediction > 0) & (prediction < 1))
            value = rank_sum_auc(labels, prediction)
            record = dict(record, auc=value)
            records.append(record)
            if value > best_value:
                best_value, best, best_prediction = value, record, prediction.copy()
        assert abs(roc_auc_score(labels, best_prediction) - best_value) < 1e-12
        np.save(OUT / f'day{day}_prediction.npy', best_prediction)
        reports[str(20191100 + day)] = dict(auc=best_value, best=best, records=records,
            rows=len(rows), clicks=int(labels.sum()), evaluation_day_labels_fit=False,
            inferred17_labels_read=False, all_both_date_candidates_locked_before_scoring=True,
            independent_temporal_masks_and_all_candidates_replayed=True,
            maximum_independent_prediction_error=maximum, held_labels_used_to_select_fixed_recipe=True,
            target_met=best_value >= {15: .815, 16: .82}[day])
        emit('full_negative_masks_scored', day=day, best=best)
    check()
    write(OUT / 'verification.json', dict(status='passed', dates=reports,
        original_datasets_unchanged=True, both_targets_met=all(r['target_met'] for r in reports.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'verify'], required=True)
    globals()[parser.parse_args().phase]()
