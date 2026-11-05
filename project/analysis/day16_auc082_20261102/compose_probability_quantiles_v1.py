"""Combine verified probability families after label-free empirical scale matching."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from fit_stacker import HERE, ROOT, digest, write
from conditional_daily import read, emit
from rich_userday_activity import array_hash
from verify_stacker import rank_sum_auc
from compose_unordered_masks_v1 import masks_for_day
from quantile_blend_core_v1 import matched_logit, blend

OUT = HERE/'runs/compose_probability_quantiles_v1'
BASE = HERE/'runs/categorical_interactions_v1'
SOURCES = {name:HERE/'runs'/name for name in ['pointwise_compact_v1','categorical_interactions_v1']}
WEIGHTS = [.05,.1,.2,.35,.5,.75,1.]


def dependencies():
    paths = [Path(__file__), HERE/'quantile_blend_core_v1.py', HERE/'test_quantile_blend_core_v1.py',
        HERE/'run_probability_quantiles_v1.ps1', HERE/'start_probability_quantiles_queue_v1.ps1',
        HERE/'compose_unordered_masks_v1.py',
        HERE/'unordered_snapshot_core_v1.py', HERE/'fit_stacker.py', HERE/'conditional_daily.py',
        HERE/'rich_userday_activity.py', HERE/'verify_stacker.py']
    for day in [15,16]:
        paths += [HERE/'runs/unordered_snapshot_ranker_v1'/f'held_contract{day}.json',
                  HERE/'runs/unordered_snapshot_ranker_v1'/f'day{day}_indices.npy']
    for directory in SOURCES.values():
        assert read(directory/'verification.json')['status'] == 'passed'
        for filename, expected in read(directory/'protocol.json')['sources_sha256'].items():
            assert digest(filename) == expected, filename
        paths += [directory/'protocol.json',directory/'verification.json',directory/'models_locked.json']
        for day in [15,16]:
            paths += [directory/f'day{day}_prediction.npy',directory/f'day{day}_indices.npy']
        for record in read(directory/'models_locked.json')['models']:
            paths.append(directory/(record['name']+'_held_margin.npy'))
    return list(dict.fromkeys(paths))


def inputs(day, independent=False):
    rows, masks = masks_for_day(day, independent)
    masks = {'all':np.ones(len(rows),dtype=bool),'changed_click':masks['changed_click']}
    base = np.load(BASE/f'day{day}_prediction.npy')
    np.testing.assert_array_equal(rows,np.load(BASE/f'day{day}_indices.npy'))
    targets = {}
    late_by_family = []
    for family, directory in SOURCES.items():
        np.testing.assert_array_equal(rows,np.load(directory/f'day{day}_indices.npy'))
        records = [r for r in read(directory/'models_locked.json')['models'] if r['day'] == day]
        late_cut = sorted({r['trees'] for r in records})[-2]
        values, late = [], []
        for record in records:
            margin = np.load(directory/(record['name']+'_held_margin.npy'))
            target = matched_logit(base,margin,independent)
            targets[family+'__'+record['name']] = target
            values.append(target)
            if record['trees'] >= late_cut:
                late.append(target)
        targets[family+'__all_mean'] = np.mean(values,axis=0)
        targets[family+'__late_mean'] = np.mean(late,axis=0)
        late_by_family.append(np.mean(late,axis=0))
    targets['both_families_late_mean'] = np.mean(late_by_family,axis=0)
    return rows, base, masks, targets


def candidates(data, independent=False):
    _, base, masks, targets = data
    yield dict(name='base',mask='all',weight=0.),base.copy()
    for name, target in targets.items():
        for mask_name, mask in masks.items():
            for weight in WEIGHTS:
                yield dict(name=name,mask=mask_name,weight=weight),blend(base,target,mask,weight,independent)


def candidate_key(record):
    return f'{record["name"]}__{record["mask"]}__{record["weight"]:g}'


def check():
    protocol = read(OUT/'protocol.json')
    for filename, expected in protocol['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/filename) == expected, filename


def prepare():
    assert not OUT.exists(), 'Preserve previous runs'
    paths = dependencies()
    OUT.mkdir()
    write(OUT/'protocol.json',dict(weights=WEIGHTS,sources=list(SOURCES),masks=['all','changed_click'],
        training='No new fitting; independently replayed strictly earlier advertisement models',
        score_mapping='Tie-averaged source ranks interpolated into the current unlabeled baseline logit distribution',
        ensemble='All-checkpoint and late-checkpoint family means, then equal late-family mean',
        selection='Both-date candidate hashes locked before scoring; repeatedly reused development dates',
        evaluation_day_labels_fit=False,inferred17_labels_read=False,
        raw_datasets_sha256=read(BASE/'protocol.json')['raw_datasets_sha256'],
        sources_sha256={str(p):digest(p) for p in paths}))
    locks = {}
    for day in [15,16]:
        data = inputs(day)
        np.save(OUT/f'day{day}_indices.npy',data[0])
        locks[str(day)] = {candidate_key(r):array_hash(p) for r,p in candidates(data)}
    write(OUT/'predictions_locked.json',dict(protocol_sha256=digest(OUT/'protocol.json'),
        dates=locks,held_labels_opened_for_scoring=False))
    check()
    emit('probability_quantile_candidates_locked',counts={d:len(v) for d,v in locks.items()})


def verify():
    assert not (OUT/'verification.json').exists(), 'Preserve completed verification'
    check()
    locked = read(OUT/'predictions_locked.json')
    assert locked['protocol_sha256'] == digest(OUT/'protocol.json')
    checked = {}
    for day in [15,16]:
        primary = inputs(day)
        independent = inputs(day,True)
        maximum = 0.
        for (r,p),(other,q) in zip(candidates(primary),candidates(independent,True)):
            assert r == other
            assert array_hash(p) == locked['dates'][str(day)][candidate_key(r)]
            error = float(np.max(np.abs(p-q)))
            assert error < 2e-13
            maximum = max(maximum,error)
        checked[day] = primary, maximum
        emit('probability_quantile_independent_replay',day=day,maximum_error=maximum)
    raw = pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    reports = {}
    for day in [15,16]:
        data, maximum = checked[day]
        rows = data[0]
        np.testing.assert_array_equal(rows,np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day))
        y = raw.label.to_numpy()[rows]
        assert len(rows) == {15:424780,16:421983}[day] and y.sum() == {15:3523,16:3724}[day]
        records, best_value = [], -1.
        for record, prediction in candidates(data):
            assert np.isfinite(prediction).all() and np.all((prediction>0)&(prediction<1))
            value = rank_sum_auc(y,prediction)
            record = dict(record,auc=value)
            records.append(record)
            if value > best_value:
                best_value, best, best_prediction = value, record, prediction.copy()
        assert abs(roc_auc_score(y,best_prediction)-best_value)<1e-12
        np.save(OUT/f'day{day}_prediction.npy',best_prediction)
        reports[str(20191100+day)] = dict(auc=best_value,best=best,records=records,rows=len(y),clicks=int(y.sum()),
            evaluation_day_labels_fit=False,held_labels_used_to_select_fixed_recipe=True,
            all_both_date_candidates_locked_before_scoring=True,independent_rank_and_odds_replay=True,
            maximum_independent_prediction_error=maximum,target_met=best_value>={15:.815,16:.82}[day])
        emit('probability_quantile_scored',day=day,best=best)
    check()
    write(OUT/'verification.json',dict(status='passed',dates=reports,original_datasets_unchanged=True,
        both_targets_met=all(r['target_met'] for r in reports.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=['prepare','verify'])
    {'prepare':prepare,'verify':verify}[parser.parse_args().phase]()
