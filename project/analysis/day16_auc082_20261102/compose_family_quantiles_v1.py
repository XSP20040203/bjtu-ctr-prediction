"""Preregistered family means, replayed against historical prediction locks."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '2'
import argparse
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash
from verify_stacker import rank_sum_auc
from quantile_blend_core_v1 import matched_logit, blend
from compose_unordered_masks_v1 import masks_for_day
from cold_task_core import cold_mask

OUT = HERE / 'runs/compose_family_quantiles_v1'
BASE = HERE / 'runs/compose_transition_priors_v1'
FAMILIES = ['compact_full_sample_ranker_v1', 'unordered_snapshot_ranker_v1',
            'pointwise_compact_v1', 'task_audience_row_ranker_v1',
            'profile_response_ranker_v1', 'cold_task_ranker_v1']
WEIGHTS = [.025, .05, .1, .2, .35, .5]


def dependencies():
    paths = [Path(__file__), HERE / 'run_family_quantiles_v1.ps1',
             HERE / 'quantile_blend_core_v1.py', HERE / 'test_quantile_blend_core_v1.py',
             HERE / 'compose_unordered_masks_v1.py', HERE / 'unordered_snapshot_core_v1.py',
             HERE / 'cold_task_core.py', HERE / 'rich_userday_activity.py',
             HERE / 'fit_stacker.py', HERE / 'conditional_daily.py', HERE / 'verify_stacker.py']
    assert read(BASE / 'verification.json')['status'] == 'passed'
    paths += [BASE / 'protocol.json', BASE / 'verification.json']
    for day in [15, 16]:
        paths += [BASE / f'day{day}_indices.npy', BASE / f'day{day}_prediction.npy',
                  HERE / 'runs/unordered_snapshot_ranker_v1' / f'held_contract{day}.json']
    for family in FAMILIES:
        folder = HERE / 'runs' / family
        assert read(folder / 'verification.json')['status'] == 'passed'
        protocol = read(folder / 'protocol.json')
        assert protocol['inferred17_labels_read'] is False and protocol['evaluation_day_labels_fit'] is False
        for filename, expected in protocol['sources_sha256'].items():
            if Path(filename).suffix == '.py':
                assert digest(filename) == expected, filename
                paths.append(Path(filename))
        paths += [folder / filename for filename in ['protocol.json', 'verification.json',
                  'models_locked.json', 'predictions_locked.json']]
        for day in [15, 16]:
            contract_path = folder / f'input_contract{day}.json'
            contract = read(contract_path)
            assert contract['max_training_date'] < 20191100 + day
            paths += [contract_path, folder / f'held_contract{day}.json',
                      folder / f'day{day}_indices.npy', folder / f'day{day}_prediction.npy']
        models = read(folder / 'models_locked.json')
        assert models['protocol_sha256'] == digest(folder / 'protocol.json')
        for record in models['models']:
            model = folder / (record['name'] + '.cbm')
            assert digest(model) == record['model_sha256']
            assert digest(folder / f'input_contract{record["day"]}.json') == record['input_contract_sha256']
            paths += [model, folder / (record['name'] + '_held_margin.npy')]
    return list(dict.fromkeys(paths))


def check():
    protocol = read(OUT / 'protocol.json')
    for filename, expected in protocol['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename


def inputs(day, independent=False):
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'user_id', 'task_id'], dtype='int32')
    query = raw.loc[raw.pt_d.eq(20191100 + day)].reset_index(drop=True)
    rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day)
    np.testing.assert_array_equal(rows, np.load(BASE / f'day{day}_indices.npy'))
    groups, _ = pd.factorize(query.user_id, sort=True)
    snapshot_rows, snapshot = masks_for_day(day, independent)
    np.testing.assert_array_equal(rows, snapshot_rows)
    cold = cold_mask(raw[['pt_d', 'task_id']], query[['pt_d', 'task_id']], 20191100 + day)
    masks = dict(all=np.ones(len(rows), dtype=bool), cold_task=cold, changed_click=snapshot['changed_click'])
    base = np.load(BASE / f'day{day}_prediction.npy')
    targets = {}
    late_families = []
    for family in FAMILIES:
        folder = HERE / 'runs' / family
        np.testing.assert_array_equal(rows, np.load(folder / f'day{day}_indices.npy'))
        records = [record for record in read(folder / 'models_locked.json')['models'] if record['day'] == day]
        margins = {record['name']: np.load(folder / (record['name'] + '_held_margin.npy')) for record in records}
        parent = importlib.import_module(family)
        frozen = read(folder / 'predictions_locked.json')
        assert frozen['protocol_sha256'] == digest(folder / 'protocol.json')
        # Historical candidate locks establish that each margin's ordering is the same
        # ordering independently replayed during its original full-cohort verification.
        count = 0
        for record, prediction in parent.candidates(day, margins, groups):
            assert array_hash(prediction) == frozen['candidates_sha256'][str(day)][parent.candidate_key(record)]
            count += 1
        assert count == len(frozen['candidates_sha256'][str(day)])
        late_cut = sorted({record['trees'] for record in records})[-2]
        values, late = [], []
        for record in records:
            target = matched_logit(base, margins[record['name']], independent)
            values.append(target)
            if record['trees'] >= late_cut:
                late.append(target)
        targets[family + '__all_mean'] = np.mean(values, axis=0)
        targets[family + '__late_mean'] = np.mean(late, axis=0)
        late_families.append(targets[family + '__late_mean'])
    targets['equal_late_family_mean'] = np.mean(late_families, axis=0)
    emit('family_quantiles_inputs_replayed', day=day, independent=independent, families=len(FAMILIES))
    return rows, base, masks, targets


def candidates(data, independent=False):
    _, base, masks, targets = data
    yield dict(name='base', mask='all', weight=0.), base.copy()
    for name, target in targets.items():
        for mask_name, mask in masks.items():
            for weight in WEIGHTS:
                yield dict(name=name, mask=mask_name, weight=weight), blend(base, target, mask, weight, independent)


def key(record):
    return f'{record["name"]}__{record["mask"]}__{record["weight"]:g}'


def prepare():
    assert not OUT.exists(), 'Preserve all completed and partial runs'
    paths = dependencies()
    OUT.mkdir()
    write(OUT / 'protocol.json', dict(families=FAMILIES, weights=WEIGHTS,
        masks=['all', 'cold_task', 'changed_click'], new_model_training=False,
        score_mapping='Tie-averaged model rank into the current label-free baseline logit distribution',
        averaging='All and late checkpoints per family; equal late-family mean; no learned ensemble coefficients',
        selection='Both-date candidate hashes locked before scoring; dates repeatedly reused',
        evaluation_day_labels_fit=False, inferred17_labels_read=False,
        raw_datasets_sha256=read(BASE / 'protocol.json')['raw_datasets_sha256'],
        sources_sha256={str(path): digest(path) for path in paths}))
    locks = {}
    for day in [15, 16]:
        data = inputs(day)
        np.save(OUT / f'day{day}_indices.npy', data[0])
        locks[str(day)] = {key(record): array_hash(prediction) for record, prediction in candidates(data)}
    write(OUT / 'predictions_locked.json', dict(protocol_sha256=digest(OUT / 'protocol.json'),
        dates=locks, held_labels_opened_for_scoring=False))
    check()
    emit('family_quantile_candidates_locked', candidates={day: len(values) for day, values in locks.items()})


def verify():
    check()
    assert not (OUT / 'verification.json').exists()
    locked = read(OUT / 'predictions_locked.json')
    assert locked['protocol_sha256'] == digest(OUT / 'protocol.json')
    checked = {}
    for day in [15, 16]:
        primary = inputs(day)
        independent = inputs(day, True)
        maximum = 0.
        for (record, prediction), (other, replay) in zip(candidates(primary), candidates(independent, True)):
            assert record == other and array_hash(prediction) == locked['dates'][str(day)][key(record)]
            error = float(np.max(np.abs(prediction - replay)))
            assert error < 2e-13
            maximum = max(maximum, error)
        checked[day] = primary, maximum
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    reports = {}
    for day in [15, 16]:
        data, maximum = checked[day]
        rows = data[0]
        np.testing.assert_array_equal(rows, np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day))
        y = raw.label.to_numpy()[rows]
        assert len(y) == {15: 424780, 16: 421983}[day] and y.sum() == {15: 3523, 16: 3724}[day]
        records, best_value = [], -1.
        for record, prediction in candidates(data):
            assert np.isfinite(prediction).all() and np.all((prediction > 0) & (prediction < 1))
            value = rank_sum_auc(y, prediction)
            record = dict(record, auc=value)
            records.append(record)
            if value > best_value:
                best_value, best, best_prediction = value, record, prediction.copy()
        assert abs(roc_auc_score(y, best_prediction) - best_value) < 1e-12
        np.save(OUT / f'day{day}_prediction.npy', best_prediction)
        reports[str(20191100 + day)] = dict(auc=best_value, best=best, records=records, rows=len(y), clicks=int(y.sum()),
            evaluation_day_labels_fit=False, inferred17_labels_read=False, held_labels_used_to_select_fixed_recipe=True,
            all_both_date_candidates_locked_before_scoring=True, all_historical_parent_candidate_locks_replayed=True,
            independent_rank_mapping_and_odds_replayed=True, maximum_independent_prediction_error=maximum,
            target_met=best_value >= {15: .815, 16: .82}[day])
        emit('family_quantiles_scored', day=day, best=best)
    check()
    write(OUT / 'verification.json', dict(status='passed', dates=reports,
        original_datasets_unchanged=True, both_targets_met=all(record['target_met'] for record in reports.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['prepare', 'verify'])
    {'prepare': prepare, 'verify': verify}[parser.parse_args().phase]()
