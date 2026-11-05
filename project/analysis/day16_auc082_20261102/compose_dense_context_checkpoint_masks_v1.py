"""Transfer all dense context checkpoints onto the frozen history-affinity winner."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash
from full_negative_mask_core_v1 import masks as build_masks, NAMES
import compose_full_negative_masks_v1 as engine

OUT = HERE / 'runs/compose_dense_context_checkpoint_masks_v1'
BASE = HERE / 'runs/compose_history_dot_embedding_masks_v1'
DENSE = HERE / 'runs/full_negative_dense_context_lgb_v1'
INPUT = HERE / 'runs/full_negative_context_v1'
MODEL_ARMS = ['dense_context_leaf31_w5', 'dense_context_leaf63_w25']
CHECKPOINTS = [400, 1200, 2400]
ARMS = [f'{arm}_trees{trees}' for arm in MODEL_ARMS for trees in CHECKPOINTS] + ['mean_last']
WEIGHTS = [.0125, .025, .05, .1, .2, .35, .5, .75]
MODES = ['replace_global', 'add_within', 'rank_global']
MASKS = NAMES + ['all']
PER_DATE = 2 + len(ARMS) * len(MODES) * len(WEIGHTS) * len(MASKS)
assert len(ARMS) == 7 and PER_DATE == 1178
engine.OUT, engine.BASE, engine.ANCHOR = OUT, BASE, BASE
engine.ARMS, engine.WEIGHTS, engine.MODES, engine.NAMES = ARMS, WEIGHTS, MODES, MASKS
engine.emit = lambda name, **values: emit(name.replace('full_negative_masks', 'dense_context_checkpoint_masks'), **values)
original_candidates = engine.candidates


def inputs(day, independent=False):
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'user_id', 'task_id'], dtype='int32')
    directory = INPUT / f'input_day{day}'
    rows = np.load(directory / 'rows.npy')
    training = rows[np.load(directory / 'train.npy')]
    query = rows[np.load(directory / 'held.npy')]
    np.testing.assert_array_equal(query, np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day))
    for folder in (BASE, DENSE):
        np.testing.assert_array_equal(query, np.load(folder / f'day{day}_indices.npy'))
    cohorts = build_masks(raw.pt_d.to_numpy(), raw.user_id.to_numpy(), raw.task_id.to_numpy(),
        training, np.load(directory / 'labels.npy'), query, 20191100 + day, independent)
    cohorts['all'] = np.ones(len(query), dtype=bool)
    if independent:
        _, groups = np.unique(raw.user_id.to_numpy()[query], return_inverse=True)
    else:
        groups, _ = pd.factorize(raw.user_id.to_numpy()[query], sort=True)
    margins = {arm: np.load(DENSE / f'day{day}_{arm}_held_margin.npy') for arm in ARMS[:-1]}
    last = [margins[f'{arm}_trees2400'] for arm in MODEL_ARMS]
    margins['mean_last'] = (last[0] + last[1]) / 2 if independent else np.mean(last, axis=0)
    current = np.load(BASE / f'day{day}_prediction.npy')
    source = np.load(DENSE / f'day{day}_prediction.npy')
    assert all(value.shape == current.shape for value in [source, *margins.values(), *cohorts.values()])
    assert len(margins) == len(ARMS) and len(cohorts) == len(MASKS)
    contract = dict(rows_sha256=array_hash(query), groups_sha256=array_hash(groups),
        masks_sha256={name: array_hash(value) for name, value in cohorts.items()},
        mask_rows={name: int(value.sum()) for name, value in cohorts.items()},
        margins_sha256={name: array_hash(value) for name, value in margins.items()},
        current_sha256=array_hash(current), dense_winner_sha256=array_hash(source),
        training_rows_sha256=array_hash(training), max_mask_label_date=20191100 + day - 1,
        all_training_rows_strictly_past=True, inferred17_labels_read=False)
    return dict(current=current, anchor=current.copy(), dense=source, groups=groups,
                margins=margins, masks=cohorts, rows=query), contract


def candidates(data, independent=False):
    yield from original_candidates(data, independent)
    yield dict(name='dense_verified_best', mask='all', mode='source', weight=1.), data['dense'].copy()


engine.inputs, engine.candidates = inputs, candidates


def preflight():
    fixture = dict(current=np.array([.01, .08, .03, .2, .1]),
        anchor=np.array([.01, .08, .03, .2, .1]), dense=np.array([.02, .07, .04, .15, .12]),
        groups=np.array([0, 0, 1, 1, 2]),
        margins={name: np.array([-.1, -.1, .4, .7, -.3]) + index * np.array([.03, -.02, .01, .04, -.01])
                 for index, name in enumerate(ARMS)},
        masks={name: np.array([True, False, True, False, True]) for name in MASKS})
    fixture['masks']['all'][:] = True
    seen, maximum = set(), 0.
    for (record, primary), (other, independent) in zip(candidates(fixture), candidates(fixture, True)):
        assert record == other
        identity = engine.key(record)
        assert identity not in seen
        seen.add(identity)
        difference = float(np.max(np.abs(primary - independent)))
        assert difference < 2e-13
        maximum = max(maximum, difference)
        if record['name'] not in ('base', 'dense_verified_best'):
            mask = fixture['masks'][record['mask']]
            np.testing.assert_array_equal(primary[~mask], fixture['current'][~mask])
    assert len(seen) == PER_DATE
    return dict(status='passed', fixture_candidates=len(seen), candidate_keys_unique=True,
        untouched_mask_complement_exact=True, maximum_independent_prediction_error=maximum,
        both_verified_source_controls_included=True, held_labels_used=False,
        no_model_training=True, inferred17_labels_read=False)


def prepare():
    assert not OUT.exists(), 'Preserve all prior compositions and failures'
    plan_path = HERE / 'dense_context_checkpoint_cohort_transfer_queued_v1.json'
    plan = read(plan_path)
    assert plan['total_candidates'] == 2 * PER_DATE
    for path, expected in plan['sources_sha256'].items():
        assert digest(path) == expected, path
    probe = preflight()
    paths = [Path(__file__), plan_path] + [HERE / name for name in (
        'run_compose_dense_context_checkpoint_masks_v1.ps1', 'compose_full_negative_masks_v1.py',
        'publish_progress_v4.py', 'record_two_day_progress_v4.py',
        'full_negative_mask_core_v1.py', 'test_full_negative_mask_core_v1.py',
        'conditional_daily.py', 'fit_stacker.py', 'rich_userday_activity.py',
        'cross_user_pair_ranker_v1.py', 'quantile_blend_core_v1.py', 'verify_stacker.py')]
    hashes = read(BASE / 'protocol.json')['raw_datasets_sha256']
    for run in (BASE, DENSE):
        report = read(run / 'verification.json')
        assert report['status'] == 'passed'
        protocol = read(run / 'protocol.json')
        assert protocol['raw_datasets_sha256'] == hashes and not protocol['inferred17_labels_read']
        for path, expected in protocol['sources_sha256'].items():
            assert digest(path) == expected, path
            if Path(path).suffix == '.py':
                paths.append(Path(path))
        paths += [run / 'protocol.json', run / 'verification.json']
        for day in (15, 16):
            assert report['dates'][str(20191100 + day)]['evaluation_day_labels_fit'] is False
            paths += [run / f'day{day}_{name}.npy' for name in ('prediction', 'indices')]
    assert all(record['independent_all_saved_models_and_candidates_replayed']
               for record in read(DENSE / 'verification.json')['dates'].values())
    models = read(DENSE / 'models_locked.json')
    assert len(models['models']) == 12
    assert models['protocol_sha256'] == digest(DENSE / 'protocol.json')
    paths += [DENSE / name for name in ('models_locked.json', 'predictions_locked.json', 'inputs_locked.json')]
    for day in (15, 16):
        paths += [DENSE / f'day{day}_{arm}_held_margin.npy' for arm in ARMS[:-1]]
        paths += [INPUT / f'input_day{day}/{name}.npy' for name in ('rows', 'train', 'held', 'labels')]
    for filename, expected in hashes.items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    OUT.mkdir()
    write(OUT / 'preflight.json', probe)
    write(OUT / 'protocol.json', dict(dates=[20191115, 20191116], arms=ARMS, modes=MODES,
        weights=WEIGHTS, masks=MASKS, baseline=str(BASE), dense_source=str(DENSE),
        candidates_per_date=PER_DATE, total_candidates=2 * PER_DATE,
        new_model_training=False, evaluation_day_labels_fit=False, inferred17_labels_read=False,
        rationale='Test all six independently verified dense context checkpoints and the last-checkpoint mean against the frozen history-affinity cohort baseline; retain the baseline and dense global winner as controls.',
        mask_training_labels='Only complete strictly earlier populations; query ad labels unavailable to mask builder',
        selection='All2356 complete-cohort candidate hashes frozen across both dates before scoring; both dates repeatedly reused for development',
        raw_datasets_sha256=hashes, sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)}))
    contracts, locked = {}, {}
    for day in (15, 16):
        data, contract = inputs(day)
        contracts[str(day)] = contract
        locked[str(day)] = {engine.key(record): array_hash(prediction) for record, prediction in candidates(data)}
        assert len(locked[str(day)]) == PER_DATE
        np.save(OUT / f'day{day}_indices.npy', data['rows'])
        emit('dense_context_checkpoint_mask_candidates_locked', day=day, candidates=PER_DATE)
    write(OUT / 'inputs_locked.json', contracts)
    write(OUT / 'predictions_locked.json', dict(protocol_sha256=digest(OUT / 'protocol.json'),
        dates=locked, all_both_date_candidates_locked_before_scoring=True))
    engine.check()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['preflight', 'prepare', 'verify'], required=True)
    phase = parser.parse_args().phase
    if phase == 'preflight':
        print(preflight(), flush=True)
    elif phase == 'prepare':
        prepare()
    else:
        engine.verify()
