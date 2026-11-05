"""Historical-cohort transfer of the independently replayed class-weight models."""
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

OUT = HERE / 'runs/compose_class_balance_masks_v1'
BASE = HERE / 'runs/full_negative_class_balance_v1'
ANCHOR = HERE / 'runs/compose_full_negative_masks_v1'
INPUT = HERE / 'runs/full_negative_context_v1'
ARMS = ['natural_d6', 'positive25_d4', 'positive25_d6', 'mean_last']
WEIGHTS = [0., .025, .05, .1, .2, .35, .5]
MODES = ['replace_global', 'add_within', 'rank_global']
engine.OUT, engine.BASE, engine.ANCHOR = OUT, BASE, ANCHOR
engine.ARMS, engine.WEIGHTS, engine.MODES = ARMS, WEIGHTS, MODES
engine.emit = lambda name, **values: emit(name.replace('full_negative_masks', 'class_balance_masks'), **values)


def inputs(day, independent=False):
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'user_id', 'task_id'], dtype='int32')
    directory = INPUT / f'input_day{day}'
    global_rows = np.load(directory / 'rows.npy')
    training = global_rows[np.load(directory / 'train.npy')]
    query = global_rows[np.load(directory / 'held.npy')]
    np.testing.assert_array_equal(query, np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day))
    np.testing.assert_array_equal(query, np.load(BASE / f'day{day}_indices.npy'))
    np.testing.assert_array_equal(query, np.load(ANCHOR / f'day{day}_indices.npy'))
    masks = build_masks(raw.pt_d.to_numpy(), raw.user_id.to_numpy(), raw.task_id.to_numpy(),
        training, np.load(directory / 'labels.npy'), query, 20191100 + day, independent)
    groups, _ = pd.factorize(raw.user_id.to_numpy()[query], sort=True)
    margins = {arm: np.load(BASE / f'day{day}_{arm}_trees3200_held_margin.npy') for arm in ARMS[:3]}
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
        all_training_rows_strictly_past=True, inferred17_labels_read=False)
    return dict(current=current, anchor=anchor, groups=groups, margins=margins, masks=masks, rows=query), contract


engine.inputs = inputs


def preflight():
    fixture = dict(current=np.array([.01, .08, .03, .2, .1]),
        anchor=np.array([.02, .07, .04, .15, .12]), groups=np.array([0, 0, 1, 1, 2]),
        margins={'probe': np.array([-.1, -.1, .4, .7, -.3])},
        masks={name: np.array([True, False, True, False, True]) for name in NAMES})
    count, maximum = 0, 0.
    for (record, primary), (other, independent) in zip(engine.candidates(fixture), engine.candidates(fixture, True)):
        assert record == other
        difference = float(np.max(np.abs(primary - independent)))
        assert difference < 2e-13
        maximum = max(maximum, difference)
        if record['name'] != 'base':
            np.testing.assert_array_equal(primary[~fixture['masks'][record['mask']]],
                fixture['current'][~fixture['masks'][record['mask']]])
        count += 1
    assert count == 1 + len(MODES) * len(WEIGHTS) * len(NAMES)
    return dict(status='passed', fixture_candidates=count,
        untouched_mask_complement_exact=True, maximum_independent_prediction_error=maximum,
        tied_margins_and_repeated_users_checked=True, no_model_training=True,
        held_labels_used=False, inferred17_labels_read=False)


def prepare():
    assert not OUT.exists(), 'Preserve all previous compositions and failures'
    probe = preflight()
    paths = [Path(__file__)] + [HERE / name for name in (
        'run_compose_class_balance_masks_v1.ps1', 'compose_full_negative_masks_v1.py',
        'full_negative_mask_core_v1.py', 'test_full_negative_mask_core_v1.py',
        'conditional_daily.py', 'fit_stacker.py', 'rich_userday_activity.py',
        'cross_user_pair_ranker_v1.py', 'quantile_blend_core_v1.py', 'verify_stacker.py')]
    hashes = read(BASE / 'protocol.json')['raw_datasets_sha256']
    assert Path(read(BASE / 'protocol.json')['baseline']).resolve() == ANCHOR.resolve()
    for run in (BASE, ANCHOR):
        report = read(run / 'verification.json')
        assert report['status'] == 'passed'
        protocol = read(run / 'protocol.json')
        assert protocol['raw_datasets_sha256'] == hashes
        for path, expected in protocol['sources_sha256'].items():
            assert digest(path) == expected, path
            if Path(path).suffix == '.py':
                paths.append(Path(path))
        paths += [run / 'verification.json', run / 'protocol.json']
        for day in (15, 16):
            assert report['dates'][str(20191100 + day)]['evaluation_day_labels_fit'] is False
            paths += [run / f'day{day}_{name}.npy' for name in ('prediction', 'indices')]
    assert all(value['independent_all_saved_models_and_candidates_replayed']
        for value in read(BASE / 'verification.json')['dates'].values())
    paths += [BASE / name for name in ('models_locked.json', 'predictions_locked.json', 'inputs_locked.json')]
    for day in (15, 16):
        paths += [BASE / f'day{day}_{arm}_trees3200_held_margin.npy' for arm in ARMS[:3]]
        paths += [INPUT / f'input_day{day}/{name}.npy' for name in ('rows', 'train', 'held', 'labels')]
    for filename, expected in hashes.items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    OUT.mkdir()
    write(OUT / 'preflight.json', probe)
    write(OUT / 'protocol.json', dict(dates=[20191115, 20191116], arms=ARMS, modes=MODES,
        weights=WEIGHTS, masks=NAMES, baseline=str(BASE), anchor=str(ANCHOR),
        new_model_training=False, inferred17_labels_read=False, evaluation_day_labels_fit=False,
        rationale='The newly verified positive25 shallow model improved15 globally; compare cohort-specific transfer of all three class-weight arms, retaining the new verified global winner outside each mask.',
        mask_training_labels='Only complete strictly earlier populations; query ad labels unavailable to mask builder',
        selection='All1010 full-cohort candidate hashes frozen across both dates before scoring; both development dates repeatedly reused',
        raw_datasets_sha256=hashes, sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)}))
    contracts, locked = {}, {}
    for day in (15, 16):
        data, contract = inputs(day)
        contracts[str(day)] = contract
        locked[str(day)] = {engine.key(record): array_hash(prediction) for record, prediction in engine.candidates(data)}
        assert len(locked[str(day)]) == 1 + 4 * len(MODES) * len(WEIGHTS) * len(NAMES) == 505
        np.save(OUT / f'day{day}_indices.npy', data['rows'])
        emit('class_balance_mask_candidates_locked', day=day, candidates=len(locked[str(day)]))
    write(OUT / 'inputs_locked.json', contracts)
    write(OUT / 'predictions_locked.json', dict(protocol_sha256=digest(OUT / 'protocol.json'),
        dates=locked, all_both_date_candidates_locked_before_scoring=True))
    engine.check()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'verify'], required=True)
    phase = parser.parse_args().phase
    prepare() if phase == 'prepare' else engine.verify()
