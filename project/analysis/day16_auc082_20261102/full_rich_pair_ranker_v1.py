"""Complete-negative date-pair ranking on the independently rebuilt576 columns."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
import full_rich_negative_v1 as engine
from full_rich_negative_inputs_v1 import compose
from full_negative_pair_core_v1 import design
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash
from catboost_userday_residual import save_model, load_model

SOURCE = HERE / 'runs/full_rich_negative_v1'
OUT = HERE / 'runs/full_rich_pair_ranker_v1'
PREFLIGHT = HERE / 'runs/full_rich_pair_preflight_v1'
ARMS = [dict(name='complete_pair_d6', depth=6), dict(name='complete_pair_d8', depth=8)]
CHECKPOINTS = [400, 1200, 2400]
PAIR_PARAMS = dict(seed=151681, negative_block=512, positive_draws=64)
PARAMS = dict(engine.PARAMS, loss_function='PairLogit', learning_rate=.025,
              l2_leaf_reg=100., random_seed=151681)

# Reuse the frozen independent feature/model/prediction verifier with explicit run settings.
# The custom training phase below changes the loss and checks every training pair.
engine.OUT = OUT
engine.BASE = SOURCE
engine.ARMS = ARMS
engine.CHECKPOINTS = CHECKPOINTS
engine.PARAMS = PARAMS


def dependencies():
    return [Path(__file__), Path(engine.__file__), HERE / 'full_negative_pair_core_v1.py',
        HERE / 'test_full_negative_pair_core_v1.py', HERE / 'run_full_rich_pair_ranker_v1.ps1']


def pair_inputs(day):
    declared, arrays, _, _ = engine.components(day)
    dates = np.load(engine.CONTEXT / f'input_day{day}/dates.npy')[arrays['train']]
    labels = arrays['labels']
    order, groups, pairs = design(dates, labels, 20191100 + day, **PAIR_PARAMS)
    np.testing.assert_array_equal(np.sort(order), np.arange(len(labels)))
    contract = dict(order_sha256=array_hash(order), groups_sha256=array_hash(groups),
        pairs_sha256=array_hash(pairs), rows=len(order), pairs=len(pairs),
        positives=int(labels.sum()), negatives=int((labels == 0).sum()),
        maximum_query_size=int(np.bincount(groups).max()),
        every_negative_in_pairs=True, every_positive_in_pairs=True,
        max_training_date=int(dates.max()), input_contract=declared)
    return order, groups, pairs, labels[order], contract


def preflight():
    assert not PREFLIGHT.exists(), 'Preserve previous preflights'
    declared, arrays, left, right = engine.components(15)
    labels = arrays['labels']
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    chosen = np.sort(np.r_[pos[np.linspace(0, len(pos) - 1, 64, dtype='int64')],
                           neg[np.linspace(0, len(neg) - 1, 1024, dtype='int64')]])
    dates = np.load(engine.CONTEXT / 'input_day15/dates.npy')[arrays['train'][chosen]]
    order, groups, pairs = design(dates, labels[chosen], 20191115, **PAIR_PARAMS)
    positions = arrays['train'][chosen[order]]
    data = compose(left, right, positions, declared['feature_names'], declared['categories'])
    replay = compose(left, right, positions, declared['feature_names'], declared['categories'], True)
    pd.testing.assert_frame_equal(data, replay)
    params = {k: v for k, v in PARAMS.items() if k not in ('devices', 'gpu_ram_part', 'gpu_cat_features_storage', 'pinned_memory_size')}
    params.update(task_type='CPU', iterations=20, depth=4, verbose=False)
    model = CatBoostRanker(**params)
    pool = Pool(data, label=labels[chosen[order]], group_id=groups, pairs=pairs, cat_features=declared['categories'])
    model.fit(pool)
    PREFLIGHT.mkdir()
    path = PREFLIGHT / 'model.cbm'
    save_model(model, path)
    first = model.predict(pool, thread_count=2)
    second = load_model(path).predict(Pool(replay, cat_features=declared['categories']), prediction_type='RawFormulaVal', thread_count=2)
    np.testing.assert_array_equal(first, second)
    write(PREFLIGHT / 'verification.json', dict(status='passed', feature_only=True,
        rows=len(data), pairs=len(pairs), columns=576, max_training_date=int(dates.max()),
        every_negative_in_pairs=True, every_positive_in_pairs=True,
        held_labels_read=False, inferred17_labels_read=False, device='CPU', full_gpu_capacity_proven=False,
        independent_join_exact=True, serialization_error=0., model_sha256=digest(path),
        sources_sha256={str(p): digest(p) for p in dependencies()}))
    emit('full_rich_pair_real_preflight_passed', rows=len(data), pairs=len(pairs))


def prepare():
    assert not OUT.exists(), 'Preserve previous fits'
    checked = read(PREFLIGHT / 'verification.json')
    assert checked['status'] == 'passed' and checked['every_negative_in_pairs']
    for filename, expected in checked['sources_sha256'].items():
        assert digest(filename) == expected, filename
    source = read(SOURCE / 'protocol.json')
    report = read(SOURCE / 'verification.json')
    assert report['status'] == 'passed'
    paths = dependencies() + [PREFLIGHT / 'verification.json', PREFLIGHT / 'model.cbm',
        SOURCE / 'protocol.json', SOURCE / 'verification.json', SOURCE / 'inputs_locked.json',
        SOURCE / 'models_locked.json', SOURCE / 'predictions_locked.json']
    for filename, expected in source['sources_sha256'].items():
        assert digest(filename) == expected, filename
        paths.append(Path(filename))
    prepared = {}
    for day in (15, 16):
        assert report['dates'][str(20191100 + day)]['evaluation_day_labels_fit'] is False
        order, groups, pairs, labels, contract = pair_inputs(day)
        assert contract['input_contract'] == read(SOURCE / 'inputs_locked.json')[str(day)]
        assert contract['maximum_query_size'] <= 1024
        declared, arrays, _, _ = engine.components(day, True)
        assert declared == contract['input_contract']
        np.testing.assert_array_equal(arrays['rows'][arrays['held']], np.load(SOURCE / f'day{day}_indices.npy'))
        prepared[day] = (order, groups, pairs, contract)
        paths += [SOURCE / f'day{day}_prediction.npy', SOURCE / f'day{day}_indices.npy']
    OUT.mkdir()
    write(OUT / 'protocol.json', dict(dates=[20191115, 20191116], params=PARAMS,
        arms=ARMS, checkpoints=CHECKPOINTS, pair_design=PAIR_PARAMS,
        modes=engine.MODES, weights=engine.WEIGHTS, baseline=str(SOURCE),
        population='Every earlier positive and negative, retaining all576 verified features',
        objective='Date-isolated PairLogit; every negative and positive participates in at least one pair',
        held_labels_for_fitting=False, inferred17_labels_read=False, development_dates_repeatedly_reused=True,
        verification='Pair metadata independently regenerated; separate component reconstruction/column join and model replay',
        raw_datasets_sha256=source['raw_datasets_sha256'],
        sources_sha256={str(p): digest(p) for p in dict.fromkeys(paths)}))
    write(OUT / 'inputs_locked.json', read(SOURCE / 'inputs_locked.json'))
    metadata = {}
    for day, (order, groups, pairs, contract) in prepared.items():
        for name, value in [('order', order), ('groups', groups), ('pairs', pairs)]:
            np.save(OUT / f'day{day}_{name}.npy', value)
        np.save(OUT / f'day{day}_indices.npy', np.load(SOURCE / f'day{day}_indices.npy'))
        metadata[str(day)] = contract
    write(OUT / 'pairs_locked.json', metadata)
    engine.check()
    emit('full_rich_pair_protocol_locked', pairs={d: r['pairs'] for d, r in metadata.items()})


def fit():
    engine.check()
    assert not (OUT / 'models_locked.json').exists() and not list(OUT.glob('*.cbm')), 'Preserve partial fits'
    records = []
    for day in (15, 16):
        engine.wait_for_memory()
        order, groups, pairs, labels, declared = pair_inputs(day)
        assert declared == read(OUT / 'pairs_locked.json')[str(day)]
        for name, values in [('order', order), ('groups', groups), ('pairs', pairs)]:
            np.testing.assert_array_equal(values, np.load(OUT / f'day{day}_{name}.npy'))
        data, cats, _ = engine.frame(day, 'train', selected=order)
        for arm in ARMS:
            emit('full_rich_pair_fit_started', day=day, arm=arm['name'], rows=len(data), pairs=len(pairs))
            pool = Pool(data, label=labels, group_id=groups, pairs=pairs, cat_features=cats)
            model = CatBoostRanker(**dict(PARAMS, depth=arm['depth'], iterations=max(CHECKPOINTS)))
            model.fit(pool)
            del pool
            for trees in CHECKPOINTS:
                name = f'day{day}_{arm["name"]}_trees{trees}'
                saved = model.copy()
                saved.shrink(trees)
                path = OUT / (name + '.cbm')
                save_model(saved, path)
                records.append(dict(arm, day=day, trees=trees, name=name,
                    model_sha256=digest(path), pair_contract_sha256=digest(OUT / 'pairs_locked.json'),
                    max_training_date=declared['max_training_date']))
                del saved
            del model
            gc.collect()
            emit('full_rich_pair_fit_complete', day=day, arm=arm['name'])
        del data, order, groups, pairs, labels
        gc.collect()
    write(OUT / 'models_locked.json', dict(models=records, protocol_sha256=digest(OUT / 'protocol.json'),
        inputs_locked_sha256=digest(OUT / 'inputs_locked.json'), pairs_locked_sha256=digest(OUT / 'pairs_locked.json')))


def verify():
    engine.check()
    locked = read(OUT / 'models_locked.json')
    assert locked['pairs_locked_sha256'] == digest(OUT / 'pairs_locked.json')
    checked = {}
    for day in (15, 16):
        order, groups, pairs, labels, declared = pair_inputs(day)
        assert declared == read(OUT / 'pairs_locked.json')[str(day)]
        for name, values in [('order', order), ('groups', groups), ('pairs', pairs)]:
            np.testing.assert_array_equal(values, np.load(OUT / f'day{day}_{name}.npy'))
        for model in locked['models']:
            assert model['pair_contract_sha256'] == digest(OUT / 'pairs_locked.json')
        checked[str(day)] = {k: v for k, v in declared.items() if k != 'input_contract'}
        del order, groups, pairs, labels
        gc.collect()
    write(OUT / 'pair_verification.json', dict(status='passed', dates=checked,
        full_pair_metadata_recreated=True, no_held_or_future_labels=True, inferred17_labels_read=False))
    engine.verify()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['preflight', 'prepare', 'fit', 'evaluate', 'verify'])
    phase = parser.parse_args().phase
    engine.evaluate() if phase == 'evaluate' else globals()[phase]()
