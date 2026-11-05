"""GPU CatBoost on744 strict-past features; source verification gates scoring."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'

import argparse
import gc
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from catboost import CatBoost, Pool
from fit_stacker import ROOT, HERE, digest, write
from rich_userday_activity import array_hash
from conditional_daily import read, emit
from catboost_userday_residual import save_model, load_model
from verify_stacker import rank_sum_auc
import global_row_context as rows16
import global_row_context15 as rows15

SOURCE = HERE/'runs/rich_augmented_rows_v1'
OUT = HERE/'runs/rich_row_catboost_v1'
PREFLIGHT = HERE/'runs/rich_row_catboost_preflight_v1'
BASES = {15: HERE/'runs/conditional_daily_trees_v1', 16: HERE/'runs/group_temperature_v1'}
ARMS = [{'name': 'depth6_weight5', 'depth': 6}, {'name': 'depth8_weight5', 'depth': 8}]
CHECKS = [600, 1200, 2400]
WEIGHTS = [.025, .05, .1, .2, .35, .5]
PARAMS = {'loss_function': 'Logloss', 'learning_rate': .025, 'l2_leaf_reg': 60.,
          'random_strength': 1., 'bootstrap_type': 'Bernoulli', 'subsample': .8,
          'border_count': 32, 'one_hot_max_size': 16, 'max_ctr_complexity': 1,
          'thread_count': 2, 'random_seed': 151640, 'task_type': 'GPU', 'devices': '0',
          'gpu_ram_part': .6, 'allow_writing_files': False, 'verbose': False}


def selection(day, labels, dates):
    safe = np.asarray(labels, dtype='int32').copy()
    safe[np.asarray(dates) >= 20191100+day] = -1
    module = rows15 if day == 15 else rows16
    chosen, weights = module.selected_training(safe, dates)
    held_rows = np.flatnonzero(np.asarray(dates) == 20191100+day)
    rows = np.sort(np.r_[chosen, held_rows])
    train = np.flatnonzero(np.asarray(dates)[rows] < 20191100+day)
    held = np.flatnonzero(np.asarray(dates)[rows] == 20191100+day)
    np.testing.assert_array_equal(rows[train], chosen)
    assert np.all(np.asarray(dates)[chosen] < 20191100+day)
    assert np.isin(safe[chosen], [0, 1]).all()
    return chosen, weights, held_rows, train, held, safe[chosen]


def frame_from_matrix(matrix, positions, categories):
    values = np.asarray(matrix[positions], dtype='float32')
    frame = pd.DataFrame(values, columns=[f'f{i}' for i in range(values.shape[1])], copy=False)
    for column in categories:
        values = frame.iloc[:, column].to_numpy()
        if not np.array_equal(values, values.astype('int64')):
            raise ValueError('Categorical inputs must be exact integer codes')
        frame[f'f{column}'] = values.astype('int64')
    return frame


def frame_hash(frame):
    value = hashlib.sha256()
    for start in range(0, len(frame), 20000):
        block = np.ascontiguousarray(frame.iloc[start:start+20000].to_numpy(dtype='float32'))
        value.update(memoryview(block).cast('B'))
    return value.hexdigest()


def inputs(day, partition, independent=False, limit=None):
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    chosen, weights, rows, training, held, labels = selection(day, raw.label.to_numpy(), raw.pt_d.to_numpy())
    contract = read(SOURCE/f'input_contract{day}.json')
    assert array_hash(chosen) == contract['training_rows_sha256']
    assert array_hash(labels) == contract['training_labels_sha256']
    assert array_hash(weights) == contract['weights_sha256']
    assert array_hash(rows) == contract['held_rows_sha256']
    directory = SOURCE/(f'independent_day{day}' if independent else f'input_day{day}')
    path = directory/'features.npy'
    assert digest(path) == contract['matrix_sha256']
    matrix = np.load(path, mmap_mode='r')
    assert list(matrix.shape) == contract['shape'] and matrix.shape[1] == 744
    positions = training if partition == 'train' else held
    if limit is not None:
        positions = positions[:limit]
    frame = frame_from_matrix(matrix, positions, contract['categorical_columns'])
    cats = [f'f{i}' for i in contract['categorical_columns']]
    audit = {'shape': list(frame.shape), 'source_matrix_sha256': contract['matrix_sha256'],
             'frame_sha256': frame_hash(frame),
             'training_rows_sha256': array_hash(chosen), 'training_labels_sha256': array_hash(labels),
             'weights_sha256': array_hash(weights), 'held_rows_sha256': array_hash(rows),
             'max_training_date': int(raw.pt_d.to_numpy()[chosen].max()),
             'own_held_targets_fit': False, 'categories': cats}
    del matrix, raw
    gc.collect()
    return frame, labels[:len(positions)], weights[:len(positions)], rows, cats, audit


def preflight():
    assert not PREFLIGHT.exists(), 'Preserve earlier GPU preflights'
    PREFLIGHT.mkdir(parents=True)
    frame, y, weights, _, cats, audit = inputs(15, 'train', limit=100000)
    pool = Pool(frame, label=y, weight=weights*np.where(y == 1, 5., 1.), cat_features=cats)
    model = CatBoost(dict(PARAMS, depth=6, iterations=20))
    model.fit(pool)
    sample = Pool(frame.iloc[:2000], cat_features=cats)
    before = np.asarray(model.predict(sample, prediction_type='RawFormulaVal'))
    save_model(model, PREFLIGHT/'smoke_model.cbm')
    after = load_model(PREFLIGHT/'smoke_model.cbm').predict(sample, prediction_type='RawFormulaVal')
    error = float(np.max(np.abs(before-after)))
    assert error < 1e-10 and audit['max_training_date'] == 20191114
    write(PREFLIGHT/'verification.json', {'status': 'passed', 'audit': audit,
          'iterations': 20, 'device': 'GPU0', 'serialization_maximum_error': error,
          'held_labels_used_for_fitting_or_evaluation': False,
          'full_training_memory_capacity_proven': False, 'source_code_sha256': digest(__file__)})
    emit('rich_catboost_preflight_passed', shape=audit['shape'], serialization_error=error)


def check_sources():
    protocol = read(OUT/'protocol.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    return protocol


def fit():
    assert not OUT.exists(), 'Preserve all prior GPU fits and failures'
    preflight_report = read(PREFLIGHT/'verification.json')
    assert preflight_report['status'] == 'passed'
    assert preflight_report['source_code_sha256'] == digest(__file__)
    parent = read(SOURCE/'protocol.json')
    for path, expected in parent['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in parent['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    paths = [Path(__file__), HERE/'test_rich_row_catboost.py', HERE/'run_rich_row_catboost.ps1',
             HERE/'conditional_daily.py', HERE/'fit_stacker.py', HERE/'rich_userday_activity.py',
             HERE/'verify_stacker.py', HERE/'catboost_userday_residual.py',
             HERE/'global_row_context.py', HERE/'global_row_context15.py',
             PREFLIGHT/'verification.json', SOURCE/'protocol.json']
    for day in [15, 16]:
        assert read(BASES[day]/'verification.json')['status'] == 'passed'
        paths += [SOURCE/f'input_contract{day}.json', SOURCE/f'input_day{day}/features.npy',
                  BASES[day]/f'day{day}_prediction.npy', BASES[day]/f'day{day}_indices.npy',
                  BASES[day]/'verification.json']
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json', {'dates': [20191115, 20191116], 'arms': ARMS, 'params': PARAMS,
        'checkpoints': CHECKS, 'blend_weights': WEIGHTS, 'positive_weight': 5.,
        'source_features': '744 columns from frozen rich row matrices',
        'training': 'All strictly earlier positives, one-third earlier negatives, inverse inclusion weights',
        'source_independent_verification_required_before_scoring': True,
        'source_verification_deferred_until_parent_finishes': True,
        'selection': 'Both-date model checkpoints lock before held-label scoring; reused development dates',
        'own_held_ad_labels_fit': False, 'inferred17_labels_read': False,
        'raw_datasets_sha256': parent['raw_datasets_sha256'],
        'sources_sha256': {str(p): digest(p) for p in paths}})
    locks = []
    for day in [15, 16]:
        frame, y, weights, rows, cats, audit = inputs(day, 'train')
        write(OUT/f'training_contract{day}.json', audit)
        np.save(OUT/f'day{day}_indices.npy', rows)
        emit('rich_catboost_training_ready', day=day, shape=audit['shape'])
        for arm in ARMS:
            pool = Pool(frame, label=y, weight=weights*np.where(y == 1, 5., 1.), cat_features=cats)
            model = CatBoost(dict(PARAMS, depth=arm['depth'], iterations=max(CHECKS)))
            model.fit(pool)
            del pool
            for trees in CHECKS:
                saved = model.copy()
                saved.shrink(trees)
                name = f'day{day}_{arm["name"]}_trees{trees}'
                path = OUT/(name+'.cbm')
                save_model(saved, path)
                locks.append(dict(arm, day=day, trees=trees, name=name, model_sha256=digest(path)))
                del saved
            emit('rich_catboost_arm_complete', day=day, name=arm['name'])
            del model
            gc.collect()
        del frame
        gc.collect()
    write(OUT/'models_locked.json', {'models': locks, 'protocol_sha256': digest(OUT/'protocol.json')})


def parent_gate(verifying):
    parent_file = SOURCE/'verification.json'
    report = read(parent_file)
    assert report['status'] == 'passed'
    sources = {str(parent_file): digest(parent_file)}
    for day in [15, 16]:
        path = SOURCE/f'independent_day{day}/features.npy'
        expected = read(SOURCE/f'input_contract{day}.json')['matrix_sha256']
        assert digest(path) == expected
        sources[str(path)] = expected
        assert report['dates'][str(20191100+day)]['maximum_replay_error'] < 2e-7
    record = {'status': 'passed_before_new_held_label_scoring', 'sources_sha256': sources}
    path = OUT/'parent_verification_binding.json'
    if verifying:
        assert record == read(path)
    else:
        assert not path.exists()
        write(path, record)


def evaluate(verifying):
    check_sources()
    locks = read(OUT/'models_locked.json')
    assert locks['protocol_sha256'] == digest(OUT/'protocol.json')
    parent_gate(verifying)
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    results = {}
    for day in [15, 16]:
        frame, _, _, rows, cats, audit = inputs(day, 'held', independent=verifying)
        np.testing.assert_array_equal(rows, np.load(OUT/f'day{day}_indices.npy'))
        np.testing.assert_array_equal(rows, np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day))
        np.testing.assert_array_equal(rows, np.load(BASES[day]/f'day{day}_indices.npy'))
        y = raw.label.to_numpy()[rows]
        assert len(y) == {15: 424780, 16: 421983}[day] and int(y.sum()) == {15: 3523, 16: 3724}[day]
        contract_file = OUT/f'held_contract{day}.json'
        if verifying:
            assert audit == read(contract_file)
        else:
            write(contract_file, audit)
        pool = Pool(frame, cat_features=cats)
        base = np.load(BASES[day]/f'day{day}_prediction.npy')
        a = logit(base.clip(1e-6, 1-1e-6))
        best = {'name': 'base', 'recipe': 'base', 'auc': rank_sum_auc(y, base)}
        winner = base.copy()
        records, predictions, last, replay_error = [best], {}, [], 0.
        for arm in locks['models']:
            if arm['day'] != day:
                continue
            path = OUT/(arm['name']+'.cbm')
            assert digest(path) == arm['model_sha256']
            z = load_model(path).predict(pool, prediction_type='RawFormulaVal')-np.log(5.)
            p = expit(z)
            prediction_file = OUT/(arm['name']+'_prediction.npy')
            if verifying:
                error = float(np.max(np.abs(p-np.load(prediction_file))))
                assert error < 2e-7
                replay_error = max(replay_error, error)
            else:
                np.save(prediction_file, p)
            predictions[arm['name']] = p
            if arm['trees'] == max(CHECKS):
                last.append(p)
        predictions['mean_last_two'] = np.mean(last, axis=0)
        for name, p in predictions.items():
            b = logit(p.clip(1e-6, 1-1e-6))
            b = (b-b.mean())*a.std()/max(b.std(), 1e-6)+a.mean()
            candidates = [('single', None, p)]
            for weight in WEIGHTS:
                candidates += [('normalized_logit_blend', weight, expit((1-weight)*a+weight*b)),
                               ('probability_blend', weight, (1-weight)*base+weight*p)]
            for recipe, weight, q in candidates:
                record = {'name': name, 'recipe': recipe, 'weight': weight, 'auc': rank_sum_auc(y, q)}
                records.append(record)
                if record['auc'] > best['auc']:
                    best, winner = record, q.copy()
        if verifying:
            error = float(np.max(np.abs(winner-np.load(OUT/f'day{day}_prediction.npy'))))
            assert error < 2e-7
            replay_error = max(replay_error, error)
        else:
            np.save(OUT/f'day{day}_prediction.npy', winner)
        results[str(20191100+day)] = {'auc': best['auc'], 'best': best, 'records': records,
            'rows': len(y), 'clicks': int(y.sum()), 'maximum_replay_error': replay_error,
            'evaluation_day_labels_fit': False, 'independent_source_matrix_used': verifying,
            'target_met': best['auc'] >= {15: .815, 16: .82}[day]}
        emit('rich_catboost_scored', day=day, best=best, verifying=verifying)
        del frame, pool
        gc.collect()
    check_sources()
    write(OUT/('verification.json' if verifying else 'result.json'), {
        'status': 'passed' if verifying else 'evaluated', 'dates': results,
        'original_datasets_unchanged': True, 'both_targets_met': all(r['target_met'] for r in results.values())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['preflight', 'fit', 'evaluate', 'verify'], required=True)
    phase = parser.parse_args().phase
    if phase == 'preflight':
        preflight()
    elif phase == 'fit':
        fit()
    else:
        evaluate(phase == 'verify')
