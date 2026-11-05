"""Regularized dense-category rows and context-focused transfer, strict-past fits."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from fit_stacker import ROOT, HERE, write, digest
from rich_userday_activity import array_hash, lgb
from verify_stacker import rank_sum_auc
import global_row_context as rows16
import global_row_context15 as rows15

SOURCE = HERE/'runs/rich_augmented_rows_v1'
OUT = HERE/'runs/dense_context_rows_v1'
QUEUE = HERE/'dense_context_rows_queued_v1.json'
CHECKS = [600, 1400, 2600]
WEIGHTS = [.025, .05, .1, .2, .35, .5]
ARMS = [
    {'name': 'dense_regularized', 'focus': False, 'num_leaves': 31,
     'min_data_in_leaf': 700, 'lambda_l2': 60., 'cat_l2': 200.,
     'cat_smooth': 250., 'min_data_per_group': 300},
    {'name': 'context_numeric', 'focus': True, 'num_leaves': 31,
     'min_data_in_leaf': 350, 'lambda_l2': 40., 'cat_l2': 60.,
     'cat_smooth': 100., 'min_data_per_group': 120},
]
PARAMS = {'objective': 'binary', 'learning_rate': .025, 'max_bin': 63,
    'max_cat_threshold': 24, 'max_cat_to_onehot': 4, 'feature_fraction': .9,
    'bagging_fraction': .9, 'bagging_freq': 1, 'scale_pos_weight': 5.,
    'num_threads': 4, 'seed': 151637, 'verbosity': -1, 'deterministic': True,
    'force_col_wise': True, 'feature_pre_filter': False}


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def dense_codes(values, vocabulary):
    """Only training values define the dictionary; unseen/negative values are missing."""
    values = np.asarray(values, dtype='int64')
    vocabulary = np.asarray(vocabulary, dtype='int64')
    result = np.full(values.shape, -1, dtype='int32')
    if len(vocabulary):
        positions = np.searchsorted(vocabulary, values)
        inside = (positions < len(vocabulary)) & (values >= 0)
        good = np.flatnonzero(inside)
        good = good[vocabulary[positions[good]] == values[good]]
        result[good] = positions[good]
    return result


def schema(x, training, names, categorical, focused):
    vocabularies = {column: np.unique(np.asarray(x[training, column], dtype='int64'))
                    for column in categorical}
    vocabularies = {column: values[values >= 0] for column, values in vocabularies.items()}
    excluded = set()
    if focused:
        excluded.update(i for i, name in enumerate(names) if name.startswith('original__target_stat__'))
        excluded.update(column for column, values in vocabularies.items() if len(values) > 500)
    columns = [i for i in range(x.shape[1]) if i not in excluded]
    vocabularies = {columns.index(i): values for i, values in vocabularies.items() if i not in excluded}
    return columns, vocabularies


def input_data(day, arm, verifying=False):
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    dates = raw.pt_d.to_numpy(copy=True)
    # Training code never retains the held or later ad outcomes.
    labels = raw.label.to_numpy(copy=True)
    labels[dates >= 20191100+day] = -1
    del raw
    module = rows15 if day == 15 else rows16
    chosen, weights = module.selected_training(labels, dates)
    held_rows = np.flatnonzero(dates == 20191100+day)
    rows = np.sort(np.r_[chosen, held_rows])
    training = np.flatnonzero(dates[rows] < 20191100+day)
    held = np.flatnonzero(dates[rows] == 20191100+day)
    original = read(SOURCE/f'input_contract{day}.json')
    np.testing.assert_array_equal(rows[training], chosen)
    assert array_hash(chosen) == original['training_rows_sha256']
    assert array_hash(labels[chosen]) == original['training_labels_sha256']
    assert array_hash(weights) == original['weights_sha256']
    assert array_hash(held_rows) == original['held_rows_sha256']
    directory = SOURCE/(f'independent_day{day}' if verifying else f'input_day{day}')
    assert digest(directory/'features.npy') == original['matrix_sha256']
    x = np.load(directory/'features.npy', mmap_mode='r')
    columns, vocabularies = schema(x, training, original['feature_names'],
                                   original['categorical_columns'], arm['focus'])
    destination = OUT/('independent' if verifying else 'input')/f'day{day}_{arm["name"]}'
    destination.mkdir(parents=True, exist_ok=False)
    z = np.lib.format.open_memmap(destination/'features.npy', mode='w+', dtype='float32',
                                  shape=(len(rows), len(columns)))
    for start in range(0, len(rows), 20000):
        stop = min(start+20000, len(rows))
        block = np.array(x[start:stop, columns], dtype='float32', copy=True)
        for column, vocabulary in vocabularies.items():
            block[:, column] = dense_codes(block[:, column], vocabulary)
        assert np.isfinite(block).all()
        z[start:stop] = block
    z.flush()
    np.savez_compressed(destination/'category_maps.npz', **{f'f{i}': v for i, v in vocabularies.items()})
    contract = {'shape': list(z.shape), 'matrix_sha256': digest(destination/'features.npy'),
        'source_matrix_sha256': original['matrix_sha256'], 'columns': columns,
        'feature_names': [original['feature_names'][i] for i in columns],
        'categorical_columns': sorted(vocabularies),
        'vocabularies_sha256': {f'f{i}': array_hash(v) for i, v in vocabularies.items()},
        'max_training_date': int(dates[chosen].max()), 'training_rows_sha256': array_hash(chosen),
        'training_labels_sha256': array_hash(labels[chosen]), 'weights_sha256': array_hash(weights),
        'held_rows_sha256': array_hash(held_rows), 'own_held_ad_labels_fit': False,
        'unknown_categories_map_to_missing': True, 'dictionaries_use_training_inputs_only': True}
    expected = OUT/f'contract_day{day}_{arm["name"]}.json'
    if verifying:
        assert contract == read(expected)
    else:
        write(expected, contract)
    emit('dense_context_input_ready', day=day, arm=arm['name'], shape=list(z.shape), verifying=verifying)
    return z, training, held, labels[chosen], weights, held_rows, contract


def check_sources():
    protocol = read(OUT/'protocol.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    return protocol


def fit():
    assert not OUT.exists(), 'Preserve all earlier and failed experiments'
    queued = read(QUEUE)
    assert queued['arms'] == ARMS and queued['params'] == PARAMS
    assert queued['checkpoints'] == CHECKS and queued['blend_weights'] == WEIGHTS
    for path, expected in queued['sources_sha256'].items():
        assert digest(path) == expected, path
    verified = read(SOURCE/'verification.json')
    assert verified['status'] == 'passed'
    previous = read(SOURCE/'protocol.json')
    for path, expected in previous['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in previous['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    paths = [Path(__file__), QUEUE, HERE/'test_dense_context_rows.py', SOURCE/'verification.json',
             SOURCE/'protocol.json', HERE/'global_row_context.py', HERE/'global_row_context15.py',
             HERE/'fit_stacker.py', HERE/'rich_userday_activity.py', HERE/'verify_stacker.py']
    for day in [15, 16]:
        paths += [SOURCE/f'input_contract{day}.json', SOURCE/f'day{day}_prediction.npy',
            SOURCE/f'input_day{day}/features.npy', SOURCE/f'independent_day{day}/features.npy']
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json', {'dates': [20191115, 20191116], 'arms': ARMS,
        'params': PARAMS, 'checkpoints': CHECKS, 'blend_weights': WEIGHTS,
        'category_encoding': 'Dense dictionaries learned solely on prior-ad training inputs; unseen categories -> -1',
        'context_mask': 'Drop categorical columns with more than500 training values and36 original grouped-OOF target-stat columns',
        'sampling': 'All earlier positives and fixed one-third earlier negatives; inverse inclusion weights',
        'selection': 'All both-date model checkpoints lock before held ad label scoring; development dates reused',
        'inferred17_labels_read': False, 'held_ad_outcomes_fit': False,
        'raw_datasets_sha256': previous['raw_datasets_sha256'],
        'sources_sha256': {str(p): digest(p) for p in dict.fromkeys(paths)}})
    locks = []
    for day in [15, 16]:
        for arm in ARMS:
            x, training, held, y, weights, held_rows, contract = input_data(day, arm)
            dataset = lgb.Dataset(np.asarray(x[training]), label=y, weight=weights,
                categorical_feature=contract['categorical_columns'], params=PARAMS, free_raw_data=True)
            params = dict(PARAMS, **{k: v for k, v in arm.items() if k not in ['name', 'focus']})
            model = lgb.train(params, dataset, num_boost_round=max(CHECKS))
            for trees in CHECKS:
                name = f'day{day}_{arm["name"]}_trees{trees}'
                path = OUT/(name+'.txt')
                path.write_text(model.model_to_string(num_iteration=trees), encoding='utf8')
                locks.append(dict(arm, day=day, trees=trees, model_name=name, model_sha256=digest(path)))
            emit('dense_context_arm_complete', day=day, arm=arm['name'])
            del x, dataset, model
            gc.collect()
    write(OUT/'models_locked.json', {'models': locks, 'protocol_sha256': digest(OUT/'protocol.json')})
    evaluate(False)


def evaluate(verifying):
    check_sources()
    locks = read(OUT/'models_locked.json')
    assert locks['protocol_sha256'] == digest(OUT/'protocol.json')
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    results = {}
    for day in [15, 16]:
        held_rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
        y = raw.label.to_numpy()[held_rows]
        assert len(y) == {15: 424780, 16: 421983}[day] and int(y.sum()) == {15: 3523, 16: 3724}[day]
        base = np.load(SOURCE/f'day{day}_prediction.npy')
        a = logit(base.clip(1e-6, 1-1e-6))
        best = {'name': 'base', 'recipe': 'base', 'auc': rank_sum_auc(y, base)}
        winner = base.copy()
        records, forecasts, last = [best], {}, []
        for arm in ARMS:
            if verifying:
                x, _, held, _, _, rows, contract = input_data(day, arm, True)
            else:
                x = np.load(OUT/f'input/day{day}_{arm["name"]}/features.npy', mmap_mode='r')
                module = rows15 if day == 15 else rows16
                labels = raw.label.to_numpy(copy=True)
                labels[raw.pt_d.to_numpy() >= 20191100+day] = -1
                chosen, _ = module.selected_training(labels, raw.pt_d.to_numpy())
                rows = np.sort(np.r_[chosen, held_rows])
                held = np.flatnonzero(raw.pt_d.to_numpy()[rows] == 20191100+day)
                rows = rows[held]
            np.testing.assert_array_equal(rows, held_rows)
            for record in locks['models']:
                if record['day'] != day or record['name'] != arm['name']:
                    continue
                path = OUT/(record['model_name']+'.txt')
                assert digest(path) == record['model_sha256']
                model = lgb.Booster(model_str=path.read_text(encoding='utf8'))
                z = model.predict(x[held], raw_score=True, num_threads=4)-np.log(5.)
                p = expit(z)
                item = {'name': record['model_name'], 'recipe': 'single', 'auc': rank_sum_auc(y, p)}
                records.append(item)
                if item['auc'] > best['auc']:
                    best, winner = item, p.copy()
                z = (z-z.mean())*a.std()/max(z.std(), 1e-6)+a.mean()
                forecasts[record['model_name']] = z
                if record['trees'] == max(CHECKS):
                    last.append(z)
            del x
            gc.collect()
        assert len(last) == len(ARMS)
        forecasts['mean_last_two'] = np.mean(last, axis=0)
        for name, z in forecasts.items():
            for weight in WEIGHTS:
                p = expit((1-weight)*a+weight*z)
                item = {'name': name, 'recipe': 'normalized_logit_blend', 'weight': weight,
                        'auc': rank_sum_auc(y, p)}
                records.append(item)
                if item['auc'] > best['auc']:
                    best, winner = item, p.copy()
        error = 0.
        if verifying:
            error = float(np.max(np.abs(winner-np.load(OUT/f'day{day}_prediction.npy'))))
            assert error < 1e-12
        else:
            np.save(OUT/f'day{day}_prediction.npy', winner)
            np.save(OUT/f'day{day}_indices.npy', held_rows)
        results[str(20191100+day)] = {'auc': best['auc'], 'best': best, 'records': records,
            'rows': len(y), 'clicks': int(y.sum()), 'maximum_replay_error': error,
            'evaluation_day_labels_fit': False, 'own_held_day_ad_labels_fit': False,
            'target_met': best['auc'] >= {15: .815, 16: .82}[day]}
        emit('dense_context_scored', day=day, best=best, verifying=verifying)
    report = {'status': 'passed' if verifying else 'evaluated', 'dates': results,
        'original_datasets_unchanged': True,
        'both_targets_met': all(r['target_met'] for r in results.values())}
    write(OUT/('verification.json' if verifying else 'result.json'), report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    evaluate(True) if parser.parse_args().verify else fit()
