"""Add all original row numerics and ordered click-history tokens to row trees."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import gc
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from catboost import CatBoost, Pool
from fit_stacker import ROOT, HERE, digest, write
from rich_userday_activity import array_hash
from catboost_userday_residual import save_model, load_model
from catboost_global_rows import predict, recipes
from verify_stacker import rank_sum_auc
sys.path.insert(0, str(ROOT/'analysis/strong_interest_control_20261031'))
from features import prepare
import global_row_context as rows16
import global_row_context15 as rows15

SOURCES = {15: HERE/'runs/global_row_context15_v1', 16: HERE/'runs/global_row_context_v1'}
BASES = {15: HERE/'runs/rank_finetune_v2_day15/prediction.npy',
         16: HERE/'runs/compose_rich_response_v1/day16_prediction.npy'}
ARMS = [{'name': 'depth6_weight5', 'depth': 6, 'positive_weight': 5.},
        {'name': 'depth8_weight5', 'depth': 8, 'positive_weight': 5.}]
CHECKS = [400, 800, 1200]
PARAMS = {'loss_function': 'Logloss', 'learning_rate': .035, 'l2_leaf_reg': 30.,
          'random_strength': .5, 'bootstrap_type': 'Bernoulli', 'subsample': .8,
          'border_count': 32, 'one_hot_max_size': 16, 'max_ctr_complexity': 1,
          'thread_count': 2, 'random_seed': 151633, 'task_type': 'GPU', 'devices': '0',
          'gpu_ram_part': .65, 'allow_writing_files': False, 'verbose': False}


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def build(day, directory):
    directory.mkdir(parents=True, exist_ok=False)
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    labels, dates = raw.label.to_numpy().copy(), raw.pt_d.to_numpy()
    module = rows15 if day == 15 else rows16
    chosen, weights = module.selected_training(labels, dates)
    held_rows = np.flatnonzero(dates == 20191100+day)
    rows = np.sort(np.r_[chosen, held_rows])
    train = np.flatnonzero(dates[rows] < 20191100+day)
    held = np.flatnonzero(dates[rows] == 20191100+day)
    labels[dates >= 20191100+day] = -1
    source = SOURCES[day]
    source_contract = json.loads((source/'input_contract.json').read_text(encoding='utf8'))
    matrix_path = source/'independent_input/features.npy'
    assert digest(matrix_path) == source_contract['features_sha256']
    assert array_hash(rows) == source_contract['rows_sha256']
    assert array_hash(chosen) == source_contract['train_rows_sha256']
    assert array_hash(labels[chosen]) == source_contract['training_labels_sha256']
    assert array_hash(weights) == source_contract['weights_sha256']
    assert source_contract['max_fit_date'] < 20191100+day
    core = prepare(20191100+day, directory)
    # Legacy source returns uint8; mask through signed storage immediately.
    core['label'] = core['label'].astype('int16')
    core['label'][core['valid']] = -1
    assert np.all(core['label'][core['valid']] == -1)
    positions = pd.Index(core['global_rows']).get_indexer(rows)
    assert np.all(positions >= 0)
    np.testing.assert_array_equal(core['global_rows'][positions], rows)
    np.testing.assert_array_equal(core['label'][positions[train]], labels[rows[train]])
    np.testing.assert_array_equal(core['global_rows'][core['valid']], held_rows)
    existing = np.load(matrix_path, mmap_mode='r')
    assert existing.shape == (len(rows), len(source_contract['feature_names']))
    nbase = existing.shape[1]
    numeric = core['numeric'][positions]
    # The first three sequences are supplied click histories for task/advertiser/app.
    sequence_names = core['meta']['sequence_fields'][:3]
    tokens = core['sequences'][positions, :3, :].reshape(len(rows), -1)
    values = np.empty((len(rows), nbase+numeric.shape[1]+tokens.shape[1]), dtype='float32')
    values[:, :nbase] = existing
    values[:, nbase:nbase+numeric.shape[1]] = numeric
    values[:, nbase+numeric.shape[1]:] = tokens
    np.testing.assert_array_equal(values[:, nbase+numeric.shape[1]:].astype('int32'), tokens)
    names = source_contract['feature_names']+['original__'+n for n in core['feature_manifest']['numeric_fields']]
    names += [f'ordered_click_history__{name}__{i}' for name in sequence_names for i in range(tokens.shape[1]//3)]
    assert len(names) == values.shape[1] and np.isfinite(values).all()
    numerical_hash = array_hash(values)
    metadata = core['feature_manifest']['metadata_sha256']
    del core, existing, numeric, tokens
    gc.collect()
    frame = pd.DataFrame(values, columns=[f'f{i}' for i in range(values.shape[1])])
    categories = list(range(len(module.CAT)))+list(range(values.shape[1]-15, values.shape[1]))
    for col in categories:
        frame[f'f{col}'] = frame.iloc[:, col].to_numpy(dtype='int64')
    contract = {'feature_names': names, 'feature_shape': list(frame.shape), 'matrix_sha256': numerical_hash,
                'training_rows_sha256': array_hash(rows[train]), 'training_labels_sha256': array_hash(labels[rows[train]]),
                'weights_sha256': array_hash(weights), 'held_rows_sha256': array_hash(held_rows),
                'training_max_date': int(dates[rows[train]].max()), 'held_label_values': [-1],
                'original_feature_metadata_sha256': metadata, 'categorical_columns': categories}
    emit('augmented_rows_ready', day=day, shape=list(frame.shape), training_rows=len(train))
    return frame, train, held, labels[rows[train]], weights, held_rows, [f'f{i}' for i in categories], contract


def fit(day):
    out = HERE/f'runs/augmented_rows_v1_day{day}'
    assert not out.exists(), 'Preserving augmented row experiment'
    for directory in [SOURCES[day], BASES[day].parent]:
        assert json.loads((directory/'verification.json').read_text(encoding='utf8'))['status'] == 'passed'
    expected = json.loads((HERE/'runs/weighted_v1_day16/protocol.json').read_text(encoding='utf8'))['raw_datasets_sha256']
    for name, h in expected.items():
        assert digest(ROOT/'2026ml'/name) == h, name
    out.mkdir(parents=True)
    frame, train, held, y, weights, rows, categories, contract = build(day, out/'input')
    paths = [Path(__file__), HERE/'global_row_context.py', HERE/'global_row_context15.py',
             HERE/'catboost_global_rows.py', HERE/'catboost_userday_residual.py',
             HERE/'fit_stacker.py', HERE/'verify_stacker.py',
             SOURCES[day]/'input_contract.json', SOURCES[day]/'verification.json',
             BASES[day], BASES[day].parent/'verification.json']
    paths += [Path(path) for path in contract['original_feature_metadata_sha256']]
    write(out/'protocol.json', {'day': 20191100+day, 'arms': ARMS, 'params': PARAMS,
          'checkpoints': CHECKS, 'sources_sha256': {str(path): digest(path) for path in paths},
          'raw_datasets_sha256': expected, f'evaluation{day}_labels_fit': False,
          'selection': 'Fixed endpoints and original declared row blend recipes; reused full development date',
          'features': 'Verified strict-past row cache + all166 original row numerics +15 ordered click-history tokens',
          'training': 'Strictly earlier positives and one-third earlier negatives with inverse inclusion weights',
          'inferred17_labels_read': False})
    write(out/'input_contract.json', contract)
    np.save(out/'valid_indices.npy', rows)
    locks = []
    for arm in ARMS:
        pool = Pool(frame.iloc[train], label=y, weight=weights*np.where(y == 1, arm['positive_weight'], 1.),
                    cat_features=categories)
        model = CatBoost(dict(PARAMS, depth=arm['depth'], iterations=max(CHECKS)))
        model.fit(pool)
        del pool
        for trees in CHECKS:
            name = arm['name']+f'_trees{trees}'
            saved = model.copy()
            saved.shrink(trees)
            path = out/(name+'.cbm')
            save_model(saved, path)
            locks.append(dict(arm, name=name, trees=trees, model_sha256=digest(path)))
            del saved
        emit('augmented_rows_arm_complete', day=day, arm=arm['name'])
        del model
        gc.collect()
    write(out/'models_locked.json', {'models': locks, 'protocol_sha256': digest(out/'protocol.json')})
    evaluate(day, False, frame, held, rows, categories)


def evaluate(day, verifying, frame=None, held=None, rows=None, categories=None):
    out = HERE/f'runs/augmented_rows_v1_day{day}'
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf8'))
    locks = json.loads((out/'models_locked.json').read_text(encoding='utf8'))
    assert locks['protocol_sha256'] == digest(out/'protocol.json')
    for path, h in protocol['sources_sha256'].items():
        assert digest(path) == h, path
    for name, h in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == h, name
    if verifying:
        frame, _, held, _, _, rows, categories, contract = build(day, out/'independent_input')
        assert contract == json.loads((out/'input_contract.json').read_text(encoding='utf8'))
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    np.testing.assert_array_equal(rows, np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day))
    np.testing.assert_array_equal(rows, np.load(out/'valid_indices.npy'))
    y = raw.label.to_numpy()[rows]
    assert len(y) == {15: 424780, 16: 421983}[day] and int(y.sum()) == {15: 3523, 16: 3724}[day]
    base = np.load(BASES[day])
    best = {'name': 'base', 'recipe': 'base', 'auc': rank_sum_auc(y, base)}
    best_p = base.copy()
    records = []
    maximum_error = 0.
    pool = Pool(frame.iloc[held], cat_features=categories)
    for model in locks['models']:
        path = out/(model['name']+'.cbm')
        assert digest(path) == model['model_sha256']
        p = predict(load_model(path), pool, model['positive_weight'])
        if verifying:
            error = float(np.max(np.abs(p-np.load(out/(model['name']+'_prediction.npy')))))
            assert error < 2e-7, error
            maximum_error = max(maximum_error, error)
        else:
            np.save(out/(model['name']+'_prediction.npy'), p)
        for recipe, values in recipes(p, base):
            record = {'name': model['name'], 'recipe': recipe, 'auc': rank_sum_auc(y, values)}
            records.append(record)
            if record['auc'] > best['auc']:
                best = record
                best_p = values.copy()
    if verifying:
        error = float(np.max(np.abs(best_p-np.load(out/'prediction.npy'))))
        assert error < 2e-7
        maximum_error = max(maximum_error, error)
    else:
        np.save(out/'prediction.npy', best_p)
    report = {'status': 'passed' if verifying else 'evaluated', 'auc': best['auc'], 'best': best,
              'records': records, 'rows': len(y), 'clicks': int(y.sum()), 'maximum_replay_error': maximum_error,
              f'evaluation{day}_labels_fit': False, 'development_day_reused': True,
              'original_datasets_unchanged': True, 'independent_feature_matrix_used': verifying,
              'target_met': best['auc'] >= {15: .815, 16: .82}[day]}
    write(out/('verification.json' if verifying else 'result.json'), report)
    emit('augmented_rows_result', day=day, status=report['status'], best=best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=int, choices=[15, 16], required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    evaluate(args.day, True) if args.verify else fit(args.day)
