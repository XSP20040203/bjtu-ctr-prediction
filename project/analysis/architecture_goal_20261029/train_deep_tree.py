"""Bounded CatBoost interaction control on the existing architecture input cache.

Adapted from daily_interest_din/train_chrono_tree.py. Uses complete future-day
validation, unchanged categorical IDs, and 341 of the 500 cached numeric inputs.
The 159 multiscale statistics are excluded before allocating feature columns.
No cache is rebuilt, no scaler is fitted, and final-test data is never opened.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key, '3')

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = ROOT / 'analysis/nextday_goal_20261029/duplicate_multiplicity'
BLEND_WEIGHTS = [.05, .1, .2, .3]
THREADS = 3


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scores(y, probability):
    if probability.shape != y.shape or not np.isfinite(probability).all():
        raise AssertionError('Predictions must be finite and label-aligned')
    if np.any((probability < 0) | (probability > 1)):
        raise AssertionError('Probabilities outside [0, 1]')
    return {'auc': float(roc_auc_score(y, probability)),
            'logloss': float(log_loss(y, probability, labels=[0, 1])),
            'average_precision': float(average_precision_score(y, probability)),
            'mean_probability': float(probability.mean())}


def numeric_selection(input_meta, meta):
    names = input_meta['numeric_fields']
    if len(names) != 500 or len(set(names)) != 500:
        raise AssertionError('Expected architecture schema with 500 unique numeric inputs')
    if len(meta['numeric_fields']) != 75 or names[:75] != meta['numeric_fields']:
        raise AssertionError('Original 75 numeric fields differ from architecture cache')
    blocks = [(75, 118, ('day_exposure__',)), (118, 166, ('snapshot_',)),
              (166, 325, ('past_multiscale__',)),
              (325, 420, ('history_input__', 'history_metadata__')),
              (420, 500, ('source_context__',))]
    for start, stop, prefixes in blocks:
        if not all(name.startswith(prefixes) for name in names[start:stop]):
            raise AssertionError('Architecture numeric block order changed')
    selected = np.r_[0:166, 325:500]
    if len(selected) != 341:
        raise AssertionError('Expected 341 retained numeric inputs')
    return selected


def make_schema(meta, input_meta, selected_numeric):
    if len(meta['categorical_fields']) != 27 or len(meta['sequence_fields']) != 12:
        raise AssertionError('Expected all 27 raw categories and 12 sequences')
    categorical = []
    for source, fields in [('categories', meta['categorical_fields']),
                           ('sequences', meta['sequence_fields'])]:
        for column, field in enumerate(fields):
            group = meta['group_for_field'][field]
            positions = [None] if source == 'categories' else range(5)
            for position in positions:
                categorical.append({'name': field if position is None else f'{field}_token{position}',
                                    'source': source, 'source_column': column,
                                    'sequence_position': position, 'vocabulary_group': group,
                                    'vocabulary_size': int(meta['vocabulary_sizes'][group])})
    numeric = [{'name': input_meta['numeric_fields'][int(column)],
                'source_column': int(column), 'dtype': 'float32'} for column in selected_numeric]
    names = [entry['name'] for entry in categorical + numeric]
    if len(categorical) != 87 or len(names) != 428 or len(set(names)) != len(names):
        raise AssertionError('Expected unique schema with 87 categorical + 341 numeric fields')
    return {'feature_names': names, 'categorical': categorical, 'numeric': numeric,
            'categorical_indices': list(range(87)),
            'categorical_representation': 'pandas.Categorical with integer category values, unordered',
            'categorical_id_rule': 'Original cached integer values are passed to CatBoost; pandas codes are never used',
            'categorical_domain': 'range(vocabulary_sizes[group]); same metadata domain in fit and validation',
            'numeric_rule': 'Unchanged architecture float32 values; outer-train standardized, clipped [-8, 8]',
            'excluded_numeric_fields': input_meta['numeric_fields'][166:325],
            'numeric_block_counts': {'original': 75, 'exposure': 43, 'snapshot': 48,
                                     'history': 95, 'source_context': 80}}


def make_pool(arrays, numeric, global_rows, local_rows, labels, schema, *, borders=None):
    """Build one split at a time; never copy the full 500-column matrix."""
    columns = {}
    for entry in schema['categorical']:
        source = arrays[entry['source']]
        if entry['sequence_position'] is None:
            values = source[global_rows, entry['source_column']]
        else:
            values = source[global_rows, entry['source_column'], entry['sequence_position']]
        size = entry['vocabulary_size']
        if values.dtype.kind not in 'iu' or np.any(values < 0) or np.any(values >= size):
            raise AssertionError('Invalid cached categorical IDs: ' + entry['name'])
        # Explicit domains preserve actual integer IDs across the two dataframes.
        columns[entry['name']] = pd.Categorical(values, categories=np.arange(size, dtype='int32'))
    for entry in schema['numeric']:
        values = np.asarray(numeric[local_rows, entry['source_column']], dtype='float32')
        if not np.isfinite(values).all() or np.any(np.abs(values) > 8.00001):
            raise AssertionError('Architecture numeric values must be finite and clipped: ' + entry['name'])
        columns[entry['name']] = values
    data = pd.DataFrame(columns, copy=False)
    del columns, values
    if list(data.columns) != schema['feature_names']:
        raise AssertionError('Feature dataframe order changed')
    for entry in schema['categorical']:
        if not isinstance(data[entry['name']].dtype, pd.CategoricalDtype):
            raise AssertionError('Categorical metadata was lost')
    print('DEEP_TREE_POOL', len(data), len(data.columns),
          'dataframe_bytes', int(data.memory_usage(index=True, deep=True).sum()), flush=True)
    pool = Pool(data, labels, cat_features=schema['categorical_indices'], thread_count=THREADS)
    del data
    gc.collect()
    # Quantize and discard the raw split before constructing the next one.
    # Validation uses only borders learned from the actual fit rows.
    options = {'task_type': 'GPU', 'nan_mode': 'Forbidden'}
    if borders is None:
        options.update(border_count=64, random_seed=42)
    else:
        options['input_borders'] = str(borders)
    pool.quantize(**options)
    gc.collect()
    if pool.get_feature_names() != schema['feature_names'] or pool.get_cat_feature_indices() != schema['categorical_indices']:
        raise AssertionError('CatBoost Pool schema differs from saved categorical metadata')
    return pool


def main(args):
    if args.recent_days < 0 or not 1 <= args.depth <= 10:
        raise ValueError('recent-days must be nonnegative and depth must be in [1, 10]')
    if not 0 < args.gpu_ram_part <= 1:
        raise ValueError('gpu-ram-part must be in (0, 1]')
    if not args.tag or Path(args.tag).name != args.tag or args.tag in {'.', '..'}:
        raise ValueError('tag must be a single result-directory name')
    out = HERE / 'results' / args.tag
    if out.exists():
        raise FileExistsError('Preserving existing deep-tree run: ' + str(out))
    cache = ROOT / ('analysis/user_personalization/cache_day15_stats' if args.day == 20191115
                    else 'analysis/auc085/cache_rich_stats_time16')
    architecture = HERE / 'cache' / f'day{args.day % 100}_oof'
    meta, input_meta = read(cache / 'manifest.json'), read(architecture / 'manifest.json')
    if meta['validation_day'] != args.day or input_meta['day'] != args.day or input_meta['statistics'] != 'oof':
        raise AssertionError('Input validation day or grouped-OOF protocol differs')
    if input_meta.get('advertisement_validation_labels_used') is not False:
        raise AssertionError('Architecture cache must exclude validation labels')
    if input_meta.get('scaler_fit_scope') != 'Only outer-training rows' or input_meta.get('clip') != [-8, 8]:
        raise AssertionError('Architecture numeric scaling provenance differs')
    arrays = {name: np.load(cache / f'{name}.npy', mmap_mode='r') for name in
              ['categories', 'sequences', 'date', 'label', 'train_indices', 'valid_indices']}
    outer_train, valid = arrays['train_indices'], arrays['valid_indices']
    dates = arrays['date']
    np.testing.assert_array_equal(outer_train, np.flatnonzero(dates < args.day))
    np.testing.assert_array_equal(valid, np.flatnonzero(dates == args.day))
    np.testing.assert_array_equal(outer_train, np.load(architecture / 'train_indices.npy'))
    np.testing.assert_array_equal(valid, np.load(architecture / 'valid_indices.npy'))
    if input_meta['train_rows'] != len(outer_train) or input_meta['valid_rows'] != len(valid):
        raise AssertionError('Architecture cache row counts differ')
    if arrays['categories'].shape != (len(dates), 27) or arrays['sequences'].shape != (len(dates), 12, 5):
        raise AssertionError('Raw category/sequence array shape changed')
    if arrays['label'].shape != dates.shape:
        raise AssertionError('Label/date alignment differs')
    numeric = np.load(architecture / 'numeric.npy', mmap_mode='r')
    if numeric.shape != (len(outer_train) + len(valid), 500) or numeric.dtype != np.dtype('float32'):
        raise AssertionError('Architecture matrix shape/dtype differs')
    selected_numeric = numeric_selection(input_meta, meta)
    schema = make_schema(meta, input_meta, selected_numeric)
    local_train = np.arange(len(outer_train))
    cutoff = None
    if args.recent_days:
        cutoff = int((pd.Timestamp(str(args.day)) - pd.Timedelta(days=args.recent_days)).strftime('%Y%m%d'))
        local_train = local_train[dates[outer_train] >= cutoff]
    train = outer_train[local_train]
    if not len(train) or not len(valid) or dates[train].max() >= dates[valid].min():
        raise AssertionError('Fit rows must strictly predate the complete validation day')
    y_train, y_valid = np.asarray(arrays['label'][train]), np.asarray(arrays['label'][valid])
    if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_valid)) != {0, 1}:
        raise AssertionError('Expected both binary label classes in each split')
    baseline_path = BASELINE / f'day{args.day % 100}_prediction.npy'
    baseline = np.load(baseline_path)
    np.testing.assert_array_equal(valid, np.load(BASELINE / f'day{args.day % 100}_indices.npy'))
    baseline_metrics = scores(y_valid, baseline)
    recorded_baseline = read(BASELINE / 'metrics.json')['results'][str(args.day)]['corrected_auc']
    if abs(baseline_metrics['auc'] - recorded_baseline) > 1e-10:
        raise AssertionError('Duplicate-corrected baseline prediction/metric mismatch')
    params = dict(iterations=1800, depth=args.depth, learning_rate=.03, l2_leaf_reg=30,
                  loss_function='Logloss', eval_metric='AUC', random_seed=42, task_type='GPU',
                  devices='0', border_count=64, one_hot_max_size=12, max_ctr_complexity=2,
                  gpu_ram_part=args.gpu_ram_part, gpu_cat_features_storage='CpuPinnedMemory',
                  pinned_memory_size='512mb', thread_count=THREADS, allow_writing_files=False,
                  counter_calc_method='SkipTest')
    protocol = {'status': 'prepared', 'day': args.day, 'tag': args.tag,
                'model': 'CatBoostClassifier', 'params': params, 'statistics': 'oof',
                'outer_train_rows': len(outer_train), 'train_rows': len(train), 'valid_rows': len(valid),
                'recent_days': args.recent_days, 'recent_cutoff_inclusive': cutoff,
                'train_dates': [int(day) for day in np.unique(dates[train])],
                'validation_dates': [int(args.day)],
                'split': 'All rows before held day (or complete recent dates); all rows of the future held day',
                'category_cache': str(cache.resolve()), 'architecture_cache': str(architecture.resolve()),
                'architecture_manifest_sha256': sha256(architecture / 'manifest.json'),
                'category_manifest_sha256': sha256(cache / 'manifest.json'),
                'vocabulary_sha256': sha256(cache / 'vocabs.json'),
                'trainer_sha256': sha256(Path(__file__)),
                'adapted_from': str((ROOT / 'analysis/daily_interest_din/train_chrono_tree.py').resolve()),
                'numeric_fields': [entry['name'] for entry in schema['numeric']],
                'numeric_count': 341, 'categorical_count': 87, 'total_features': 428,
                'numeric_scaler_fit_scope': input_meta['scaler_fit_scope'],
                'numeric_scaler_note': 'Reused unchanged; recent-day runs retain the original outer-train scaler',
                'quantization_fit_scope': 'Actual fit rows only; same saved borders applied to validation',
                'memory_policy': 'One split dataframe at a time, only selected columns, then CPU quantization',
                'validation_labels_used_for_features_or_optimizer': False,
                'validation_usage': 'CatBoost eval_set for early stopping and checkpoint selection; metrics and fixed blends',
                'early_stopping_rounds': 100, 'use_best_model': True,
                'source_availability': input_meta['availability'],
                'per_row_strictly_historical': input_meta['per_row_strictly_historical'],
                'baseline_prediction': str(baseline_path.resolve()), 'baseline_metrics': baseline_metrics,
                'baseline_sha256': sha256(baseline_path), 'fixed_blend_tree_weights': BLEND_WEIGHTS,
                'structural_control': 'Depth 8 by default and pairwise categorical CTRs (2), versus existing depth 6/CTR 1',
                'development_caveat': 'Held days previously reused for model development; not a blind test',
                'final_test_read': False}
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'schema.json', schema)
    write(out / 'protocol.json', protocol)
    write(out / 'input_manifest.json', input_meta)
    (out / 'trainer_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (out / 'vocabulary.json').write_bytes((cache / 'vocabs.json').read_bytes())
    np.save(out / 'train_indices.npy', train)
    np.save(out / 'valid_indices.npy', valid)
    started = time.perf_counter()
    print('DEEP_TREE_PREPARE', len(train), len(valid), 'categorical=87 numeric=341', flush=True)
    fit = make_pool(arrays, numeric, train, local_train, y_train, schema)
    borders = out / 'quantization_borders.tsv'
    fit.save_quantization_borders(str(borders))
    local_valid = slice(len(outer_train), len(outer_train) + len(valid))
    validation = make_pool(arrays, numeric, valid, local_valid, y_valid, schema, borders=borders)
    protocol['status'] = 'training'
    protocol['quantization_borders_sha256'] = sha256(borders)
    write(out / 'protocol.json', protocol)
    print('DEEP_TREE_TRAIN', json.dumps(params), flush=True)
    fit_started = time.perf_counter()
    model = CatBoostClassifier(**params)
    model.fit(fit, eval_set=validation, early_stopping_rounds=100, verbose=50, use_best_model=True)
    fit_seconds = time.perf_counter() - fit_started
    probability = model.predict_proba(validation, thread_count=THREADS)[:, 1]
    scores(y_valid, probability)
    np.save(out / 'validation_prediction.npy', probability)
    model.save_model(str(out / 'model.cbm'))
    write(out / 'history.json', model.get_evals_result())
    write(out / 'effective_parameters.json', model.get_all_params())
    del fit
    gc.collect()
    restored = CatBoostClassifier()
    restored.load_model(str(out / 'model.cbm'))
    if restored.feature_names_ != schema['feature_names'] or restored.get_cat_feature_indices() != schema['categorical_indices']:
        raise AssertionError('Reloaded model categorical schema differs')
    replay = restored.predict_proba(validation, thread_count=THREADS)[:, 1]
    error = float(np.max(np.abs(replay - probability)))
    if not np.isfinite(error) or error > 1e-10:
        raise AssertionError('Tree checkpoint replay error')
    comparisons = [{'tree_weight': weight, **scores(y_valid, (1 - weight) * baseline + weight * probability)}
                   for weight in BLEND_WEIGHTS]
    result = {'status': 'complete_and_verified', 'day': args.day, 'tag': args.tag,
              **scores(y_valid, probability), 'train_rows': len(train), 'valid_rows': len(valid),
              'validation_clicks': int(y_valid.sum()), 'trees': int(model.tree_count_),
              'best_iteration': int(model.get_best_iteration()), 'features': len(schema['feature_names']),
              'numeric_features': 341, 'categorical_features': 87, 'params': params,
              'maximum_reload_error': error, 'model_sha256': sha256(out / 'model.cbm'),
              'fit_seconds': fit_seconds, 'seconds': time.perf_counter() - started,
              'baseline_auc': baseline_metrics['auc'], 'baseline_metrics': baseline_metrics,
              'fixed_blends': comparisons, 'recent_days': args.recent_days,
              'validation_labels_used_for_features_or_optimizer': False, 'final_test_read': False,
              'development_caveat': protocol['development_caveat']}
    write(out / 'metrics.json', result)
    pd.DataFrame({'feature': model.feature_names_, 'importance': model.get_feature_importance()}).sort_values(
        'importance', ascending=False).to_csv(out / 'importance.csv', index=False)
    protocol['status'] = 'complete_and_verified'
    write(out / 'protocol.json', protocol)
    print('DEEP_TREE_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--day', type=int, choices=[20191115, 20191116], default=20191115)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--recent-days', type=int, default=0)
    parser.add_argument('--gpu-ram-part', type=float, default=.5)
    main(parser.parse_args())
