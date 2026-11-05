"""Full historical row features, preserving every available negative exposure."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'
import gc
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash
from rich_augmented_rows import select_daily
from global_row_context import CAT, HISTORY, histories
from rich_row_catboost import selection
from catboost_userday_residual import load_model
from news_quality_features import READ, build_quality
from full_negative_context_core_v1 import population, positions
from unordered_snapshot_core_v1 import build as snapshot, NAMES as SNAP_NAMES, FIELDS as SNAP_FIELDS
sys.path.insert(0, str(ROOT / 'analysis/auc085_next'))
from prepare_multiscale_stats import build_multiscale_statistics, feature_names
sys.path.insert(0, str(ROOT / 'analysis/daily_interest_din'))
from prepare_exposure_features import build_exposure_features, SINGLE

NATIVE = HERE / 'runs/rich_augmented_rows_v1'
QUALITY = HERE / 'runs/news_quality_daily_v1'
SELECTOR = HERE / 'runs/task_audience_row_ranker_v1'
HISTORY_NAMES = [name + suffix for name in HISTORY for suffix in ('_log_unique', '_match')]


def schema(day):
    source = read(NATIVE / f'input_contract{day}.json')
    quality = read(QUALITY / f'input_contract{day}.json')
    daily = [name for name in source['feature_names']
             if name.startswith('rich_daily__') and not name.startswith('rich_daily__quality__')]
    assert len(daily) == 128 and len(quality['quality_names']) == 337
    common = CAT + feature_names() + daily + quality['quality_names'] + HISTORY_NAMES
    assert len(common) == len(set(common)) == 660
    native_keep = [name for name in source['feature_names'] if not name.startswith('rich_daily__quality__')]
    selector = load_model(SELECTOR / f'day{day}_audience_row_d6_trees1600.cbm')
    importance = selector.get_feature_importance(type='PredictionValuesChange', thread_count=2)
    mapping = {name: float(importance[i]) for i, name in enumerate(native_keep + quality['quality_names'])}
    del selector
    numeric = [i for i, name in enumerate(common) if name not in CAT]
    chosen = sorted(sorted(numeric, key=lambda i: (-mapping[common[i]], i))[:256])
    selected = CAT + [common[i] for i in chosen]
    dummy = pd.DataFrame({name: [1] for name in ['pt_d'] + SINGLE})
    _, exposure = build_exposure_features(dummy)
    names = selected + exposure + SNAP_NAMES
    assert len(names) == len(set(names)) == 395
    return dict(columns=names, categories=CAT, common=selected, daily_names=daily,
                quality_names=quality['quality_names'], exposure_names=exposure,
                selector_max_training_date=20191100 + day - 1,
                selector_model_sha256=digest(SELECTOR / f'day{day}_audience_row_d6_trees1600.cbm'))


def store(matrix, target_names, values, source_names, lookup=None):
    destination = [i for i, name in enumerate(target_names) if name in source_names]
    source = [source_names.index(target_names[i]) for i in destination]
    if not destination:
        return []
    for start in range(0, len(matrix), 10000):
        stop = min(start + 10000, len(matrix))
        local = np.arange(start, stop) if lookup is None else lookup[start:stop]
        block = np.asarray(values[np.ix_(local, source)], dtype='float32')
        assert np.isfinite(block).all()
        matrix[start:stop, destination] = block
    return destination


def build(day, destination, independent=False):
    destination.mkdir(parents=True, exist_ok=False)
    declared = schema(day)
    columns = list(dict.fromkeys(CAT + ['pt_d', 'label'] + HISTORY))
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=columns,
                      dtype={name: 'category' if name in HISTORY else 'int32' for name in columns})
    dates = raw.pt_d.to_numpy()
    rows, train, held, safe = population(dates, raw.label.to_numpy(), 20191100 + day)
    raw['label'] = safe
    assert np.all(raw.label.to_numpy()[dates >= 20191100 + day] == -1)
    sampled, _, reference_held, _, _, _ = selection(day, safe, dates)
    np.testing.assert_array_equal(rows[held], reference_held)
    np.testing.assert_array_equal(rows[train][np.searchsorted(rows[train], sampled)], sampled)
    np.save(destination / 'rows.npy', rows)
    np.save(destination / 'train.npy', train)
    np.save(destination / 'held.npy', held)
    np.save(destination / 'labels.npy', safe[rows[train]])
    np.save(destination / 'dates.npy', dates[rows])
    matrix = np.lib.format.open_memmap(destination / 'features.npy', mode='w+', dtype='float32',
                                     shape=(len(rows), len(declared['columns'])))
    filled = set(store(matrix, declared['columns'], raw[CAT].to_numpy(), CAT, rows))
    stats = np.lib.format.open_memmap(destination / 'past_rates.npy', mode='w+', dtype='float32',
                                    shape=(len(raw), len(feature_names())), fortran_order=True)
    def progress(index, total, fields, size):
        emit('full_negative_past_rates', day=day, independent=independent, group=index, total=total)
    stats, names = build_multiscale_statistics(raw, 20191100 + day, output=stats, progress=progress)
    stats.flush()
    filled.update(store(matrix, declared['columns'], stats, names, rows))
    del stats
    gc.collect()
    emit('full_negative_daily_start', day=day, independent=independent)
    daily, keys, daily_names, _, _ = select_daily(day)
    names = ['rich_daily__' + name for name in daily_names]
    query = raw.iloc[rows][['pt_d', 'user_id']].to_numpy()
    lookup = positions(keys, query, independent)
    filled.update(store(matrix, declared['columns'], daily, names, lookup))
    del daily, keys, lookup
    gc.collect()
    with np.load(QUALITY / f'quality{day}.npz') as cache:
        keys, values = cache['keys'], cache['values']
    expected = read(QUALITY / f'input_contract{day}.json')
    assert array_hash(values) == expected['quality_sha256']
    if independent:
        news = pd.read_csv(ROOT / '2026ml/train_feeds.csv', usecols=READ,
                           dtype={'i_docId': str, 'i_s_sourceId': str})
        rebuilt, quality_names = build_quality(news, keys, 20191100 + day)
        np.testing.assert_array_equal(rebuilt, values)
        assert quality_names == declared['quality_names']
        values = rebuilt
        del news
        gc.collect()
    lookup = positions(keys, query, independent)
    filled.update(store(matrix, declared['columns'], values, declared['quality_names'], lookup))
    del keys, values, lookup, query
    gc.collect()
    local = raw.iloc[rows].drop(columns='label').reset_index(drop=True)
    values = histories(local)
    filled.update(store(matrix, declared['columns'], values, HISTORY_NAMES))
    del values
    values, names = build_exposure_features(local[['pt_d'] + SINGLE])
    assert names == declared['exposure_names']
    filled.update(store(matrix, declared['columns'], values, names))
    del values
    gc.collect()
    values, names = snapshot(local[SNAP_FIELDS], independent)
    filled.update(store(matrix, declared['columns'], values, names))
    del values, local
    assert filled == set(range(len(declared['columns'])))
    matrix.flush()
    # Check every common feature on every previously audited sampled/held row.
    contract = read(NATIVE / f'input_contract{day}.json')
    old_rows = np.sort(np.r_[sampled, reference_held])
    old = np.load(NATIVE / f'independent_day{day}/features.npy', mmap_mode='r')
    common = [name for name in declared['common'] if name in contract['feature_names']]
    source_columns = [contract['feature_names'].index(name) for name in common]
    target_columns = [declared['columns'].index(name) for name in common]
    local_positions = np.searchsorted(rows, old_rows)
    np.testing.assert_array_equal(rows[local_positions], old_rows)
    for start in range(0, len(old_rows), 10000):
        stop = min(start + 10000, len(old_rows))
        np.testing.assert_array_equal(matrix[np.ix_(local_positions[start:stop], target_columns)],
                                      old[np.ix_(np.arange(start, stop), source_columns)])
    negative = int((safe[rows[train]] == 0).sum())
    result = dict(schema=declared, shape=list(matrix.shape), matrix_sha256=digest(destination / 'features.npy'),
        rows_sha256=array_hash(rows), training_rows_sha256=array_hash(rows[train]),
        training_labels_sha256=array_hash(safe[rows[train]]), held_rows_sha256=array_hash(rows[held]),
        training_rows=len(train), training_positives=int(safe[rows[train]].sum()), training_negatives=negative,
        added_training_negatives=len(train) - len(sampled), all_historical_negatives_retained=True,
        max_training_date=int(dates[rows[train]].max()), own_held_labels_masked=True,
        all_prior_audited_common_rows_checked=True, common_reference_rows=len(old_rows),
        common_reference_columns=len(common), inferred17_labels_read=False,
        files_sha256={name: digest(destination / name) for name in
                      ('rows.npy', 'train.npy', 'held.npy', 'labels.npy', 'dates.npy')})
    assert negative == int(((dates < 20191100 + day) & (safe == 0)).sum())
    write(destination / 'contract.json', result)
    emit('full_negative_inputs_ready', day=day, independent=independent, shape=result['shape'],
         training_rows=len(train), additional_negatives=result['added_training_negatives'])
    return result


def frame(folder, partition):
    declared = read(folder / 'contract.json')
    positions_ = np.load(folder / ('train.npy' if partition == 'train' else 'held.npy'))
    matrix = np.load(folder / 'features.npy', mmap_mode='r')
    values = np.asarray(matrix[positions_], dtype='float32')
    data = pd.DataFrame(values, columns=declared['schema']['columns'], copy=False)
    for name in declared['schema']['categories']:
        numeric = data[name].to_numpy()
        np.testing.assert_array_equal(numeric, numeric.astype('int64'))
        data[name] = numeric.astype('int64')
    assert np.isfinite(values).all()
    return data, declared['schema']['categories']
