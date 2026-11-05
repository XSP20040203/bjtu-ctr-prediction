"""Full historical context LightGBM with train-fitted dense categorical IDs."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'

import argparse
import faulthandler
faulthandler.enable()
import gc
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash, lgb
from full_population_original_v1 import wait_for_memory
from dense_categorical_context_core_v1 import fit_dictionaries, encode, frame
import full_negative_class_balance_v1 as shared

OUT = HERE / 'runs/full_negative_dense_context_lgb_v1'
SOURCE = HERE / 'runs/full_negative_context_v1'
INDEPENDENT = HERE / 'runs/full_negative_context_v2'
BASE = HERE / 'runs/compose_semantic_embedding_masks_v1'
PROBE = HERE / 'runs/dense_context_category_probe_v1'
ARMS = [dict(name='dense_context_leaf31_w5', num_leaves=31, scale_pos_weight=5.),
        dict(name='dense_context_leaf63_w25', num_leaves=63, scale_pos_weight=25.)]
CHECKPOINTS = [400, 1200, 2400]
PARAMS = dict(objective='binary', learning_rate=.025, min_data_in_leaf=128,
    lambda_l2=40., lambda_l1=0., max_bin=63, feature_fraction=.8,
    bagging_fraction=.8, bagging_freq=1, cat_l2=40., cat_smooth=40.,
    max_cat_threshold=64, min_data_per_group=50, num_threads=2, seed=151751,
    verbosity=-1, deterministic=True, force_col_wise=True, feature_pre_filter=False,
    metric='None', device_type='cpu')
shared.OUT, shared.BASE, shared.CHECKPOINTS = OUT, BASE, CHECKPOINTS
shared.engine.OUT, shared.engine.BASE, shared.engine.CHECKPOINTS = OUT, BASE, CHECKPOINTS
shared.emit = lambda name, **values: emit(name.replace('class_balance', 'dense_context_lgb'), **values)


def folder(day, independent=False):
    return (INDEPENDENT if independent else SOURCE) / f'{"independent" if independent else "input"}_day{day}'


def native_path(path):
    assert Path.cwd().resolve() == ROOT.resolve()
    value = Path(path).resolve().relative_to(ROOT).as_posix()
    value.encode('ascii')
    return value


def parameters(arm):
    return dict(PARAMS, **{name: value for name, value in arm.items() if name != 'name'})


def dictionary_path(day, independent=False):
    return OUT / f'category_dictionaries_day{day}{"_independent" if independent else ""}.npz'


def dictionaries(day, independent=False):
    path = dictionary_path(day, independent)
    locked = read(OUT / 'inputs_locked.json')[str(day)]
    assert digest(path) == locked['independent_dictionary_sha256' if independent else 'dictionary_sha256']
    with np.load(path) as saved:
        result = {name: saved[name] for name in saved.files}
    assert {name: array_hash(value) for name, value in result.items()} == locked['dictionary_values_sha256']
    return result


def validate_inputs(day, independent=False):
    directory = folder(day, independent)
    locked = read(OUT / 'inputs_locked.json')[str(day)]
    contract = read(directory / 'contract.json')
    assert contract == locked['source_contract']
    assert digest(directory / 'features.npy') == contract['matrix_sha256']
    for name, expected in contract['files_sha256'].items():
        assert digest(directory / name) == expected, name
    assert contract['shape'][1] == 395 and len(contract['schema']['columns']) == 395
    assert contract['all_historical_negatives_retained'] and contract['own_held_labels_masked']
    assert contract['max_training_date'] < 20191100 + day
    dictionaries(day, independent)
    return contract


def fit_independent_dictionaries(day):
    directory = folder(day, True)
    contract = read(directory / 'contract.json')
    positions = np.load(directory / 'train.npy')
    dates = np.load(directory / 'dates.npy')
    assert np.all(dates[positions] < 20191100 + day)
    matrix = np.load(directory / 'features.npy', mmap_mode='r')
    result = {}
    for name in contract['schema']['categories']:
        column = contract['schema']['columns'].index(name)
        numeric = np.asarray(matrix[:, column][positions], dtype='float32')
        integer = numeric.astype('int64')
        np.testing.assert_array_equal(numeric, integer)
        # Separate fitting implementation on the independently rebuilt source.
        result[name] = pd.Index(integer[integer >= 0]).unique().sort_values().to_numpy(dtype='int64')
    return result


def full_encoding_replay(day, partition):
    primary_tables, independent_tables = dictionaries(day), dictionaries(day, True)
    count = len(np.load(folder(day) / ('train.npy' if partition == 'train' else 'held.npy')))
    left_hash, right_hash = hashlib.sha256(), hashlib.sha256()
    for start in range(0, count, 20000):
        chosen = np.arange(start, min(start + 20000, count))
        left, categories = frame(folder(day), partition, primary_tables, chosen)
        right, other = frame(folder(day, True), partition, independent_tables, chosen, independent=True)
        assert categories == other and list(left) == list(right)
        left_values, right_values = left.to_numpy(dtype='float32'), right.to_numpy(dtype='float32')
        np.testing.assert_array_equal(left_values, right_values)
        left_hash.update(left_values.tobytes(order='C'))
        right_hash.update(right_values.tobytes(order='C'))
        del left, right, left_values, right_values
    assert left_hash.hexdigest() == right_hash.hexdigest()
    emit('dense_context_complete_encoding_replayed', day=day, partition=partition, rows=count, columns=395)
    return dict(rows=count, columns=395, row_major_float32_sha256=left_hash.hexdigest(),
        independent_raw_source_dictionary_fit_mapping_and_column_gather_bitwise_exact=True)


def categorical_splits(model):
    def count(node):
        if 'split_index' not in node:
            return 0
        return int(node['decision_type'] == '==') + count(node['left_child']) + count(node['right_child'])
    return sum(count(tree['tree_structure']) for tree in model.dump_model()['tree_info'])


def native_preflight():
    day = 15
    labels = np.load(folder(day) / 'labels.npy')
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    selected = np.sort(np.r_[positives[np.linspace(0, len(positives)-1, 512, dtype='int64')],
        negatives[np.linspace(0, len(negatives)-1, 32256, dtype='int64')]])
    tables = dictionaries(day)
    data, cats = frame(folder(day), 'train', tables, selected)
    other, other_cats = frame(folder(day, True), 'train', dictionaries(day, True), selected, independent=True)
    assert cats == other_cats
    np.testing.assert_array_equal(data.to_numpy(), other.to_numpy())
    held_rows = np.arange(1500)
    held, _ = frame(folder(day), 'held', tables, held_rows)
    held_other, _ = frame(folder(day, True), 'held', dictionaries(day, True), held_rows, independent=True)
    np.testing.assert_array_equal(held.to_numpy(), held_other.to_numpy())
    records, baseline = [], None
    for arm in ARMS:
        params = dict(parameters(arm), min_data_in_leaf=32, min_data_per_group=10)
        dataset = lgb.Dataset(data, label=labels[selected], categorical_feature=cats, params=params)
        def callback(environment):
            trees = environment.iteration + 1
            if trees in (5, 12, 32):
                environment.model.save_model(native_path(OUT / f'{arm["name"]}_preflight_trees{trees}.txt'), num_iteration=trees)
        model = lgb.train(params, dataset, num_boost_round=32, callbacks=[callback], keep_training_booster=True)
        maximum_error = 0.
        for trees in (5, 12, 32):
            path = OUT / f'{arm["name"]}_preflight_trees{trees}.txt'
            restored = lgb.Booster(model_file=native_path(path))
            assert restored.current_iteration() == trees
            before = model.predict(held, raw_score=True, num_iteration=trees, num_threads=2)
            after = restored.predict(held_other, raw_score=True, num_threads=2)
            np.testing.assert_array_equal(before, after)
            maximum_error = max(maximum_error, float(np.max(np.abs(before - after))))
            del restored
        split_count = categorical_splits(model)
        record = dict(arm, rows=len(data), columns=395, trees=32, categorical_split_count=split_count,
            final_checkpoint_bytes=(OUT / f'{arm["name"]}_preflight_trees32.txt').stat().st_size,
            native_callback_restore_and_independent_prediction_maximum_error=maximum_error)
        records.append(record)
        if arm == ARMS[0]:
            baseline = dict(train=model.predict(data, raw_score=True, num_threads=2),
                held=model.predict(held, raw_score=True, num_threads=2), params=params,
                bytes=record['final_checkpoint_bytes'], categorical_split_count=split_count)
        del model, dataset
        gc.collect()
    raw, _ = frame(folder(day), 'train', tables, selected, recode=False)
    raw_held, _ = frame(folder(day), 'held', tables, held_rows, recode=False)
    dataset = lgb.Dataset(raw, label=labels[selected], categorical_feature=cats, params=baseline['params'])
    model = lgb.train(baseline['params'], dataset, num_boost_round=32, keep_training_booster=True)
    path = OUT / 'raw_category_comparison_preflight.txt'
    model.save_model(native_path(path))
    restored = lgb.Booster(model_file=native_path(path))
    raw_train = restored.predict(raw, raw_score=True, num_threads=2)
    raw_prediction = restored.predict(raw_held, raw_score=True, num_threads=2)
    np.testing.assert_array_equal(raw_prediction, model.predict(raw_held, raw_score=True, num_threads=2))
    # Equivalence and compression are observations, not assumptions or score gains.
    comparison = dict(rows=len(raw), trees=32, raw_bytes=path.stat().st_size,
        dense_bytes=baseline['bytes'], dense_to_raw_size_ratio=baseline['bytes']/path.stat().st_size,
        raw_categorical_split_count=categorical_splits(model),
        dense_categorical_split_count=baseline['categorical_split_count'],
        training_prediction_raw_vs_dense_maximum_error=float(np.max(np.abs(raw_train-baseline['train']))),
        held_prediction_raw_vs_dense_maximum_error=float(np.max(np.abs(raw_prediction-baseline['held']))),
        held_probe_labels_used=False)
    write(OUT / 'representative_encoding_comparison.json', comparison)
    assert not torch.cuda.is_initialized()
    write(OUT / 'preflight.json', dict(status='passed', device='CPU', rows=len(data), columns=395,
        arms=records, representative_encoding_comparison=comparison,
        full_complete_training_and_held_encodings_independently_replayed=True,
        complete_source_labels_dates_and_populations_checked=True,
        native_callback_ascii_file_save_restore_passed=True, gpu_memory_allocated=False,
        production_capacity_proven=False, held_labels_used=False, inferred17_labels_read=False,
        protocol_sha256=digest(OUT / 'protocol.json'), input_lock_sha256=digest(OUT / 'inputs_locked.json'),
        full_projection_verification_sha256=digest(OUT / 'full_projection_verification.json')))
    shared.engine.check()
    emit('dense_context_lgb_preflight_passed', rows=len(data), columns=395, comparison=comparison)


def prepare():
    assert not OUT.exists(), 'Preserve previous sources, models and failures'
    assert not torch.cuda.is_initialized()
    probe = read(PROBE / 'verification.json')
    assert probe['status'] == 'passed' and probe['raw_vs_dense_predictions_identical']
    assert digest(PROBE / 'protocol.json') == probe['protocol_sha256']
    upstream = read(INDEPENDENT / 'verification.json')
    assert upstream['status'] == 'passed'
    assert all(entry['independent_full_population_features_and_models_replayed'] for entry in upstream['dates'].values())
    paths = [Path(__file__)] + [HERE / name for name in (
        'run_full_negative_dense_context_lgb_v1.ps1', 'dense_categorical_context_core_v1.py',
        'publish_progress_v2.py', 'record_two_day_progress_v2.py',
        'probe_dense_context_categories_v1.py', 'full_negative_class_balance_v1.py',
        'full_negative_context_v2.py', 'full_population_original_v1.py',
        'cross_user_pair_ranker_v1.py', 'quantile_blend_core_v1.py',
        'conditional_daily.py', 'fit_stacker.py', 'rich_userday_activity.py', 'verify_stacker.py')]
    paths += [PROBE / name for name in ('protocol.json', 'verification.json')]
    paths += [HERE / 'native_lgb_sparse_category_diagnostic_v1.json']
    hashes = read(BASE / 'protocol.json')['raw_datasets_sha256']
    for source in (INDEPENDENT, BASE):
        assert read(source / 'verification.json')['status'] == 'passed'
        protocol = read(source / 'protocol.json')
        assert protocol['raw_datasets_sha256'] == hashes
        for filename, expected in protocol['sources_sha256'].items():
            assert digest(filename) == expected, filename
            if Path(filename).suffix == '.py':
                paths.append(Path(filename))
        paths += [source / 'protocol.json', source / 'verification.json']
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    contracts = {}
    for day in (15, 16):
        declared = read(folder(day) / 'contract.json')
        for independent in (False, True):
            directory = folder(day, independent)
            assert read(directory / 'contract.json') == declared
            assert digest(directory / 'features.npy') == declared['matrix_sha256']
            for filename, expected in declared['files_sha256'].items():
                assert digest(directory / filename) == expected, filename
            paths += [directory / name for name in ('contract.json', 'features.npy', *declared['files_sha256'])]
        rows, train, held = [np.load(folder(day) / (name + '.npy')) for name in ('rows', 'train', 'held')]
        np.testing.assert_array_equal(rows[train], np.flatnonzero(raw.pt_d.to_numpy() < 20191100 + day))
        np.testing.assert_array_equal(rows[held], np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day))
        np.testing.assert_array_equal(np.load(folder(day) / 'labels.npy'), raw.label.to_numpy()[rows[train]])
        np.testing.assert_array_equal(rows[held], np.load(BASE / f'day{day}_indices.npy'))
        assert declared['max_training_date'] < 20191100 + day
        contracts[str(day)] = dict(source_contract=declared)
        paths += [BASE / f'day{day}_{name}.npy' for name in ('indices', 'prediction')]
    del raw
    gc.collect()
    for name, expected in hashes.items():
        assert digest(ROOT / '2026ml' / name) == expected, name
    OUT.mkdir()
    for day in (15, 16):
        primary, independent = fit_dictionaries(folder(day)), fit_independent_dictionaries(day)
        assert primary.keys() == independent.keys()
        for name in primary:
            np.testing.assert_array_equal(primary[name], independent[name])
        np.savez(dictionary_path(day), **primary)
        np.savez(dictionary_path(day, True), **independent)
        contracts[str(day)].update(dictionary_sha256=digest(dictionary_path(day)),
            independent_dictionary_sha256=digest(dictionary_path(day, True)),
            dictionary_values_sha256={name: array_hash(value) for name, value in primary.items()},
            dictionary_max_training_date=20191100 + day - 1)
        paths += [dictionary_path(day), dictionary_path(day, True)]
        np.save(OUT / f'day{day}_indices.npy', np.load(BASE / f'day{day}_indices.npy'))
        emit('dense_context_dictionaries_independently_fitted', day=day, categories=len(primary))
    write(OUT / 'inputs_locked.json', contracts)
    write(OUT / 'protocol.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        dates=[20191115, 20191116], arms=ARMS, params=PARAMS, checkpoints=CHECKPOINTS,
        baseline=str(BASE), input_feature_columns=395, added_neural_features=0,
        training='Every strictly earlier positive and negative row; no negative subsampling',
        category_encoding='Sorted nonnegative raw IDs fitted on all strictly earlier feature rows, mapped to consecutive0..n-1; all negative values -> -1, unseen nonnegative -> n',
        dictionary_fit_uses_advertisement_labels=False,
        rationale='Full533 neural coordinates did not improve standalone discrimination; test verified395-column context alone with leaf31/weight5 and leaf63/weight25. Sparse raw user IDs create measured large category bitsets, so train-fitted dense dictionaries bound native category indices.',
        checkpoint_reliability='keep_training_booster=True, native callback checkpoints, ASCII relative file paths and exact native file restore',
        prior_native_crash_root_cause_proven=False,
        evaluation_day_labels_fit=False, inferred17_labels_read=False,
        selection='All12 checkpoints and both-date candidate hashes locked before complete-cohort scoring;15/16 repeatedly reused development dates',
        lightgbm_version=lgb.__version__, device='CPU',
        raw_datasets_sha256=hashes, sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)}))
    projections = {}
    for day in (15, 16):
        validate_inputs(day)
        validate_inputs(day, True)
        projections[str(day)] = {partition: full_encoding_replay(day, partition) for partition in ('train', 'held')}
    write(OUT / 'full_projection_verification.json', dict(status='passed', dates=projections,
        all_full_training_and_held_rows_replayed=True, columns=395,
        advertisement_held_labels_read=False, inferred17_labels_read=False,
        protocol_sha256=digest(OUT / 'protocol.json')))
    native_preflight()


def fit():
    shared.engine.check()
    assert read(OUT / 'preflight.json')['status'] == 'passed'
    assert not (OUT / 'models_locked.json').exists() and not list(OUT.glob('day*.txt'))
    records = []
    for day in (15, 16):
        wait_for_memory()
        contract = validate_inputs(day)
        tables = dictionaries(day)
        data, cats = frame(folder(day), 'train', tables)
        labels = np.load(folder(day) / 'labels.npy')
        dates = np.load(folder(day) / 'dates.npy')[np.load(folder(day) / 'train.npy')]
        assert np.all(dates < 20191100 + day)
        assert array_hash(labels) == contract['training_labels_sha256']
        assert len(labels) == contract['training_rows']
        for arm in ARMS:
            emit('dense_context_lgb_fit_started', day=day, arm=arm['name'], rows=len(data), columns=395)
            params = parameters(arm)
            dataset = lgb.Dataset(data, label=labels, categorical_feature=cats, params=params, free_raw_data=True)
            def callback(environment):
                trees = environment.iteration + 1
                if trees in CHECKPOINTS:
                    name = f'day{day}_{arm["name"]}_trees{trees}'
                    path = OUT / (name + '.txt')
                    assert not path.exists()
                    write(OUT / 'checkpoint_save_journal.json', dict(stage='before_native_save',
                        day=day, arm=arm['name'], trees=trees, path=str(path)))
                    environment.model.save_model(native_path(path), num_iteration=trees)
                    write(OUT / 'checkpoint_save_journal.json', dict(stage='native_save_complete',
                        day=day, arm=arm['name'], trees=trees, path=str(path), sha256=digest(path)))
                    emit('dense_context_lgb_checkpoint_saved', day=day, arm=arm['name'], trees=trees, bytes=path.stat().st_size)
                if trees % 200 == 0:
                    emit('dense_context_lgb_iteration', day=day, arm=arm['name'], trees=trees)
            model = lgb.train(params, dataset, num_boost_round=max(CHECKPOINTS),
                              callbacks=[callback], keep_training_booster=True)
            for trees in CHECKPOINTS:
                name = f'day{day}_{arm["name"]}_trees{trees}'
                path = OUT / (name + '.txt')
                assert path.is_file()
                restored = lgb.Booster(model_file=native_path(path))
                assert restored.current_iteration() == trees and restored.feature_name() == list(data)
                np.testing.assert_array_equal(model.predict(data.iloc[:2000], raw_score=True,
                    num_iteration=trees, num_threads=2), restored.predict(data.iloc[:2000], raw_score=True, num_threads=2))
                records.append(dict(arm, day=day, trees=trees, name=name,
                    model_sha256=digest(path), max_training_date=int(dates.max()),
                    training_rows=len(labels), training_labels_sha256=array_hash(labels),
                    serialization_maximum_error=0., dictionary_sha256=digest(dictionary_path(day))))
                del restored
            del model, dataset
            gc.collect()
            emit('dense_context_lgb_fit_complete', day=day, arm=arm['name'])
        del data, labels, dates, tables
        gc.collect()
    write(OUT / 'models_locked.json', dict(models=records, protocol_sha256=digest(OUT / 'protocol.json'),
        inputs_locked_sha256=digest(OUT / 'inputs_locked.json')))
    shared.engine.check()


def forecasts(day, independent=False):
    contract = validate_inputs(day, independent)
    data, _ = frame(folder(day, independent), 'held', dictionaries(day, independent), independent=independent)
    locked = read(OUT / 'models_locked.json')
    assert locked['protocol_sha256'] == digest(OUT / 'protocol.json')
    assert locked['inputs_locked_sha256'] == digest(OUT / 'inputs_locked.json')
    expected = read(OUT / 'full_projection_verification.json')['dates'][str(day)]['held']['row_major_float32_sha256']
    assert array_hash(data.to_numpy(dtype='float32')) == expected
    margins = {}
    for record in locked['models']:
        if record['day'] != day:
            continue
        path = OUT / (record['name'] + '.txt')
        assert digest(path) == record['model_sha256']
        assert record['max_training_date'] == contract['max_training_date'] < 20191100 + day
        assert record['dictionary_sha256'] == digest(dictionary_path(day))
        model = lgb.Booster(model_file=native_path(path))
        assert model.feature_name() == list(data)
        margin = model.predict(data, raw_score=True, num_threads=2)
        assert margin.shape == (len(data),) and np.isfinite(margin).all()
        target = OUT / (record['name'] + '_held_margin.npy')
        if independent:
            np.testing.assert_array_equal(margin, np.load(target))
        else:
            assert not target.exists()
            np.save(target, margin)
        margins[record['name']] = margin
        del model
    return margins


shared.forecasts = forecasts
shared.engine.forecasts = forecasts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['prepare', 'fit', 'evaluate', 'verify'])
    phase = parser.parse_args().phase
    if phase == 'evaluate':
        shared.engine.evaluate()
    elif phase == 'verify':
        shared.verify()
    else:
        globals()[phase]()
