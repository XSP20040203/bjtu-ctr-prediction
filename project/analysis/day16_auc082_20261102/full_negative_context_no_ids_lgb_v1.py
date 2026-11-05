"""Sequential 389-column ablation of six dominant raw categorical identifiers."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'

import argparse
import gc
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import full_negative_dense_context_lgb_v1 as dense
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash, lgb
from dense_categorical_context_core_v1 import encode
from full_population_original_v1 import wait_for_memory

OUT = HERE / 'runs/full_negative_context_no_ids_lgb_v1'
PARENT = HERE / 'runs/full_negative_dense_context_lgb_v1'
BASE = HERE / 'runs/compose_history_dot_embedding_masks_v1'
DIAG = HERE / 'dense_context_feature_use_diagnostic_v1.json'
REMOVE = ['user_id', 'task_id', 'adv_id', 'city', 'device_name', 'device_size']
COLUMNS = 389
ARMS = [dict(name='no_ids_leaf31_w5', num_leaves=31, scale_pos_weight=5.),
        dict(name='no_ids_leaf63_w25', num_leaves=63, scale_pos_weight=25.)]
CHECKPOINTS = [400, 1200, 2400]
PARAMS = dict(dense.PARAMS)

dense.OUT, dense.BASE, dense.ARMS, dense.PARAMS = OUT, BASE, ARMS, PARAMS
dense.shared.OUT, dense.shared.BASE, dense.shared.ARMS = OUT, BASE, ARMS
dense.shared.engine.OUT, dense.shared.engine.BASE, dense.shared.engine.ARMS = OUT, BASE, ARMS
dense.shared.emit = lambda name, **values: emit(name.replace('class_balance', 'context_no_ids_lgb'), **values)


def projection(directory):
    schema = read(directory / 'contract.json')['schema']
    assert len(schema['columns']) == 395 and set(REMOVE) <= set(schema['categories'])
    names = [name for name in schema['columns'] if name not in REMOVE]
    cats = [name for name in schema['categories'] if name not in REMOVE]
    assert len(names) == COLUMNS and len(cats) == 18
    return names, cats, [schema['columns'].index(name) for name in names]


def frame(directory, partition, tables, selected=None, independent=False):
    names, cats, columns = projection(directory)
    positions = np.load(directory / ('train.npy' if partition == 'train' else 'held.npy'))
    if selected is not None:
        positions = positions[np.asarray(selected)]
    matrix = np.load(directory / 'features.npy', mmap_mode='r')
    if independent:
        data = {name: np.asarray(matrix[:, column][positions], dtype='float32')
                for name, column in zip(names, columns)}
        result = pd.DataFrame(data, copy=False)
    else:
        # Bound gather memory even for the complete two-million-row training set.
        values = np.empty((len(positions), COLUMNS), dtype='float32')
        for start in range(0, len(positions), 20000):
            stop = min(start + 20000, len(positions))
            values[start:stop] = matrix[np.ix_(positions[start:stop], columns)]
        result = pd.DataFrame(values, columns=names, copy=False)
    for name in cats:
        result[name] = encode(result[name].to_numpy(), tables[name], independent)
    assert list(result) == names and not set(REMOVE) & set(result)
    # Keep validation bounded instead of materializing a second full mixed frame.
    for name in names:
        assert np.isfinite(result[name].to_numpy()).all(), name
    return result, cats


dense.frame = frame


def full_projection_replay(day, partition):
    count = len(np.load(dense.folder(day) / ('train.npy' if partition == 'train' else 'held.npy')))
    left_hash, right_hash = hashlib.sha256(), hashlib.sha256()
    primary, independent = dense.dictionaries(day), dense.dictionaries(day, True)
    for start in range(0, count, 20000):
        chosen = np.arange(start, min(start + 20000, count))
        left, cats = frame(dense.folder(day), partition, primary, chosen)
        right, other = frame(dense.folder(day, True), partition, independent, chosen, True)
        assert cats == other and list(left) == list(right)
        a, b = left.to_numpy(dtype='float32'), right.to_numpy(dtype='float32')
        np.testing.assert_array_equal(a, b)
        left_hash.update(a.tobytes(order='C'))
        right_hash.update(b.tobytes(order='C'))
        del left, right, a, b
    assert left_hash.hexdigest() == right_hash.hexdigest()
    emit('context_no_ids_full_projection_replayed', day=day, partition=partition, rows=count, columns=COLUMNS)
    return dict(rows=count, columns=COLUMNS, row_major_float32_sha256=left_hash.hexdigest(),
        independent_raw_source_dictionary_fit_mapping_and_column_gather_bitwise_exact=True)


def native_preflight():
    labels = np.load(dense.folder(15) / 'labels.npy')
    positives, negatives = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    chosen = np.sort(np.r_[positives[np.linspace(0, len(positives)-1, 512, dtype='int64')],
        negatives[np.linspace(0, len(negatives)-1, 3584, dtype='int64')]])
    data, cats = frame(dense.folder(15), 'train', dense.dictionaries(15), chosen)
    other, other_cats = frame(dense.folder(15, True), 'train', dense.dictionaries(15, True), chosen, True)
    assert cats == other_cats and list(data) == list(other)
    np.testing.assert_array_equal(data.to_numpy(), other.to_numpy())
    held, _ = frame(dense.folder(15), 'held', dense.dictionaries(15), np.arange(1000))
    held_other, _ = frame(dense.folder(15, True), 'held', dense.dictionaries(15, True), np.arange(1000), True)
    np.testing.assert_array_equal(held.to_numpy(), held_other.to_numpy())
    records = []
    for arm in ARMS:
        params = dict(dense.parameters(arm), min_data_in_leaf=16, min_data_per_group=4,
                      lambda_l2=4., cat_l2=4., cat_smooth=4.)
        dataset = lgb.Dataset(data, label=labels[chosen], categorical_feature=cats, params=params)
        def callback(environment):
            trees = environment.iteration + 1
            if trees in (5, 12, 20):
                environment.model.save_model(dense.native_path(OUT / f'{arm["name"]}_preflight_trees{trees}.txt'), num_iteration=trees)
        model = lgb.train(params, dataset, num_boost_round=20, callbacks=[callback], keep_training_booster=True)
        maximum = 0.
        for trees in (5, 12, 20):
            restored = lgb.Booster(model_file=dense.native_path(OUT / f'{arm["name"]}_preflight_trees{trees}.txt'))
            assert restored.current_iteration() == trees and restored.feature_name() == list(data)
            for partition, left, right in [('train', data, other), ('held_features_only', held, held_other)]:
                expected = model.predict(left, raw_score=True, num_iteration=trees, num_threads=2)
                actual = restored.predict(right, raw_score=True, num_threads=2)
                np.testing.assert_array_equal(expected, actual)
                maximum = max(maximum, float(np.max(np.abs(expected-actual))))
            del restored
        records.append(dict(arm, saved_tree_counts=[5,12,20], serialization_maximum_error=maximum,
                            native_categorical_splits=dense.categorical_splits(model)))
        del model, dataset
    assert not torch.cuda.is_initialized()
    write(OUT / 'preflight.json', dict(status='passed', rows=len(data), columns=COLUMNS,
        removed_features=REMOVE, retained_categories=cats, arms=records, device='CPU',
        all_complete_training_and_held_feature_projections_independently_replayed=True,
        held_probe_labels_read=False, inferred17_labels_read=False, gpu_memory_allocated=False,
        production_capacity_proven=False, protocol_sha256=digest(OUT / 'protocol.json'),
        input_lock_sha256=digest(OUT / 'inputs_locked.json'),
        full_projection_verification_sha256=digest(OUT / 'full_projection_verification.json')))
    dense.shared.engine.check()
    emit('context_no_ids_preflight_passed', rows=len(data), columns=COLUMNS)


def prepare():
    assert not OUT.exists(), 'Preserve all frozen runs, sources, models and failures'
    assert not torch.cuda.is_initialized()
    diag = read(DIAG)
    assert diag['status'] == 'checked' and diag['proposed_removed_features'] == REMOVE
    assert len(diag['models']) == 2 and not diag['held_labels_read']
    original = read(PARENT / 'protocol.json')
    original_inputs = read(PARENT / 'inputs_locked.json')
    assert read(PARENT / 'preflight.json')['status'] == 'passed'
    replay = read(PARENT / 'full_projection_verification.json')
    assert replay['status'] == 'passed' and replay['all_full_training_and_held_rows_replayed']
    assert read(BASE / 'verification.json')['status'] == 'passed'
    hashes = original['raw_datasets_sha256']
    assert read(BASE / 'protocol.json')['raw_datasets_sha256'] == hashes
    paths = [Path(__file__), DIAG, HERE / 'diagnose_dense_context_feature_use_v1.py']
    paths += [HERE / name for name in ('run_full_negative_context_no_ids_lgb_v1.ps1',
        'publish_progress_v5.py', 'record_two_day_progress_v5.py')]
    for parent in (PARENT, BASE):
        protocol = read(parent / 'protocol.json')
        for filename, expected in protocol['sources_sha256'].items():
            assert digest(filename) == expected, filename
            paths.append(Path(filename))
        paths += [parent / 'protocol.json', parent / 'inputs_locked.json']
    paths += [PARENT / name for name in ('preflight.json', 'full_projection_verification.json')]
    paths += [BASE / 'verification.json']
    for day in (15,16):
        paths += [BASE / f'day{day}_{name}.npy' for name in ('indices','prediction')]
        for independent in (False,True):
            directory = dense.folder(day, independent)
            source = original_inputs[str(day)]['source_contract']
            assert read(directory / 'contract.json') == source
            assert digest(directory / 'features.npy') == source['matrix_sha256']
            for name, expected in source['files_sha256'].items():
                assert digest(directory / name) == expected, name
            paths += [directory / name for name in ('contract.json','features.npy',*source['files_sha256'])]
        assert original_inputs[str(day)]['dictionary_max_training_date'] < 20191100+day
    for filename, expected in hashes.items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    OUT.mkdir()
    for day in (15,16):
        for independent in (False,True):
            name = f'category_dictionaries_day{day}{"_independent" if independent else ""}.npz'
            expected = original_inputs[str(day)]['independent_dictionary_sha256' if independent else 'dictionary_sha256']
            assert digest(PARENT / name) == expected
            shutil.copy2(PARENT / name, OUT / name)
            assert digest(OUT / name) == expected
            paths.append(OUT / name)
        np.save(OUT / f'day{day}_indices.npy', np.load(BASE / f'day{day}_indices.npy'))
    write(OUT / 'inputs_locked.json', original_inputs)
    write(OUT / 'protocol.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        dates=[20191115,20191116], arms=ARMS, params=PARAMS, checkpoints=CHECKPOINTS,
        baseline=str(BASE), input_feature_columns=COLUMNS, source_context_columns=395,
        removed_raw_features=REMOVE, added_neural_features=0,
        retained_features='All original 371 numeric historical/context features and 18 remaining raw categories; no raw user/task/ad/city/device-name/device-size columns',
        rationale='Completed day15 native models concentrate splits on six raw categorical identifiers. This is an ablation hypothesis, not evidence of overfitting or improved AUC.',
        training='Every strictly earlier positive and negative row; no negative downsampling; CPU two threads; sequential after existing fit and cohort composition',
        dictionary_fit_uses_advertisement_labels=False,
        category_encoding=original['category_encoding'],
        evaluation_day_labels_fit=False, inferred17_labels_read=False,
        selection='All12 model checkpoints and both-date candidate hashes locked before full-cohort scoring;15/16 are repeatedly reused development dates',
        checkpoint_reliability=original['checkpoint_reliability'],
        raw_datasets_sha256=hashes, sources_sha256={str(path):digest(path) for path in dict.fromkeys(paths)}))
    projections = {}
    for day in (15,16):
        dense.validate_inputs(day)
        dense.validate_inputs(day,True)
        projections[str(day)] = {partition:full_projection_replay(day,partition) for partition in ('train','held')}
    write(OUT / 'full_projection_verification.json', dict(status='passed', dates=projections,
        all_full_training_and_held_rows_replayed=True, columns=COLUMNS, removed_features=REMOVE,
        advertisement_held_labels_read=False, inferred17_labels_read=False,
        protocol_sha256=digest(OUT / 'protocol.json')))
    native_preflight()


def fit():
    dense.shared.engine.check()
    assert read(OUT / 'preflight.json')['status'] == 'passed'
    assert not (OUT / 'models_locked.json').exists() and not list(OUT.glob('day*.txt'))
    records = []
    for day in (15,16):
        wait_for_memory()
        contract = dense.validate_inputs(day)
        data, cats = frame(dense.folder(day),'train',dense.dictionaries(day))
        labels = np.load(dense.folder(day) / 'labels.npy')
        dates = np.load(dense.folder(day) / 'dates.npy')[np.load(dense.folder(day) / 'train.npy')]
        assert np.all(dates < 20191100+day)
        assert len(labels) == contract['training_rows'] and array_hash(labels) == contract['training_labels_sha256']
        for arm in ARMS:
            emit('context_no_ids_lgb_fit_started',day=day,arm=arm['name'],rows=len(data),columns=COLUMNS)
            params = dense.parameters(arm)
            dataset = lgb.Dataset(data,label=labels,categorical_feature=cats,params=params,free_raw_data=True)
            def callback(environment):
                trees = environment.iteration+1
                if trees in CHECKPOINTS:
                    path = OUT / f'day{day}_{arm["name"]}_trees{trees}.txt'
                    assert not path.exists()
                    write(OUT / 'checkpoint_save_journal.json',dict(stage='before_native_save',day=day,arm=arm['name'],trees=trees,path=str(path)))
                    environment.model.save_model(dense.native_path(path),num_iteration=trees)
                    write(OUT / 'checkpoint_save_journal.json',dict(stage='native_save_complete',day=day,arm=arm['name'],trees=trees,path=str(path),sha256=digest(path)))
                    emit('context_no_ids_lgb_checkpoint_saved',day=day,arm=arm['name'],trees=trees,bytes=path.stat().st_size)
                if trees % 200 == 0:
                    emit('context_no_ids_lgb_iteration',day=day,arm=arm['name'],trees=trees)
            model = lgb.train(params,dataset,num_boost_round=max(CHECKPOINTS),callbacks=[callback],keep_training_booster=True)
            for trees in CHECKPOINTS:
                name = f'day{day}_{arm["name"]}_trees{trees}'
                path = OUT / (name+'.txt')
                restored = lgb.Booster(model_file=dense.native_path(path))
                assert restored.current_iteration() == trees and restored.feature_name() == list(data)
                np.testing.assert_array_equal(model.predict(data.iloc[:2000],raw_score=True,num_iteration=trees,num_threads=2),
                    restored.predict(data.iloc[:2000],raw_score=True,num_threads=2))
                records.append(dict(arm,day=day,trees=trees,name=name,model_sha256=digest(path),
                    max_training_date=int(dates.max()),training_rows=len(labels),training_labels_sha256=array_hash(labels),
                    serialization_maximum_error=0.,dictionary_sha256=digest(dense.dictionary_path(day))))
                del restored
            del model,dataset
            gc.collect()
            emit('context_no_ids_lgb_fit_complete',day=day,arm=arm['name'])
        del data,labels,dates
        gc.collect()
    write(OUT / 'models_locked.json',dict(models=records,protocol_sha256=digest(OUT / 'protocol.json'),
        inputs_locked_sha256=digest(OUT / 'inputs_locked.json')))
    dense.shared.engine.check()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=['prepare','fit','evaluate','verify'])
    phase = parser.parse_args().phase
    if phase == 'evaluate':
        dense.shared.engine.evaluate()
    elif phase == 'verify':
        dense.shared.verify()
    else:
        globals()[phase]()
