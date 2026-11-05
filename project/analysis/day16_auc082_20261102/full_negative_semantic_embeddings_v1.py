"""Controlled ablation of user-identity neural features in full-negative trees."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'
import argparse
import gc
import hashlib
from pathlib import Path
import numpy as np
import torch
from catboost import CatBoost, Pool
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from catboost_userday_residual import load_model, save_model
import full_negative_class_balance_v1 as experiment
import frozen_neural_embedding_inputs_v2 as source
import semantic_embedding_inputs_v1 as features

OUT = HERE / 'runs/full_negative_semantic_embeddings_v1'
BASE = HERE / 'runs/compose_neural_embedding_masks_v1'
FULL = source.OUT
ARMS = [dict(name='semantic_d4', depth=4, positive_weight=5.),
        dict(name='semantic_d6', depth=6, positive_weight=5.)]
CHECKPOINTS = [400, 1200, 2400]
PARAMS = read(FULL / 'protocol.json')['params']
experiment.OUT, experiment.BASE, experiment.ARMS = OUT, BASE, ARMS
experiment.CHECKPOINTS, experiment.PARAMS = CHECKPOINTS, PARAMS
experiment.features, experiment.input_folder = features, source.folder
experiment.engine.OUT, experiment.engine.BASE, experiment.engine.ARMS = OUT, BASE, ARMS
experiment.engine.CHECKPOINTS, experiment.engine.PARAMS = CHECKPOINTS, PARAMS
experiment.emit = lambda name, **values: emit(name.replace('class_balance', 'semantic_embeddings'), **values)


def verify_full_projection(day, partition):
    positions = np.load(source.folder(day) / ('train.npy' if partition == 'train' else 'held.npy'))
    primary_hash, independent_hash = hashlib.sha256(), hashlib.sha256()
    for first in range(0, len(positions), 20000):
        chosen = np.arange(first, min(first + 20000, len(positions)))
        primary, cats = features.frame(source.folder(day), partition, chosen)
        independent, other_cats = features.frame(source.folder(day, True), partition, chosen)
        assert cats == other_cats and list(primary) == list(independent)
        left = primary.to_numpy(dtype='float32')
        right = independent.to_numpy(dtype='float32')
        np.testing.assert_array_equal(left, right)
        primary_hash.update(left.tobytes(order='C'))
        independent_hash.update(right.tobytes(order='C'))
        del primary, independent, left, right
    assert primary_hash.hexdigest() == independent_hash.hexdigest()
    gc.collect()
    emit('semantic_full_projection_replayed', day=day, partition=partition, rows=len(positions), columns=489)
    return dict(rows=len(positions), columns=489, row_major_float32_sha256=primary_hash.hexdigest(),
        independent_row_block_and_column_gather_bitwise_exact=True)


def prepare():
    assert not OUT.exists(), 'Preserve all previous models, sources and failures'
    assert not torch.cuda.is_initialized()
    full_report = read(FULL / 'verification.json')
    assert full_report['status'] == 'passed'
    assert all(r['independent_all_saved_models_and_candidates_replayed'] for r in full_report['dates'].values())
    diagnostic = HERE / 'neural_embedding_usage_day15_v1.json'
    assert read(diagnostic)['status'] == 'passed'
    paths = [Path(__file__)] + [HERE / name for name in (
        'run_full_negative_semantic_embeddings_v1.ps1', 'semantic_embedding_inputs_v1.py',
        'full_negative_class_balance_v1.py', 'full_negative_context_v2.py',
        'frozen_neural_embedding_inputs_v2.py', 'catboost_userday_residual.py',
        'full_population_original_v1.py', 'cross_user_pair_ranker_v1.py',
        'quantile_blend_core_v1.py', 'conditional_daily.py', 'fit_stacker.py',
        'rich_userday_activity.py', 'verify_stacker.py')]
    paths.append(diagnostic)
    hashes = read(FULL / 'protocol.json')['raw_datasets_sha256']
    for run in (FULL, BASE):
        assert read(run / 'verification.json')['status'] == 'passed'
        protocol = read(run / 'protocol.json')
        assert protocol['raw_datasets_sha256'] == hashes
        for path, expected in protocol['sources_sha256'].items():
            assert digest(path) == expected, path
            if Path(path).suffix == '.py':
                paths.append(Path(path))
        paths += [run / 'protocol.json', run / 'verification.json']
    contracts = read(FULL / 'inputs_locked.json')
    designs = {}
    for day in (15, 16):
        contract = contracts[str(day)]
        kept, omitted = features.selected_columns(contract['schema']['columns'])
        assert contract['all_historical_negatives_retained'] and contract['own_held_labels_masked']
        assert contract['max_training_date'] < 20191100 + day
        designs[str(day)] = dict(columns=kept, omitted=omitted, categories=contract['schema']['categories'],
            source_fused_columns=533, selected_fused_columns=489, retained_embedding_columns=94,
            direct_user_embedding_and_user_interactions_excluded=True)
        for independent in (False, True):
            directory = source.folder(day, independent)
            assert read(directory / 'contract.json') == contract
            assert digest(directory / 'features.npy') == contract['matrix_sha256']
            for name, expected in contract['files_sha256'].items():
                assert digest(directory / name) == expected, name
            context = source.context_folder(day, independent)
            assert read(context / 'contract.json') == contract['context_contract']
            assert digest(context / 'features.npy') == contract['context_matrix_sha256']
            paths += [directory / name for name in ('contract.json', 'features.npy', *contract['files_sha256'])]
            paths += [context / 'contract.json', context / 'features.npy']
        for run in (FULL, BASE):
            paths += [run / f'day{day}_{name}.npy' for name in ('prediction', 'indices')]
        np.testing.assert_array_equal(np.load(FULL / f'day{day}_indices.npy'), np.load(BASE / f'day{day}_indices.npy'))
    for name, expected in hashes.items():
        assert digest(ROOT / '2026ml' / name) == expected, name
    OUT.mkdir()
    write(OUT / 'protocol.json', dict(dates=[20191115, 20191116], arms=ARMS,
        checkpoints=CHECKPOINTS, params=PARAMS, baseline=str(BASE), full_embedding_control=str(FULL),
        input_feature_columns=489, unchanged_source_fused_columns=533, source_added_features_stored_columns=138,
        retained_neural_features=94, omitted_user_neural_features=44,
        feature_designs=designs,
        hypothesis='Direct user embeddings receive over half of model importance while full-source standalone AUC falls to .745/.763; remove the44 direct user features to test overfitting while keeping candidate/history semantics.',
        controlled_training='Same full-negative populations, fixed teachers, positive weight5, seeds, CatBoost parameters, depths and checkpoints as the full533-column experiment; only44 declared user-neural features removed.',
        source_contracts='inputs_locked.json preserves the unchanged full533-column source contracts; feature_designs declares the489-column production projection.',
        training_embeddings_fit_on_outer_training_labels=True,
        historical_embeddings_are_not_per_date_out_of_time=True,
        evaluation_day_labels_fit=False, inferred17_labels_read=False,
        independent_assembly='Row-block gathering versus column-wise gathering from independently rebuilt complete sources',
        selection='All12 checkpoints and both-date candidate hashes frozen before complete-cohort scoring; development dates repeatedly reused',
        raw_datasets_sha256=hashes, sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)}))
    write(OUT / 'inputs_locked.json', contracts)
    for day in (15, 16):
        np.save(OUT / f'day{day}_indices.npy', np.load(BASE / f'day{day}_indices.npy'))
    projections = {str(day): {partition: verify_full_projection(day, partition)
        for partition in ('train', 'held')} for day in (15, 16)}
    write(OUT / 'full_projection_verification.json', dict(status='passed', dates=projections,
        all_full_training_and_held_rows_replayed=True, columns=489,
        advertisement_held_labels_read=False, inferred17_labels_read=False,
        protocol_sha256=digest(OUT / 'protocol.json')))
    labels = np.load(source.folder(15) / 'labels.npy')
    selected = np.sort(np.r_[np.flatnonzero(labels == 1)[-64:],
        np.flatnonzero(labels == 0)[np.linspace(0, int((labels == 0).sum()) - 1, 1024, dtype='int64')]])
    frame, categories = features.frame(source.folder(15), 'train', selected)
    independent, other_categories = features.frame(source.folder(15, True), 'train', selected)
    assert categories == other_categories and list(frame) == list(independent) == designs['15']['columns']
    np.testing.assert_array_equal(frame.to_numpy(), independent.to_numpy())
    full_frame, _ = source.frame(source.folder(15), 'train', selected)
    np.testing.assert_array_equal(frame.to_numpy(), full_frame[designs['15']['columns']].to_numpy())
    pool = Pool(frame, label=labels[selected], weight=np.where(labels[selected] == 1, 5., 1.), cat_features=categories)
    reports = []
    for arm in ARMS:
        gpu_probe = CatBoost(dict(PARAMS, iterations=20, depth=arm['depth']))
        gpu_probe._prepare_train_params(pool)
        parameters = {key: value for key, value in PARAMS.items() if key not in
            ('devices', 'gpu_ram_part', 'gpu_cat_features_storage', 'pinned_memory_size')}
        parameters.update(task_type='CPU', iterations=20, depth=arm['depth'], verbose=False)
        model = CatBoost(parameters)
        model.fit(pool)
        path = OUT / (arm['name'] + '_preflight.cbm')
        save_model(model, path)
        np.testing.assert_array_equal(model.predict(pool, prediction_type='RawFormulaVal'),
            load_model(path).predict(pool, prediction_type='RawFormulaVal', thread_count=2))
        reports.append(dict(arm, native_gpu_parameter_preparation='passed', serialization_error=0.))
    assert not torch.cuda.is_initialized()
    write(OUT / 'preflight.json', dict(status='passed', rows=len(frame), columns=489,
        source_columns=533, omitted_user_neural_columns=44, retained_neural_columns=94,
        arms=reports, device='CPU', gpu_memory_allocated=False,
        independent_projection_exact=True, projection_matches_explicit_full_source_columns=True,
        all_full_training_and_held_projection_rows_independently_replayed=True,
        full_projection_verification_sha256=digest(OUT / 'full_projection_verification.json'),
        actual_full_schema_native_fits_and_save_reload_passed=True,
        production_gpu_capacity_proven=False, evaluation_day_labels_fit=False, inferred17_labels_read=False,
        protocol_sha256=digest(OUT / 'protocol.json'), input_lock_sha256=digest(OUT / 'inputs_locked.json')))
    experiment.engine.check()
    emit('full_negative_semantic_embedding_preflight_passed', rows=len(frame), columns=489)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['prepare', 'fit', 'evaluate', 'verify'])
    phase = parser.parse_args().phase
    if phase == 'prepare':
        prepare()
    elif phase == 'evaluate':
        experiment.engine.evaluate()
    else:
        getattr(experiment, phase)()
