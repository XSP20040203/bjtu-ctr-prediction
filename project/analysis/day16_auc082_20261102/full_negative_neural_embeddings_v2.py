"""Frozen neural semantic features followed by complete-negative row trees."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'
import argparse
from pathlib import Path
import numpy as np
import torch
from catboost import CatBoost, Pool
from fit_stacker import HERE, ROOT, digest, write
from conditional_daily import read, emit
from catboost_userday_residual import save_model, load_model
import full_negative_class_balance_v1 as experiment
import frozen_neural_embedding_inputs_v2 as inputs

OUT = inputs.OUT
BASE = HERE / 'runs/full_negative_prior_offset_v1'
ARMS = [dict(name='neural_embedding_d4', depth=4, positive_weight=5.),
        dict(name='neural_embedding_d6', depth=6, positive_weight=5.)]
CHECKPOINTS = [400, 1200, 2400]
PARAMS = dict(experiment.PARAMS, l2_leaf_reg=80., random_seed=151711, border_count=32)
# Reuse the frozen full-population fit/checkpoint/candidate/replay engine with
# explicit533-column inputs. No sources from the earlier experiment are edited.
experiment.OUT, experiment.BASE, experiment.ARMS = OUT, BASE, ARMS
experiment.CHECKPOINTS, experiment.PARAMS = CHECKPOINTS, PARAMS
experiment.features, experiment.input_folder = inputs, inputs.folder
experiment.engine.OUT, experiment.engine.BASE, experiment.engine.ARMS = OUT, BASE, ARMS
experiment.engine.CHECKPOINTS, experiment.engine.PARAMS = CHECKPOINTS, PARAMS
experiment.emit = lambda name, **values: emit(name.replace('class_balance', 'neural_embeddings'), **values)


def prepare():
    assert not OUT.exists(), 'Preserve earlier feature builds, fits and failures'
    assert not torch.cuda.is_initialized()
    native_preflight = HERE / 'runs/frozen_neural_teacher_preflight_v2/verification.json'
    assert read(native_preflight)['status'] == 'passed'
    for path, expected in read(native_preflight)['sources_sha256'].items():
        assert digest(path) == expected, path
    torch.set_num_threads(2)
    paths = [Path(__file__)] + [HERE / name for name in (
        'run_full_negative_neural_embeddings_v2.ps1', 'frozen_neural_embedding_core_v1.py',
        'test_frozen_neural_embedding_core_v1.py', 'frozen_neural_embedding_inputs_v2.py',
        'full_negative_class_balance_v1.py', 'full_negative_context_v2.py',
        'full_negative_context_inputs_v1.py', 'full_population_original_v1.py',
        'catboost_userday_residual.py', 'cross_user_pair_ranker_v1.py',
        'quantile_blend_core_v1.py', 'fit_stacker.py', 'conditional_daily.py',
        'rich_userday_activity.py', 'verify_stacker.py')]
    paths += [ROOT / name for name in (
        'analysis/architecture_goal_20261029/interaction_models.py',
        'analysis/model_comparison/id_regularized_models.py',
        'analysis/model_comparison/mainstream_models.py',
        'analysis/model_comparison/benchmark_core.py',
        'analysis/userday_aux_20261101/verification.json')]
    paths += [native_preflight, HERE / 'runs/full_negative_neural_embeddings_v1/failure.json',
              HERE / 'runs/full_negative_neural_embeddings_v1/protocol.json']
    hashes = read(BASE / 'protocol.json')['raw_datasets_sha256']
    for run in (BASE, inputs.INDEPENDENT):
        assert read(run / 'verification.json')['status'] == 'passed'
        protocol = read(run / 'protocol.json')
        assert protocol['raw_datasets_sha256'] == hashes
        paths += [run / 'protocol.json', run / 'verification.json']
        for path, expected in protocol['sources_sha256'].items():
            assert digest(path) == expected, path
            if Path(path).suffix == '.py':
                paths.append(Path(path))
    for day in (15, 16):
        root, cache = inputs.teacher_paths(day)
        paths += [root / name for name in ('feature_manifest.json', 'protocol.json',
            'both/metrics.json', 'both/final_model.pt')]
        paths += [cache / name for name in ('manifest.json', 'vocabs.json')]
        for path, expected in read(root / 'protocol.json')['sources'].items():
            assert digest(path) == expected, path
            paths.append(Path(path))
        for independent in (False, True):
            folder = inputs.context_folder(day, independent)
            contract = read(folder / 'contract.json')
            paths += [folder / name for name in ('contract.json', 'features.npy', *contract['files_sha256'])]
        paths += [BASE / f'day{day}_{name}.npy' for name in ('indices', 'prediction')]
    for name, expected in hashes.items():
        assert digest(ROOT / '2026ml' / name) == expected, name
    OUT.mkdir()
    write(OUT / 'protocol.json', dict(dates=[20191115, 20191116], arms=ARMS,
        repair='Read held counts from actual held.npy indices; preserve the v1 pre-feature failure; use32 numeric borders as in proven576-column GPU fits',
        checkpoints=CHECKPOINTS, params=PARAMS, baseline=str(BASE), input_feature_columns=533,
        original_context_columns=395, added_neural_embedding_columns=138,
        teacher='Existing fixed final-epoch protected GatedCross8 embedding tables, separately trained strictly before each prediction date',
        features='Six categorical embeddings, six masked-mean ad-history embeddings, four user/candidate products and ten dot products',
        canonical_tables='Original native encoder norm caps checked against independent NumPy reconstruction for every vocabulary ID; then frozen as constants',
        independent_features='Pandas ID maps/custom sequence parser/NumPy versus sorted ID searches/original sequence encoder/CPU Torch; every full-population row must match exactly',
        training='All earlier original positives and negatives; class weight5; no downsampling',
        training_embeddings_fit_on_outer_training_labels=True,
        historical_embeddings_are_not_per_date_out_of_time=True,
        evaluation_day_labels_fit=False, inferred17_labels_read=False,
        selection='All12 checkpoints and both-date candidate hashes frozen before complete-cohort development scoring; dates repeatedly reused',
        raw_datasets_sha256=hashes, sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)}))
    contracts = {}
    for day in (15, 16):
        contracts[str(day)] = inputs.build(day)
        np.testing.assert_array_equal(np.load(inputs.folder(day) / 'rows.npy')[np.load(inputs.folder(day) / 'held.npy')],
            np.load(BASE / f'day{day}_indices.npy'))
        np.save(OUT / f'day{day}_indices.npy', np.load(BASE / f'day{day}_indices.npy'))
    write(OUT / 'inputs_locked.json', contracts)
    labels = np.load(inputs.folder(15) / 'labels.npy')
    selected = np.sort(np.r_[np.flatnonzero(labels == 1)[-64:],
        np.flatnonzero(labels == 0)[np.linspace(0, int((labels == 0).sum()) - 1, 1024, dtype='int64')]])
    frame, cats = inputs.frame(inputs.folder(15), 'train', selected)
    independent, other_cats = inputs.frame(inputs.folder(15, True), 'train', selected)
    assert cats == other_cats and list(frame) == list(independent)
    np.testing.assert_array_equal(frame.to_numpy(), independent.to_numpy())
    reports = []
    pool = Pool(frame, label=labels[selected], weight=np.where(labels[selected] == 1, 5., 1.), cat_features=cats)
    for arm in ARMS:
        gpu_probe = CatBoost(dict(PARAMS, iterations=20, depth=arm['depth']))
        gpu_probe._prepare_train_params(pool)
        params = {key: value for key, value in PARAMS.items() if key not in
            ('devices', 'gpu_ram_part', 'gpu_cat_features_storage', 'pinned_memory_size')}
        params.update(task_type='CPU', iterations=20, depth=arm['depth'], verbose=False)
        model = CatBoost(params)
        model.fit(pool)
        path = OUT / (arm['name'] + '_preflight.cbm')
        save_model(model, path)
        np.testing.assert_array_equal(model.predict(pool, prediction_type='RawFormulaVal'),
            load_model(path).predict(pool, prediction_type='RawFormulaVal', thread_count=2))
        reports.append(dict(arm, native_gpu_parameter_preparation='passed', serialization_error=0.))
    assert not torch.cuda.is_initialized()
    write(OUT / 'preflight.json', dict(status='passed', device='CPU', rows=len(frame), columns=533,
        added_embedding_features=138, arms=reports, gpu_memory_allocated=False,
        held_labels_read_for_preflight_fit=False, inferred17_labels_read=False,
        all_full_population_embedding_features_rebuilt_independently_and_exact=True,
        actual_small_full_schema_fits_and_serialization_passed=True,
        full_gpu_capacity_proven=False,
        protocol_sha256=digest(OUT / 'protocol.json'), input_lock_sha256=digest(OUT / 'inputs_locked.json')))
    experiment.engine.check()
    emit('full_negative_neural_embedding_preflight_passed', rows=len(frame), columns=533)


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
