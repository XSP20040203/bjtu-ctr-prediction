"""Complete historical negatives and frozen neural features in leaf-wise trees."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash, lgb
from full_population_original_v1 import wait_for_memory
import full_negative_class_balance_v1 as shared
import frozen_neural_embedding_inputs_v2 as inputs

OUT = HERE / 'runs/full_negative_embedding_lgb_v1'
BASE = HERE / 'runs/full_negative_prior_offset_v1'
ARMS = [dict(name='embedding_leaf31', num_leaves=31, min_data_in_leaf=128, extra_trees=False),
        dict(name='embedding_random63', num_leaves=63, min_data_in_leaf=128, extra_trees=True)]
CHECKPOINTS = [400, 800, 1600]
PARAMS = dict(objective='binary', learning_rate=.025, lambda_l2=40., lambda_l1=0.,
    scale_pos_weight=5., max_bin=63, feature_fraction=.8,
    bagging_fraction=.8, bagging_freq=1, cat_l2=40., cat_smooth=40.,
    max_cat_threshold=64, min_data_per_group=50, num_threads=2,
    seed=151721, verbosity=-1, deterministic=True, force_col_wise=True,
    feature_pre_filter=False, metric='None', device_type='cpu')
shared.OUT, shared.BASE, shared.CHECKPOINTS = OUT, BASE, CHECKPOINTS
shared.engine.OUT, shared.engine.BASE, shared.engine.CHECKPOINTS = OUT, BASE, CHECKPOINTS
shared.emit = lambda name, **values: emit(name.replace('class_balance', 'embedding_lgb'), **values)


def parameters(arm):
    return dict(PARAMS, **{key: value for key, value in arm.items() if key != 'name'})


def validate_inputs(day, independent=False):
    directory = inputs.folder(day, independent)
    contract = read(directory / 'contract.json')
    assert contract == read(OUT / 'inputs_locked.json')[str(day)]
    assert digest(directory / 'features.npy') == contract['matrix_sha256']
    context = inputs.context_folder(day, independent)
    assert digest(context / 'features.npy') == contract['context_matrix_sha256']
    assert read(context / 'contract.json') == contract['context_contract']
    for name, expected in contract['files_sha256'].items():
        assert digest(directory / name) == expected, name
    assert contract['fused_shape'][1] == 533
    assert contract['all_historical_negatives_retained'] and contract['own_held_labels_masked']
    assert contract['independent_raw_encoding_and_numpy_torch_features_exact']
    assert contract['max_training_date'] < 20191100 + day
    return contract


def prepare():
    assert not OUT.exists(), 'Preserve all prior sources, artifacts and failures'
    assert not torch.cuda.is_initialized()
    upstream = read(inputs.OUT / 'preflight.json')
    assert upstream['status'] == 'passed'
    assert upstream['all_full_population_embedding_features_rebuilt_independently_and_exact']
    assert digest(inputs.OUT / 'protocol.json') == upstream['protocol_sha256']
    assert digest(inputs.OUT / 'inputs_locked.json') == upstream['input_lock_sha256']
    assert read(BASE / 'verification.json')['status'] == 'passed'
    paths = [Path(__file__), HERE / 'run_full_negative_embedding_lgb_v1.ps1',
        HERE / 'full_negative_class_balance_v1.py', HERE / 'full_negative_context_v2.py',
        HERE / 'frozen_neural_embedding_inputs_v2.py', HERE / 'rich_userday_activity.py',
        HERE / 'full_population_original_v1.py', HERE / 'fit_stacker.py',
        HERE / 'conditional_daily.py', HERE / 'verify_stacker.py',
        HERE / 'cross_user_pair_ranker_v1.py', HERE / 'quantile_blend_core_v1.py',
        inputs.OUT / 'preflight.json', inputs.OUT / 'protocol.json',
        inputs.OUT / 'inputs_locked.json', BASE / 'protocol.json', BASE / 'verification.json']
    for source in (inputs.OUT, BASE):
        for path, expected in read(source / 'protocol.json')['sources_sha256'].items():
            assert digest(path) == expected, path
            paths.append(Path(path))
    hashes = read(BASE / 'protocol.json')['raw_datasets_sha256']
    for name, expected in hashes.items():
        assert digest(ROOT / '2026ml' / name) == expected, name
    contracts = read(inputs.OUT / 'inputs_locked.json')
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    for day in (15, 16):
        for independent in (False, True):
            directory = inputs.folder(day, independent)
            contract = read(directory / 'contract.json')
            assert contract == contracts[str(day)]
            assert digest(directory / 'features.npy') == contract['matrix_sha256']
            for name, expected in contract['files_sha256'].items():
                assert digest(directory / name) == expected, name
            context = inputs.context_folder(day, independent)
            assert digest(context / 'features.npy') == contract['context_matrix_sha256']
            assert read(context / 'contract.json') == contract['context_contract']
            paths += [directory / name for name in ('contract.json', 'features.npy', *contract['files_sha256'])]
            paths += [context / 'features.npy', context / 'contract.json']
        directory = inputs.folder(day)
        rows, training, held = [np.load(directory / (name + '.npy')) for name in ('rows', 'train', 'held')]
        np.testing.assert_array_equal(rows[training], np.flatnonzero(raw.pt_d.to_numpy() < 20191100 + day))
        np.testing.assert_array_equal(rows[held], np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day))
        np.testing.assert_array_equal(np.load(directory / 'labels.npy'), raw.label.to_numpy()[rows[training]])
        np.testing.assert_array_equal(rows[held], np.load(BASE / f'day{day}_indices.npy'))
        paths += [BASE / f'day{day}_{name}.npy' for name in ('indices', 'prediction')]
    del raw
    gc.collect()
    OUT.mkdir()
    write(OUT / 'protocol.json', dict(dates=[20191115, 20191116], arms=ARMS,
        checkpoints=CHECKPOINTS, params=PARAMS, baseline=str(BASE), input_feature_columns=533,
        training='All strictly earlier original positive and negative rows; no subsampling; positive class weight5',
        rationale='The previous LightGBM experiment used a one-third negative sample and423 features; this tests all negatives and138 frozen neural semantic features with395 verified context columns.',
        feature_teacher='Existing fixed final-epoch neural encoder fitted on all outer-training labels; historical embeddings are not per-date out-of-time',
        evaluation_day_labels_fit=False, inferred17_labels_read=False,
        selection='All12 models and both-date candidate hashes frozen before development scoring; both dates repeatedly reused',
        device='CPU', lightgbm_version=lgb.__version__,
        raw_datasets_sha256=hashes,
        sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)}))
    write(OUT / 'inputs_locked.json', contracts)
    for day in (15, 16):
        validate_inputs(day)
        validate_inputs(day, True)
        np.save(OUT / f'day{day}_indices.npy', np.load(BASE / f'day{day}_indices.npy'))
    labels = np.load(inputs.folder(15) / 'labels.npy')
    selected = np.sort(np.r_[np.flatnonzero(labels == 1)[-64:],
        np.flatnonzero(labels == 0)[np.linspace(0, int((labels == 0).sum()) - 1, 1024, dtype='int64')]])
    frame, categories = inputs.frame(inputs.folder(15), 'train', selected)
    independent, other_categories = inputs.frame(inputs.folder(15, True), 'train', selected)
    assert categories == other_categories and list(frame) == list(independent)
    np.testing.assert_array_equal(frame.to_numpy(), independent.to_numpy())
    reports = []
    for arm in ARMS:
        params = parameters(arm)
        params['min_data_in_leaf'] = 16
        dataset = lgb.Dataset(frame, label=labels[selected], categorical_feature=categories,
            params=params, free_raw_data=False)
        model = lgb.train(params, dataset, num_boost_round=20)
        path = OUT / (arm['name'] + '_preflight.txt')
        path.write_text(model.model_to_string(), encoding='utf8')
        restored = lgb.Booster(model_str=path.read_text(encoding='utf8'))
        before = model.predict(frame, raw_score=True, num_threads=2)
        after = restored.predict(independent, raw_score=True, num_threads=2)
        np.testing.assert_array_equal(before, after)
        assert model.num_trees() > 0 and np.isfinite(after).all()
        assert list(frame) == restored.feature_name()
        reports.append(dict(arm, serialization_error=0., real_rows=len(frame),
            preflight_min_data_in_leaf=16, production_min_data_in_leaf=arm['min_data_in_leaf']))
        del dataset, model, restored
        gc.collect()
    assert not torch.cuda.is_initialized()
    write(OUT / 'preflight.json', dict(status='passed', rows=len(frame), columns=533,
        arms=reports, device='CPU', gpu_memory_allocated=False,
        actual_fits_and_full_schema_serialization_passed=True,
        all_original_training_rows_and_labels_checked=True,
        complete_primary_and_independent_feature_hashes_match=True,
        full_cpu_capacity_proven=False, evaluation_day_labels_fit=False, inferred17_labels_read=False,
        protocol_sha256=digest(OUT / 'protocol.json'), input_lock_sha256=digest(OUT / 'inputs_locked.json')))
    shared.engine.check()
    emit('full_negative_embedding_lgb_preflight_passed', rows=len(frame), columns=533)


def fit():
    shared.engine.check()
    assert read(OUT / 'preflight.json')['status'] == 'passed'
    assert not (OUT / 'models_locked.json').exists() and not list(OUT.glob('day*.txt'))
    records = []
    for day in (15, 16):
        wait_for_memory()
        contract = validate_inputs(day)
        frame, categories = inputs.frame(inputs.folder(day), 'train')
        labels = np.load(inputs.folder(day) / 'labels.npy')
        dates = np.load(inputs.folder(day) / 'dates.npy')[np.load(inputs.folder(day) / 'train.npy')]
        assert np.all(dates < 20191100 + day)
        assert array_hash(labels) == contract['training_labels_sha256']
        assert len(labels) == contract['training_rows']
        for arm in ARMS:
            emit('full_negative_embedding_lgb_fit_started', day=day, arm=arm['name'], rows=len(frame))
            params = parameters(arm)
            dataset = lgb.Dataset(frame, label=labels, categorical_feature=categories,
                params=params, free_raw_data=True)
            def callback(environment):
                if (environment.iteration + 1) % 200 == 0:
                    emit('full_negative_embedding_lgb_iteration', day=day,
                        arm=arm['name'], trees=environment.iteration + 1)
            model = lgb.train(params, dataset, num_boost_round=max(CHECKPOINTS), callbacks=[callback])
            for trees in CHECKPOINTS:
                name = f'day{day}_{arm["name"]}_trees{trees}'
                path = OUT / (name + '.txt')
                path.write_text(model.model_to_string(num_iteration=trees), encoding='utf8')
                restored = lgb.Booster(model_str=path.read_text('utf8'))
                assert restored.current_iteration() == trees
                before = model.predict(frame.iloc[:2000], raw_score=True, num_iteration=trees, num_threads=2)
                after = restored.predict(frame.iloc[:2000], raw_score=True, num_threads=2)
                np.testing.assert_array_equal(before, after)
                records.append(dict(arm, day=day, trees=trees, name=name,
                    model_sha256=digest(path), max_training_date=int(dates.max()),
                    training_rows=len(labels), training_labels_sha256=array_hash(labels),
                    serialization_maximum_error=0.))
                del restored
            del model, dataset
            gc.collect()
            emit('full_negative_embedding_lgb_fit_complete', day=day, arm=arm['name'])
        del frame, labels, dates
        gc.collect()
    write(OUT / 'models_locked.json', dict(models=records, protocol_sha256=digest(OUT / 'protocol.json'),
        inputs_locked_sha256=digest(OUT / 'inputs_locked.json')))
    shared.engine.check()


def forecasts(day, independent=False):
    validate_inputs(day, independent)
    frame, categories = inputs.frame(inputs.folder(day, independent), 'held')
    locked = read(OUT / 'models_locked.json')
    assert locked['protocol_sha256'] == digest(OUT / 'protocol.json')
    assert locked['inputs_locked_sha256'] == digest(OUT / 'inputs_locked.json')
    margins = {}
    for record in locked['models']:
        if record['day'] != day:
            continue
        path = OUT / (record['name'] + '.txt')
        assert digest(path) == record['model_sha256']
        assert record['max_training_date'] < 20191100 + day
        model = lgb.Booster(model_str=path.read_text('utf8'))
        assert model.feature_name() == list(frame)
        margin = model.predict(frame, raw_score=True, num_threads=2)
        assert margin.shape == (len(frame),) and np.isfinite(margin).all()
        target = OUT / (record['name'] + '_held_margin.npy')
        if independent:
            np.testing.assert_array_equal(margin, np.load(target))
        else:
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
