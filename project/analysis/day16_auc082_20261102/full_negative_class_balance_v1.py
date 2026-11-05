"""Controlled class-weight experiments on independently verified full populations."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoost, Pool
from sklearn.metrics import roc_auc_score
import full_negative_context_v2 as engine
import full_negative_context_inputs_v1 as features
from full_population_original_v1 import wait_for_memory
from catboost_userday_residual import save_model, load_model
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash
from verify_stacker import rank_sum_auc

OUT = HERE / 'runs/full_negative_class_balance_v1'
SOURCE = HERE / 'runs/full_negative_context_v1'
VERIFIED = HERE / 'runs/full_negative_context_v2'
BASE = HERE / 'runs/compose_full_negative_masks_v1'
ARMS = [dict(name='natural_d6', depth=6, positive_weight=1.),
        dict(name='positive25_d6', depth=6, positive_weight=25.),
        dict(name='positive25_d4', depth=4, positive_weight=25.)]
CHECKPOINTS = [400, 1200, 3200]
PARAMS = dict(engine.PARAMS, random_seed=151701, gpu_ram_part=.55,
    gpu_cat_features_storage='CpuPinnedMemory', pinned_memory_size='1GB')
engine.OUT, engine.BASE, engine.ARMS = OUT, BASE, ARMS
engine.CHECKPOINTS, engine.PARAMS = CHECKPOINTS, PARAMS


def input_folder(day, independent=False):
    return (VERIFIED if independent else SOURCE) / f'{"independent" if independent else "input"}_day{day}'


def prepare():
    assert not OUT.exists(), 'Preserve all earlier fits and failures'
    paths = [Path(__file__)] + [HERE / name for name in (
        'run_full_negative_class_balance_v1.ps1', 'full_negative_context_v2.py',
        'full_negative_context_inputs_v1.py', 'full_population_original_v1.py',
        'catboost_userday_residual.py', 'cross_user_pair_ranker_v1.py',
        'quantile_blend_core_v1.py', 'conditional_daily.py', 'fit_stacker.py',
        'rich_userday_activity.py', 'verify_stacker.py')]
    hashes = read(VERIFIED / 'protocol.json')['raw_datasets_sha256']
    for run in (VERIFIED, BASE):
        report = read(run / 'verification.json')
        assert report['status'] == 'passed'
        assert read(run / 'protocol.json')['raw_datasets_sha256'] == hashes
        paths += [run / 'protocol.json', run / 'verification.json']
        for record in report['dates'].values():
            assert record['evaluation_day_labels_fit'] is False
        for filename, expected in read(run / 'protocol.json')['sources_sha256'].items():
            assert digest(filename) == expected, filename
            if Path(filename).suffix == '.py':
                paths.append(Path(filename))
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    contracts = {}
    for day in (15, 16):
        declared = None
        for independent in (False, True):
            folder = input_folder(day, independent)
            contract = read(folder / 'contract.json')
            if declared is None:
                declared = contract
            assert contract == declared
            assert digest(folder / 'features.npy') == contract['matrix_sha256']
            for filename, expected in contract['files_sha256'].items():
                assert digest(folder / filename) == expected, filename
            paths += [folder / name for name in ('contract.json', 'features.npy', *contract['files_sha256'])]
        folder = input_folder(day)
        rows, train, held = [np.load(folder / (name + '.npy')) for name in ('rows', 'train', 'held')]
        np.testing.assert_array_equal(rows[train], np.flatnonzero(raw.pt_d.to_numpy() < 20191100 + day))
        np.testing.assert_array_equal(rows[held], np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day))
        np.testing.assert_array_equal(np.load(folder / 'labels.npy'), raw.label.to_numpy()[rows[train]])
        np.testing.assert_array_equal(rows[held], np.load(BASE / f'day{day}_indices.npy'))
        assert declared['max_training_date'] < 20191100 + day
        assert declared['all_historical_negatives_retained'] and declared['own_held_labels_masked']
        paths += [BASE / f'day{day}_{name}.npy' for name in ('indices', 'prediction')]
        contracts[str(day)] = declared
    for filename, expected in hashes.items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    OUT.mkdir()
    write(OUT / 'protocol.json', dict(dates=[20191115, 20191116], arms=ARMS,
        checkpoints=CHECKPOINTS, params=PARAMS, baseline=str(BASE), input_feature_columns=395,
        training='Every strictly earlier positive and negative; no downsampling; constant class weights 1 and 25',
        rationale='Earlier full-population pointwise arms all used positive weight5; test natural and substantially increased positive weights with shallow regularization.',
        selection='All18 saved checkpoints and both-date candidates locked before scoring; dates repeatedly reused for development',
        inferred17_labels_read=False, evaluation_day_labels_fit=False,
        raw_datasets_sha256=hashes, sources_sha256={str(p): digest(p) for p in dict.fromkeys(paths)}))
    write(OUT / 'inputs_locked.json', contracts)
    for day in (15, 16):
        np.save(OUT / f'day{day}_indices.npy', np.load(BASE / f'day{day}_indices.npy'))
    folder = input_folder(15)
    labels, train = np.load(folder / 'labels.npy'), np.load(folder / 'train.npy')
    selected = np.sort(np.r_[np.flatnonzero(labels == 1)[-64:],
        np.flatnonzero(labels == 0)[np.linspace(0, int((labels == 0).sum()) - 1, 1024, dtype='int64')]])
    declared = contracts['15']['schema']
    matrix = np.load(folder / 'features.npy', mmap_mode='r')
    frame = pd.DataFrame(np.array(matrix[train[selected]]), columns=declared['columns'])
    for name in declared['categories']:
        frame[name] = frame[name].to_numpy(dtype='int64')
    reports = []
    for arm in ARMS:
        weights = np.where(labels[selected] == 1, arm['positive_weight'], 1.)
        pool = Pool(frame, label=labels[selected], weight=weights, cat_features=declared['categories'])
        np.testing.assert_array_equal(np.asarray(pool.get_weight()), weights)
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
        reports.append(dict(arm, native_gpu_parameter_preparation='passed',
            serialization_error=0., training_weight_values=np.unique(weights).tolist()))
    write(OUT / 'preflight.json', dict(status='passed', device='CPU', rows=len(frame), columns=395,
        arms=reports, held_labels_read=False, gpu_memory_allocated=False,
        full_populations_and_independent_feature_hashes_match=True,
        protocol_sha256=digest(OUT / 'protocol.json'), input_lock_sha256=digest(OUT / 'inputs_locked.json')))
    engine.check()
    emit('full_negative_class_balance_preflight_passed', rows=len(frame), arms=len(ARMS))


def fit():
    engine.check()
    assert read(OUT / 'preflight.json')['status'] == 'passed'
    assert not (OUT / 'models_locked.json').exists() and not list(OUT.glob('day*.cbm'))
    records = []
    for day in (15, 16):
        wait_for_memory()
        folder = input_folder(day)
        frame, cats = features.frame(folder, 'train')
        labels = np.load(folder / 'labels.npy')
        dates = np.load(folder / 'dates.npy')[np.load(folder / 'train.npy')]
        assert np.all(dates < 20191100 + day)
        for arm in ARMS:
            emit('full_negative_class_balance_fit_started', day=day, arm=arm['name'], rows=len(frame))
            weights = np.where(labels == 1, arm['positive_weight'], 1.)
            pool = Pool(frame, label=labels, weight=weights, cat_features=cats)
            assert len(pool.get_label()) == len(labels)
            model = CatBoost(dict(PARAMS, depth=arm['depth'], iterations=max(CHECKPOINTS)))
            model.fit(pool)
            del pool
            for trees in CHECKPOINTS:
                name = f'day{day}_{arm["name"]}_trees{trees}'
                saved = model.copy()
                saved.shrink(trees)
                path = OUT / (name + '.cbm')
                save_model(saved, path)
                records.append(dict(arm, day=day, trees=trees, name=name,
                    model_sha256=digest(path), max_training_date=int(dates.max())))
                del saved
            del model, weights
            gc.collect()
            emit('full_negative_class_balance_fit_complete', day=day, arm=arm['name'])
        del frame, labels, dates
        gc.collect()
    write(OUT / 'models_locked.json', dict(models=records, protocol_sha256=digest(OUT / 'protocol.json'),
        inputs_locked_sha256=digest(OUT / 'inputs_locked.json')))


def forecasts(day, independent=False):
    folder = input_folder(day, independent)
    declared = read(folder / 'contract.json')
    assert declared == read(OUT / 'inputs_locked.json')[str(day)]
    assert digest(folder / 'features.npy') == declared['matrix_sha256']
    frame, cats = features.frame(folder, 'held')
    pool = Pool(frame, cat_features=cats)
    lock = read(OUT / 'models_locked.json')
    assert lock['protocol_sha256'] == digest(OUT / 'protocol.json')
    assert lock['inputs_locked_sha256'] == digest(OUT / 'inputs_locked.json')
    margins = {}
    for record in lock['models']:
        if record['day'] != day:
            continue
        path = OUT / (record['name'] + '.cbm')
        assert digest(path) == record['model_sha256']
        assert record['max_training_date'] < 20191100 + day
        model = load_model(path)
        assert model.feature_names_ == list(frame)
        margin = np.asarray(model.predict(pool, prediction_type='RawFormulaVal', thread_count=2))
        assert margin.shape == (len(frame),) and np.isfinite(margin).all()
        target = OUT / (record['name'] + '_held_margin.npy')
        if independent:
            np.testing.assert_array_equal(margin, np.load(target))
        else:
            np.save(target, margin)
        margins[record['name']] = margin
    return margins


engine.forecasts = forecasts


def verify():
    engine.check()
    assert not (OUT / 'verification.json').exists()
    locked = read(OUT / 'predictions_locked.json')
    assert locked['models_locked_sha256'] == digest(OUT / 'models_locked.json')
    # The independently rebuilt upstream matrices cover both complete populations.
    checked = {day: forecasts(day, True) for day in (15, 16)}
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    reports = {}
    for day in (15, 16):
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day)
        np.testing.assert_array_equal(rows, np.load(OUT / f'day{day}_indices.npy'))
        labels = raw.label.to_numpy()[rows]
        assert len(rows) == {15: 424780, 16: 421983}[day] and labels.sum() == {15: 3523, 16: 3724}[day]
        records, best_auc, maximum = [], -1., 0.
        for (record, probability), (other, replay) in zip(engine.candidates(day, checked[day]), engine.candidates(day, checked[day], True)):
            assert record == other and array_hash(probability) == locked['dates'][str(day)][engine.key(record)]
            error = float(np.max(np.abs(probability - replay)))
            assert error < 2e-13
            maximum = max(maximum, error)
            assert np.isfinite(probability).all() and np.all((probability > 0) & (probability < 1))
            auc = rank_sum_auc(labels, probability)
            record = dict(record, auc=auc)
            records.append(record)
            if auc > best_auc:
                best_auc, best, best_prediction = auc, record, probability.copy()
        assert abs(roc_auc_score(labels, best_prediction) - best_auc) < 1e-12
        np.save(OUT / f'day{day}_prediction.npy', best_prediction)
        reports[str(20191100 + day)] = dict(auc=best_auc, best=best, records=records,
            rows=len(rows), clicks=int(labels.sum()), evaluation_day_labels_fit=False,
            inferred17_labels_read=False, held_labels_used_to_select_fixed_recipe=True,
            all_both_date_candidates_locked_before_scoring=True,
            complete_historical_negative_population=True,
            upstream_complete_features_independently_rebuilt=True,
            independent_all_saved_models_and_candidates_replayed=True,
            maximum_independent_prediction_error=maximum,
            target_met=best_auc >= {15: .815, 16: .82}[day])
        emit('full_negative_class_balance_scored', day=day, best=best)
    engine.check()
    write(OUT / 'verification.json', dict(status='passed', dates=reports,
        original_datasets_unchanged=True, both_targets_met=all(r['target_met'] for r in reports.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['prepare', 'fit', 'evaluate', 'verify'])
    phase = parser.parse_args().phase
    engine.evaluate() if phase == 'evaluate' else globals()[phase]()
