"""Complete-negative row residuals around strictly historical hierarchical priors."""
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
from full_negative_prior_offset_core_v1 import FIELDS, build as build_prior
from full_population_original_v1 import wait_for_memory
from catboost_userday_residual import save_model, load_model
from full_quality_row_catboost import recency_weights
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash
from verify_stacker import rank_sum_auc

OUT = HERE / 'runs/full_negative_prior_offset_v1'
SOURCE = HERE / 'runs/full_negative_context_v1'
VERIFIED = HERE / 'runs/full_negative_context_v2'
BASE = HERE / 'runs/compose_full_negative_masks_v1'
ARMS = [dict(name='prior_offset_all_d4', depth=4, half_life=None),
        dict(name='prior_offset_recent2_d6', depth=6, half_life=2.)]
CHECKPOINTS = [400, 1200, 2400]
PARAMS = dict(engine.PARAMS, l2_leaf_reg=80., random_seed=151691,
    gpu_ram_part=.55, gpu_cat_features_storage='CpuPinnedMemory', pinned_memory_size='1GB')
engine.OUT, engine.BASE, engine.ARMS, engine.PARAMS = OUT, BASE, ARMS, PARAMS


def offsets(day, independent=False):
    folder = (VERIFIED if independent else SOURCE) / f'{"independent" if independent else "input"}_day{day}'
    declared = read(folder / 'contract.json')
    arrays = {name: np.load(folder / (name + '.npy')) for name in ('rows', 'train', 'held', 'labels')}
    metadata = pd.read_csv(ROOT / '2026ml/train.csv', usecols=FIELDS, dtype='int32')
    dates = metadata.pt_d.to_numpy()
    np.testing.assert_array_equal(arrays['rows'][arrays['train']], np.flatnonzero(dates < 20191100 + day))
    np.testing.assert_array_equal(arrays['rows'][arrays['held']], np.flatnonzero(dates == 20191100 + day))
    local = metadata.iloc[arrays['rows']][FIELDS].reset_index(drop=True)
    hidden = np.full(len(local), -1, dtype='int32')
    hidden[arrays['train']] = arrays['labels']
    result, audit = build_prior(local, hidden, 20191100 + day, independent)
    contract = dict(rows_sha256=array_hash(arrays['rows']), training_rows_sha256=array_hash(arrays['rows'][arrays['train']]),
        training_labels_sha256=array_hash(arrays['labels']), offset_sha256=array_hash(result),
        metadata_sha256=array_hash(local.to_numpy(dtype='int32')), shape=list(result.shape),
        audit=audit, max_training_date=declared['max_training_date'], positive_weight=5.,
        every_own_date_prior_excludes_own_labels=True, inferred17_labels_read=False)
    assert all(r['max_label_date'] is None or r['max_label_date'] < r['query_date'] for r in audit)
    emit('full_negative_prior_built', day=day, independent=independent, rows=len(result))
    return result, contract


def prepare():
    assert not OUT.exists(), 'Preserve earlier fits and failures'
    paths = [Path(__file__)] + [HERE / name for name in (
        'full_negative_prior_offset_core_v1.py', 'test_full_negative_prior_offset_core_v1.py',
        'run_full_negative_prior_offset_v1.ps1', 'full_negative_context_v2.py',
        'full_negative_context_inputs_v1.py', 'full_population_original_v1.py',
        'catboost_userday_residual.py', 'full_quality_row_catboost.py',
        'conditional_daily.py', 'fit_stacker.py', 'rich_userday_activity.py',
        'verify_stacker.py', 'cross_user_pair_ranker_v1.py', 'quantile_blend_core_v1.py')]
    hashes = read(VERIFIED / 'protocol.json')['raw_datasets_sha256']
    for run in (VERIFIED, BASE):
        report = read(run / 'verification.json')
        assert report['status'] == 'passed'
        assert read(run / 'protocol.json')['raw_datasets_sha256'] == hashes
        paths += [run / 'protocol.json', run / 'verification.json']
        for date, record in report['dates'].items():
            assert record['evaluation_day_labels_fit'] is False
        for filename, expected in read(run / 'protocol.json')['sources_sha256'].items():
            assert digest(filename) == expected, filename
            if Path(filename).suffix == '.py':
                paths.append(Path(filename))
    for day in (15, 16):
        for run, phase in ((SOURCE, 'input'), (VERIFIED, 'independent')):
            folder = run / f'{phase}_day{day}'
            declared = read(folder / 'contract.json')
            assert digest(folder / 'features.npy') == declared['matrix_sha256']
            for filename, expected in declared['files_sha256'].items():
                assert digest(folder / filename) == expected
            paths += [folder / n for n in ('contract.json', 'features.npy', *declared['files_sha256'])]
        np.testing.assert_array_equal(np.load(BASE / f'day{day}_indices.npy'), np.load(VERIFIED / f'day{day}_indices.npy'))
        paths += [BASE / f'day{day}_indices.npy', BASE / f'day{day}_prediction.npy']
    for filename, expected in hashes.items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    OUT.mkdir()
    write(OUT / 'protocol.json', dict(dates=[20191115,20191116], arms=ARMS, checkpoints=CHECKPOINTS,
        params=PARAMS, baseline=str(BASE), input_feature_columns=395,
        training='Every strictly earlier positive and negative; positive weight5; past-only per-row prior supplied as CatBoost baseline',
        prior='Hierarchical user/app/user-app/slot rates; each historical row also excludes its own date labels',
        prediction='Saved model residual plus explicit float32 per-row prior; no Pool baseline used during production prediction',
        selection='Both-date models and candidate hashes locked before full-cohort scoring; dates repeatedly reused',
        inferred17_labels_read=False, evaluation_day_labels_fit=False,
        raw_datasets_sha256=hashes, sources_sha256={str(p):digest(p) for p in dict.fromkeys(paths)}))
    contracts = {}
    for day in (15, 16):
        primary, contract = offsets(day)
        independent, other = offsets(day, True)
        assert contract == other
        np.testing.assert_array_equal(primary, independent)
        np.save(OUT / f'day{day}_offset.npy', primary)
        np.save(OUT / f'day{day}_independent_offset.npy', independent)
        np.save(OUT / f'day{day}_indices.npy', np.load(BASE / f'day{day}_indices.npy'))
        contracts[str(day)] = contract
    write(OUT / 'inputs_locked.json', contracts)
    # A real-data CPU fit checks native baseline semantics without allocating GPU memory.
    folder = SOURCE / 'input_day15'
    labels, train = np.load(folder / 'labels.npy'), np.load(folder / 'train.npy')
    chosen = np.sort(np.r_[np.flatnonzero(labels == 1)[-64:],
                          np.flatnonzero(labels == 0)[np.linspace(0, int((labels == 0).sum())-1, 1024, dtype='int64')]])
    declared = read(folder / 'contract.json')['schema']
    x = np.load(folder / 'features.npy', mmap_mode='r')
    frame = pd.DataFrame(np.array(x[train[chosen]]), columns=declared['columns'])
    for name in declared['categories']:
        frame[name] = frame[name].to_numpy(dtype='int64')
    prior = np.load(OUT / 'day15_offset.npy')[train[chosen]]
    pool = Pool(frame, label=labels[chosen], baseline=prior, cat_features=declared['categories'])
    params = {k:v for k,v in PARAMS.items() if k not in ('devices','gpu_ram_part','gpu_cat_features_storage','pinned_memory_size')}
    params.update(task_type='CPU', iterations=20, depth=4, verbose=False)
    model = CatBoost(params)
    gpu_probe = CatBoost(dict(PARAMS, iterations=20, depth=4))
    gpu_probe._prepare_train_params(pool)
    del gpu_probe
    model.fit(pool)
    plain = Pool(frame, cat_features=declared['categories'])
    residual = model.predict(plain, prediction_type='RawFormulaVal', thread_count=2)
    combined = model.predict(pool, prediction_type='RawFormulaVal', thread_count=2)
    np.testing.assert_allclose(combined, residual + prior.astype('float64'), rtol=0, atol=1e-13)
    save_model(model, OUT / 'preflight.cbm')
    np.testing.assert_array_equal(residual, load_model(OUT / 'preflight.cbm').predict(plain, prediction_type='RawFormulaVal', thread_count=2))
    write(OUT / 'preflight.json', dict(status='passed', device='CPU', rows=len(frame), columns=395,
        native_gpu_training_parameter_preparation='passed', gpu_memory_allocated=False,
        explicit_prediction_offset_semantics_verified=True, serialization_error=0.,
        primary_independent_full_population_offsets_exact=True, held_labels_read=False,
        protocol_sha256=digest(OUT / 'protocol.json'), input_lock_sha256=digest(OUT / 'inputs_locked.json')))
    engine.check()
    emit('full_negative_prior_preflight_passed', rows=len(frame), columns=395)


def fit():
    engine.check()
    assert read(OUT / 'preflight.json')['status'] == 'passed'
    assert not (OUT / 'models_locked.json').exists() and not list(OUT.glob('day*.cbm'))
    records = []
    for day in (15, 16):
        wait_for_memory()
        folder = SOURCE / f'input_day{day}'
        frame, cats = features.frame(folder, 'train')
        labels, train = np.load(folder / 'labels.npy'), np.load(folder / 'train.npy')
        dates = np.load(folder / 'dates.npy')[train]
        prior = np.load(OUT / f'day{day}_offset.npy')[train]
        assert np.all(dates < 20191100+day)
        for arm in ARMS:
            emit('full_negative_prior_fit_started', day=day, arm=arm['name'], rows=len(frame))
            weights = recency_weights(dates, 20191100+day, arm['half_life']) * np.where(labels == 1, 5., 1.)
            pool = Pool(frame, label=labels, weight=weights, baseline=prior, cat_features=cats)
            model = CatBoost(dict(PARAMS, depth=arm['depth'], iterations=max(CHECKPOINTS)))
            model.fit(pool)
            del pool
            for trees in CHECKPOINTS:
                name = f'day{day}_{arm["name"]}_trees{trees}'
                saved = model.copy(); saved.shrink(trees)
                path = OUT / (name + '.cbm'); save_model(saved, path)
                records.append(dict(arm, day=day, trees=trees, name=name, model_sha256=digest(path),
                    offset_sha256=digest(OUT / f'day{day}_offset.npy'), max_training_date=int(dates.max())))
                del saved
            del model, weights
            gc.collect()
            emit('full_negative_prior_fit_complete', day=day, arm=arm['name'])
        del frame, labels, train, dates, prior
        gc.collect()
    write(OUT / 'models_locked.json', dict(models=records, protocol_sha256=digest(OUT/'protocol.json'),
        inputs_locked_sha256=digest(OUT/'inputs_locked.json')))


def forecasts(day, independent=False):
    folder = (VERIFIED if independent else SOURCE) / f'{"independent" if independent else "input"}_day{day}'
    declared = read(folder / 'contract.json')
    assert digest(folder / 'features.npy') == declared['matrix_sha256']
    offset = np.load(OUT / f'day{day}_{"independent_" if independent else ""}offset.npy')
    assert array_hash(offset) == read(OUT/'inputs_locked.json')[str(day)]['offset_sha256']
    frame, cats = features.frame(folder, 'held')
    prior = offset[np.load(folder / 'held.npy')].astype('float64')
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
        assert digest(OUT / f'day{day}_offset.npy') == record['offset_sha256']
        model = load_model(path)
        assert model.feature_names_ == list(frame)
        margin = np.asarray(model.predict(pool, prediction_type='RawFormulaVal', thread_count=2)) + prior
        assert np.isfinite(margin).all()
        if independent:
            np.testing.assert_array_equal(margin, np.load(OUT/(record['name']+'_held_margin.npy')))
        else:
            np.save(OUT/(record['name']+'_held_margin.npy'), margin)
        margins[record['name']] = margin
        del model
    return margins


engine.forecasts = forecasts


def verify():
    engine.check()
    assert not (OUT / 'verification.json').exists()
    locked = read(OUT / 'predictions_locked.json')
    assert locked['models_locked_sha256'] == digest(OUT / 'models_locked.json')
    checked = {}
    for day in (15, 16):
        replay, contract = offsets(day, True)
        assert contract == read(OUT / 'inputs_locked.json')[str(day)]
        np.testing.assert_array_equal(replay, np.load(OUT / f'day{day}_offset.npy'))
        checked[day] = forecasts(day, True)
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d','label'], dtype='int32')
    reports = {}
    for day in (15, 16):
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
        np.testing.assert_array_equal(rows, np.load(OUT / f'day{day}_indices.npy'))
        labels = raw.label.to_numpy()[rows]
        assert len(rows) == {15:424780,16:421983}[day] and labels.sum() == {15:3523,16:3724}[day]
        records, best_auc, maximum = [], -1., 0.
        for (record, prediction), (other, replay) in zip(engine.candidates(day,checked[day]), engine.candidates(day,checked[day],True)):
            assert record == other and array_hash(prediction) == locked['dates'][str(day)][engine.key(record)]
            error = float(np.max(np.abs(prediction-replay)))
            assert error < 2e-13
            maximum = max(maximum,error)
            assert np.isfinite(prediction).all() and np.all((prediction > 0)&(prediction < 1))
            auc = rank_sum_auc(labels,prediction); record = dict(record,auc=auc); records.append(record)
            if auc > best_auc:
                best_auc,best,best_prediction = auc,record,prediction.copy()
        assert abs(roc_auc_score(labels,best_prediction)-best_auc) < 1e-12
        np.save(OUT / f'day{day}_prediction.npy', best_prediction)
        reports[str(20191100+day)] = dict(auc=best_auc,best=best,records=records,rows=len(rows),clicks=int(labels.sum()),
            evaluation_day_labels_fit=False,inferred17_labels_read=False,held_labels_used_to_select_fixed_recipe=True,
            all_both_date_candidates_locked_before_scoring=True,upstream_complete_features_independently_rebuilt=True,
            independent_full_population_prior_offsets_and_all_models_replayed=True,
            maximum_independent_prediction_error=maximum,target_met=best_auc >= {15:.815,16:.82}[day])
        emit('full_negative_prior_scored',day=day,best=best)
    engine.check()
    write(OUT/'verification.json',dict(status='passed',dates=reports,original_datasets_unchanged=True,
        both_targets_met=all(r['target_met'] for r in reports.values())))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True,choices=['prepare','fit','evaluate','verify'])
    phase=parser.parse_args().phase
    engine.evaluate() if phase == 'evaluate' else globals()[phase]()
