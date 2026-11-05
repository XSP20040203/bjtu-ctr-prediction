"""Full-negative row models retaining all original181 and context395 columns."""
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
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash
from catboost_userday_residual import save_model, load_model
from full_quality_row_catboost import recency_weights
from cross_user_pair_ranker_v1 import prediction as row_prediction
from quantile_blend_core_v1 import matched_logit, blend
from verify_stacker import rank_sum_auc
from full_rich_negative_inputs_v1 import compose, validate_alignment
from full_population_original_v1 import wait_for_memory

OUT = HERE / 'runs/full_rich_negative_v2'
PREFLIGHT = HERE / 'runs/full_rich_negative_preflight_v2'
CONTEXT = HERE / 'runs/full_negative_context_v1'
CONTEXT_VERIFIER = HERE / 'runs/full_negative_context_v2'
ORIGINAL = HERE / 'runs/full_population_original_v1'
BASE = HERE / 'runs/group_event_posterior_v1'
ARMS = [dict(name='rich_all_d6', depth=6, half_life=None),
        dict(name='rich_recent2_d8', depth=8, half_life=2.)]
CHECKPOINTS = [400, 1200, 2400]
WEIGHTS = [.025, .05, .1, .2, .35, .5, .75, 1.]
MODES = ['replace_global', 'add_within', 'rank_global']
PARAMS = dict(loss_function='Logloss', learning_rate=.025, l2_leaf_reg=40.,
    random_strength=1., bootstrap_type='Bernoulli', subsample=.8, border_count=32,
    one_hot_max_size=16, max_ctr_complexity=1, thread_count=2, random_seed=151671,
    task_type='GPU', devices='0', gpu_ram_part=.55, gpu_cat_features_storage='CpuPinnedMemory',
    pinned_memory_size='1GB', boosting_type='Plain', allow_writing_files=False, verbose=200)


def folders(day, independent=False):
    return ((CONTEXT_VERIFIER if independent else CONTEXT) / f'{"independent" if independent else "input"}_day{day}',
            ORIGINAL / f'{"independent" if independent else "input"}_day{day}')


def components(day, independent=False, hashes=True):
    left, right = folders(day, independent)
    context, original = read(left / 'contract.json'), read(right / 'contract.json')
    arrays = [{name: np.load(folder / (name + '.npy')) for name in ('rows', 'train', 'held', 'labels')}
              for folder in (left, right)]
    validate_alignment(*arrays)
    names = context['schema']['columns'] + original['feature_names']
    categories = context['schema']['categories'] + [original['feature_names'][i] for i in original['categorical_columns']]
    assert len(names) == len(set(names)) == 576 and len(categories) == len(set(categories)) == 39
    assert context['max_training_date'] == original['max_training_date'] < 20191100 + day
    assert context['all_historical_negatives_retained'] and original['all_prior_negative_rows_retained']
    assert context['own_held_labels_masked'] and original['held_labels_masked_in_retained_core']
    assert context['inferred17_labels_read'] is original['inferred17_labels_read'] is False
    for folder, declared in ((left, context), (right, original)):
        if hashes:
            assert digest(folder / 'features.npy') == declared['matrix_sha256']
            for name, expected in declared['files_sha256'].items():
                assert digest(folder / name) == expected, name
    matrix_left = np.load(left / 'features.npy', mmap_mode='r')
    matrix_right = np.load(right / 'features.npy', mmap_mode='r')
    assert list(matrix_left.shape) == context['shape'] and list(matrix_right.shape) == original['shape']
    contract = dict(context_contract=context, original_contract=original,
        feature_names=names, categories=categories, shape=[len(arrays[0]['rows']), len(names)],
        training_rows=len(arrays[0]['train']), held_rows=len(arrays[0]['held']),
        training_rows_sha256=array_hash(arrays[0]['rows'][arrays[0]['train']]),
        training_labels_sha256=array_hash(arrays[0]['labels']),
        max_training_date=context['max_training_date'], all_historical_negatives_retained=True,
        original181_and_context395_retained=True, inferred17_labels_read=False)
    return contract, arrays[0], matrix_left, matrix_right


def frame(day, partition, independent=False, selected=None):
    declared, arrays, left, right = components(day, independent)
    positions = arrays[partition]
    if selected is not None:
        positions = positions[selected]
    result = compose(left, right, positions, declared['feature_names'], declared['categories'], independent)
    emit('full_rich_frame_ready', day=day, partition=partition, independent=independent,
         rows=len(result), columns=len(result.columns))
    return result, declared['categories'], arrays


def preflight():
    assert not PREFLIGHT.exists(), 'Preserve previous preflights'
    labels = np.load(CONTEXT / 'input_day15/labels.npy')
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    selected = np.r_[positive[-64:], negative[np.linspace(0, len(negative) - 1, 1024, dtype='int64')]]
    primary, cats, arrays = frame(15, 'train', selected=selected)
    declared, _, left, right = components(15)
    replay = compose(left, right, arrays['train'][selected], declared['feature_names'], cats, True)
    pd.testing.assert_frame_equal(primary, replay)
    params = {k: v for k, v in PARAMS.items() if k not in ('devices', 'gpu_ram_part', 'gpu_cat_features_storage', 'pinned_memory_size')}
    params.update(task_type='CPU', iterations=20, depth=4, verbose=False)
    model = CatBoost(params)
    pool = Pool(primary, label=labels[selected], cat_features=cats)
    gpu_probe = CatBoost(dict(PARAMS, iterations=20, depth=4))
    gpu_probe._prepare_train_params(pool)
    del gpu_probe
    model.fit(pool)
    PREFLIGHT.mkdir()
    save_model(model, PREFLIGHT / 'model.cbm')
    first = model.predict(pool, prediction_type='RawFormulaVal', thread_count=2)
    second = load_model(PREFLIGHT / 'model.cbm').predict(Pool(replay, cat_features=cats), prediction_type='RawFormulaVal', thread_count=2)
    np.testing.assert_array_equal(first, second)
    dates = np.load(CONTEXT / 'input_day15/dates.npy')[arrays['train'][selected]]
    assert np.all(dates < 20191115)
    sources = [Path(__file__), HERE / 'full_rich_negative_inputs_v1.py',
               HERE / 'test_full_rich_negative_inputs_v1.py', HERE / 'run_full_rich_negative_v2.ps1']
    write(PREFLIGHT / 'verification.json', dict(status='passed', feature_only=True, rows=len(primary),
        columns=576, categories=len(cats), max_training_date=int(dates.max()), device='CPU',
        full_gpu_capacity_proven=False, held_labels_read=False, serialization_error=0.,
        gpu_native_training_parameter_preparation='passed',
        independent_join_exact=True, sources_sha256={str(path): digest(path) for path in sources}))
    emit('full_rich_real_preflight_passed', rows=len(primary), columns=576)


def check():
    protocol = read(OUT / 'protocol.json')
    for filename, expected in protocol['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    return protocol


def prepare():
    assert not OUT.exists(), 'Preserve completed or partial models'
    verified = read(PREFLIGHT / 'verification.json')
    assert verified['status'] == 'passed' and verified['independent_join_exact']
    paths = [Path(__file__), HERE / 'full_rich_negative_inputs_v1.py', HERE / 'test_full_rich_negative_inputs_v1.py',
        HERE / 'run_full_rich_negative_v2.ps1', PREFLIGHT / 'verification.json', PREFLIGHT / 'model.cbm']
    paths += [HERE / 'full_rich_gpu_native_preflight_v2.json',
              HERE / 'full_rich_gpu_recovery_v2.json', Path(__file__).with_name('prepare_full_rich_gpu_recovery_v2.py')]
    paths += [HERE / name for name in ('conditional_daily.py', 'fit_stacker.py', 'rich_userday_activity.py',
        'catboost_userday_residual.py', 'full_quality_row_catboost.py', 'cross_user_pair_ranker_v1.py',
        'quantile_blend_core_v1.py', 'verify_stacker.py', 'full_population_original_v1.py')]
    for run in (CONTEXT_VERIFIER, ORIGINAL, BASE):
        report = read(run / 'verification.json')
        assert report['status'] == 'passed'
        protocol = read(run / 'protocol.json')
        for filename, expected in protocol['sources_sha256'].items():
            assert digest(filename) == expected, filename
            paths.append(Path(filename))
        paths += [run / 'verification.json', run / 'protocol.json']
    assert read(ORIGINAL / 'verification.json')['full_population_rebuilt_twice']
    contracts = {}
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    for day in (15, 16):
        contract, arrays, _, _ = components(day)
        other, replay_arrays, _, _ = components(day, True)
        assert contract == other
        np.testing.assert_array_equal(arrays['rows'][arrays['train']], np.flatnonzero(raw.pt_d.to_numpy() < 20191100 + day))
        np.testing.assert_array_equal(arrays['rows'][arrays['held']], np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day))
        np.testing.assert_array_equal(arrays['labels'], raw.label.to_numpy()[arrays['rows'][arrays['train']]])
        np.testing.assert_array_equal(arrays['rows'][arrays['held']], np.load(BASE / f'day{day}_indices.npy'))
        contracts[str(day)] = contract
        for folder in (*folders(day), *folders(day, True)):
            paths += [folder / 'contract.json', folder / 'features.npy']
            paths += [folder / filename for filename in read(folder / 'contract.json')['files_sha256']]
        paths += [BASE / f'day{day}_prediction.npy', BASE / f'day{day}_indices.npy']
    raw_hashes = read(CONTEXT_VERIFIER / 'protocol.json')['raw_datasets_sha256']
    for filename, expected in raw_hashes.items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    OUT.mkdir()
    write(OUT / 'protocol.json', dict(dates=[20191115, 20191116], arms=ARMS, checkpoints=CHECKPOINTS,
        params=PARAMS, modes=MODES, weights=WEIGHTS, positive_weight=5.,
        feature_policy='All395 context columns plus all166 original numeric and15 ordered history token columns',
        population='Every strictly earlier positive and negative; complete held date, original row order',
        held_labels_for_fitting=False, inferred17_labels_read=False, development_dates_repeatedly_reused=True,
        selection='All both-date model checkpoints and predictions locked before full-cohort scoring',
        verification='Both upstream full populations reconstructed independently; separate column-wise join and saved-model replay',
        raw_datasets_sha256=raw_hashes, sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)}))
    write(OUT / 'inputs_locked.json', contracts)
    for day in (15, 16):
        np.save(OUT / f'day{day}_indices.npy', np.load(BASE / f'day{day}_indices.npy'))
    check()
    emit('full_rich_protocol_locked', columns=576, categories=39)


def fit():
    check()
    assert not (OUT / 'models_locked.json').exists() and not list(OUT.glob('*.cbm')), 'Preserve partial fits'
    records = []
    for day in (15, 16):
        wait_for_memory()
        declared, _, _, _ = components(day)
        assert declared == read(OUT / 'inputs_locked.json')[str(day)]
        data, cats, arrays = frame(day, 'train')
        labels = arrays['labels']
        dates = np.load(CONTEXT / f'input_day{day}/dates.npy')[arrays['train']]
        assert np.all(dates < 20191100 + day)
        for arm in ARMS:
            emit('full_rich_fit_started', day=day, arm=arm['name'], rows=len(data), columns=len(data.columns))
            weights = recency_weights(dates, 20191100 + day, arm['half_life']) * np.where(labels == 1, 5., 1.)
            pool = Pool(data, label=labels, weight=weights, cat_features=cats)
            model = CatBoost(dict(PARAMS, depth=arm['depth'], iterations=max(CHECKPOINTS)))
            model.fit(pool)
            del pool
            for trees in CHECKPOINTS:
                name = f'day{day}_{arm["name"]}_trees{trees}'
                saved = model.copy()
                saved.shrink(trees)
                path = OUT / (name + '.cbm')
                save_model(saved, path)
                records.append(dict(arm, day=day, trees=trees, name=name, model_sha256=digest(path),
                    input_contract_sha256=digest(OUT / 'inputs_locked.json'), max_training_date=declared['max_training_date']))
                del saved
            del model, weights
            gc.collect()
            emit('full_rich_fit_complete', day=day, arm=arm['name'])
        del data, arrays, labels
        gc.collect()
    write(OUT / 'models_locked.json', dict(models=records, protocol_sha256=digest(OUT / 'protocol.json'),
        inputs_locked_sha256=digest(OUT / 'inputs_locked.json')))


def forecasts(day, independent=False):
    declared, _, _, _ = components(day, independent)
    assert declared == read(OUT / 'inputs_locked.json')[str(day)]
    data, cats, _ = frame(day, 'held', independent)
    pool = Pool(data, cat_features=cats)
    locked = read(OUT / 'models_locked.json')
    assert locked['protocol_sha256'] == digest(OUT / 'protocol.json')
    assert locked['inputs_locked_sha256'] == digest(OUT / 'inputs_locked.json')
    margins = {}
    for record in locked['models']:
        if record['day'] != day:
            continue
        path = OUT / (record['name'] + '.cbm')
        assert digest(path) == record['model_sha256']
        model = load_model(path)
        assert model.feature_names_ == list(data)
        margin = np.asarray(model.predict(pool, prediction_type='RawFormulaVal', thread_count=2))
        assert np.isfinite(margin).all() and margin.shape == (len(data),)
        if independent:
            np.testing.assert_array_equal(margin, np.load(OUT / (record['name'] + '_held_margin.npy')))
        else:
            np.save(OUT / (record['name'] + '_held_margin.npy'), margin)
        margins[record['name']] = margin
        del model
    return margins


def candidates(day, margins, independent=False):
    base = np.load(BASE / f'day{day}_prediction.npy')
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'user_id'], dtype='int32')
    groups, _ = pd.factorize(raw.loc[raw.pt_d.eq(20191100 + day), 'user_id'], sort=True)
    values = dict(margins)
    values['mean_last'] = np.mean([m for n, m in margins.items() if n.endswith(f'trees{max(CHECKPOINTS)}')], axis=0)
    yield dict(name='base', mode='base', weight=0.), base.copy()
    for name, margin in values.items():
        target = matched_logit(base, margin, independent)
        for mode in MODES:
            for weight in WEIGHTS:
                result = (blend(base, target, np.ones(len(base), dtype=bool), weight, independent)
                          if mode == 'rank_global' else row_prediction(base, margin, groups, weight, mode, independent))
                yield dict(name=name, mode=mode, weight=weight), result


def key(record):
    return f'{record["name"]}__{record["mode"]}__{record["weight"]:g}'


def evaluate():
    check()
    assert not (OUT / 'predictions_locked.json').exists()
    locked = {}
    for day in (15, 16):
        margins = forecasts(day)
        locked[str(day)] = {key(record): array_hash(prediction) for record, prediction in candidates(day, margins)}
    write(OUT / 'predictions_locked.json', dict(models_locked_sha256=digest(OUT / 'models_locked.json'),
        dates=locked, all_both_date_candidates_locked_before_scoring=True))
    emit('full_rich_candidates_locked')


def verify():
    check()
    assert not (OUT / 'verification.json').exists()
    locked = read(OUT / 'predictions_locked.json')
    assert locked['models_locked_sha256'] == digest(OUT / 'models_locked.json')
    checked = {day: forecasts(day, True) for day in (15, 16)}
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    reports = {}
    for day in (15, 16):
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day)
        np.testing.assert_array_equal(rows, np.load(OUT / f'day{day}_indices.npy'))
        y = raw.label.to_numpy()[rows]
        assert len(y) == {15: 424780, 16: 421983}[day] and y.sum() == {15: 3523, 16: 3724}[day]
        records, best_value, maximum = [], -1., 0.
        for (record, prediction), (other, replay) in zip(candidates(day, checked[day]), candidates(day, checked[day], True)):
            assert record == other and array_hash(prediction) == locked['dates'][str(day)][key(record)]
            error = float(np.max(np.abs(prediction - replay)))
            assert error < 2e-13
            maximum = max(maximum, error)
            assert np.isfinite(prediction).all() and np.all((prediction > 0) & (prediction < 1))
            value = rank_sum_auc(y, prediction)
            record = dict(record, auc=value)
            records.append(record)
            if value > best_value:
                best_value, best, best_prediction = value, record, prediction.copy()
        assert abs(roc_auc_score(y, best_prediction) - best_value) < 1e-12
        np.save(OUT / f'day{day}_prediction.npy', best_prediction)
        reports[str(20191100 + day)] = dict(auc=best_value, best=best, records=records,
            rows=len(y), clicks=int(y.sum()), evaluation_day_labels_fit=False, inferred17_labels_read=False,
            held_labels_used_to_select_fixed_recipe=True, all_both_date_candidates_locked_before_scoring=True,
            upstream_full_population_features_independently_rebuilt=True,
            independent_column_join_and_all_models_replayed=True,
            maximum_independent_prediction_error=maximum, target_met=best_value >= {15: .815, 16: .82}[day])
        emit('full_rich_scored', day=day, best=best)
    check()
    write(OUT / 'verification.json', dict(status='passed', dates=reports,
        original_datasets_unchanged=True, both_targets_met=all(r['target_met'] for r in reports.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['preflight', 'prepare', 'fit', 'evaluate', 'verify'])
    globals()[parser.parse_args().phase]()
