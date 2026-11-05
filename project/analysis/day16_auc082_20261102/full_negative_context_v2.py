"""Complete-negative row trees, locked before full-cohort development scoring."""
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
import full_negative_context_inputs_v1 as source

OUT = HERE / 'runs/full_negative_context_v2'
INPUT = HERE / 'runs/full_negative_context_v1'
BASE = HERE / 'runs/group_event_posterior_v1'
ARMS = [dict(name='all_negative_d6', depth=6, half_life=None),
        dict(name='all_negative_d8_recent2', depth=8, half_life=2.)]
CHECKPOINTS = [400, 1200, 2400]
WEIGHTS = [.025, .05, .1, .2, .35, .5, .75, 1.]
MODES = ['replace_global', 'add_within', 'rank_global']
PARAMS = dict(loss_function='Logloss', learning_rate=.025, l2_leaf_reg=40.,
    random_strength=1., bootstrap_type='Bernoulli', subsample=.8, border_count=64,
    one_hot_max_size=16, max_ctr_complexity=1, thread_count=2, random_seed=151661,
    task_type='GPU', devices='0', gpu_ram_part=.60, allow_writing_files=False, verbose=200)


def dependencies():
    paths = [Path(__file__), HERE / 'prepare_full_negative_context_v2.py'] + [HERE / name for name in
        ('full_negative_context_core_v1.py', 'test_full_negative_context_core_v1.py',
         'full_negative_context_inputs_v1.py', 'run_full_negative_context_v2.ps1',
         'rich_augmented_rows.py', 'global_row_context.py', 'rich_row_catboost.py',
         'catboost_userday_residual.py', 'full_quality_row_catboost.py',
         'cross_user_pair_ranker_v1.py', 'quantile_blend_core_v1.py',
         'unordered_snapshot_core_v1.py', 'news_quality_features.py',
         'rich_userday_activity.py', 'fit_stacker.py', 'conditional_daily.py', 'verify_stacker.py')]
    paths += [ROOT / name for name in
              ('analysis/auc085_next/prepare_multiscale_stats.py',
               'analysis/daily_interest_din/prepare_exposure_features.py')]
    for folder in (BASE, source.NATIVE, source.QUALITY, source.SELECTOR):
        assert read(folder / 'verification.json')['status'] == 'passed'
        parent = read(folder / 'protocol.json')
        for filename, expected in parent['sources_sha256'].items():
            assert digest(filename) == expected, filename
            paths.append(Path(filename))
        paths += [folder / 'protocol.json', folder / 'verification.json']
    for day in (15, 16):
        paths += [BASE / f'day{day}_prediction.npy', BASE / f'day{day}_indices.npy',
                  source.NATIVE / f'input_contract{day}.json',
                  source.NATIVE / f'independent_day{day}/features.npy',
                  source.QUALITY / f'input_contract{day}.json', source.QUALITY / f'quality{day}.npz',
                  source.SELECTOR / f'input_contract{day}.json',
                  source.SELECTOR / f'day{day}_audience_row_d6_trees1600.cbm']
    return list(dict.fromkeys(paths))


def check():
    protocol = read(OUT / 'protocol.json')
    for filename, expected in protocol['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    return protocol


def prepare():
    assert not OUT.exists(), 'Preserve all earlier runs and API failure evidence'
    previous = read(INPUT / 'protocol.json')
    for filename, expected in previous['sources_sha256'].items():
        assert digest(filename) == expected, filename
    assert previous['params'] == PARAMS and previous['arms'] == ARMS
    assert previous['checkpoints'] == CHECKPOINTS
    paths = dependencies() + [INPUT / 'protocol.json', INPUT / 'inputs_locked.json']
    locked = read(INPUT / 'inputs_locked.json')
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    contracts = {}
    for day in (15, 16):
        folder = INPUT / f'input_day{day}'
        contract = read(folder / 'contract.json')
        assert contract == locked[str(day)]['contract']
        assert digest(folder / 'contract.json') == locked[str(day)]['contract_sha256']
        assert digest(folder / 'features.npy') == contract['matrix_sha256']
        for name, expected in contract['files_sha256'].items():
            assert digest(folder / name) == expected, name
        rows, train, held = [np.load(folder / (name + '.npy')) for name in ('rows', 'train', 'held')]
        np.testing.assert_array_equal(rows[train], np.flatnonzero(raw.pt_d.to_numpy() < 20191100 + day))
        np.testing.assert_array_equal(rows[held], np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day))
        np.testing.assert_array_equal(np.load(folder / 'labels.npy'), raw.label.to_numpy()[rows[train]])
        assert contract['max_training_date'] < 20191100 + day
        assert contract['all_historical_negatives_retained'] and contract['all_prior_audited_common_rows_checked']
        paths += [folder / name for name in ('contract.json', 'features.npy', *contract['files_sha256'])]
        contracts[str(day)] = locked[str(day)]
    for filename, expected in previous['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    OUT.mkdir()
    protocol = {key: value for key, value in previous.items() if key != 'sources_sha256'}
    protocol.update(input_adoption=str(INPUT),
        repair='CatBoost base constructor takes one parameter dictionary; v1 feature matrices reused unchanged',
        sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)})
    write(OUT / 'protocol.json', protocol)
    write(OUT / 'inputs_locked.json', contracts)
    for day in (15, 16):
        np.save(OUT / f'day{day}_indices.npy', np.load(INPUT / f'day{day}_indices.npy'))
    folder = INPUT / 'input_day16'
    declared = read(folder / 'contract.json')
    labels = np.load(folder / 'labels.npy')
    train = np.load(folder / 'train.npy')
    selected = np.r_[np.flatnonzero(labels == 1)[:64], np.flatnonzero(labels == 0)[:1024]]
    matrix = np.load(folder / 'features.npy', mmap_mode='r')
    frame = pd.DataFrame(np.asarray(matrix[train[selected]]), columns=declared['schema']['columns'])
    cats = declared['schema']['categories']
    for name in cats:
        frame[name] = frame[name].to_numpy(dtype='int64')
    pool = Pool(frame, label=labels[selected], cat_features=cats)
    model = CatBoost(dict(iterations=20, depth=4, loss_function='Logloss', task_type='CPU',
                         random_seed=151661, thread_count=2, allow_writing_files=False, verbose=False))
    model.fit(pool)
    (OUT / 'preflight').mkdir()
    path = OUT / 'preflight/model.cbm'
    save_model(model, path)
    np.testing.assert_array_equal(model.predict(pool, prediction_type='RawFormulaVal'),
                                  load_model(path).predict(pool, prediction_type='RawFormulaVal', thread_count=2))
    dates = np.load(folder / 'dates.npy')[train[selected]]
    write(OUT / 'preflight.json', dict(status='passed', rows=len(frame), device='CPU',
        max_training_date=int(dates.max()), held_labels_read=False,
        serialization_maximum_error=0., full_gpu_capacity_proven=False,
        original_full_negative_inputs_reused_unchanged=True,
        protocol_sha256=digest(OUT / 'protocol.json'), input_lock_sha256=digest(OUT / 'inputs_locked.json')))
    check()
    emit('full_negative_v2_adopted_and_preflight_passed',
         population15=contracts['15']['contract']['training_rows'],
         population16=contracts['16']['contract']['training_rows'])


def fit():
    check()
    assert read(OUT / 'preflight.json')['status'] == 'passed'
    assert not (OUT / 'models_locked.json').exists() and not list(OUT.glob('*.cbm')), 'Preserve partial fits'
    records = []
    for day in (15, 16):
        folder = INPUT / f'input_day{day}'
        contract = read(folder / 'contract.json')
        assert digest(folder / 'contract.json') == read(OUT / 'inputs_locked.json')[str(day)]['contract_sha256']
        assert digest(folder / 'features.npy') == contract['matrix_sha256']
        frame, cats = source.frame(folder, 'train')
        labels = np.load(folder / 'labels.npy')
        dates = np.load(folder / 'dates.npy')[np.load(folder / 'train.npy')]
        assert np.all(dates < 20191100 + day) and len(labels) == len(frame)
        assert array_hash(labels) == contract['training_labels_sha256']
        for arm in ARMS:
            emit('full_negative_fit_started', day=day, arm=arm['name'], rows=len(frame))
            weights = recency_weights(dates, 20191100 + day, arm['half_life']) * np.where(labels == 1, 5., 1.)
            pool = Pool(frame, label=labels, weight=weights, cat_features=cats)
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
                    input_contract_sha256=digest(folder / 'contract.json'), max_training_date=contract['max_training_date']))
                del saved
            del model, weights
            gc.collect()
            emit('full_negative_fit_complete', day=day, arm=arm['name'])
        del frame, labels
        gc.collect()
    write(OUT / 'models_locked.json', dict(models=records, protocol_sha256=digest(OUT / 'protocol.json'),
                                          inputs_locked_sha256=digest(OUT / 'inputs_locked.json')))


def forecasts(day, independent=False):
    folder = (OUT / f'independent_day{day}' if independent else INPUT / f'input_day{day}')
    contract = read(folder / 'contract.json')
    assert contract == read(INPUT / f'input_day{day}/contract.json')
    assert digest(folder / 'features.npy') == contract['matrix_sha256']
    frame, cats = source.frame(folder, 'held')
    pool = Pool(frame, cat_features=cats)
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
        assert model.feature_names_ == list(frame)
        margin = np.asarray(model.predict(pool, prediction_type='RawFormulaVal', thread_count=2))
        assert np.isfinite(margin).all() and margin.shape == (len(frame),)
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
    last = [margin for name, margin in margins.items() if name.endswith(f'trees{max(CHECKPOINTS)}')]
    values['mean_last'] = np.mean(last, axis=0)
    yield dict(name='base', mode='base', weight=0.), base.copy()
    for name, margin in values.items():
        target = matched_logit(base, margin, independent)
        for mode in MODES:
            for weight in WEIGHTS:
                prediction = (blend(base, target, np.ones(len(base), dtype=bool), weight, independent)
                              if mode == 'rank_global' else row_prediction(base, margin, groups, weight, mode, independent))
                yield dict(name=name, mode=mode, weight=weight), prediction


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
    emit('full_negative_candidates_locked')


def verify():
    check()
    assert not (OUT / 'verification.json').exists()
    locked = read(OUT / 'predictions_locked.json')
    assert locked['models_locked_sha256'] == digest(OUT / 'models_locked.json')
    checked = {}
    for day in (15, 16):
        rebuilt = source.build(day, OUT / f'independent_day{day}', True)
        assert rebuilt == read(INPUT / f'input_day{day}/contract.json')
        checked[day] = forecasts(day, True)
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
            independent_full_population_features_and_models_replayed=True,
            maximum_independent_prediction_error=maximum, target_met=best_value >= {15: .815, 16: .82}[day])
        emit('full_negative_scored', day=day, best=best)
    check()
    write(OUT / 'verification.json', dict(status='passed', dates=reports,
        original_datasets_unchanged=True, both_targets_met=all(r['target_met'] for r in reports.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['prepare', 'fit', 'evaluate', 'verify'])
    {'prepare': prepare, 'fit': fit, 'evaluate': evaluate, 'verify': verify}[parser.parse_args().phase]()
