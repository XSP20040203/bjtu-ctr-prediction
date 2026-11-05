"""Strict-past GPU row models with all337 content-quality features."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'

import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoost, Pool
from scipy.special import expit, logit
from fit_stacker import ROOT, HERE, digest, write
from conditional_daily import read, emit
from rich_userday_activity import array_hash
from rich_row_catboost import selection, frame_hash, PARAMS as PARENT_PARAMS
from catboost_userday_residual import save_model, load_model
from news_quality_features import READ, build_quality
from verify_stacker import rank_sum_auc

SOURCE = HERE/'runs/rich_augmented_rows_v1'
QUALITY = HERE/'runs/news_quality_daily_v1'
BASE = HERE/'runs/rich_row_catboost_v1'
OUT = HERE/'runs/full_quality_row_catboost_v1'
PREFLIGHT = HERE/'runs/full_quality_row_catboost_preflight_v1'
ARMS = [dict(name='all_quality_depth8', depth=8, half_life=None),
        dict(name='recent2_quality_depth8', depth=8, half_life=2.)]
CHECKS = [800, 1800, 3600]
WEIGHTS = [.025, .05, .1, .2, .35, .5]
PARAMS = dict(PARENT_PARAMS, random_seed=151641)


def quality_rows(keys, values, query):
    index = pd.MultiIndex.from_arrays(np.asarray(keys).T)
    if not index.is_unique:
        raise ValueError('Quality user/date keys must be unique')
    positions = index.get_indexer(pd.MultiIndex.from_arrays(np.asarray(query).T))
    if np.any(positions < 0):
        raise ValueError('A requested advertisement user/date has no quality record')
    return np.asarray(values)[positions]


def full_frame(matrix, positions, names, categories, quality):
    removed = [i for i, name in enumerate(names) if name.startswith('rich_daily__quality__')]
    if len(removed) != 64 or quality.shape != (len(positions), 337):
        raise ValueError('Expected64 replaced columns and337 aligned quality columns')
    keep = [i for i in range(len(names)) if i not in set(removed)]
    if any(i not in keep for i in categories):
        raise ValueError('A categorical source column cannot be replaced')
    values = np.empty((len(positions), len(keep)+337), dtype='float32')
    for start in range(0, len(positions), 20000):
        stop = min(len(positions), start+20000)
        values[start:stop, :len(keep)] = np.asarray(matrix[positions[start:stop]])[:, keep]
        values[start:stop, len(keep):] = quality[start:stop]
    if not np.isfinite(values).all():
        raise ValueError('Features must be finite')
    frame = pd.DataFrame(values, columns=[f'f{i}' for i in range(values.shape[1])], copy=False)
    cats = []
    for column in categories:
        updated = keep.index(column)
        category = frame.iloc[:, updated].to_numpy()
        if not np.array_equal(category, category.astype('int64')):
            raise ValueError('Categorical inputs must be integer codes')
        name = f'f{updated}'
        frame[name] = category.astype('int64')
        cats.append(name)
    return frame, cats, keep


def recency_weights(dates, cutoff, half_life):
    dates = np.asarray(dates)
    if np.any(dates >= cutoff):
        raise ValueError('Held or future advertisement dates are forbidden')
    if half_life is None:
        return np.ones(len(dates), dtype='float32')
    if half_life <= 0:
        raise ValueError('Half life must be positive')
    age = (pd.Timestamp(str(cutoff))-pd.to_datetime(dates.astype(str))).days.to_numpy()-1
    result = np.exp2(-age/half_life)
    return (result/result.mean()).astype('float32')


def inputs(day, partition, independent=False, limit=None):
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'user_id', 'label'], dtype='int32')
    chosen, weights, rows, train, held, y = selection(day, raw.label.to_numpy(), raw.pt_d.to_numpy())
    contract = read(SOURCE/f'input_contract{day}.json')
    assert array_hash(chosen) == contract['training_rows_sha256']
    assert array_hash(y) == contract['training_labels_sha256']
    assert array_hash(weights) == contract['weights_sha256']
    assert array_hash(rows) == contract['held_rows_sha256']
    selected = np.sort(np.r_[chosen, rows])
    positions = train if partition == 'train' else held
    if limit is not None:
        positions = positions[:limit]
    query = raw.iloc[selected[positions]][['pt_d', 'user_id']].to_numpy()
    cached = np.load(QUALITY/f'quality{day}.npz', allow_pickle=False)
    keys, quality = cached['keys'], cached['values']
    native = read(QUALITY/f'input_contract{day}.json')
    assert array_hash(quality) == native['quality_sha256']
    assert array_hash(keys) == native['native_contract']['native_contract']['keys_sha256']
    assert quality.shape == (len(keys), 337) and np.all(keys[:, 0] <= 20191100+day)
    if independent:
        news = pd.read_csv(ROOT/'2026ml/train_feeds.csv', usecols=READ,
                           dtype={'i_docId': str, 'i_s_sourceId': str})
        replay, replay_names = build_quality(news, keys, 20191100+day)
        np.testing.assert_array_equal(replay, quality)
        assert replay_names == native['quality_names']
        quality = replay
        del news
        gc.collect()
    quality = quality_rows(keys, quality, query)
    directory = SOURCE/(f'independent_day{day}' if independent else f'input_day{day}')
    matrix_path = directory/'features.npy'
    assert digest(matrix_path) == contract['matrix_sha256']
    matrix = np.load(matrix_path, mmap_mode='r')
    frame, cats, keep = full_frame(matrix, positions, contract['feature_names'],
                                   contract['categorical_columns'], quality)
    assert frame.shape[1] == 1017
    feature_names = [contract['feature_names'][i] for i in keep]+native['quality_names']
    audit = dict(shape=list(frame.shape), frame_sha256=frame_hash(frame),
        source_matrix_sha256=contract['matrix_sha256'], quality_sha256=array_hash(quality),
        query_keys_sha256=array_hash(query), training_rows_sha256=array_hash(chosen),
        training_labels_sha256=array_hash(y), weights_sha256=array_hash(weights),
        held_rows_sha256=array_hash(rows), max_training_date=int(raw.pt_d.to_numpy()[chosen].max()),
        feature_names=feature_names, categories=cats, own_held_targets_fit=False)
    dates = raw.pt_d.to_numpy()[chosen]
    del matrix, raw, quality, cached
    gc.collect()
    return frame, y, weights, dates, rows, cats, audit


def preflight():
    assert not PREFLIGHT.exists(), 'Preserve prior preflights'
    PREFLIGHT.mkdir(parents=True)
    frame, y, weights, dates, _, cats, audit = inputs(15, 'train', limit=100000)
    pool = Pool(frame, label=y[:len(frame)],
                weight=weights[:len(frame)]*np.where(y[:len(frame)] == 1, 5., 1.), cat_features=cats)
    model = CatBoost(dict(PARAMS, depth=8, iterations=20))
    model.fit(pool)
    sample = Pool(frame.iloc[:2000], cat_features=cats)
    before = np.asarray(model.predict(sample, prediction_type='RawFormulaVal'))
    save_model(model, PREFLIGHT/'smoke_model.cbm')
    after = load_model(PREFLIGHT/'smoke_model.cbm').predict(sample, prediction_type='RawFormulaVal')
    error = float(np.max(np.abs(before-after)))
    assert error < 1e-10 and audit['max_training_date'] == 20191114
    write(PREFLIGHT/'verification.json', dict(status='passed', audit=audit,
        source_code_sha256=digest(__file__), serialization_maximum_error=error,
        held_labels_used_for_fitting_or_evaluation=False, full_training_memory_capacity_proven=False))
    emit('full_quality_preflight_passed', shape=audit['shape'], serialization_error=error)


def check_sources():
    protocol = read(OUT/'protocol.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    return protocol


def fit():
    assert not OUT.exists(), 'Preserve prior models and failures'
    smoke = read(PREFLIGHT/'verification.json')
    assert smoke['status'] == 'passed' and smoke['source_code_sha256'] == digest(__file__)
    assert read(SOURCE/'verification.json')['status'] == 'passed'
    assert read(QUALITY/'verification.json')['status'] == 'passed'
    assert read(BASE/'verification.json')['status'] == 'passed'
    parent = read(SOURCE/'protocol.json')
    for path, expected in parent['sources_sha256'].items():
        assert digest(path) == expected, path
    paths = [Path(__file__), HERE/'test_full_quality_row_catboost.py', HERE/'run_full_quality_row_catboost.ps1',
        HERE/'rich_row_catboost.py', HERE/'news_quality_features.py', HERE/'conditional_daily.py',
        HERE/'fit_stacker.py', HERE/'rich_userday_activity.py', HERE/'verify_stacker.py',
        HERE/'catboost_userday_residual.py', HERE/'global_row_context.py', HERE/'global_row_context15.py',
        SOURCE/'verification.json', SOURCE/'protocol.json', QUALITY/'verification.json',
        QUALITY/'protocol.json', BASE/'verification.json', BASE/'protocol.json', PREFLIGHT/'verification.json']
    for day in [15, 16]:
        paths += [SOURCE/f'input_contract{day}.json', SOURCE/f'input_day{day}/features.npy',
            SOURCE/f'independent_day{day}/features.npy', QUALITY/f'quality{day}.npz',
            QUALITY/f'input_contract{day}.json', BASE/f'day{day}_prediction.npy', BASE/f'day{day}_indices.npy']
    for name, expected in parent['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json', dict(dates=[20191115, 20191116], params=PARAMS, arms=ARMS,
        checkpoints=CHECKS, blend_weights=WEIGHTS, positive_weight=5.,
        source_features='Original744 minus64 selected quality features plus all337 quality features',
        source_quality_selection='All declared content groups and smoothing values; no held-label feature selection',
        training='All strictly earlier positives, fixed one-third negatives, inverse inclusion weights',
        selection='Both-date models locked before full held-cohort scoring; reused development dates',
        own_held_ad_labels_fit=False, inferred17_labels_read=False,
        raw_datasets_sha256=parent['raw_datasets_sha256'], sources_sha256={str(p): digest(p) for p in paths}))
    locks = []
    for day in [15, 16]:
        frame, y, weights, dates, rows, cats, audit = inputs(day, 'train')
        write(OUT/f'training_contract{day}.json', audit)
        np.save(OUT/f'day{day}_indices.npy', rows)
        emit('full_quality_training_ready', day=day, shape=audit['shape'])
        for arm in ARMS:
            recent = recency_weights(dates, 20191100+day, arm['half_life'])
            fit_weights = weights*recent*np.where(y == 1, 5., 1.)
            pool = Pool(frame, label=y, weight=fit_weights, cat_features=cats)
            model = CatBoost(dict(PARAMS, depth=arm['depth'], iterations=max(CHECKS)))
            model.fit(pool)
            del pool
            for trees in CHECKS:
                saved = model.copy()
                saved.shrink(trees)
                name = f'day{day}_{arm["name"]}_trees{trees}'
                path = OUT/(name+'.cbm')
                save_model(saved, path)
                locks.append(dict(arm, day=day, trees=trees, name=name, model_sha256=digest(path),
                                  fitting_weights_sha256=array_hash(fit_weights)))
                del saved
            emit('full_quality_arm_complete', day=day, name=arm['name'])
            del model, fit_weights
            gc.collect()
        del frame
        gc.collect()
    write(OUT/'models_locked.json', dict(models=locks, protocol_sha256=digest(OUT/'protocol.json')))


def evaluate(verifying):
    check_sources()
    locks = read(OUT/'models_locked.json')
    assert locks['protocol_sha256'] == digest(OUT/'protocol.json')
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    results = {}
    for day in [15, 16]:
        frame, _, _, _, rows, cats, audit = inputs(day, 'held', independent=verifying)
        np.testing.assert_array_equal(rows, np.load(OUT/f'day{day}_indices.npy'))
        np.testing.assert_array_equal(rows, np.load(BASE/f'day{day}_indices.npy'))
        np.testing.assert_array_equal(rows, np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day))
        y = raw.label.to_numpy()[rows]
        assert len(y) == {15: 424780, 16: 421983}[day] and int(y.sum()) == {15: 3523, 16: 3724}[day]
        contract_file = OUT/f'held_contract{day}.json'
        if verifying:
            assert audit == read(contract_file)
        else:
            write(contract_file, audit)
        pool = Pool(frame, cat_features=cats)
        base = np.load(BASE/f'day{day}_prediction.npy')
        a = logit(base.clip(1e-6, 1-1e-6))
        best = dict(name='base', recipe='base', auc=rank_sum_auc(y, base))
        winner = base.copy()
        records, forecasts, last, replay_error = [best], {}, [], 0.
        for arm in locks['models']:
            if arm['day'] != day:
                continue
            path = OUT/(arm['name']+'.cbm')
            assert digest(path) == arm['model_sha256']
            p = expit(load_model(path).predict(pool, prediction_type='RawFormulaVal')-np.log(5.))
            prediction_file = OUT/(arm['name']+'_prediction.npy')
            if verifying:
                error = float(np.max(np.abs(p-np.load(prediction_file))))
                assert error < 2e-7
                replay_error = max(replay_error, error)
            else:
                np.save(prediction_file, p)
            forecasts[arm['name']] = p
            if arm['trees'] == max(CHECKS):
                last.append(p)
        forecasts['mean_last_two'] = np.mean(last, axis=0)
        for name, p in forecasts.items():
            b = logit(p.clip(1e-6, 1-1e-6))
            b = (b-b.mean())*a.std()/max(b.std(), 1e-6)+a.mean()
            candidates = [('single', None, p)]
            for weight in WEIGHTS:
                candidates += [('normalized_logit_blend', weight, expit((1-weight)*a+weight*b)),
                               ('probability_blend', weight, (1-weight)*base+weight*p)]
            for recipe, weight, q in candidates:
                record = dict(name=name, recipe=recipe, weight=weight, auc=rank_sum_auc(y, q))
                records.append(record)
                if record['auc'] > best['auc']:
                    best, winner = record, q.copy()
        if verifying:
            error = float(np.max(np.abs(winner-np.load(OUT/f'day{day}_prediction.npy'))))
            assert error < 2e-7
            replay_error = max(replay_error, error)
        else:
            np.save(OUT/f'day{day}_prediction.npy', winner)
        results[str(20191100+day)] = dict(auc=best['auc'], best=best, records=records, rows=len(y),
            clicks=int(y.sum()), maximum_replay_error=replay_error, evaluation_day_labels_fit=False,
            independent_source_matrix_used=verifying, quality_rebuilt_from_raw=verifying,
            target_met=best['auc'] >= {15: .815, 16: .82}[day])
        emit('full_quality_scored', day=day, best=best, verifying=verifying)
        del frame, pool
        gc.collect()
    check_sources()
    write(OUT/('verification.json' if verifying else 'result.json'), dict(
        status='passed' if verifying else 'evaluated', dates=results, original_datasets_unchanged=True,
        both_targets_met=all(r['target_met'] for r in results.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['preflight', 'fit', 'evaluate', 'verify'], required=True)
    phase = parser.parse_args().phase
    if phase == 'preflight':
        preflight()
    elif phase == 'fit':
        fit()
    else:
        evaluate(phase == 'verify')
