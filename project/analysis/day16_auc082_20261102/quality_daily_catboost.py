"""Complete news-quality user/day response models with strictly past Bayesian offsets."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoost, Pool
from scipy.special import expit
from news_quality_daily import build as daily_build
from bayesian_daily_offset import offsets
from catboost_userday_residual import save_model, load_model
from conditional_daily import read, emit
from fit_stacker import ROOT, HERE, digest, write
from rich_userday_activity import array_hash, correct
from verify_stacker import rank_sum_auc

SOURCE = HERE/'runs/news_quality_daily_v1'
BASE = HERE/'runs/grouped_catboost_residual_v1'
OUT = HERE/'runs/quality_daily_catboost_v1'
QUEUE = HERE/'quality_daily_catboost_queued_v1.json'
ARMS = [dict(name='binary_d6_offset', objective='binary', depth=6),
        dict(name='poisson_d5_offset', objective='poisson', depth=5)]
CHECKS = [300, 800, 1600]
POWERS = [.0625, .125, .25, .5]
PARAMS = dict(learning_rate=.025, l2_leaf_reg=60., random_strength=1.,
    bootstrap_type='Bernoulli', subsample=.8, border_count=32, one_hot_max_size=10,
    max_ctr_complexity=1, thread_count=2, random_seed=151642, task_type='GPU', devices='0',
    gpu_ram_part=.5, allow_writing_files=False, verbose=False)


def training_targets(keys, counts, cutoff, objective):
    keys = np.asarray(keys)
    counts = np.asarray(counts)
    training = np.flatnonzero(keys[:, 0] < cutoff)
    if np.any(counts[keys[:, 0] >= cutoff] != -1):
        raise ValueError('Held advertisement group counts must be masked')
    values = counts[training].astype('float64')
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Only observed earlier advertisement counts can be fitted')
    if objective == 'binary':
        values = (values > 0).astype('float64')
    elif objective != 'poisson':
        raise ValueError('Only binary/count objectives are declared')
    return training, values


def queue_check():
    record = read(QUEUE)
    for path, expected in record['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in record['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    emit('quality_daily_queue_passed', boundary_tests=record['boundary_tests_passed'])
    return record


def inputs(day):
    data = daily_build(day)
    contract = read(SOURCE/f'input_contract{day}.json')
    assert data['contract'] == contract
    cached = np.load(SOURCE/f'quality{day}.npz', allow_pickle=False)
    np.testing.assert_array_equal(cached['keys'], data['keys'])
    np.testing.assert_array_equal(cached['values'], data['quality'])
    training, _ = training_targets(data['keys'], data['counts'], 20191100+day, 'poisson')
    assert data['keys'][training, 0].max() < 20191100+day
    assert len(data['frame']) == len(data['keys']) and np.isfinite(data['frame'].to_numpy()).all()
    quality_names = [name for name in data['frame'] if name.startswith('quality__')]
    assert len(quality_names) == 337
    audit = dict(source_contract=contract, feature_names=data['frame'].columns.tolist(),
        frame_sha256=array_hash(data['frame'].to_numpy(dtype='float32')),
        keys_sha256=array_hash(data['keys']), counts_sha256=array_hash(data['counts']),
        categories=data['categories'], full_shape=list(data['frame'].shape),
        training_groups=len(training), max_training_date=int(data['keys'][training, 0].max()),
        quality_features=337, own_held_targets_masked=True)
    return data, audit


def check_sources():
    protocol = read(OUT/'protocol.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    return protocol


def fit():
    assert not OUT.exists(), 'Preserve all earlier daily quality runs and failures'
    queue = queue_check()
    assert read(SOURCE/'verification.json')['status'] == 'passed'
    assert read(BASE/'verification.json')['status'] == 'passed'
    parent = read(SOURCE/'protocol.json')
    for path, expected in parent['sources_sha256'].items():
        assert digest(path) == expected, path
    paths = [Path(__file__), HERE/'test_quality_daily_catboost.py', HERE/'run_quality_daily_catboost.ps1',
        HERE/'news_quality_daily.py', HERE/'news_quality_features.py', HERE/'bayesian_daily_offset.py',
        HERE/'catboost_userday_residual.py', HERE/'conditional_daily.py', HERE/'fit_stacker.py',
        HERE/'rich_userday_activity.py', HERE/'verify_stacker.py', SOURCE/'verification.json',
        SOURCE/'protocol.json', BASE/'verification.json', BASE/'protocol.json', QUEUE]
    paths += [Path(path) for path in parent['sources_sha256']]
    for day in [15, 16]:
        paths += [SOURCE/f'quality{day}.npz', SOURCE/f'input_contract{day}.json',
            BASE/f'day{day}_prediction.npy', BASE/f'day{day}_indices.npy']
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json', dict(dates=[20191115, 20191116], arms=ARMS, params=PARAMS,
        checkpoints=CHECKS, powers=POWERS, quality_features=337,
        baseline='Strictly earlier hierarchical user/app/slot rate mass; Pool baseline only while fitting',
        training='One complete advertisement-count target per strictly earlier user/day',
        own_held_ad_labels_fit=False, inferred17_labels_read=False,
        selection='All both-date checkpoints lock before held scoring; reused development dates',
        raw_datasets_sha256=queue['raw_datasets_sha256'],
        sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)}))
    locks = []
    for day in [15, 16]:
        data, audit = inputs(day)
        write(OUT/f'input_contract{day}.json', audit)
        np.save(OUT/f'day{day}_indices.npy', data['rows'])
        emit('quality_daily_training_ready', day=day, shape=audit['full_shape'], training_groups=audit['training_groups'])
        for arm in ARMS:
            training, target = training_targets(data['keys'], data['counts'], 20191100+day, arm['objective'])
            baseline = offsets(data, arm['objective'])[training]
            pool = Pool(data['frame'].iloc[training], label=target, cat_features=data['categories'], baseline=baseline)
            model = CatBoost(dict(PARAMS, iterations=max(CHECKS), depth=arm['depth'],
                loss_function='Logloss' if arm['objective'] == 'binary' else 'Poisson'))
            model.fit(pool)
            del pool
            for trees in CHECKS:
                saved = model.copy()
                saved.shrink(trees)
                name = f'day{day}_{arm["name"]}_trees{trees}'
                path = OUT/(name+'.cbm')
                save_model(saved, path)
                locks.append(dict(arm, day=day, trees=trees, name=name, model_sha256=digest(path),
                    target_sha256=array_hash(target), baseline_sha256=array_hash(baseline)))
                del saved
            emit('quality_daily_arm_complete', day=day, name=arm['name'])
            del model
            gc.collect()
        del data
        gc.collect()
    write(OUT/'models_locked.json', dict(protocol_sha256=digest(OUT/'protocol.json'), models=locks))


def evaluate(verifying):
    check_sources()
    locks = read(OUT/'models_locked.json')
    assert locks['protocol_sha256'] == digest(OUT/'protocol.json')
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    results = {}
    for day in [15, 16]:
        data, audit = inputs(day)
        assert audit == read(OUT/f'input_contract{day}.json')
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
        np.testing.assert_array_equal(rows, data['rows'])
        np.testing.assert_array_equal(rows, np.load(OUT/f'day{day}_indices.npy'))
        np.testing.assert_array_equal(rows, np.load(BASE/f'day{day}_indices.npy'))
        y = raw.label.to_numpy()[rows]
        assert len(y) == {15:424780, 16:421983}[day] and int(y.sum()) == {15:3523, 16:3724}[day]
        held = np.flatnonzero(data['keys'][:, 0] == 20191100+day)
        base = np.load(BASE/f'day{day}_prediction.npy')
        best = dict(name='base', recipe='base', power=0., auc=rank_sum_auc(y, base))
        winner, records, error = base.copy(), [best], 0.
        pool = Pool(data['frame'].iloc[held], cat_features=data['categories'])
        for arm in locks['models']:
            if arm['day'] != day:
                continue
            path = OUT/(arm['name']+'.cbm')
            assert digest(path) == arm['model_sha256']
            margin = load_model(path).predict(pool, prediction_type='RawFormulaVal')+offsets(data, arm['objective'])[held]
            predicted = expit(margin) if arm['objective'] == 'binary' else np.exp(margin.clip(-15, 8))
            path = OUT/(arm['name']+'_held_group_prediction.npy')
            if verifying:
                delta = float(np.max(np.abs(predicted-np.load(path))))
                assert delta < 2e-7
                error = max(error, delta)
            else:
                np.save(path, predicted)
            groups = np.zeros(len(data['keys']))
            groups[held] = predicted
            for power in POWERS:
                p = correct(base, groups, data['row_groups'], arm['objective'], power)
                record = dict(name=arm['name'], objective=arm['objective'], power=power, auc=rank_sum_auc(y, p))
                records.append(record)
                if record['auc'] > best['auc']:
                    best, winner = record, p.copy()
        if verifying:
            delta = float(np.max(np.abs(winner-np.load(OUT/f'day{day}_prediction.npy'))))
            assert delta < 2e-7
            error = max(error, delta)
        else:
            np.save(OUT/f'day{day}_prediction.npy', winner)
        results[str(20191100+day)] = dict(auc=best['auc'], best=best, records=records, rows=len(y), clicks=int(y.sum()),
            maximum_replay_error=error, features_and_quality_rebuilt_from_raw=True, evaluation_day_labels_fit=False,
            target_met=best['auc'] >= {15:.815, 16:.82}[day])
        emit('quality_daily_scored', day=day, best=best, verifying=verifying)
        del data, pool
        gc.collect()
    check_sources()
    write(OUT/('verification.json' if verifying else 'result.json'), dict(status='passed' if verifying else 'evaluated',
        dates=results, original_datasets_unchanged=True, both_targets_met=all(r['target_met'] for r in results.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['queue-check', 'fit', 'evaluate', 'verify'], required=True)
    phase = parser.parse_args().phase
    if phase == 'queue-check':
        queue_check()
    elif phase == 'fit':
        fit()
    else:
        evaluate(phase == 'verify')
