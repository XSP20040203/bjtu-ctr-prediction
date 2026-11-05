"""Nonlinear within-user date effects with a coupled conditional objective."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'

import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from conditional_daily import (build, read, emit, daily_targets, apply_target,
                               BASE, QUALITY, MODES, POWERS)
from fit_stacker import ROOT, HERE, digest, write
from rich_userday_activity import array_hash, lgb
from verify_stacker import rank_sum_auc

OUT = HERE/'runs/conditional_daily_trees_v1'
ARMS = [{'name': f'{scope}_l{leaves}', 'scope': scope, 'num_leaves': leaves}
        for scope in ['activity', 'quality_activity'] for leaves in [7, 15]]
CHECKS = [200, 600, 1000]
PARAMS = {'learning_rate': .02, 'min_data_in_leaf': 180, 'lambda_l2': 30.,
          'max_bin': 31, 'feature_fraction': .8, 'num_threads': 2,
          'seed': 151639, 'verbosity': -1, 'deterministic': True,
          'force_col_wise': True, 'feature_pre_filter': False, 'boost_from_average': False}


def conditional_loss(prediction, y, groups, starts, totals, offset):
    z = prediction+offset
    maxima = np.maximum.reduceat(z, starts)
    numerator = np.exp(z-maxima[groups])
    normalizer = np.bincount(groups, weights=numerator, minlength=len(totals))
    probability = numerator/normalizer[groups]
    expected = totals[groups]*probability
    loss = float(totals@(maxima+np.log(normalizer))-y@z)
    return loss, expected-y, np.maximum(expected*(1-probability), 1e-6)


def chosen_columns(data, arm):
    width = data['contract']['activity_columns'] if arm['scope'] == 'activity' else data['x'].shape[1]
    return np.arange(width)


def fit_arm(data, day, arm):
    training, users = data['training'], data['users']
    nusers = int(users.max())+1
    totals = np.bincount(users[training], weights=data['counts'][training], minlength=nusers)
    days = np.bincount(users[training], minlength=nusers)
    chosen = np.flatnonzero(training & (days[users] >= 2) & (totals[users] > 0))
    chosen = chosen[np.argsort(users[chosen], kind='stable')]
    groups = pd.factorize(users[chosen], sort=True)[0]
    starts = np.r_[0, np.flatnonzero(np.diff(groups))+1]
    y = data['counts'][chosen]
    totals = np.bincount(groups, weights=y)
    offset = np.log(data['mass'][chosen])
    x = data['x'][np.ix_(chosen, chosen_columns(data, arm))]
    def coupled_objective(prediction, dataset):
        _, gradient, diagonal = conditional_loss(prediction, y, groups, starts, totals, offset)
        return gradient, diagonal
    dataset = lgb.Dataset(x, label=y, params=PARAMS, free_raw_data=True)
    model = lgb.train(dict(PARAMS, num_leaves=arm['num_leaves'], objective=coupled_objective),
                      dataset, num_boost_round=max(CHECKS))
    locks = []
    initial_loss = conditional_loss(np.zeros(len(y)), y, groups, starts, totals, offset)[0]
    for trees in CHECKS:
        name = f'day{day}_{arm["name"]}_trees{trees}'
        path = OUT/(name+'.txt')
        path.write_text(model.model_to_string(num_iteration=trees), encoding='utf8')
        fitted = model.predict(x, num_iteration=trees, raw_score=True, num_threads=2)
        loss = conditional_loss(fitted, y, groups, starts, totals, offset)[0]
        assert np.isfinite(fitted).all() and loss < initial_loss
        locks.append(dict(arm, day=day, trees=trees, name=name, model_sha256=digest(path),
                          fit_rows=len(y), fit_clicks=int(y.sum()), fit_groups=len(totals),
                          fit_max_date=int(data['keys'][chosen, 0].max()),
                          fit_keys_sha256=array_hash(data['keys'][chosen]),
                          fit_labels_sha256=array_hash(y),
                          initial_conditional_loss=initial_loss, training_conditional_loss=loss))
    emit('conditional_daily_trees_arm_complete', day=day, name=arm['name'],
         initial_loss=initial_loss, final_loss=locks[-1]['training_conditional_loss'])
    return locks


def tree_targets(data, model, arm, base):
    latent = model.predict(data['x'][:, chosen_columns(data, arm)], raw_score=True, num_threads=2)
    # Reuse the verified target conversion with the tree score as its one-column latent effect.
    target_data = dict(data, x=latent[:, None], contract={'activity_columns': 1})
    return daily_targets(target_data, np.ones(1), {'scope': 'activity'}, base)


def check_sources():
    protocol = read(OUT/'protocol.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    return protocol


def fit():
    assert not OUT.exists(), 'Preserve prior and failed nonlinear daily runs'
    assert read(HERE/'runs/conditional_daily_v1/verification.json')['status'] == 'passed'
    raw_hashes = read(BASE/'protocol.json')['raw_datasets_sha256']
    for name, expected in raw_hashes.items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    paths = [Path(__file__), HERE/'conditional_daily.py', HERE/'test_conditional_daily_trees.py',
             HERE/'run_conditional_daily_trees.ps1', HERE/'rich_userday_activity.py',
             HERE/'bayesian_daily_offset.py', HERE/'news_quality_features.py',
             HERE/'fit_stacker.py', HERE/'verify_stacker.py',
             BASE/'protocol.json', BASE/'verification.json', QUALITY/'verification.json',
             HERE/'runs/conditional_daily_v1/verification.json']
    for day in [15, 16]:
        paths += [BASE/f'day{day}_prediction.npy', BASE/f'day{day}_indices.npy',
                  QUALITY/f'quality{day}.npz', QUALITY/f'input_contract{day}.json',
                  HERE/f'runs/conditional_daily_v1/input_contract{day}.json']
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json', {
        'dates': [20191115, 20191116], 'arms': ARMS, 'checkpoints': CHECKS,
        'params': PARAMS, 'powers': POWERS, 'modes': MODES,
        'objective': 'Conditional Poisson across dates within each user, exact gradient and diagonal Hessian approximation',
        'source_features': '350 training-centered daily quality/activity features, no user identifier predictor',
        'selection': 'Both-date trees and checkpoints locked before any held-label scoring; reused development dates',
        'evaluation_day_labels_fit': False, 'inferred17_labels_read': False,
        'raw_datasets_sha256': raw_hashes, 'sources_sha256': {str(p): digest(p) for p in paths},
    })
    locks = []
    for day in [15, 16]:
        data = build(day)
        assert data['contract'] == read(HERE/f'runs/conditional_daily_v1/input_contract{day}.json')
        write(OUT/f'input_contract{day}.json', data['contract'])
        np.save(OUT/f'day{day}_indices.npy', data['rows'])
        for arm in ARMS:
            locks += fit_arm(data, day, arm)
        del data
        gc.collect()
    write(OUT/'models_locked.json', {'models': locks, 'protocol_sha256': digest(OUT/'protocol.json')})
    evaluate(False)


def evaluate(verifying):
    check_sources()
    locked = read(OUT/'models_locked.json')
    assert locked['protocol_sha256'] == digest(OUT/'protocol.json')
    labels = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    results = {}
    for day in [15, 16]:
        data = build(day, rebuild_quality=verifying)
        assert data['contract'] == read(OUT/f'input_contract{day}.json')
        rows = np.flatnonzero(labels.pt_d.to_numpy() == 20191100+day)
        np.testing.assert_array_equal(rows, data['rows'])
        np.testing.assert_array_equal(rows, np.load(OUT/f'day{day}_indices.npy'))
        y = labels.label.to_numpy()[rows]
        assert len(y) == {15: 424780, 16: 421983}[day] and int(y.sum()) == {15: 3523, 16: 3724}[day]
        base = np.load(BASE/f'day{day}_prediction.npy')
        best = {'name': 'base', 'mode': 'base', 'power': 0., 'auc': rank_sum_auc(y, base)}
        winner = base.copy()
        records = [best]
        for arm in locked['models']:
            if arm['day'] != day:
                continue
            path = OUT/(arm['name']+'.txt')
            assert digest(path) == arm['model_sha256']
            model = lgb.Booster(model_str=path.read_text(encoding='utf8'))
            targets = tree_targets(data, model, arm, base)
            for mode in MODES:
                for power in POWERS:
                    p = apply_target(base, data['row_groups'], targets[mode], power)
                    record = {'name': arm['name'], 'mode': mode, 'power': power, 'auc': rank_sum_auc(y, p)}
                    records.append(record)
                    if record['auc'] > best['auc']:
                        best, winner = record, p.copy()
        error = 0.
        if verifying:
            error = float(np.max(np.abs(winner-np.load(OUT/f'day{day}_prediction.npy'))))
            assert error < 1e-12
        else:
            np.save(OUT/f'day{day}_prediction.npy', winner)
        results[str(20191100+day)] = {
            'auc': best['auc'], 'best': best, 'records': records, 'rows': len(y), 'clicks': int(y.sum()),
            'maximum_replay_error': error, 'evaluation_day_labels_fit': False,
            'quality_features_rebuilt_from_raw_for_verification': verifying,
            'target_met': best['auc'] >= {15: .815, 16: .82}[day],
        }
        emit('conditional_daily_trees_scored', day=day, best=best, verifying=verifying)
        del data
        gc.collect()
    check_sources()
    write(OUT/('verification.json' if verifying else 'result.json'), {
        'status': 'passed' if verifying else 'evaluated', 'dates': results,
        'original_datasets_unchanged': True, 'both_targets_met': all(r['target_met'] for r in results.values()),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    evaluate(True) if parser.parse_args().verify else fit()
