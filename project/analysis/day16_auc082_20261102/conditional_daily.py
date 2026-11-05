"""Conditional Poisson effects for changes across dates within the same user."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'

import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from fit_stacker import ROOT, HERE, digest, write
from rich_userday_activity import array_hash
from bayesian_daily_offset import prior_mass
from news_quality_features import READ, build_quality
from verify_stacker import rank_sum_auc

OUT = HERE/'runs/conditional_daily_v1'
QUALITY = HERE/'runs/news_quality_daily_v1'
BASE = HERE/'runs/group_temperature_v1'
FIELDS = ['pt_d', 'user_id', 'spread_app_id', 'slot_id', 'label']
ARMS = [{'name': f'{scope}_ridge{ridge:g}', 'scope': scope, 'ridge': ridge}
        for scope in ['activity', 'quality_activity'] for ridge in [.005, .05, .5]]
POWERS = [.0625, .125, .25, .5, 1.]
MODES = ['relative_dynamic', 'posterior_absolute']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def mask_ad(raw, cutoff):
    if set(raw.columns) != set(FIELDS):
        raise ValueError('Only declared advertisement fields are accepted')
    safe = raw.loc[raw.pt_d <= cutoff].copy()
    safe.loc[safe.pt_d >= cutoff, 'label'] = -1
    assert safe.loc[safe.pt_d < cutoff, 'label'].isin([0, 1]).all()
    return safe


def activity(news, keys, sizes, cutoff):
    source = news.loc[news.e_et <= cutoff, ['e_et', 'u_userId', 'label', 'cillabel', 'pro']]
    assert source.label.isin([-1, 1]).all()
    code, index = pd.factorize(pd.MultiIndex.from_frame(source[['e_et', 'u_userId']]), sort=True)
    n = np.bincount(code, minlength=len(index)).astype('float64')
    clicks = np.bincount(code, weights=source.label.eq(1), minlength=len(index))
    likes = np.bincount(code, weights=source.cillabel.eq(1), minlength=len(index))
    read_mass = np.bincount(code, weights=np.clip(source.pro.to_numpy(dtype='float64'), 0, 100)/100*
                           source.label.eq(1).to_numpy(), minlength=len(index))
    positions = index.get_indexer(pd.MultiIndex.from_arrays(keys.T))
    aligned = np.zeros((len(keys), 4), dtype='float64')
    known = positions >= 0
    aligned[known] = np.column_stack([n, clicks, likes, read_mass])[positions[known]]
    n, clicks, likes, read_mass = aligned.T
    values = np.column_stack([
        np.log1p(sizes), np.log1p(n), np.log1p(clicks), (clicks+.5)/(n+5.),
        np.log1p(likes), (likes+.5)/(n+10.), np.log1p(read_mass), read_mass/(n+1.),
        read_mass/(clicks+1.), np.log((n+1.)/(sizes+1.)),
        np.log((clicks+.5)/(sizes+1.)), np.log((read_mass+.5)/(sizes+1.)), ~known,
    ]).astype('float32')
    names = ['log_ad_exposures', 'log_news_exposures', 'log_news_clicks', 'news_ctr',
             'log_news_likes', 'news_like_rate', 'log_news_read_mass', 'news_read_per_exposure',
             'news_read_per_click', 'log_news_per_ad', 'log_news_clicks_per_ad',
             'log_news_read_per_ad', 'news_missing']
    assert np.isfinite(values).all()
    return values, names


def standardize(raw_features, users, training):
    """Every centering and scaling statistic is estimated from earlier dates."""
    nusers = int(users.max())+1
    counts = np.bincount(users[training], minlength=nusers)
    means = np.empty((nusers, raw_features.shape[1]), dtype='float64')
    for column in range(raw_features.shape[1]):
        values = raw_features[:, column].astype('float64')
        total = np.bincount(users[training], weights=values[training], minlength=nusers)
        means[:, column] = np.divide(total, counts, out=np.full(nusers, values[training].mean()), where=counts > 0)
    centered = raw_features.astype('float64')-means[users]
    scale = np.sqrt(np.mean(centered[training]**2, axis=0))
    scale[scale < 1e-6] = 1.
    result = np.clip(centered/scale, -8., 8.)
    assert np.isfinite(result).all()
    return result, means, scale


def build(day, rebuild_quality=False):
    cutoff = 20191100+day
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=FIELDS, dtype='int32')
    raw = mask_ad(raw, cutoff)
    groups, index = pd.factorize(pd.MultiIndex.from_frame(raw[['pt_d', 'user_id']]), sort=True)
    keys = np.asarray(index.tolist(), dtype='int64')
    sizes = np.bincount(groups, minlength=len(keys)).astype('float64')
    training = keys[:, 0] < cutoff
    earlier = raw.pt_d.to_numpy() < cutoff
    counts = np.full(len(keys), -1., dtype='float64')
    totals = np.bincount(groups[earlier], weights=raw.label.to_numpy()[earlier], minlength=len(keys))
    counts[training] = totals[training]
    rows = raw.index.to_numpy()[~earlier]
    row_groups = groups[~earlier]
    np.testing.assert_array_equal(rows, np.load(BASE/f'day{day}_indices.npy'))
    cached = np.load(QUALITY/f'quality{day}.npz')
    np.testing.assert_array_equal(cached['keys'], keys)
    names_quality = read(QUALITY/f'input_contract{day}.json')['quality_names']
    news = pd.read_csv(ROOT/'2026ml/train_feeds.csv', usecols=READ,
                       dtype={'i_docId': str, 'i_s_sourceId': str})
    extra, names_activity = activity(news, keys, sizes, cutoff)
    if rebuild_quality:
        quality, rebuilt_names = build_quality(news, keys, cutoff)
        assert rebuilt_names == names_quality
        np.testing.assert_array_equal(quality, cached['values'])
    else:
        quality = cached['values']
    features = np.column_stack([extra, quality])
    user_codes = pd.factorize(keys[:, 1], sort=True)[0]
    x, means, scale = standardize(features, user_codes, training)
    mass, audit = prior_mass(raw, keys, cutoff)
    assert all(row['label_max_date'] is None or row['label_max_date'] < row['query_date'] for row in audit)
    assert np.all(counts[~training] == -1)
    contract = {
        'shape': list(x.shape), 'feature_names': names_activity+names_quality,
        'activity_columns': len(names_activity), 'keys_sha256': array_hash(keys),
        'design_sha256': array_hash(x), 'activity_sha256': array_hash(extra),
        'quality_sha256': array_hash(quality), 'sizes_sha256': array_hash(sizes),
        'masked_counts_sha256': array_hash(counts), 'prior_mass_sha256': array_hash(mass),
        'training_means_sha256': array_hash(means), 'training_scale_sha256': array_hash(scale),
        'training_max_date': int(keys[training, 0].max()),
        'held_targets_masked': True, 'prior_audit': audit,
    }
    del raw, news, quality, features, cached, means, scale
    gc.collect()
    emit('conditional_daily_inputs', day=day, shape=contract['shape'], quality_rebuilt=rebuild_quality)
    return {'x': x, 'keys': keys, 'counts': counts, 'sizes': sizes, 'users': user_codes,
            'training': training, 'mass': mass, 'rows': rows, 'row_groups': row_groups,
            'contract': contract}


def objective(beta, x, y, groups, starts, totals, offset, ridge):
    z = offset+x@beta
    maxima = np.maximum.reduceat(z, starts)
    numerator = np.exp(z-maxima[groups])
    normalizer = np.bincount(groups, weights=numerator, minlength=len(totals))
    expected = totals[groups]*numerator/normalizer[groups]
    total_clicks = y.sum()
    value = (totals@(maxima+np.log(normalizer))-y@z)/total_clicks+.5*ridge*np.dot(beta, beta)
    gradient = x.T@(expected-y)/total_clicks+ridge*beta
    return float(value), gradient


def columns(data, arm):
    width = data['contract']['activity_columns'] if arm['scope'] == 'activity' else data['x'].shape[1]
    return np.arange(width)


def fit_arm(data, day, arm):
    training, users = data['training'], data['users']
    nusers = int(users.max())+1
    user_clicks = np.bincount(users[training], weights=data['counts'][training], minlength=nusers)
    days = np.bincount(users[training], minlength=nusers)
    chosen = np.flatnonzero(training & (days[users] >= 2) & (user_clicks[users] > 0))
    chosen = chosen[np.argsort(users[chosen], kind='stable')]
    groups = pd.factorize(users[chosen], sort=True)[0]
    starts = np.r_[0, np.flatnonzero(np.diff(groups))+1]
    x = np.ascontiguousarray(data['x'][np.ix_(chosen, columns(data, arm))])
    y = data['counts'][chosen]
    totals = np.bincount(groups, weights=y)
    offset = np.log(data['mass'][chosen])
    result = minimize(objective, np.zeros(x.shape[1]),
                      args=(x, y, groups, starts, totals, offset, arm['ridge']), jac=True,
                      method='L-BFGS-B', options={'maxiter': 500, 'maxcor': 15,
                                                'ftol': 1e-12, 'gtol': 2e-7})
    assert result.success and np.isfinite(result.x).all(), str(result.message)
    name = f'day{day}_{arm["name"]}'
    path = OUT/(name+'.npy')
    np.save(path, result.x)
    record = dict(arm, day=day, name=name, model_sha256=digest(path),
                  fit_rows=len(chosen), fit_groups=len(totals), fit_clicks=int(y.sum()),
                  fit_max_date=int(data['keys'][chosen, 0].max()),
                  fit_keys_sha256=array_hash(data['keys'][chosen]),
                  fit_labels_sha256=array_hash(y), optimizer_success=bool(result.success),
                  iterations=int(result.nit), loss=float(result.fun),
                  maximum_gradient=float(np.max(np.abs(result.jac))))
    assert record['fit_max_date'] < 20191100+day
    emit('conditional_daily_model_complete', **record)
    return record


def daily_targets(data, beta, arm, base):
    training, users = data['training'], data['users']
    nusers = int(users.max())+1
    effect = np.exp(np.clip(data['x'][:, columns(data, arm)]@beta, -4., 4.))
    weighted = np.bincount(users[training], weights=data['mass'][training]*effect[training], minlength=nusers)
    prior = np.bincount(users[training], weights=data['mass'][training], minlength=nusers)
    history_effect = np.divide(weighted, prior, out=np.ones(nusers), where=prior > 0)
    clicks = np.bincount(users[training], weights=data['counts'][training], minlength=nusers)
    relative = np.clip(effect/history_effect[users], .05, 20.)
    absolute = data['mass']*effect*(clicks[users]+10.)/(weighted[users]+10.)
    base_mass = np.bincount(data['row_groups'], weights=-np.log1p(-base), minlength=len(users))
    return {'relative_dynamic': base_mass*relative, 'posterior_absolute': absolute}


def apply_target(base, row_groups, target, power):
    hazard = -np.log1p(-base)
    mass = np.bincount(row_groups, weights=hazard, minlength=len(target))
    ratio = np.clip(target/mass.clip(1e-12), .05, 20.)**power
    return -np.expm1(-hazard*ratio[row_groups])


def check_sources():
    protocol = read(OUT/'protocol.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    return protocol


def fit():
    assert not OUT.exists(), 'Preserve prior and failed conditional daily runs'
    assert read(BASE/'verification.json')['status'] == read(QUALITY/'verification.json')['status'] == 'passed'
    raw_hashes = read(BASE/'protocol.json')['raw_datasets_sha256']
    for name, expected in raw_hashes.items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    paths = [Path(__file__), HERE/'test_conditional_daily.py', HERE/'run_conditional_daily.ps1',
             HERE/'bayesian_daily_offset.py', HERE/'news_quality_features.py', HERE/'fit_stacker.py',
             HERE/'rich_userday_activity.py', HERE/'verify_stacker.py',
             BASE/'protocol.json', BASE/'verification.json', QUALITY/'verification.json']
    for day in [15, 16]:
        paths += [BASE/f'day{day}_prediction.npy', BASE/f'day{day}_indices.npy',
                  QUALITY/f'quality{day}.npz', QUALITY/f'input_contract{day}.json']
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json', {
        'dates': [20191115, 20191116], 'arms': ARMS, 'powers': POWERS, 'modes': MODES,
        'model': 'Within-user conditional Poisson across earlier dates, hierarchical causal mass offset',
        'fit_scope': 'Users with at least two earlier dates and at least one earlier ad click',
        'standardization': 'Subtract training-only per-user means, divide by training-only within-user RMS, clip to [-8,8]',
        'relative_forecast': 'Query effect divided by earlier mass-weighted mean effect for this user',
        'absolute_forecast': 'Causal expected mass times effect times Gamma(10,10) posterior user factor',
        'selection': 'Both-date coefficients lock before either held-date ad-label evaluation; reused development dates',
        'held_date_labels_used_for_fitting': False, 'inferred17_labels_read': False,
        'raw_datasets_sha256': raw_hashes, 'sources_sha256': {str(p): digest(p) for p in paths},
    })
    locks = []
    for day in [15, 16]:
        data = build(day)
        write(OUT/f'input_contract{day}.json', data['contract'])
        np.save(OUT/f'day{day}_indices.npy', data['rows'])
        for arm in ARMS:
            locks.append(fit_arm(data, day, arm))
        del data
        gc.collect()
    write(OUT/'models_locked.json', {'models': locks, 'protocol_sha256': digest(OUT/'protocol.json')})
    evaluate(False)


def evaluate(verifying):
    check_sources()
    locked = read(OUT/'models_locked.json')
    assert locked['protocol_sha256'] == digest(OUT/'protocol.json')
    # Model coefficients for both dates are already locked before this scoring read.
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
            path = OUT/(arm['name']+'.npy')
            assert digest(path) == arm['model_sha256']
            targets = daily_targets(data, np.load(path), arm, base)
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
        emit('conditional_daily_scored', day=day, best=best, verifying=verifying)
        del data
        gc.collect()
    check_sources()
    report = {'status': 'passed' if verifying else 'evaluated', 'dates': results,
              'original_datasets_unchanged': True,
              'both_targets_met': all(r['target_met'] for r in results.values())}
    write(OUT/('verification.json' if verifying else 'result.json'), report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    evaluate(True) if parser.parse_args().verify else fit()
