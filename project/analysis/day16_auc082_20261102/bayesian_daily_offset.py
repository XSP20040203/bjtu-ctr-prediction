"""Learn rich daily response residuals around strictly prior hierarchical rates."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from fit_stacker import ROOT, HERE, digest, write
from rich_group_residual import lgb
from rich_userday_activity import correct, array_hash
from rolling_rich_daily_catboost import inputs, contract, dependencies
from verify_stacker import rank_sum_auc

OUT = HERE/'runs/bayesian_daily_offset_v1'
BASES = {15: HERE/'runs/augmented_rows_v1_day15/prediction.npy',
         16: HERE/'runs/augmented_rows_v1_day16/prediction.npy'}
ARMS = [{'name': 'poisson_l15', 'objective': 'poisson'},
        {'name': 'binary_l15', 'objective': 'binary'}]
CHECKS = [250, 600, 1000]
POWERS = [0., .125, .25, .5, .75]
PARAMS = {'learning_rate': .025, 'num_leaves': 15, 'min_data_in_leaf': 200,
          'lambda_l2': 40., 'cat_l2': 40., 'cat_smooth': 60., 'max_bin': 31,
          'feature_fraction': .8, 'num_threads': 4, 'seed': 151634,
          'verbosity': -1, 'deterministic': True, 'force_col_wise': True,
          'feature_pre_filter': False, 'boost_from_average': False}


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def lookup(past, query, fields):
    if not len(past):
        return np.zeros(len(query)), np.zeros(len(query))
    grouped = past.groupby(fields, sort=False).label.agg(['sum', 'size'])
    if len(fields) == 1:
        index = pd.Index(query[fields[0]])
    else:
        index = pd.MultiIndex.from_frame(query[fields])
    aligned = grouped.reindex(index, fill_value=0)
    return aligned['sum'].to_numpy(), aligned['size'].to_numpy()


def prior_mass(raw, keys, cutoff):
    """Own-date labels never affect own-date rates, including historical dates."""
    raw = raw.loc[raw.pt_d <= cutoff].copy()
    raw.loc[raw.pt_d >= cutoff, 'label'] = -1
    index = pd.MultiIndex.from_arrays(keys.T)
    mass = np.zeros(len(keys), dtype='float64')
    audit = []
    for day in sorted(raw.pt_d.unique()):
        past = raw.loc[raw.pt_d < day]
        query = raw.loc[raw.pt_d == day]
        assert np.isin(past.label, [0, 1]).all()
        assert len(past) == 0 or int(past.pt_d.max()) < int(day)
        global_rate = (float(past.label.sum())+8.4)/(len(past)+1000.)
        user_click, user_size = lookup(past, query, ['user_id'])
        app_click, app_size = lookup(past, query, ['spread_app_id'])
        slot_click, slot_size = lookup(past, query, ['slot_id'])
        pair_click, pair_size = lookup(past, query, ['user_id', 'spread_app_id'])
        user_rate = (user_click+80.*global_rate)/(user_size+80.)
        app_rate = (app_click+100.*global_rate)/(app_size+100.)
        slot_rate = (slot_click+100.*global_rate)/(slot_size+100.)
        pair_prior = (user_rate*app_rate/global_rate).clip(1e-5, .25)
        pair_rate = (pair_click+40.*pair_prior)/(pair_size+40.)
        rate = (pair_rate*slot_rate/global_rate).clip(1e-5, .25)
        positions = index.get_indexer(pd.MultiIndex.from_frame(query[['pt_d', 'user_id']]))
        assert np.all(positions >= 0)
        mass += np.bincount(positions, weights=rate, minlength=len(keys))
        audit.append({'query_date': int(day), 'label_max_date': int(past.pt_d.max()) if len(past) else None,
                      'fit_rows': len(past), 'query_rows': len(query), 'global_rate': global_rate,
                      'query_labels_used': False})
    assert np.all(mass > 0) and np.isfinite(mass).all()
    return mass, audit


def build(day):
    cutoff = 20191100+day
    data = inputs(day)
    raw = pd.read_csv(ROOT/'2026ml/train.csv',
                       usecols=['pt_d', 'label', 'user_id', 'spread_app_id', 'slot_id'], dtype='int32')
    mass, audit = prior_mass(raw, data['keys'], cutoff)
    data['frame']['log_bayesian_click_mass'] = np.log(mass)
    data['bayesian_mass'] = mass
    data['bayesian_audit'] = audit
    data['contract'] = {'native_contract': contract(data, day),
                        'bayesian_mass_sha256': array_hash(mass), 'bayesian_audit': audit,
                        'original_raw_rows_count': int(np.sum(raw.pt_d.to_numpy() <= cutoff))}
    assert np.all(data['counts'][data['keys'][:, 0] >= cutoff] == -1)
    return data


def offsets(data, objective):
    if objective == 'poisson':
        return np.log(data['bayesian_mass'])
    return logit((-np.expm1(-data['bayesian_mass'])).clip(1e-6, 1-1e-6))


def fit():
    assert not OUT.exists(), 'Preserving Bayesian daily experiment'
    expected = json.loads((HERE/'runs/weighted_v1_day16/protocol.json').read_text(encoding='utf8'))['raw_datasets_sha256']
    for name, h in expected.items():
        assert digest(ROOT/'2026ml'/name) == h, name
    paths = list(dict.fromkeys([Path(__file__), HERE/'rich_group_residual.py']+dependencies()
                             +list(BASES.values())+[path.parent/'verification.json' for path in BASES.values()]))
    for path in BASES.values():
        assert json.loads((path.parent/'verification.json').read_text(encoding='utf8'))['status'] == 'passed'
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json', {'dates': [20191115, 20191116], 'params': PARAMS, 'arms': ARMS,
          'checkpoints': CHECKS, 'powers': POWERS, 'user_prior_exposures': 80.,
          'app_slot_prior_exposures': 100., 'user_app_prior_exposures': 40.,
          'global_initial_prior': [8.4, 1000.], 'prediction_rate_bounds': [1e-5, .25],
          'raw_datasets_sha256': expected, 'sources_sha256': {str(path): digest(path) for path in paths},
          'selection': 'All models lock before scoring either full held cohort; dates repeatedly reused for development',
          'baseline': 'Hierarchical user, app, user/app and slot rates using only strictly earlier ad labels',
          'own_held_date_ad_outcomes_fit': False, 'inferred17_labels_read': False})
    locks = []
    for day in [15, 16]:
        data = build(day)
        write(OUT/f'input_contract{day}.json', data['contract'])
        np.save(OUT/f'day{day}_indices.npy', data['rows'])
        train = np.flatnonzero(data['keys'][:, 0] < 20191100+day)
        assert np.all(data['counts'][train] >= 0)
        emit('bayesian_daily_ready', day=day, shape=list(data['frame'].shape), training_groups=len(train))
        for arm in ARMS:
            target = data['counts'][train].astype('float64')
            if arm['objective'] == 'binary':
                target = (target > 0).astype('float64')
            dataset = lgb.Dataset(data['frame'].iloc[train], label=target,
                                  init_score=offsets(data, arm['objective'])[train],
                                  categorical_feature=data['categories'], params=PARAMS,
                                  free_raw_data=True)
            model = lgb.train(dict(PARAMS, objective=arm['objective']), dataset, num_boost_round=max(CHECKS))
            for trees in CHECKS:
                name = f'day{day}_{arm["name"]}_trees{trees}'
                path = OUT/(name+'.txt')
                path.write_text(model.model_to_string(num_iteration=trees), encoding='utf8')
                locks.append(dict(arm, day=day, trees=trees, name=name, model_sha256=digest(path)))
            emit('bayesian_daily_arm_complete', day=day, name=arm['name'])
            del dataset, model
            gc.collect()
        del data
        gc.collect()
    write(OUT/'models_locked.json', {'models': locks, 'protocol_sha256': digest(OUT/'protocol.json')})
    evaluate(False)


def evaluate(verifying):
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    locked = json.loads((OUT/'models_locked.json').read_text(encoding='utf8'))
    assert locked['protocol_sha256'] == digest(OUT/'protocol.json')
    for path, h in protocol['sources_sha256'].items():
        assert digest(path) == h, path
    for name, h in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == h, name
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    results = {}
    for day in [15, 16]:
        data = build(day)
        assert data['contract'] == json.loads((OUT/f'input_contract{day}.json').read_text(encoding='utf8'))
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
        np.testing.assert_array_equal(rows, data['rows'])
        np.testing.assert_array_equal(rows, np.load(OUT/f'day{day}_indices.npy'))
        y = raw.label.to_numpy()[rows]
        assert len(y) == {15: 424780, 16: 421983}[day] and int(y.sum()) == {15: 3523, 16: 3724}[day]
        base = np.load(BASES[day])
        best = {'name': 'base', 'power': 0., 'auc': rank_sum_auc(y, base)}
        best_p = base.copy()
        records = []
        held = np.flatnonzero(data['valid'])
        for arm in locked['models']:
            if arm['day'] != day:
                continue
            path = OUT/(arm['name']+'.txt')
            assert digest(path) == arm['model_sha256']
            model = lgb.Booster(model_str=path.read_text(encoding='utf8'))
            z = model.predict(data['frame'].iloc[held], raw_score=True, num_threads=4)
            z += offsets(data, arm['objective'])[held]
            group = np.zeros(len(data['keys']))
            group[held] = expit(z) if arm['objective'] == 'binary' else np.exp(z.clip(-15, 8))
            for power in POWERS:
                p = correct(base, group, data['row_groups'], arm['objective'], power)
                record = {'name': arm['name'], 'power': power, 'auc': rank_sum_auc(y, p)}
                records.append(record)
                if record['auc'] > best['auc']:
                    best = record
                    best_p = p.copy()
        error = 0.
        if verifying:
            error = float(np.max(np.abs(best_p-np.load(OUT/f'day{day}_prediction.npy'))))
            assert error < 1e-12, error
        else:
            np.save(OUT/f'day{day}_prediction.npy', best_p)
        results[str(20191100+day)] = {'auc': best['auc'], 'best': best, 'records': records,
            'rows': len(y), 'clicks': int(y.sum()), 'maximum_replay_error': error,
            'evaluation_day_labels_fit': False, 'own_held_day_ad_labels_fit': False,
            'target_met': best['auc'] >= {15: .815, 16: .82}[day]}
        emit('bayesian_daily_scored', day=day, best=best)
        del data
        gc.collect()
    report = {'status': 'passed' if verifying else 'evaluated', 'dates': results,
              'original_datasets_unchanged': True,
              'both_targets_met': all(record['target_met'] for record in results.values())}
    write(OUT/('verification.json' if verifying else 'result.json'), report)
    emit('bayesian_daily_finished', status=report['status'], dates={d: r['auc'] for d, r in results.items()})


def self_test():
    raw = pd.DataFrame({'pt_d': [20191111, 20191111, 20191112, 20191112, 20191113],
                        'user_id': [1, 2, 1, 3, 1], 'spread_app_id': [4, 4, 4, 5, 4],
                        'slot_id': [2, 2, 2, 3, 2], 'label': [1, 0, 0, 0, 1]})
    keys = np.array([[20191111, 1], [20191111, 2], [20191112, 1], [20191112, 3]])
    mass, audit = prior_mass(raw, keys, 20191112)
    changed = raw.copy()
    changed.loc[changed.pt_d >= 20191112, 'label'] = 1-changed.loc[changed.pt_d >= 20191112, 'label']
    replay, audit2 = prior_mass(changed, keys, 20191112)
    np.testing.assert_array_equal(mass, replay)
    assert audit == audit2
    np.testing.assert_allclose(mass[:2], [.0084, .0084], rtol=0, atol=1e-15)
    assert mass[2] > mass[3]
    for record in audit:
        assert record['label_max_date'] is None or record['label_max_date'] < record['query_date']
    emit('bayesian_daily_boundary_self_test_passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else evaluate(True) if args.verify else fit()
