"""Add content-adjusted news response to strict-past rich daily ad models."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from bayesian_daily_offset import build as native_build, offsets, lgb
from news_quality_features import READ, GROUPS, ALPHAS, build_quality, self_test
from rich_userday_activity import array_hash, correct
from fit_stacker import ROOT, HERE, digest, write
from verify_stacker import rank_sum_auc

OUT = HERE/'runs/news_quality_daily_v1'
BASE = HERE/'runs/bayesian_daily_offset_v1'
ORIGINAL = {15: HERE/'runs/augmented_rows_v1_day15', 16: HERE/'runs/augmented_rows_v1_day16'}
ARMS = [{'name': 'binary_l15', 'objective': 'binary', 'num_leaves': 15},
        {'name': 'binary_l31', 'objective': 'binary', 'num_leaves': 31},
        {'name': 'poisson_l15', 'objective': 'poisson', 'num_leaves': 15}]
CHECKS = [400, 900, 1600]
POWERS = [.0625, .125, .25, .5]
PARAMS = {'learning_rate': .025, 'min_data_in_leaf': 180, 'lambda_l2': 40.,
          'cat_l2': 40., 'cat_smooth': 60., 'max_bin': 31, 'feature_fraction': .8,
          'num_threads': 4, 'seed': 151635, 'verbosity': -1, 'deterministic': True,
          'force_col_wise': True, 'feature_pre_filter': False, 'boost_from_average': False}


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def build(day):
    data = native_build(day)
    news = pd.read_csv(ROOT/'2026ml/train_feeds.csv', usecols=READ,
                       dtype={'i_docId': str, 'i_s_sourceId': str})
    quality, names = build_quality(news, data['keys'], 20191100+day)
    extra = pd.DataFrame(quality, columns=names)
    data['frame'] = pd.concat([data['frame'], extra], axis=1)
    data['contract'] = {'native_contract': data['contract'], 'quality_sha256': array_hash(quality),
                        'quality_names': names, 'shape': list(data['frame'].shape),
                        'training_max_date': int(data['keys'][data['training'], 0].max()),
                        'own_held_day_labels_masked': bool(np.all(data['counts'][data['valid']] == -1))}
    data['quality'] = quality
    assert data['contract']['training_max_date'] < 20191100+day
    assert data['contract']['own_held_day_labels_masked']
    emit('news_quality_ready', day=day, shape=list(data['frame'].shape))
    return data


def check_sources():
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    return protocol


def fit():
    assert not OUT.exists(), 'Preserving all previous quality experiments'
    assert json.loads((BASE/'verification.json').read_text(encoding='utf8'))['status'] == 'passed'
    native_protocol = json.loads((BASE/'protocol.json').read_text(encoding='utf8'))
    paths = [Path(__file__), HERE/'news_quality_features.py', HERE/'bayesian_daily_offset.py',
             BASE/'verification.json', BASE/'protocol.json']
    paths += [Path(p) for p in native_protocol['sources_sha256']]
    for day in [15, 16]:
        paths += [BASE/f'day{day}_prediction.npy', BASE/f'day{day}_indices.npy',
                  ORIGINAL[day]/'prediction.npy', ORIGINAL[day]/'verification.json']
    for name, expected in native_protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    tests = self_test()
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json', {'dates': [20191115, 20191116], 'params': PARAMS,
        'arms': ARMS, 'checkpoints': CHECKS, 'powers': POWERS, 'content_groups': GROUPS,
        'content_smoothing_alphas': ALPHAS, 'self_test': tests,
        'raw_datasets_sha256': native_protocol['raw_datasets_sha256'],
        'sources_sha256': {str(p): digest(p) for p in dict.fromkeys(paths)},
        'source_protocol': 'Same-day supplied news feedback, content expectation excludes own user/day outcomes; no future source dates',
        'ad_training': 'Strictly earlier advertisement labels only; current-date targets masked',
        'recipes': ['correct_current_best', 'replace_daily_correction_on_original_augmented'],
        'inferred17_labels_read': False,
        'selection': 'All date/model checkpoints locked before either full held ad cohort is evaluated; reused development dates'})
    locks = []
    for day in [15, 16]:
        data = build(day)
        write(OUT/f'input_contract{day}.json', data['contract'])
        np.save(OUT/f'day{day}_indices.npy', data['rows'])
        np.savez_compressed(OUT/f'quality{day}.npz', keys=data['keys'], values=data['quality'])
        train = np.flatnonzero(data['training'])
        for arm in ARMS:
            target = data['counts'][train].astype('float64')
            if arm['objective'] == 'binary':
                target = (target > 0).astype('float64')
            dataset = lgb.Dataset(data['frame'].iloc[train], label=target,
                        init_score=offsets(data, arm['objective'])[train],
                        categorical_feature=data['categories'], params=PARAMS, free_raw_data=True)
            model = lgb.train(dict(PARAMS, objective=arm['objective'], num_leaves=arm['num_leaves']),
                              dataset, num_boost_round=max(CHECKS))
            for trees in CHECKS:
                name = f'day{day}_{arm["name"]}_trees{trees}'
                path = OUT/(name+'.txt')
                path.write_text(model.model_to_string(num_iteration=trees), encoding='utf8')
                locks.append(dict(arm, day=day, trees=trees, name=name, model_sha256=digest(path)))
            emit('news_quality_arm_complete', day=day, name=arm['name'])
            del dataset, model
            gc.collect()
        del data
        gc.collect()
    write(OUT/'models_locked.json', {'models': locks, 'protocol_sha256': digest(OUT/'protocol.json')})
    evaluate(False)


def evaluate(verifying):
    check_sources()
    locks = json.loads((OUT/'models_locked.json').read_text(encoding='utf8'))
    assert locks['protocol_sha256'] == digest(OUT/'protocol.json')
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    results = {}
    for day in [15, 16]:
        data = build(day)
        assert data['contract'] == json.loads((OUT/f'input_contract{day}.json').read_text(encoding='utf8'))
        cached = np.load(OUT/f'quality{day}.npz')
        np.testing.assert_array_equal(cached['keys'], data['keys'])
        np.testing.assert_array_equal(cached['values'], data['quality'])
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
        np.testing.assert_array_equal(rows, data['rows'])
        np.testing.assert_array_equal(rows, np.load(BASE/f'day{day}_indices.npy'))
        np.testing.assert_array_equal(rows, np.load(OUT/f'day{day}_indices.npy'))
        y = raw.label.to_numpy()[rows]
        assert len(y) == {15: 424780, 16: 421983}[day] and int(y.sum()) == {15: 3523, 16: 3724}[day]
        base = np.load(BASE/f'day{day}_prediction.npy')
        anchor = np.load(ORIGINAL[day]/'prediction.npy')
        best = {'name': 'base', 'recipe': 'base', 'power': 0., 'auc': rank_sum_auc(y, base)}
        winner = base.copy()
        records = [best]
        held = np.flatnonzero(data['valid'])
        for arm in locks['models']:
            if arm['day'] != day:
                continue
            path = OUT/(arm['name']+'.txt')
            assert digest(path) == arm['model_sha256']
            model = lgb.Booster(model_str=path.read_text(encoding='utf8'))
            margin = model.predict(data['frame'].iloc[held], raw_score=True, num_threads=4)
            margin += offsets(data, arm['objective'])[held]
            group = np.zeros(len(data['keys']))
            group[held] = expit(margin) if arm['objective'] == 'binary' else np.exp(margin.clip(-15, 8))
            for power in POWERS:
                for recipe, reference in [('correct_current_best', base), ('replace_daily_correction_on_original_augmented', anchor)]:
                    p = correct(reference, group, data['row_groups'], arm['objective'], power)
                    record = {'name': arm['name'], 'objective': arm['objective'], 'recipe': recipe,
                              'power': power, 'auc': rank_sum_auc(y, p)}
                    records.append(record)
                    if record['auc'] > best['auc']:
                        best, winner = record, p.copy()
        error = 0.
        if verifying:
            error = float(np.max(np.abs(winner-np.load(OUT/f'day{day}_prediction.npy'))))
            assert error < 1e-12
        else:
            np.save(OUT/f'day{day}_prediction.npy', winner)
        results[str(20191100+day)] = {'auc': best['auc'], 'best': best, 'records': records,
            'rows': len(y), 'clicks': int(y.sum()), 'maximum_replay_error': error,
            'evaluation_day_labels_fit': False, 'own_held_day_ad_labels_fit': False,
            'target_met': best['auc'] >= {15: .815, 16: .82}[day]}
        emit('news_quality_scored', day=day, best=best, verifying=verifying)
        del data
        gc.collect()
    report = {'status': 'passed' if verifying else 'evaluated', 'dates': results,
              'original_datasets_unchanged': True,
              'both_targets_met': all(item['target_met'] for item in results.values())}
    write(OUT/('verification.json' if verifying else 'result.json'), report)
    emit('news_quality_finished', status=report['status'], dates={day: item['auc'] for day, item in results.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    evaluate(True) if args.verify else fit()
