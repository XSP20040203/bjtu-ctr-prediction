"""Chronological row boosting with original numerics and supervised daily context."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import gc
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from fit_stacker import ROOT, HERE, digest, write
from rich_group_residual import lgb
from rich_userday_activity import array_hash
from augmented_row_catboost import build as augmented_build
import global_row_context as rows16
import global_row_context15 as rows15
from news_quality_daily import build as daily_build
from verify_stacker import rank_sum_auc
sys.path.insert(0, str(ROOT/'analysis/auc085_next'))
from prepare_source_context import (iter_feature_columns, _read_csv, AD_COLUMNS,
                                    SOURCE_INPUT_COLUMNS, feature_names as exact_source_names)

OUT = HERE/'runs/rich_augmented_rows_v1'
DAILY = HERE/'runs/news_quality_daily_v1'
ARMS = [{'name': 'leaves31_weight5', 'num_leaves': 31, 'min_data_in_leaf': 250, 'positive_weight': 5.},
        {'name': 'leaves63_weight5', 'num_leaves': 63, 'min_data_in_leaf': 150, 'positive_weight': 5.}]
CHECKS = [400, 900, 1800]
WEIGHTS = [.05, .1, .2, .35, .5]
PARAMS = {'objective': 'binary', 'learning_rate': .03, 'lambda_l2': 30.,
          'cat_l2': 40., 'cat_smooth': 60., 'max_bin': 63, 'max_cat_threshold': 32,
          'feature_fraction': .85, 'bagging_fraction': .8, 'bagging_freq': 1,
          'num_threads': 4, 'seed': 151636, 'verbosity': -1, 'deterministic': True,
          'force_col_wise': True, 'feature_pre_filter': False}


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def select_daily(day):
    data = daily_build(day)
    locks = json.loads((DAILY/'models_locked.json').read_text(encoding='utf8'))['models']
    scores = np.zeros(data['frame'].shape[1])
    # These fixed last-checkpoint selectors were fitted only on earlier advertisement dates.
    for objective in ['binary', 'poisson']:
        name = f'day{day}_{objective}_l15_trees1600'
        record = next(item for item in locks if item['name'] == name)
        path = DAILY/(name+'.txt')
        assert digest(path) == record['model_sha256']
        model = lgb.Booster(model_str=path.read_text(encoding='utf8'))
        gain = model.feature_importance('gain')
        assert len(gain) == len(scores)
        scores += gain/max(gain.sum(), 1e-8)
    names = data['frame'].columns.tolist()
    categorical = set(data['categories'])
    ordinary = [i for i, name in enumerate(names) if name not in categorical and not name.startswith('quality__')]
    quality = [i for i, name in enumerate(names) if name.startswith('quality__')]
    ordinary = sorted(ordinary, key=lambda i: (-scores[i], i))[:128]
    quality = sorted(quality, key=lambda i: (-scores[i], i))[:64]
    columns = ordinary+quality
    assert len(columns) == len(set(columns)) == 192
    chosen = data['frame'].iloc[:, columns].to_numpy(dtype='float32')
    result = chosen.copy(), data['keys'].copy(), [names[i] for i in columns], columns, data['contract']
    del data
    gc.collect()
    return result


def build(day, directory):
    directory.mkdir(parents=True, exist_ok=False)
    daily, keys, extra_names, columns, daily_contract = select_daily(day)
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'user_id', 'label'], dtype='int32')
    module = rows15 if day == 15 else rows16
    chosen, expected_weights = module.selected_training(raw.label.to_numpy(), raw.pt_d.to_numpy())
    held_rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
    rows = np.sort(np.r_[chosen, held_rows])
    ads = _read_csv(ROOT/'2026ml/train.csv', AD_COLUMNS).iloc[rows].reset_index(drop=True)
    source = _read_csv(ROOT/'2026ml/train_feeds.csv', SOURCE_INPUT_COLUMNS)
    source = source.loc[source.e_et <= 20191100+day]
    exact = np.lib.format.open_memmap(directory/'exact_source.npy', mode='w+', dtype='float32', shape=(len(rows), 80))
    source_names = []
    for i, (name, values) in enumerate(iter_feature_columns(ads, source)):
        exact[:, i] = values
        source_names.append(name)
    exact.flush()
    assert source_names == exact_source_names() and len(source_names) == 80
    del ads, source, values
    gc.collect()
    frame, train, held, y, weights, actual_held_rows, categories, original_contract = augmented_build(day, directory/'original')
    np.testing.assert_array_equal(held_rows, actual_held_rows)
    np.testing.assert_array_equal(rows[train], chosen)
    np.testing.assert_array_equal(weights, expected_weights)
    np.testing.assert_array_equal(y, raw.label.to_numpy()[chosen])
    assert raw.pt_d.to_numpy()[chosen].max() < 20191100+day
    positions = pd.MultiIndex.from_arrays(keys.T).get_indexer(pd.MultiIndex.from_frame(raw.iloc[rows][['pt_d', 'user_id']]))
    assert np.all(positions >= 0)
    width = frame.shape[1]
    x = np.lib.format.open_memmap(directory/'features.npy', mode='w+', dtype='float32',
                                 shape=(len(rows), width+len(extra_names)+len(source_names)))
    for start in range(0, len(rows), 20000):
        stop = min(len(rows), start+20000)
        x[start:stop, :width] = frame.iloc[start:stop].to_numpy(dtype='float32')
        x[start:stop, width:width+len(extra_names)] = daily[positions[start:stop]]
        x[start:stop, width+len(extra_names):] = exact[start:stop]
        assert np.isfinite(x[start:stop]).all()
    x.flush()
    names = original_contract['feature_names']+['rich_daily__'+name for name in extra_names]+source_names
    assert len(names) == x.shape[1] == 744
    categories = original_contract['categorical_columns']
    contract = {'shape': list(x.shape), 'feature_names': names, 'categorical_columns': categories,
        'matrix_sha256': digest(directory/'features.npy'), 'original_contract': original_contract,
        'exact_source_sha256': digest(directory/'exact_source.npy'),
        'daily_contract': daily_contract, 'selector_columns': columns,
        'training_rows_sha256': array_hash(rows[train]), 'training_labels_sha256': array_hash(y),
        'weights_sha256': array_hash(weights), 'held_rows_sha256': array_hash(held_rows),
        'max_training_date': int(raw.pt_d.to_numpy()[chosen].max()),
        'own_held_ad_labels_used_to_select_daily_columns': False}
    del frame, daily, raw, exact
    gc.collect()
    emit('rich_augmented_rows_ready', day=day, shape=list(x.shape), training_rows=len(train))
    return x, train, held, y, weights, held_rows, categories, contract


def check_sources():
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    return protocol


def fit():
    assert not OUT.exists(), 'Preserving earlier rich row experiments'
    source = json.loads((DAILY/'verification.json').read_text(encoding='utf8'))
    assert source['status'] == 'passed'
    native_protocol = json.loads((DAILY/'protocol.json').read_text(encoding='utf8'))
    paths = [Path(__file__), DAILY/'verification.json', DAILY/'models_locked.json']
    paths += [Path(path) for path in native_protocol['sources_sha256']]
    paths += [HERE/'augmented_row_catboost.py', HERE/'global_row_context.py', HERE/'global_row_context15.py',
              ROOT/'analysis/auc085_next/prepare_source_context.py']
    for day in [15, 16]:
        paths += [DAILY/f'day{day}_prediction.npy', DAILY/f'day{day}_indices.npy']
        augmented = HERE/f'runs/augmented_rows_v1_day{day}'
        protocol = json.loads((augmented/'protocol.json').read_text(encoding='utf8'))
        paths += [Path(path) for path in protocol['sources_sha256']]
        for objective in ['binary', 'poisson']:
            paths += [DAILY/f'day{day}_{objective}_l15_trees1600.txt']
    for name, expected in native_protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json', {'dates': [20191115, 20191116], 'params': PARAMS, 'arms': ARMS,
        'checkpoints': CHECKS, 'blend_weights': WEIGHTS,
        'features': '472 augmented row columns +128 rich numeric daily columns +64 content-quality columns +80 exact news snapshot row-match columns',
        'selector': 'Mean normalized gains of fixed prior-ad-trained binary/Poisson l15 checkpoints; no held AUC used to pick selector',
        'sampling': 'All prior positives and fixed one-third prior negatives; inverse inclusion weights',
        'sources_sha256': {str(path): digest(path) for path in dict.fromkeys(paths)},
        'raw_datasets_sha256': native_protocol['raw_datasets_sha256'],
        'own_held_ad_labels_fit': False, 'own_held_ad_labels_feature_selection': False,
        'inferred17_labels_read': False,
        'selection': 'All models for both dates lock before full development-date scoring'})
    locks = []
    for day in [15, 16]:
        x, train, held, y, weights, rows, categories, contract = build(day, OUT/f'input_day{day}')
        write(OUT/f'input_contract{day}.json', contract)
        np.save(OUT/f'day{day}_indices.npy', rows)
        for arm in ARMS:
            dataset = lgb.Dataset(np.asarray(x[train]), label=y, weight=weights,
                        categorical_feature=categories, params=PARAMS, free_raw_data=True)
            model = lgb.train(dict(PARAMS, num_leaves=arm['num_leaves'],
                         min_data_in_leaf=arm['min_data_in_leaf'], scale_pos_weight=arm['positive_weight']),
                         dataset, num_boost_round=max(CHECKS))
            for trees in CHECKS:
                name = f'day{day}_{arm["name"]}_trees{trees}'
                path = OUT/(name+'.txt')
                path.write_text(model.model_to_string(num_iteration=trees), encoding='utf8')
                locks.append(dict(arm, day=day, trees=trees, name=name, model_sha256=digest(path)))
            emit('rich_augmented_rows_arm_complete', day=day, name=arm['name'])
            del dataset, model
            gc.collect()
        del x
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
        if verifying:
            x, _, held, _, _, rows, _, contract = build(day, OUT/f'independent_day{day}')
            assert contract == json.loads((OUT/f'input_contract{day}.json').read_text(encoding='utf8'))
        else:
            x = np.load(OUT/f'input_day{day}/features.npy', mmap_mode='r')
            module = rows15 if day == 15 else rows16
            chosen, _ = module.selected_training(raw.label.to_numpy(), raw.pt_d.to_numpy())
            rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
            selected = np.sort(np.r_[chosen, rows])
            held = np.flatnonzero(raw.pt_d.to_numpy()[selected] == 20191100+day)
        np.testing.assert_array_equal(rows, np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day))
        np.testing.assert_array_equal(rows, np.load(OUT/f'day{day}_indices.npy'))
        np.testing.assert_array_equal(rows, np.load(DAILY/f'day{day}_indices.npy'))
        y = raw.label.to_numpy()[rows]
        assert len(y) == {15: 424780, 16: 421983}[day] and int(y.sum()) == {15: 3523, 16: 3724}[day]
        base = np.load(DAILY/f'day{day}_prediction.npy')
        a = logit(base.clip(1e-6, 1-1e-6))
        winner = base.copy()
        best = {'name': 'base', 'recipe': 'base', 'auc': rank_sum_auc(y, base)}
        records, forecasts, last = [best], {}, []
        for arm in locks['models']:
            if arm['day'] != day:
                continue
            path = OUT/(arm['name']+'.txt')
            assert digest(path) == arm['model_sha256']
            model = lgb.Booster(model_str=path.read_text(encoding='utf8'))
            p = expit(model.predict(x[held], raw_score=True, num_threads=4)-np.log(arm['positive_weight']))
            z = logit(p.clip(1e-6, 1-1e-6))
            forecasts[arm['name']] = (z-z.mean())*a.std()/max(z.std(), 1e-6)+a.mean()
            if arm['trees'] == max(CHECKS):
                last.append(forecasts[arm['name']])
            record = {'name': arm['name'], 'recipe': 'single', 'auc': rank_sum_auc(y, p)}
            records.append(record)
            if record['auc'] > best['auc']:
                best, winner = record, p.copy()
        assert len(last) == 2
        forecasts['mean_last_two'] = np.mean(last, axis=0)
        for name, z in forecasts.items():
            for weight in WEIGHTS:
                p = expit((1-weight)*a+weight*z)
                record = {'name': name, 'recipe': 'normalized_logit_blend', 'weight': weight,
                          'auc': rank_sum_auc(y, p)}
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
        emit('rich_augmented_rows_scored', day=day, best=best, verifying=verifying)
        del x
        gc.collect()
    report = {'status': 'passed' if verifying else 'evaluated', 'dates': results,
              'original_datasets_unchanged': True,
              'both_targets_met': all(item['target_met'] for item in results.values())}
    write(OUT/('verification.json' if verifying else 'result.json'), report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    evaluate(True) if args.verify else fit()
