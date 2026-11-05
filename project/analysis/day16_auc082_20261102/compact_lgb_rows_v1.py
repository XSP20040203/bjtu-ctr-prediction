"""CPU histogram and randomized-threshold row trees on frozen prior-only inputs."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '2'
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from conditional_daily import read, emit
from fit_stacker import HERE, ROOT, digest, write
from rich_userday_activity import array_hash, lgb
from verify_stacker import rank_sum_auc
import row_deepfm_v2 as prepared
from row_deepfm_core_v1 import fit_schema, transform
from cross_user_pair_ranker_v1 import prediction as row_prediction

OUT = HERE / 'runs/compact_lgb_rows_v1'
BASE = HERE / 'runs/compose_transition_priors_v1'
ARMS = [dict(name='histogram31', num_leaves=31, min_data_in_leaf=128, extra_trees=False),
        dict(name='randomized63', num_leaves=63, min_data_in_leaf=96, extra_trees=True)]
CHECKPOINTS = [300, 800, 1600]
WEIGHTS = [.025, .05, .1, .2, .35, .5, .75, 1.]
MODES = ['replace_global', 'add_within']
PARAMS = dict(objective='binary', learning_rate=.025, lambda_l2=40., lambda_l1=0.,
              scale_pos_weight=3., max_bin=127, feature_fraction=.8,
              bagging_fraction=.8, bagging_freq=1, cat_l2=40., cat_smooth=40.,
              max_cat_threshold=64, min_data_per_group=50, num_threads=2,
              seed=151660, verbosity=-1, deterministic=True, force_col_wise=True,
              feature_pre_filter=False, metric='None')


def check():
    protocol = read(OUT / 'protocol.json')
    for filename, expected in protocol['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    return protocol


def frame_from(numeric, encoded, schema):
    values = np.column_stack([numeric, encoded.astype('float32')]).astype('float32', copy=False)
    names = schema['numeric'] + schema['categories']
    assert len(names) == len(set(names)) == values.shape[1] == 423
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(values[:, len(schema['numeric']):].astype('int64'), encoded)
    return pd.DataFrame(values, columns=names, copy=False)


def inputs(day, partition):
    numeric, encoded, schema = prepared.load_data(day, partition)
    frame = frame_from(numeric, encoded, schema)
    audit = read(prepared.OUT / 'inputs_locked.json')[str(day)][partition]['source_audit']
    assert audit['max_training_date'] < 20191100 + day
    contract = dict(shape=list(frame.shape), columns=list(frame), categories=schema['categories'],
                    frame_sha256=array_hash(frame.to_numpy()), source_audit=audit,
                    schema_sha256=digest(prepared.OUT / f'schema{day}.json'))
    del numeric, encoded
    gc.collect()
    return frame, contract


def fit():
    assert not OUT.exists(), 'Preserve prior fits'
    prepared.check()
    assert read(BASE / 'verification.json')['status'] == 'passed'
    paths = [Path(__file__), HERE / 'run_compact_lgb_rows_v1.ps1',
             HERE / 'row_deepfm_v2.py', HERE / 'row_deepfm_core_v1.py',
             BASE / 'verification.json', BASE / 'protocol.json',
             prepared.OUT / 'protocol.json', prepared.OUT / 'inputs_locked.json',
             HERE / 'cross_user_pair_ranker_v1.py', HERE / 'within_user_ranking_core.py',
             HERE / 'rich_userday_activity.py', HERE / 'fit_stacker.py',
             HERE / 'conditional_daily.py', HERE / 'verify_stacker.py']
    for filename, expected in read(prepared.OUT / 'protocol.json')['sources_sha256'].items():
        assert digest(filename) == expected, filename
        paths.append(Path(filename))
    for day in [15, 16]:
        paths += [BASE / f'day{day}_prediction.npy', BASE / f'day{day}_indices.npy',
                  prepared.OUT / f'schema{day}.json', prepared.OUT / f'target{day}.npy',
                  prepared.OUT / f'day{day}_indices.npy']
        for partition in ['train', 'held']:
            paths += [prepared.OUT / filename for filename in
                      read(prepared.OUT / 'inputs_locked.json')[str(day)][partition]['files_sha256']]
    OUT.mkdir()
    write(OUT / 'protocol.json', dict(
        params=PARAMS, arms=ARMS, checkpoints=CHECKPOINTS, weights=WEIGHTS, modes=MODES,
        inputs='384 compact numeric features normalized and clipped using strictly prior training statistics; 39 prior-only category vocabularies',
        training='Full fixed historical source sample, all positives and fixed one-third negatives; no current/future labels',
        selection='All both-date models and candidate hashes locked before scoring; development dates repeatedly reused',
        evaluation_day_labels_fit=False, inferred17_labels_read=False,
        lightgbm_version=lgb.__version__, raw_datasets_sha256=read(BASE / 'protocol.json')['raw_datasets_sha256'],
        sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)}))
    records = []
    for day in [15, 16]:
        frame, contract = inputs(day, 'train')
        write(OUT / f'input_contract{day}.json', contract)
        labels = np.load(prepared.OUT / f'target{day}.npy')
        assert labels.shape == (len(frame),) and np.isin(labels, [0, 1]).all()
        assert array_hash(labels) == contract['source_audit']['training_targets_sha256']
        np.save(OUT / f'day{day}_indices.npy', np.load(prepared.OUT / f'day{day}_indices.npy'))
        for arm in ARMS:
            emit('compact_lgb_fit_started', day=day, arm=arm['name'], rows=len(frame))
            params = dict(PARAMS, **{key: value for key, value in arm.items() if key != 'name'})
            dataset = lgb.Dataset(frame, label=labels, categorical_feature=contract['categories'],
                                  params=params, free_raw_data=True)
            model = lgb.train(params, dataset, num_boost_round=max(CHECKPOINTS))
            for trees in CHECKPOINTS:
                name = f'day{day}_{arm["name"]}_trees{trees}'
                path = OUT / (name + '.txt')
                path.write_text(model.model_to_string(num_iteration=trees), encoding='utf8')
                restored = lgb.Booster(model_str=path.read_text('utf8'))
                assert restored.current_iteration() == trees
                probe = frame.iloc[:2000]
                before = model.predict(probe, raw_score=True, num_iteration=trees, num_threads=2)
                after = restored.predict(probe, raw_score=True, num_threads=2)
                np.testing.assert_array_equal(before, after)
                records.append(dict(arm, day=day, trees=trees, name=name, model_sha256=digest(path),
                                    input_contract_sha256=digest(OUT / f'input_contract{day}.json'),
                                    max_training_date=contract['source_audit']['max_training_date'],
                                    serialization_maximum_error=0.))
                del restored
            del model, dataset
            gc.collect()
            emit('compact_lgb_fit_complete', day=day, arm=arm['name'])
        del frame, labels
        gc.collect()
    write(OUT / 'models_locked.json', dict(models=records, protocol_sha256=digest(OUT / 'protocol.json')))
    check()


def forecasts(day, independent=False):
    frame, contract = inputs(day, 'held')
    path = OUT / f'held_contract{day}.json'
    if independent:
        assert contract == read(path)
    else:
        write(path, contract)
    locked = read(OUT / 'models_locked.json')
    assert locked['protocol_sha256'] == digest(OUT / 'protocol.json')
    values = {}
    for record in locked['models']:
        if record['day'] != day:
            continue
        path = OUT / (record['name'] + '.txt')
        assert digest(path) == record['model_sha256']
        model = lgb.Booster(model_str=path.read_text('utf8'))
        assert model.feature_name() == list(frame)
        margin = model.predict(frame, raw_score=True, num_threads=2)
        assert margin.shape == (len(frame),) and np.isfinite(margin).all()
        if independent:
            np.testing.assert_array_equal(margin, np.load(OUT / (record['name'] + '_held_margin.npy')))
        else:
            np.save(OUT / (record['name'] + '_held_margin.npy'), margin)
        values[record['name']] = margin
    return values


def candidates(day, values, independent=False):
    base = np.load(BASE / f'day{day}_prediction.npy')
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'user_id'], dtype='int32')
    groups, _ = pd.factorize(raw.loc[raw.pt_d.eq(20191100 + day), 'user_id'], sort=True)
    yield dict(name='base', mode='base', weight=0.), base.copy()
    for name, margin in values.items():
        for mode in MODES:
            for weight in WEIGHTS:
                yield dict(name=name, mode=mode, weight=weight), row_prediction(base, margin, groups, weight, mode, independent)


def key(record):
    return f'{record["name"]}__{record["mode"]}__{record["weight"]:g}'


def evaluate():
    check()
    assert not (OUT / 'predictions_locked.json').exists()
    locked = {}
    for day in [15, 16]:
        values = forecasts(day)
        locked[str(day)] = {key(record): array_hash(prediction) for record, prediction in candidates(day, values)}
    write(OUT / 'predictions_locked.json', dict(models_locked_sha256=digest(OUT / 'models_locked.json'),
        dates=locked, all_both_date_models_locked=True, held_labels_opened_for_scoring=False))
    emit('compact_lgb_candidates_locked')


def verify():
    check()
    assert not (OUT / 'verification.json').exists()
    locked = read(OUT / 'predictions_locked.json')
    assert locked['models_locked_sha256'] == digest(OUT / 'models_locked.json')
    checked = {}
    for day in [15, 16]:
        for partition in ['train', 'held']:
            frame, cats, info, audit = prepared.source.inputs(day, partition, True)
            original = read(prepared.OUT / 'inputs_locked.json')[str(day)][partition]
            assert audit == original['source_audit']
            schema = read(prepared.OUT / f'schema{day}.json')
            if partition == 'train':
                assert fit_schema(frame, cats) == schema
                np.testing.assert_array_equal(info['target'], np.load(prepared.OUT / f'target{day}.npy'))
            numeric, encoded = transform(frame, schema, True)
            assert array_hash(numeric) == original['numeric_sha256']
            assert array_hash(encoded) == original['categories_sha256']
            rebuilt = frame_from(numeric, encoded, schema)
            contract = read(OUT / (f'input_contract{day}.json' if partition == 'train' else f'held_contract{day}.json'))
            assert array_hash(rebuilt.to_numpy()) == contract['frame_sha256']
            del frame, info, numeric, encoded, rebuilt
            gc.collect()
        checked[day] = forecasts(day, True)
        emit('compact_lgb_independent_inputs_and_models_verified', day=day)
    raw = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    reports = {}
    for day in [15, 16]:
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100 + day)
        np.testing.assert_array_equal(rows, np.load(OUT / f'day{day}_indices.npy'))
        y = raw.label.to_numpy()[rows]
        assert len(y) == {15: 424780, 16: 421983}[day] and y.sum() == {15: 3523, 16: 3724}[day]
        records, best_value, maximum = [], -1., 0.
        for (record, prediction), (other, replay) in zip(candidates(day, checked[day]), candidates(day, checked[day], True)):
            assert record == other and array_hash(prediction) == locked['dates'][str(day)][key(record)]
            assert np.isfinite(prediction).all() and np.all((prediction > 0) & (prediction < 1))
            error = float(np.max(np.abs(prediction - replay)))
            assert error < 2e-13
            maximum = max(maximum, error)
            value = rank_sum_auc(y, prediction)
            record = dict(record, auc=value)
            records.append(record)
            if value > best_value:
                best_value, best, best_prediction = value, record, prediction.copy()
        assert abs(roc_auc_score(y, best_prediction) - best_value) < 1e-12
        np.save(OUT / f'day{day}_prediction.npy', best_prediction)
        reports[str(20191100 + day)] = dict(auc=best_value, best=best, records=records, rows=len(y), clicks=int(y.sum()),
            evaluation_day_labels_fit=False, inferred17_labels_read=False, held_labels_used_to_select_fixed_recipe=True,
            all_both_date_candidates_locked_before_scoring=True, independent_all_features_and_models_replayed=True,
            maximum_independent_prediction_error=maximum, target_met=best_value >= {15: .815, 16: .82}[day])
        emit('compact_lgb_scored', day=day, best=best)
    check()
    write(OUT / 'verification.json', dict(status='passed', dates=reports,
        original_datasets_unchanged=True, both_targets_met=all(r['target_met'] for r in reports.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['fit', 'evaluate', 'verify'])
    {'fit': fit, 'evaluate': evaluate, 'verify': verify}[parser.parse_args().phase]()
