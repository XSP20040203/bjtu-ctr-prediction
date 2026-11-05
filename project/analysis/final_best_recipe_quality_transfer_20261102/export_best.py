"""Apply the frozen real-day16 winner to day17, then evaluate inferred labels."""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'

import argparse
from datetime import datetime, timezone
import gc
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from catboost import Pool

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT/'analysis/day16_auc082_20261102'
OLD = ROOT/'analysis/final_best_recipe_20261102'
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(OLD))
import predict_best as original
from fit_stacker import digest, write
from conditional_daily import read
from rich_userday_activity import array_hash
from personal_context_residual import personal_changes
from topic_context_residual import topic_features
from bayesian_daily_offset import prior_mass
from news_quality_features import READ, build_quality
from catboost_userday_residual import load_model
from rich_row_catboost import selection, frame_from_matrix
from grouped_catboost_residual import components, predict as grouped_predict, independent
from group_temperature import GROUPS, group_codes
sys.path.insert(0, str(ROOT/'analysis/auc085_next'))
from prepare_source_context import _read_csv, AD_COLUMNS, SOURCE_INPUT_COLUMNS, iter_feature_columns

RUN = HERE/'run_v1'
OUTPUT = ROOT/'output/best_recipe_quality_transfer_20191117_20261102'
SOURCE = EXP/'runs/rich_augmented_rows_v1'
RICH = EXP/'runs/rich_row_catboost_v1'
QUALITY = EXP/'runs/full_quality_row_catboost_v1'
DAILY = EXP/'runs/news_quality_daily_v1'
GROUPED = EXP/'runs/grouped_catboost_residual_v1'
WINNER = EXP/'runs/compose_full_quality_transfer_v1'
BASE = ROOT/'output/best_recipe_group_temperature_20191117_20261102'
CORE = ROOT/'analysis/final_best_recipe_augmented_20261102/run_v2/independent_original_input'
ROW = ROOT/'analysis/final_best_recipe_augmented_20261102/run_v2/independent_row_input'
ALT_CORE = OLD/'run_v1'
ALT_ROW = ROOT/'analysis/final_best_recipe_updated_20261102/run_v1/input'
LABEL_FILE = ROOT/'analysis/eleven_score_strategy_20261102/imported_refinement/day17_candidate_labels_11scores.csv'
LABEL_SHA = 'ab29c227c97efcc4fbef4dd57d92a8657ce5c078f1cbd03f037d8183f0ca7e24'
MODEL_NAMES = ['day16_depth6_weight5_trees2400', 'day16_depth8_weight5_trees2400',
               'day16_recent2_quality_depth8_trees3600']


def emit(event, **values):
    print(json.dumps(dict(event=event, **values), ensure_ascii=False), flush=True)


def check_sources(raw=True):
    protocol = read(RUN/'protocol.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    if raw:
        for name, expected in protocol['raw_datasets_sha256'].items():
            assert digest(ROOT/'2026ml'/name) == expected, name
    return protocol


def freeze():
    assert not RUN.exists() and not OUTPUT.exists(), 'Preserve prior exports and failed runs'
    selected = {}
    paths = [Path(__file__)]
    hashes = read(WINNER/'protocol.json')['raw_datasets_sha256']
    for name, folder in [('rich', RICH), ('quality', QUALITY), ('grouped', GROUPED), ('compose', WINNER)]:
        report = read(folder/'verification.json')
        assert report['status'] == 'passed'
        assert not report['dates']['20191116']['evaluation_day_labels_fit']
        selected[name] = report['dates']['20191116']['best']
        paths += [folder/'verification.json', folder/'protocol.json', folder/'day16_prediction.npy']
    assert selected['rich']['name'] == MODEL_NAMES[1] and selected['rich']['weight'] == .2
    assert selected['quality']['name'] == MODEL_NAMES[2] and selected['quality']['weight'] == .025
    assert selected['grouped']['source'] == 'mean_last_two'
    assert selected['grouped']['group'] == 'user_app'
    assert selected['grouped']['between'] == -.1 and selected['grouped']['within'] == .25
    assert selected['compose']['formula'] == 'transfer_increment' and selected['compose']['weight'] == 1.5
    assert read(BASE/'verification.json')['status'] == 'passed'
    assert not read(BASE/'verification.json')['inferred_labels_used_for_training_tuning_or_selection']
    for folder in [RICH, QUALITY, SOURCE, DAILY]:
        native = read(folder/'protocol.json')
        assert native['raw_datasets_sha256'] == hashes
        # Preserve and validate all dependency bindings declared by the trained runs.
        for path, expected in native['sources_sha256'].items():
            assert digest(path) == expected, path
            paths.append(Path(path))
    for name in MODEL_NAMES:
        folder = RICH if name in MODEL_NAMES[:2] else QUALITY
        model = next(item for item in read(folder/'models_locked.json')['models'] if item['name'] == name)
        assert digest(folder/(name+'.cbm')) == model['model_sha256']
        paths += [folder/'models_locked.json', folder/(name+'.cbm')]
    paths += [SOURCE/'input_contract16.json', DAILY/'input_contract16.json', DAILY/'quality16.npz',
              BASE/'verification.json', BASE/'prediction_manifest.json',
              BASE/'day16_prediction.npy', BASE/'day17_prediction.npy']
    for core, row in [(CORE, ROW), (ALT_CORE, ALT_ROW)]:
        paths += [core/name for name in ['numeric.npy', 'sequences.npy', 'dates.npy', 'all_context.npz']]
        paths.append(row/'features.npy')
    for module in [original, sys.modules['personal_context_residual'], sys.modules['topic_context_residual'],
                   sys.modules['news_quality_features'], sys.modules['bayesian_daily_offset'],
                   sys.modules['prepare_source_context'], sys.modules['grouped_catboost_residual']]:
        paths.append(Path(module.__file__))
    for name, expected in hashes.items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    audit = read(ROOT/'output/two_day_auc_targets_20261102/independent_target_audit.json')
    assert audit['audit_status'] == 'passed'
    protocol = dict(status='frozen_before_day17_prediction_and_new_inferred_label_evaluation',
        timestamp_utc=datetime.now(timezone.utc).isoformat(), selected=selected,
        development16_auc=selected['compose']['auc'],
        highest_published_real16_auc_at_freeze=audit['dates']['20191116']['auc'],
        selection_basis='Highest independently verified real-day16 development recipe at freeze',
        development_dates_repeatedly_reused=True, new_model_training=False,
        row_tree_training_max_date=20191115, day17_ad_history_max_date=20191116,
        inherited_neural_reference_training_max_date=20191116,
        same_day_supplied_news_feedback_used=True,
        inferred17_labels_used_for_training_tuning_feature_or_model_selection=False,
        inferred17_labels_previously_evaluated_on_older_frozen_exports=True,
        raw_datasets_sha256=hashes,
        sources_sha256={str(p): digest(p) for p in dict.fromkeys(paths)})
    assert abs(protocol['development16_auc']-protocol['highest_published_real16_auc_at_freeze']) < 1e-12
    RUN.mkdir(parents=True)
    write(RUN/'protocol.json', protocol)
    emit('recipe_frozen', development16_auc=protocol['development16_auc'], selected=selected)


def daily_features(day, core):
    original.RUN = core
    pipeline = original.Pipeline(day)
    assert np.all(pipeline.counts[pipeline.query] == -1)
    values, names, changes = pipeline.news()
    context_changes = personal_changes(pipeline.keys, pipeline.context)
    news = pd.concat([pd.read_csv(ROOT/'2026ml'/name,
        usecols=['e_et', 'u_userId', 'i_cat', 'label', 'pro'], dtype='int32')
        for name in ['train_feeds.csv', 'test_feeds.csv']], ignore_index=True)
    topics = np.sort(news.loc[news.e_et < 20191116, 'i_cat'].unique()).tolist()
    namespace = dict(np=np, pd=pd, sparse=sparse, personal_changes=personal_changes, cutoff=day)
    code = inspect.getsource(topic_features).replace('20191116', 'cutoff')
    exec(compile(code, '<frozen_date_bound_topic_features>', 'exec'), namespace)
    topic, topic_names, _ = namespace['topic_features'](news, pipeline.keys, topics)
    del news
    pieces = [pipeline.context, pipeline.past, pipeline.enhanced, values, changes, context_changes, topic]
    all_names = pipeline.context_names+pipeline.past_names+pipeline.enhanced_names
    all_names += ['news__'+n for n in names]+['news_previous_delta__'+n for n in names]
    all_names += ['news_personal_mean_delta__'+n for n in names]+['news_has_previous', 'news_log_previous_days']
    all_names += ['context_previous_delta__'+n for n in pipeline.context_names]
    all_names += ['context_personal_mean_delta__'+n for n in pipeline.context_names]
    all_names += ['context_has_previous', 'context_log_previous_days']+topic_names
    expected = read(DAILY/'input_contract16.json')
    native = expected['native_contract']['native_contract']
    assert all_names == native['feature_names']
    assert len(all_names) == sum(piece.shape[1] for piece in pieces)
    # Only the fixed128 selected daily columns are materialized for queried users.
    contract = read(SOURCE/'input_contract16.json')
    requested = [n.removeprefix('rich_daily__') for n in contract['feature_names'][472:664]]
    lookup = {}
    start = 0
    for piece in pieces:
        for j in range(piece.shape[1]):
            lookup['x'+str(start+j)] = (piece, j)
        start += piece.shape[1]
    ordinary = np.zeros((pipeline.query.sum(), 128), dtype='float32')
    for i, name in enumerate(requested[:128]):
        if name != 'log_bayesian_click_mass':
            array, column = lookup[name]
            ordinary[:, i] = array[pipeline.query, column]
    del pieces, lookup, topic, values, changes, context_changes
    gc.collect()
    fields = ['pt_d', 'label', 'user_id', 'spread_app_id', 'slot_id']
    raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=fields, dtype='int32')
    final = pd.read_csv(ROOT/'2026ml/final_test.csv', usecols=[n for n in fields if n != 'label'], dtype='int32')
    final['label'] = -1
    raw = pd.concat([raw, final], ignore_index=True)
    mass, audit = prior_mass(raw, pipeline.keys, day)
    assert all(row['label_max_date'] is None or row['label_max_date'] < row['query_date'] for row in audit)
    ordinary[:, requested[:128].index('log_bayesian_click_mass')] = np.log(mass[pipeline.query])
    if day == 20191116:
        assert array_hash(pipeline.keys) == native['keys_sha256']
        assert array_hash(pipeline.counts) == native['counts_sha256']
        assert array_hash(mass) == expected['native_contract']['bayesian_mass_sha256']
    del raw, final, mass
    gc.collect()
    news = pd.concat([pd.read_csv(ROOT/'2026ml'/name, usecols=READ,
        dtype={'i_docId': str, 'i_s_sourceId': str})
        for name in ['train_feeds.csv', 'test_feeds.csv']], ignore_index=True)
    quality, quality_names = build_quality(news, pipeline.keys, day)
    assert quality_names == expected['quality_names'] and quality.shape[1] == 337
    if day == 20191116:
        assert array_hash(quality) == expected['quality_sha256']
    query_quality = quality[pipeline.query].copy()
    extra = np.column_stack([ordinary, query_quality[:, [quality_names.index(name) for name in requested[128:]]]])
    keys = pipeline.qkeys.copy()
    report = dict(day=day, keys_sha256=array_hash(keys), extra_sha256=array_hash(extra),
        quality_sha256=array_hash(query_quality), bayesian_audit=audit,
        own_query_ad_labels_masked=True, max_ad_history_date=day-1,
        topic_vocabulary_fit_cutoff=20191116)
    del news, quality, pipeline
    gc.collect()
    emit('query_daily_features_ready', day=day, groups=len(keys))
    return keys, extra, query_quality, report


def build(day, verifying=False):
    check_sources(raw=False)
    directory = RUN/(f'independent_day{day%100}' if verifying else f'input_day{day%100}')
    directory.mkdir(exist_ok=False)
    core, rowdir = (ALT_CORE, ALT_ROW) if verifying else (CORE, ROW)
    keys, extra, quality, audit = daily_features(day, core)
    dates = np.load(core/'dates.npy')
    selected = np.flatnonzero(dates == day)
    raw = pd.read_csv(ROOT/'2026ml'/('train.csv' if day == 20191116 else 'final_test.csv'),
        usecols=['pt_d', 'user_id', 'log_id'], dtype='int32')
    raw = raw.loc[raw.pt_d == day].reset_index(drop=True)
    assert len(raw) == len(selected) == {20191116:421983, 20191117:404182}[day]
    positions = pd.MultiIndex.from_arrays(keys.T).get_indexer(pd.MultiIndex.from_frame(raw[['pt_d', 'user_id']]))
    assert np.all(positions >= 0)
    rows = np.load(rowdir/'features.npy', mmap_mode='r')
    numeric = np.load(core/'numeric.npy', mmap_mode='r')
    sequences = np.load(core/'sequences.npy', mmap_mode='r')
    assert rows.shape[1]+numeric.shape[1]+15 == 472
    x = np.lib.format.open_memmap(directory/'features.npy', mode='w+', dtype='float32', shape=(len(raw), 744))
    for first in range(0, len(raw), 10000):
        stop = min(len(raw), first+10000)
        subset = selected[first:stop]
        x[first:stop, :rows.shape[1]] = rows[subset]
        x[first:stop, rows.shape[1]:457] = numeric[subset]
        x[first:stop, 457:472] = sequences[subset, :3, :].reshape(len(subset), 15)
        x[first:stop, 472:664] = extra[positions[first:stop]]
    del rows, numeric, sequences
    filename = 'train.csv' if day == 20191116 else 'final_test.csv'
    ads = _read_csv(ROOT/'2026ml'/filename, AD_COLUMNS)
    ads = ads.loc[ads.pt_d == day].reset_index(drop=True)
    newsfiles = ['train_feeds.csv'] if day == 20191116 else ['test_feeds.csv']
    news = pd.concat([_read_csv(ROOT/'2026ml'/name, SOURCE_INPUT_COLUMNS) for name in newsfiles], ignore_index=True)
    news = news.loc[news.e_et == day]
    names = []
    for i, (name, values) in enumerate(iter_feature_columns(ads, news)):
        x[:, 664+i] = values
        names.append(name)
    x.flush()
    contract = read(SOURCE/'input_contract16.json')
    assert names == contract['feature_names'][664:]
    if day == 20191116:
        labels = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
        _, _, _, _, held, _ = selection(16, labels.label.to_numpy(), labels.pt_d.to_numpy())
        old = np.load(SOURCE/'independent_day16/features.npy', mmap_mode='r')
        for first in range(0, len(raw), 10000):
            stop = min(len(raw), first+10000)
            np.testing.assert_array_equal(x[first:stop], old[held[first:stop]])
        del old, labels
        emit('day16_all744_features_replayed_exactly')
    for first in range(0, len(raw), 10000):
        assert np.isfinite(x[first:first+10000]).all()
    np.savez_compressed(directory/'daily_quality.npz', keys=keys, values=quality)
    audit.update(shape=list(x.shape), matrix_sha256=digest(directory/'features.npy'),
        quality_file_sha256=digest(directory/'daily_quality.npz'), original_row_log_ids_sha256=array_hash(raw.log_id.to_numpy()))
    if verifying:
        expected = read(RUN/f'input_day{day%100}'/'contract.json')
        assert audit == expected
    write(directory/'contract.json', audit)
    emit('features_ready', day=day, verifying=verifying, shape=list(x.shape))


def normalized(base, forecast, weight):
    a = logit(base.clip(1e-6, 1-1e-6))
    b = logit(forecast.clip(1e-6, 1-1e-6))
    b = (b-b.mean())*a.std()/max(b.std(), 1e-6)+a.mean()
    return expit((1-weight)*a+weight*b)


def forecast(day, verifying=False):
    protocol = read(RUN/'protocol.json')
    directory = RUN/(f'independent_day{day%100}' if verifying else f'input_day{day%100}')
    matrix = np.load(directory/'features.npy', mmap_mode='r')
    native = read(SOURCE/'input_contract16.json')
    quality = np.load(directory/'daily_quality.npz')
    raw = pd.read_csv(ROOT/'2026ml'/('train.csv' if day == 20191116 else 'final_test.csv'),
        usecols=['pt_d', 'user_id', 'spread_app_id', 'slot_id', 'task_id'], dtype='int32')
    raw = raw.loc[raw.pt_d == day].reset_index(drop=True)
    pos = pd.MultiIndex.from_arrays(quality['keys'].T).get_indexer(pd.MultiIndex.from_frame(raw[['pt_d', 'user_id']]))
    keep = [i for i, name in enumerate(native['feature_names']) if not name.startswith('rich_daily__quality__')]
    assert len(keep) == 680
    predictions = {}
    for name in MODEL_NAMES:
        folder = RICH if name in MODEL_NAMES[:2] else QUALITY
        model = load_model(folder/(name+'.cbm'))
        chunks = []
        batch = 7000 if verifying else 12000
        for first in range(0, len(raw), batch):
            stop = min(len(raw), first+batch)
            if folder == RICH:
                values = np.asarray(matrix[first:stop])
                cats = native['categorical_columns']
            else:
                values = np.column_stack([np.asarray(matrix[first:stop])[:, keep], quality['values'][pos[first:stop]]])
                cats = [keep.index(i) for i in native['categorical_columns']]
            frame = frame_from_matrix(values, np.arange(len(values)), cats)
            pool = Pool(frame, cat_features=['f'+str(i) for i in cats])
            margin = model.predict(pool, prediction_type='RawFormulaVal', thread_count=2)
            chunks.append(expit(margin-np.log(5.)))
            del pool, frame, values
        p = np.concatenate(chunks)
        if day == 20191116:
            np.testing.assert_allclose(p, np.load(folder/(name+'_prediction.npy')), atol=2e-14, rtol=0)
        predictions[name] = p
        emit('tree_forecast_ready', day=day, model=name, verifying=verifying)
        del model
        gc.collect()
    base = np.load(BASE/f'day{day%100}_prediction.npy')
    anchor = normalized(base, predictions[MODEL_NAMES[1]], protocol['selected']['rich']['weight'])
    q = normalized(anchor, predictions[MODEL_NAMES[2]], protocol['selected']['quality']['weight'])
    mean = np.mean([predictions[name] for name in MODEL_NAMES[:2]], axis=0)
    record = protocol['selected']['grouped']
    if verifying:
        current = independent(raw, anchor, mean, record)
    else:
        groups = group_codes(raw, GROUPS[record['group']])
        current = grouped_predict(components(anchor, mean, groups), record['between'], record['within'])
    p = expit(logit(current)+protocol['selected']['compose']['weight']*(logit(q)-logit(anchor)))
    if day == 20191116:
        for value, folder in [(anchor, RICH), (q, QUALITY), (current, GROUPED), (p, WINNER)]:
            np.testing.assert_allclose(value, np.load(folder/'day16_prediction.npy'), rtol=0, atol=2e-14)
    assert np.isfinite(p).all() and np.all((p > 0)&(p < 1))
    return p


def predict():
    protocol = check_sources()
    assert not OUTPUT.exists(), 'Preserve completed exports'
    result = {day: forecast(day) for day in [20191116, 20191117]}
    OUTPUT.mkdir()
    for day, p in result.items():
        np.save(OUTPUT/f'day{day%100}_prediction.npy', p)
    raw = pd.read_csv(ROOT/'2026ml/final_test.csv', usecols=['log_id', 'pt_d'], dtype='int32')
    assert raw.pt_d.eq(20191117).all()
    pd.DataFrame(dict(id=np.arange(len(raw)), pctr=result[20191117])).to_csv(
        OUTPUT/'submission_best_20191117.csv', index=False, float_format='%.17g')
    pd.DataFrame(dict(log_id=raw.log_id, pctr=result[20191117])).to_csv(
        OUTPUT/'submission_log_id_20191117.csv', index=False, float_format='%.17g')
    write(OUTPUT/'prediction_manifest.json', dict(status='locked_before_inferred17_label_evaluation',
        protocol_sha256=digest(RUN/'protocol.json'), development16_auc=protocol['development16_auc'],
        inferred17_labels_used_for_prediction_or_selection=False,
        files_sha256={path.name:digest(path) for path in OUTPUT.iterdir() if path.is_file()}))
    emit('submission_frozen', path=str(OUTPUT/'submission_best_20191117.csv'))


def verify():
    assert not (OUTPUT/'verification.json').exists(), 'Preserve completed verification'
    protocol = check_sources()
    manifest = read(OUTPUT/'prediction_manifest.json')
    assert manifest['protocol_sha256'] == digest(RUN/'protocol.json')
    for name, expected in manifest['files_sha256'].items():
        assert digest(OUTPUT/name) == expected, name
    replay = forecast(20191117, verifying=True)
    error = float(np.max(np.abs(replay-np.load(OUTPUT/'day17_prediction.npy'))))
    assert error < 2e-14
    raw = pd.read_csv(ROOT/'2026ml/final_test.csv', usecols=['log_id', 'pt_d'], dtype='int32')
    submission = pd.read_csv(OUTPUT/'submission_best_20191117.csv', float_precision='round_trip')
    alternate = pd.read_csv(OUTPUT/'submission_log_id_20191117.csv', float_precision='round_trip')
    assert list(submission) == ['id', 'pctr'] and list(alternate) == ['log_id', 'pctr']
    assert len(submission) == len(alternate) == len(raw) == 404182
    np.testing.assert_array_equal(submission.id, np.arange(len(raw)))
    np.testing.assert_array_equal(alternate.log_id, raw.log_id)
    np.testing.assert_array_equal(submission.pctr, alternate.pctr)
    np.testing.assert_array_equal(submission.pctr, np.load(OUTPUT/'day17_prediction.npy'))
    assert submission.pctr.between(0, 1).all() and np.isfinite(submission.pctr).all()
    actual = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    auc16 = float(roc_auc_score(actual.loc[actual.pt_d.eq(20191116), 'label'], np.load(OUTPUT/'day16_prediction.npy')))
    assert abs(auc16-protocol['development16_auc']) < 1e-12
    # The sole inferred-label read occurs after immutable recipe and forecasts exist.
    assert digest(LABEL_FILE) == LABEL_SHA
    inferred = pd.read_csv(LABEL_FILE)
    np.testing.assert_array_equal(inferred.id, submission.id)
    np.testing.assert_array_equal(inferred.log_id, raw.log_id)
    assert inferred.label.isin([0, 1]).all()
    auc = float(roc_auc_score(inferred.label, submission.pctr))
    positives = int(inferred.label.sum())
    independent_auc = float((rankdata(submission.pctr)[inferred.label.eq(1)].sum()-positives*(positives+1)/2)/
        (positives*(len(inferred)-positives)))
    assert abs(auc-independent_auc) < 1e-12
    check_sources()
    report = dict(status='passed', date=20191117, rows=len(raw), development16_auc=auc16,
        inferred_label_auc=auc, independent_rank_auc=independent_auc, inferred_positive_count=positives,
        labels_are_inferred_not_observed=True, platform_auc=None,
        inferred_labels_used_for_training_tuning_feature_or_model_selection=False,
        new_model_training=False, row_tree_training_max_date=20191115, ad_history_max_date=20191116,
        day16_all744_columns_and337_quality_columns_replayed_exactly=True,
        day16_winning_pipeline_replayed=True, day17_new_features_rebuilt_from_raw_twice=True,
        inherited472_features_previously_independently_verified=True,
        maximum_prediction_replay_error=error, csv_roundtrip_exact=True,
        original_row_order_verified=True, repeated_log_ids_preserved=int(raw.log_id.duplicated().sum()),
        original_datasets_unchanged=True, raw_datasets_sha256=protocol['raw_datasets_sha256'],
        inferred_label_sha256=LABEL_SHA, protocol_sha256=digest(RUN/'protocol.json'),
        submission_sha256=digest(OUTPUT/'submission_best_20191117.csv'))
    write(OUTPUT/'verification.json', report)
    (OUTPUT/'提交说明.md').write_text(
        f'2019年11月17日提交文件：submission_best_20191117.csv\n'
        f'格式 id,pctr；共{len(raw):,}行，id为final_test原始行序的零起始行号。\n'
        f'兼容文件 submission_log_id_20191117.csv 使用原始log_id，保留重复值。\n\n'
        f'采用已验证16号开发集最优配方 compose_full_quality_transfer_v1，AUC={auc16:.12f}。\n'
        f'沿用已训练的模型，行树训练到15号，17号历史统计使用截至16号广告标签。\n'
        f'17号反推标签AUC={auc:.12f}，反推标签不是官方真实标签，平台真实AUC未知。\n'
        f'反推标签未参与训练、调参、特征选择或模型选择。15/16号曾反复用于开发。\n'
        f'已验证全量行序、CSV精度、预测复算与原始四份数据SHA256。\n', encoding='utf8')
    emit('verified_and_evaluated', **report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['freeze', 'build16', 'build17', 'predict', 'rebuild17', 'verify'])
    args = parser.parse_args()
    if args.phase == 'freeze':
        freeze()
    elif args.phase in ['build16', 'build17', 'rebuild17']:
        build(20191116 if args.phase == 'build16' else 20191117, args.phase == 'rebuild17')
    elif args.phase == 'predict':
        predict()
    else:
        verify()
