"""Transfer the frozen, verified development16 winner and evaluate only afterwards."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import gc
import inspect
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT/'analysis/day16_auc082_20261102'
OLD = ROOT/'analysis/final_best_recipe_20261102'
UPDATED = ROOT/'analysis/final_best_recipe_updated_20261102'
RUN = HERE/'run_v2'
OUTPUT = ROOT/'output/best_recipe_augmented_20191117_20261102'
for path in [EXP, OLD, UPDATED]:
    sys.path.insert(0, str(path))
import predict_best as original
import predict_and_verify_v2 as previous
from features import read, write, digest
from rich_userday_activity import array_hash, correct
from personal_context_residual import personal_changes
from topic_context_residual import topic_features
from catboost_userday_residual import CATEGORIES, load_model
from catboost_global_rows import predict as cat_predict, recipes as cat_recipes
from bayesian_daily_offset import prior_mass, lgb
from catboost import Pool

AUG = EXP/'runs/augmented_rows_v1_day16'
WINNER = EXP/'runs/bayesian_daily_offset_v1'
BINARY = EXP/'runs/rich_binary16_cpu_v1'
COUNT = EXP/'runs/rolling_rich_count_v1'
COMPOSE = EXP/'runs/compose_rich_response_v1'
PREVIOUS_OUTPUT = ROOT/'output/best_recipe_updated_20191117_20261102'
ORIGINAL_RUN = OLD/'run_v1'


def emit(event, **values):
    print(__import__('json').dumps(dict(event=event, **values), ensure_ascii=False), flush=True)


def source_paths():
    paths = [Path(__file__), OLD/'predict_best.py', UPDATED/'predict_and_verify_v2.py',
             UPDATED/'run_v2/protocol.json', UPDATED/'run_v1/input/features.npy',
             UPDATED/'run_v2/independent_input/contract.json',
             PREVIOUS_OUTPUT/'verification.json', PREVIOUS_OUTPUT/'day16_prediction.npy',
             PREVIOUS_OUTPUT/'day17_prediction.npy', ORIGINAL_RUN/'numeric.npy',
             ORIGINAL_RUN/'sequences.npy', ORIGINAL_RUN/'dates.npy', ORIGINAL_RUN/'all_context.npz',
             ORIGINAL_RUN/'feature_manifest.json',
             EXP/'augmented_row_catboost.py', EXP/'bayesian_daily_offset.py',
             EXP/'rolling_rich_daily_catboost.py', EXP/'feed_behavior_userday.py',
             EXP/'topic_context_residual.py', EXP/'personal_context_residual.py',
             EXP/'catboost_userday_residual.py', EXP/'catboost_global_rows.py',
             EXP/'rich_userday_activity.py', EXP/'extended_history_residual.py',
             EXP/'runs/feed_behavior_userday_v1/news_schema.json']
    for folder in [AUG, WINNER, BINARY, COUNT, COMPOSE]:
        paths += [folder/'verification.json', folder/'protocol.json']
    paths += [AUG/'input_contract.json', BINARY/'input_contract.json',
              COUNT/'input_contract16.json', AUG/'models_locked.json',
              BINARY/'models_locked.json', COUNT/'models_locked.json',
              WINNER/'day16_prediction.npy', WINNER/'input_contract16.json',
              WINNER/'models_locked.json', COMPOSE/'day16_prediction.npy']
    for folder in [AUG, BINARY, COUNT]:
        for model in read(folder/'models_locked.json')['models']:
            name = model['model_name'] if folder == COUNT else model['name']
            if folder == COUNT and model['day'] != 16:
                continue
            paths += [folder/(name+'.cbm')]
    selected = read(WINNER/'verification.json')['dates']['20191116']['best']
    paths += [WINNER/(selected['name']+'.txt')]
    return list(dict.fromkeys(paths))


def freeze():
    assert not RUN.exists(), 'Preserving completed and failed runs'
    verification = read(WINNER/'verification.json')
    assert verification['status'] == 'passed'
    selected = verification['dates']['20191116']
    assert read(PREVIOUS_OUTPUT/'verification.json')['status'] == 'passed'
    for folder in [AUG, BINARY, COUNT, COMPOSE]:
        assert read(folder/'verification.json')['status'] == 'passed'
    raw_hashes = read(AUG/'protocol.json')['raw_datasets_sha256']
    for name, expected in raw_hashes.items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    RUN.mkdir(parents=True)
    protocol = {'status': 'recipe_frozen_before_prediction_and_inferred_label_evaluation',
        'date': 20191117, 'development16_auc': selected['auc'], 'selected': selected['best'],
        'augmented': read(AUG/'verification.json')['best'],
        'binary': read(BINARY/'verification.json')['best'],
        'count': read(COUNT/'verification.json')['dates']['20191116']['best'],
        'compose': read(COMPOSE/'verification.json')['dates']['20191116']['best'],
        'new_predictive_model_training': False,
        'tree_checkpoint_training_max_date': 20191115,
        'base_neural_reference_training_max_date': 20191116,
        'ad_history_max_date': 20191116,
        'same_day_supplied_news_feedback_allowed': True,
        'inferred_labels_used_for_training_or_model_selection': False,
        'selection_basis': 'Verified repeatedly reused real16 development outcomes',
        'raw_datasets_sha256': raw_hashes,
        'sources_sha256': {str(p): digest(p) for p in source_paths()}}
    write(RUN/'protocol.json', protocol)
    emit('recipe_frozen', development16_auc=selected['auc'])
    return protocol


def check_sources():
    protocol = read(RUN/'protocol.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    return protocol


def rich_prediction(day, base, protocol):
    pipeline = original.Pipeline(day)
    values, names, changes = pipeline.news()
    context_changes = personal_changes(pipeline.keys, pipeline.context)
    news = pd.concat([pd.read_csv(ROOT/'2026ml'/name,
               usecols=['e_et', 'u_userId', 'i_cat', 'label', 'pro'], dtype='int32')
               for name in ['train_feeds.csv', 'test_feeds.csv']], ignore_index=True)
    topics = np.sort(news.loc[news.e_et < 20191116, 'i_cat'].unique()).tolist()
    namespace = {'np': np, 'pd': pd, 'sparse': sparse, 'personal_changes': personal_changes, 'cutoff': day}
    source = inspect.getsource(topic_features).replace('20191116', 'cutoff')
    exec(compile(source, '<frozen_topic_features_with_query_cutoff>', 'exec'), namespace)
    topic, topic_names, _ = namespace['topic_features'](news, pipeline.keys, topics)
    feature_names = pipeline.context_names + pipeline.past_names + pipeline.enhanced_names
    feature_names += ['news__'+name for name in names]
    feature_names += ['news_previous_delta__'+name for name in names]
    feature_names += ['news_personal_mean_delta__'+name for name in names]
    feature_names += ['news_has_previous', 'news_log_previous_days']
    feature_names += ['context_previous_delta__'+name for name in pipeline.context_names]
    feature_names += ['context_personal_mean_delta__'+name for name in pipeline.context_names]
    feature_names += ['context_has_previous', 'context_log_previous_days'] + topic_names
    x = np.column_stack([pipeline.context, pipeline.past, pipeline.enhanced,
                         values, changes, context_changes, topic]).astype('float32')
    contract = read(BINARY/'input_contract.json')
    assert feature_names == contract['feature_names']
    assert np.isfinite(x).all()
    mass = np.empty(len(pipeline.keys), dtype='float64')
    clicks = exposures = 0.
    for date in np.unique(pipeline.keys[:, 0]):
        selected = pipeline.keys[:, 0] == date
        prior = (clicks+8.4)/(exposures+1000.)
        mass[selected] = pipeline.sizes[selected]*prior
        if date < day:
            clicks += float(pipeline.counts[selected].sum())
            exposures += float(pipeline.sizes[selected].sum())
    # The full day16 matrix and chronological offsets must reproduce the original fit input.
    if day == 20191116:
        assert array_hash(x) == contract['x_sha256']
        assert array_hash(pipeline.keys) == contract['keys_sha256']
        assert array_hash(pipeline.counts) == contract['counts_sha256']
        assert array_hash(mass) == contract['mass_prior_sha256']
    frame = pd.DataFrame(x[pipeline.query], columns=[f'x{i}' for i in range(x.shape[1])])
    for field in CATEGORIES:
        frame[field] = pipeline.catframe[field].to_numpy(dtype='int64')
    frame['user_id'] = pipeline.qkeys[:, 1].astype('int64')
    pool = Pool(frame, cat_features=CATEGORIES+['user_id'])
    group_forecasts = {}
    for objective, folder, key in [('binary', BINARY, 'name'), ('poisson', COUNT, 'model_name')]:
        record = protocol['binary' if objective == 'binary' else 'count']
        offset = logit((-np.expm1(-mass[pipeline.query])).clip(1e-6, 1-1e-6)) if objective == 'binary' else np.log(mass[pipeline.query])
        margin = load_model(folder/(record[key]+'.cbm')).predict(pool, prediction_type='RawFormulaVal')+offset
        gp = expit(margin) if objective == 'binary' else np.exp(margin.clip(-15, 8))
        group_forecasts[objective] = correct(base, gp, pipeline.groups, objective, record['power'])
    blend = protocol['compose']
    assert blend['formula'] == 'logit_blend'
    result = expit((1-blend['weight'])*logit(group_forecasts['binary'].clip(1e-6, 1-1e-6))+
                   blend['weight']*logit(group_forecasts['poisson'].clip(1e-6, 1-1e-6)))
    if day == 20191116:
        np.testing.assert_allclose(result, np.load(COMPOSE/'day16_prediction.npy'), rtol=0, atol=2e-7)
    fields = ['pt_d', 'label', 'user_id', 'spread_app_id', 'slot_id']
    ad = pd.read_csv(ROOT/'2026ml/train.csv', usecols=fields, dtype='int32')
    final = pd.read_csv(ROOT/'2026ml/final_test.csv', usecols=[f for f in fields if f != 'label'], dtype='int32')
    final['label'] = -1
    ad = pd.concat([ad, final], ignore_index=True)
    bayesian_mass, audit = prior_mass(ad, pipeline.keys, day)
    if day == 20191116:
        assert array_hash(bayesian_mass) == read(WINNER/'input_contract16.json')['bayesian_mass_sha256']
    assert all(item['label_max_date'] is None or item['label_max_date'] < item['query_date'] for item in audit)
    frame['log_bayesian_click_mass'] = np.log(bayesian_mass[pipeline.query])
    record = protocol['selected']
    assert record['name'] == 'day16_binary_l15_trees1000'
    model_path = WINNER/(record['name']+'.txt')
    model = lgb.Booster(model_str=model_path.read_text(encoding='utf8'))
    z = model.predict(frame, raw_score=True, num_threads=2)
    z += logit((-np.expm1(-bayesian_mass[pipeline.query])).clip(1e-6, 1-1e-6))
    response = expit(z)
    emit('rich_correction_replayed', day=day, groups=len(frame))
    return result, response, pipeline.groups


def forecast(row_features, dates, core_run, protocol):
    numeric = np.load(core_run/'numeric.npy', mmap_mode='r')
    sequence = np.load(core_run/'sequences.npy', mmap_mode='r')
    np.testing.assert_array_equal(dates, np.load(core_run/'dates.npy'))
    assert numeric.shape == (len(dates), 166)
    contract = read(AUG/'input_contract.json')
    nbase = row_features.shape[1]
    assert nbase+166+15 == len(contract['feature_names']) == 472
    result = {}
    for day in [20191116, 20191117]:
        selected = dates == day
        base = np.load(PREVIOUS_OUTPUT/f'day{day%100}_prediction.npy')
        base, response, row_groups = rich_prediction(day, base, protocol)
        gc.collect()
        a = logit(base.clip(1e-6, 1-1e-6))
        features = np.column_stack([row_features[selected], numeric[selected],
                                   sequence[selected, :3, :].reshape(selected.sum(), 15)]).astype('float32')
        frame = pd.DataFrame(features, columns=[f'f{i}' for i in range(472)])
        cats = contract['categorical_columns']
        for i in cats:
            frame[f'f{i}'] = frame.iloc[:, i].to_numpy(dtype='int64')
        pool = Pool(frame, cat_features=[f'f{i}' for i in cats])
        augmented = None
        for record in read(AUG/'models_locked.json')['models']:
            if record['name'] != protocol['augmented']['name']:
                continue
            path = AUG/(record['name']+'.cbm')
            assert digest(path) == record['model_sha256']
            p = cat_predict(load_model(path), pool, record['positive_weight'])
            if day == 20191116:
                np.testing.assert_allclose(p, np.load(AUG/(record['name']+'_prediction.npy')), rtol=0, atol=2e-7)
            augmented = next(q for name, q in cat_recipes(p, base) if name == protocol['augmented']['recipe'])
        assert augmented is not None
        if day == 20191116:
            np.testing.assert_allclose(augmented, np.load(AUG/'prediction.npy'), rtol=0, atol=2e-7)
        result[day] = correct(augmented, response, row_groups, 'binary', protocol['selected']['power'])
        assert np.isfinite(result[day]).all()
        if day == 20191116:
            np.testing.assert_allclose(result[day], np.load(WINNER/'day16_prediction.npy'), rtol=0, atol=2e-7)
            raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
            score = float(roc_auc_score(raw.loc[raw.pt_d == day, 'label'], result[day]))
            assert abs(score-protocol['development16_auc']) < 1e-12
        emit('augmented_prediction_ready', day=day, rows=len(result[day]))
        del frame, features, pool
        gc.collect()
    return result


def predict():
    protocol = freeze()
    row_path = UPDATED/'run_v1/input/features.npy'
    row_features = np.load(row_path, mmap_mode='r')
    dates = np.load(ORIGINAL_RUN/'dates.npy')
    result = forecast(row_features, dates, ORIGINAL_RUN, protocol)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    raw = pd.read_csv(ROOT/'2026ml/final_test.csv', usecols=['log_id', 'pt_d'])
    assert raw.pt_d.eq(20191117).all() and len(raw) == len(result[20191117]) == 404182
    for day, p in result.items():
        np.save(OUTPUT/f'day{day%100}_prediction.npy', p)
    pd.DataFrame({'id': np.arange(len(raw)), 'pctr': result[20191117]}).to_csv(
        OUTPUT/'submission_best_20191117.csv', index=False, float_format='%.10f')
    pd.DataFrame({'log_id': raw.log_id, 'pctr': result[20191117]}).to_csv(
        OUTPUT/'submission_log_id_20191117.csv', index=False, float_format='%.10f')
    write(OUTPUT/'prediction_manifest.json', {'status': 'frozen_before_inferred_label_evaluation',
        'protocol_sha256': digest(RUN/'protocol.json'), 'development16_auc': protocol['development16_auc'],
        'inferred_labels_used_for_prediction_or_selection': False,
        'files_sha256': {p.name: digest(p) for p in OUTPUT.iterdir() if p.is_file()}})
    emit('submission_frozen', path=str(OUTPUT/'submission_best_20191117.csv'))


def verify():
    protocol = check_sources()
    manifest = read(OUTPUT/'prediction_manifest.json')
    assert manifest['protocol_sha256'] == digest(RUN/'protocol.json')
    for name, expected in manifest['files_sha256'].items():
        assert digest(OUTPUT/name) == expected, name
    # Rebuild original query numerics/sequences/context from raw, then rebuild row matrices.
    original.RUN = RUN/'independent_original_input'
    original.prepare()
    row_features, dates, _ = previous.build_features(RUN/'independent_row_input')
    result = forecast(row_features, dates, original.RUN, protocol)
    error = 0.
    for day, p in result.items():
        current = float(np.max(np.abs(p-np.load(OUTPUT/f'day{day%100}_prediction.npy'))))
        error = max(error, current)
    assert error < 2e-7, error
    raw = pd.read_csv(ROOT/'2026ml/final_test.csv', usecols=['log_id', 'pt_d'])
    submission = pd.read_csv(OUTPUT/'submission_best_20191117.csv')
    alternate = pd.read_csv(OUTPUT/'submission_log_id_20191117.csv')
    assert list(submission) == ['id', 'pctr'] and list(alternate) == ['log_id', 'pctr']
    assert len(raw) == len(submission) == len(alternate) == 404182
    np.testing.assert_array_equal(submission.id, np.arange(len(raw)))
    np.testing.assert_array_equal(alternate.log_id, raw.log_id)
    np.testing.assert_array_equal(alternate.pctr, submission.pctr)
    assert np.isfinite(submission.pctr).all() and submission.pctr.between(0, 1).all()
    rounding = float(np.max(np.abs(submission.pctr.to_numpy()-np.load(OUTPUT/'day17_prediction.npy'))))
    assert rounding < 5.01e-11
    check_sources()
    # Evaluation-only labels are first read after fixed prediction and independent reconstruction.
    label_file = ROOT/'analysis/eleven_score_strategy_20261102/imported_refinement/day17_candidate_labels_11scores.csv'
    assert digest(label_file) == 'ab29c227c97efcc4fbef4dd57d92a8657ce5c078f1cbd03f037d8183f0ca7e24'
    labels = pd.read_csv(label_file)
    np.testing.assert_array_equal(labels.id, submission.id)
    np.testing.assert_array_equal(labels.log_id, raw.log_id)
    assert labels.label.isin([0, 1]).all()
    auc = float(roc_auc_score(labels.label, submission.pctr))
    npositive = int(labels.label.sum())
    manual = float((rankdata(submission.pctr, method='average')[labels.label.eq(1)].sum()-
                    npositive*(npositive+1)/2)/(npositive*(len(labels)-npositive)))
    assert abs(auc-manual) < 1e-12
    report = {'status': 'passed', 'date': 20191117, 'rows': len(raw),
        'development16_auc': protocol['development16_auc'], 'inferred_label_auc': auc,
        'independent_rank_auc': manual, 'inferred_positive_count': npositive,
        'labels_are_inferred_not_observed': True, 'platform_auc': None,
        'inferred_labels_used_for_training_or_selection': False,
        'maximum_prediction_replay_error': error, 'maximum_csv_rounding_error': rounding,
        'day16_exact_winner_replay': True, 'independent_day17_raw_feature_rebuild': True,
        'original_row_order_verified': True, 'repeated_log_ids_preserved': int(raw.log_id.duplicated().sum()),
        'original_datasets_unchanged': True, 'inferred_label_sha256': digest(label_file),
        'submission_sha256': digest(OUTPUT/'submission_best_20191117.csv'),
        'protocol_sha256': digest(RUN/'protocol.json')}
    write(OUTPUT/'verification.json', report)
    (OUTPUT/'提交说明.md').write_text(
        f'2019年11月17日提交文件：submission_best_20191117.csv\n'
        f'格式 id,pctr，共{len(raw):,}行，按final_test原始行序，id为零起始行号。\n'
        f'兼容文件：submission_log_id_20191117.csv，格式log_id,pctr，保留重复log_id。\n\n'
        f'模型方案依据16号真实标签选择，其开发集AUC为{protocol["development16_auc"]:.10f}。\n'
        f'最新增强行特征CatBoost结合全量神经网络基础预测、每日响应修正和贝叶斯历史修正。\n'
        f'使用已冻结的树权重（训练截至15号）；基础网络和17号历史特征可使用截至16号数据。\n'
        f'17号反推标签仅用于此次评估，未用于训练、调参或选择提交。\n'
        f'反推标签AUC：{auc:.10f}；平台真实AUC未知。\n'
        f'已独立重建17号原始特征并复算，最大预测误差{error:g}；原始数据未修改。\n', encoding='utf8')
    emit('verified_and_evaluated', **report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['predict', 'verify'], required=True)
    args = parser.parse_args()
    predict() if args.phase == 'predict' else verify()
