"""Transfer the verified news-quality winner before reading inferred17 outcomes."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import gc
import inspect
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT/'analysis/day16_auc082_20261102'
PREVIOUS = ROOT/'analysis/final_best_recipe_augmented_20261102'
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(PREVIOUS))
import export_best as previous
from features import read, write, digest
from rich_userday_activity import array_hash, correct
from personal_context_residual import personal_changes
from topic_context_residual import topic_features
from catboost_userday_residual import CATEGORIES, load_model
from catboost import Pool
from catboost_global_rows import predict as cat_predict, recipes as cat_recipes
from bayesian_daily_offset import prior_mass, lgb
from news_quality_features import READ, build_quality, self_test

RUN = HERE/'run_v2'
OUTPUT = ROOT/'output/best_recipe_news_quality_20191117_20261102'
QUALITY = EXP/'runs/news_quality_daily_v1'
OLD_OUTPUT = ROOT/'output/best_recipe_augmented_20191117_20261102'
OLD_PROTOCOL = PREVIOUS/'run_v2/protocol.json'


def emit(event, **values):
    print(__import__('json').dumps(dict(event=event, **values), ensure_ascii=False), flush=True)


def source_paths():
    sources = previous.source_paths()+[Path(__file__), PREVIOUS/'export_best.py', OLD_PROTOCOL,
        OLD_OUTPUT/'verification.json', OLD_OUTPUT/'day16_prediction.npy', OLD_OUTPUT/'day17_prediction.npy',
        QUALITY/'verification.json', QUALITY/'protocol.json', QUALITY/'input_contract16.json',
        QUALITY/'day16_prediction.npy', QUALITY/'models_locked.json', QUALITY/'quality16.npz']
    sources += [Path(p) for p in read(QUALITY/'protocol.json')['sources_sha256']]
    selected = read(QUALITY/'verification.json')['dates']['20191116']['best']
    sources.append(QUALITY/(selected['name']+'.txt'))
    for row_folder, core_folder in [
        (previous.UPDATED/'run_v1/input', previous.ORIGINAL_RUN),
        (PREVIOUS/'run_v2/independent_row_input', PREVIOUS/'run_v2/independent_original_input')]:
        sources += [row_folder/'features.npy']
        if (row_folder/'dates.npy').is_file():
            sources.append(row_folder/'dates.npy')
        sources += [core_folder/name for name in ['numeric.npy', 'sequences.npy', 'dates.npy', 'all_context.npz']]
    return list(dict.fromkeys(sources))


def freeze():
    assert not RUN.exists() and not OUTPUT.exists(), 'Preserve all old exports and failures'
    report = read(QUALITY/'verification.json')
    assert report['status'] == 'passed' and read(OLD_OUTPUT/'verification.json')['status'] == 'passed'
    selected = report['dates']['20191116']['best']
    assert selected['recipe'] == 'replace_daily_correction_on_original_augmented'
    assert selected['objective'] == 'binary'
    published = read(ROOT/'output/two_day_auc_targets_20261102/independent_target_audit.json')
    assert published['audit_status'] == 'passed'
    assert abs(published['dates']['20191116']['auc']-selected['auc']) < 1e-12
    locks = read(QUALITY/'models_locked.json')['models']
    record = next(r for r in locks if r['name'] == selected['name'])
    assert digest(QUALITY/(record['name']+'.txt')) == record['model_sha256']
    raw_hashes = read(QUALITY/'protocol.json')['raw_datasets_sha256']
    for name, expected in raw_hashes.items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    for manifest_path in [OLD_PROTOCOL, QUALITY/'protocol.json']:
        for path, expected in read(manifest_path)['sources_sha256'].items():
            assert digest(path) == expected, path
    RUN.mkdir(parents=True)
    protocol = {'status': 'frozen_before_day17_prediction_and_inferred_label_evaluation',
        'frozen_utc': datetime.now(timezone.utc).isoformat(), 'date': 20191117,
        'selected': selected, 'development16_auc': selected['auc'],
        'selection_basis': 'Highest independently verified real16 development AUC at freeze time',
        'development_dates_reused_for_model_selection': True,
        'new_predictive_model_training': False, 'tree_training_max_date': 20191115,
        'base_neural_reference_training_max_date': 20191116,
        'day17_ad_history_max_date': 20191116, 'same_day_supplied_news_feedback_allowed': True,
        'inferred17_outcomes_used_for_training_tuning_or_selection': False,
        'old_recipe_protocol_sha256': digest(OLD_PROTOCOL), 'quality_boundary_self_tests': self_test(),
        'raw_datasets_sha256': raw_hashes,
        'sources_sha256': {str(p): digest(p) for p in source_paths()}}
    write(RUN/'protocol.json', protocol)
    emit('recipe_frozen', development16_auc=selected['auc'], selected=selected)
    return protocol


def check_sources():
    protocol = read(RUN/'protocol.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    return protocol


def daily(day, protocol):
    """Rebuild the trained schema, holding the vocabulary fixed at the16 fit."""
    pipeline = previous.original.Pipeline(day)
    values, names, changes = pipeline.news()
    context_changes = personal_changes(pipeline.keys, pipeline.context)
    news = pd.concat([pd.read_csv(ROOT/'2026ml'/name,
        usecols=['e_et', 'u_userId', 'i_cat', 'label', 'pro'], dtype='int32')
        for name in ['train_feeds.csv', 'test_feeds.csv']], ignore_index=True)
    topics = np.sort(news.loc[news.e_et < 20191116, 'i_cat'].unique()).tolist()
    namespace = {'np': np, 'pd': pd, 'sparse': sparse, 'personal_changes': personal_changes, 'cutoff': day}
    source = inspect.getsource(topic_features).replace('20191116', 'cutoff')
    exec(compile(source, '<frozen_date_bound_topic_features>', 'exec'), namespace)
    topic, topic_names, _ = namespace['topic_features'](news, pipeline.keys, topics)
    names_all = pipeline.context_names+pipeline.past_names+pipeline.enhanced_names
    names_all += ['news__'+n for n in names]+['news_previous_delta__'+n for n in names]
    names_all += ['news_personal_mean_delta__'+n for n in names]+['news_has_previous', 'news_log_previous_days']
    names_all += ['context_previous_delta__'+n for n in pipeline.context_names]
    names_all += ['context_personal_mean_delta__'+n for n in pipeline.context_names]
    names_all += ['context_has_previous', 'context_log_previous_days']
    names_all += topic_names
    x = np.column_stack([pipeline.context, pipeline.past, pipeline.enhanced, values, changes,
                         context_changes, topic]).astype('float32')
    native = read(QUALITY/'input_contract16.json')['native_contract']['native_contract']
    assert names_all == native['feature_names']
    assert np.isfinite(x).all() and np.all(pipeline.counts[pipeline.query] == -1)
    frame = pd.DataFrame(x[pipeline.query], columns=[f'x{i}' for i in range(x.shape[1])])
    for field in CATEGORIES:
        frame[field] = pipeline.catframe[field].to_numpy(dtype='int64')
    frame['user_id'] = pipeline.qkeys[:, 1].astype('int64')
    if day == 20191116:
        for name, array in [('x', x), ('keys', pipeline.keys), ('counts', pipeline.counts)]:
            assert array_hash(array) == native[name+'_sha256'], name
    del x, values, changes, context_changes, topic, news
    gc.collect()
    mass = np.empty(len(pipeline.keys), dtype='float64')
    clicks = exposures = 0.
    for date in np.unique(pipeline.keys[:, 0]):
        selected = pipeline.keys[:, 0] == date
        prior = (clicks+8.4)/(exposures+1000.)
        mass[selected] = pipeline.sizes[selected]*prior
        if date < day:
            clicks += float(pipeline.counts[selected].sum())
            exposures += float(pipeline.sizes[selected].sum())
    if day == 20191116:
        assert array_hash(mass) == native['mass_prior_sha256']
    old_protocol = read(OLD_PROTOCOL)
    base = np.load(previous.PREVIOUS_OUTPUT/f'day{day%100}_prediction.npy')
    pool = Pool(frame, cat_features=CATEGORIES+['user_id'])
    daily_corrected = {}
    for objective, folder, key in [('binary', previous.BINARY, 'name'), ('poisson', previous.COUNT, 'model_name')]:
        record = old_protocol['binary' if objective == 'binary' else 'count']
        offset = logit((-np.expm1(-mass[pipeline.query])).clip(1e-6, 1-1e-6)) if objective == 'binary' else np.log(mass[pipeline.query])
        margin = load_model(folder/(record[key]+'.cbm')).predict(pool, prediction_type='RawFormulaVal')+offset
        group = expit(margin) if objective == 'binary' else np.exp(margin.clip(-15, 8))
        daily_corrected[objective] = correct(base, group, pipeline.groups, objective, record['power'])
    blend = old_protocol['compose']
    assert blend['formula'] == 'logit_blend'
    rich_base = expit((1-blend['weight'])*logit(daily_corrected['binary'].clip(1e-6, 1-1e-6))+
                      blend['weight']*logit(daily_corrected['poisson'].clip(1e-6, 1-1e-6)))
    if day == 20191116:
        np.testing.assert_allclose(rich_base, np.load(previous.COMPOSE/'day16_prediction.npy'), rtol=0, atol=2e-7)
    del pool, daily_corrected
    fields = ['pt_d', 'label', 'user_id', 'spread_app_id', 'slot_id']
    ad = pd.read_csv(ROOT/'2026ml/train.csv', usecols=fields, dtype='int32')
    final = pd.read_csv(ROOT/'2026ml/final_test.csv', usecols=[n for n in fields if n != 'label'], dtype='int32')
    final['label'] = -1
    ad = pd.concat([ad, final], ignore_index=True)
    bayesian_mass, audit = prior_mass(ad, pipeline.keys, day)
    assert all(r['label_max_date'] is None or r['label_max_date'] < r['query_date'] for r in audit)
    frame['log_bayesian_click_mass'] = np.log(bayesian_mass[pipeline.query])
    offset = logit((-np.expm1(-bayesian_mass[pipeline.query])).clip(1e-6, 1-1e-6))
    old_model = lgb.Booster(model_str=(previous.WINNER/(old_protocol['selected']['name']+'.txt')).read_text(encoding='utf8'))
    old_response = expit(old_model.predict(frame, raw_score=True, num_threads=2)+offset)
    del ad, final, old_model
    news = pd.concat([pd.read_csv(ROOT/'2026ml'/name, usecols=READ,
            dtype={'i_docId': str, 'i_s_sourceId': str}) for name in ['train_feeds.csv', 'test_feeds.csv']], ignore_index=True)
    quality, quality_names = build_quality(news, pipeline.keys, day)
    trained = read(QUALITY/'input_contract16.json')
    assert quality_names == trained['quality_names']
    if day == 20191116:
        assert array_hash(quality) == trained['quality_sha256']
        assert array_hash(bayesian_mass) == trained['native_contract']['bayesian_mass_sha256']
    frame = pd.concat([frame, pd.DataFrame(quality[pipeline.query], columns=quality_names)], axis=1)
    record = protocol['selected']
    model = lgb.Booster(model_str=(QUALITY/(record['name']+'.txt')).read_text(encoding='utf8'))
    assert model.feature_name() == frame.columns.tolist()
    response = expit(model.predict(frame, raw_score=True, num_threads=2)+offset)
    contract = {'day': day, 'shape': list(frame.shape), 'keys_sha256': array_hash(pipeline.qkeys),
        'frame_sha256': array_hash(frame.to_numpy()), 'quality_sha256': array_hash(quality),
        'bayesian_audit': audit, 'inferred_ad_outcomes_used': False}
    result = rich_base, old_response, response, pipeline.groups.copy(), contract
    del news, frame, quality, pipeline, model
    gc.collect()
    emit('daily_transfer_ready', day=day, shape=contract['shape'])
    return result


def forecast(protocol, verifying=False):
    row_dir = PREVIOUS/'run_v2/independent_row_input' if verifying else previous.UPDATED/'run_v1/input'
    core_dir = PREVIOUS/'run_v2/independent_original_input' if verifying else previous.ORIGINAL_RUN
    previous.original.RUN = core_dir
    rows = np.load(row_dir/'features.npy', mmap_mode='r')
    dates = np.load(core_dir/'dates.npy')
    numeric = np.load(core_dir/'numeric.npy', mmap_mode='r')
    sequences = np.load(core_dir/'sequences.npy', mmap_mode='r')
    assert len(dates) == len(rows) == len(numeric) == len(sequences)
    if (row_dir/'dates.npy').is_file():
        np.testing.assert_array_equal(dates, np.load(row_dir/'dates.npy'))
    old_protocol = read(OLD_PROTOCOL)
    contract = read(previous.AUG/'input_contract.json')
    record = next(r for r in read(previous.AUG/'models_locked.json')['models'] if r['name'] == old_protocol['augmented']['name'])
    assert digest(previous.AUG/(record['name']+'.cbm')) == record['model_sha256']
    result = {}
    for day in [20191116, 20191117]:
        rich_base, old_response, response, groups, daily_contract = daily(day, protocol)
        selected = dates == day
        features = np.column_stack([rows[selected], numeric[selected], sequences[selected, :3, :].reshape(selected.sum(), 15)]).astype('float32')
        assert features.shape[1] == len(contract['feature_names']) == 472
        frame = pd.DataFrame(features, columns=[f'f{i}' for i in range(features.shape[1])])
        for i in contract['categorical_columns']:
            frame[f'f{i}'] = frame.iloc[:, i].to_numpy(dtype='int64')
        pool = Pool(frame, cat_features=[f'f{i}' for i in contract['categorical_columns']])
        p = cat_predict(load_model(previous.AUG/(record['name']+'.cbm')), pool, record['positive_weight'])
        augmented = next(q for name, q in cat_recipes(p, rich_base) if name == old_protocol['augmented']['recipe'])
        old_prediction = correct(augmented, old_response, groups, 'binary', old_protocol['selected']['power'])
        np.testing.assert_allclose(old_prediction, np.load(OLD_OUTPUT/f'day{day%100}_prediction.npy'), rtol=0, atol=2e-7)
        result[day] = correct(augmented, response, groups, protocol['selected']['objective'], protocol['selected']['power'])
        if day == 20191116:
            np.testing.assert_allclose(augmented, np.load(previous.AUG/'prediction.npy'), rtol=0, atol=2e-7)
            np.testing.assert_allclose(result[day], np.load(QUALITY/'day16_prediction.npy'), rtol=0, atol=2e-7)
            raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
            auc = float(roc_auc_score(raw.loc[raw.pt_d == day, 'label'], result[day]))
            assert abs(auc-protocol['development16_auc']) < 1e-12
        path = RUN/f'contract{day%100}.json'
        if verifying:
            assert daily_contract == read(path)
        else:
            write(path, daily_contract)
        emit('prediction_ready', day=day, rows=len(result[day]), verifying=verifying)
        del frame, features, pool, rich_base, augmented
        gc.collect()
    return result


def predict():
    protocol = freeze()
    result = forecast(protocol)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    raw = pd.read_csv(ROOT/'2026ml/final_test.csv', usecols=['log_id', 'pt_d'])
    assert raw.pt_d.eq(20191117).all() and len(raw) == len(result[20191117]) == 404182
    for day, p in result.items():
        np.save(OUTPUT/f'day{day%100}_prediction.npy', p)
    pd.DataFrame({'id': np.arange(len(raw)), 'pctr': result[20191117]}).to_csv(
        OUTPUT/'submission_best_20191117.csv', index=False, float_format='%.10f')
    pd.DataFrame({'log_id': raw.log_id, 'pctr': result[20191117]}).to_csv(
        OUTPUT/'submission_log_id_20191117.csv', index=False, float_format='%.10f')
    write(OUTPUT/'prediction_manifest.json', {'status': 'frozen_before_inferred17_evaluation',
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
    result = forecast(protocol, verifying=True)
    error = max(float(np.max(np.abs(p-np.load(OUTPUT/f'day{day%100}_prediction.npy')))) for day, p in result.items())
    assert error < 2e-7
    raw = pd.read_csv(ROOT/'2026ml/final_test.csv', usecols=['log_id', 'pt_d'])
    submission = pd.read_csv(OUTPUT/'submission_best_20191117.csv')
    alternate = pd.read_csv(OUTPUT/'submission_log_id_20191117.csv')
    assert list(submission) == ['id', 'pctr'] and list(alternate) == ['log_id', 'pctr']
    assert len(raw) == len(submission) == len(alternate) == 404182
    np.testing.assert_array_equal(submission.id, np.arange(len(raw)))
    np.testing.assert_array_equal(alternate.log_id, raw.log_id)
    np.testing.assert_array_equal(alternate.pctr, submission.pctr)
    assert np.isfinite(submission.pctr).all() and submission.pctr.between(0, 1).all()
    rounding = float(np.max(np.abs(submission.pctr.to_numpy()-result[20191117])))
    assert rounding < 5.01e-11
    check_sources()
    # First read of inferred17 outcomes: recipe, predictions and replay are already fixed.
    label_file = ROOT/'analysis/eleven_score_strategy_20261102/imported_refinement/day17_candidate_labels_11scores.csv'
    assert digest(label_file) == 'ab29c227c97efcc4fbef4dd57d92a8657ce5c078f1cbd03f037d8183f0ca7e24'
    labels = pd.read_csv(label_file)
    np.testing.assert_array_equal(labels.id, submission.id)
    np.testing.assert_array_equal(labels.log_id, raw.log_id)
    assert labels.label.isin([0, 1]).all()
    auc = float(roc_auc_score(labels.label, submission.pctr))
    npositive = int(labels.label.sum())
    manual = float((rankdata(submission.pctr, method='average')[labels.label.eq(1)].sum()-npositive*(npositive+1)/2)/
                   (npositive*(len(labels)-npositive)))
    assert abs(auc-manual) < 1e-12
    report = {'status': 'passed', 'date': 20191117, 'rows': len(raw),
        'development16_auc': protocol['development16_auc'], 'inferred_label_auc': auc,
        'independent_rank_auc': manual, 'inferred_positive_count': npositive,
        'labels_are_inferred_not_observed': True, 'platform_auc': None,
        'inferred_labels_used_for_training_tuning_or_selection': False,
        'maximum_prediction_replay_error': error, 'maximum_csv_rounding_error': rounding,
        'day16_verified_winner_replay': True, 'previous_day17_recipe_exactly_replayed': True,
        'changed_daily_features_rebuilt_from_raw': True,
        'row_features_independently_rebuilt_in_prior_verified_export': True,
        'original_row_order_verified': True, 'repeated_log_ids_preserved': int(raw.log_id.duplicated().sum()),
        'original_datasets_unchanged': True, 'inferred_label_sha256': digest(label_file),
        'submission_sha256': digest(OUTPUT/'submission_best_20191117.csv'),
        'protocol_sha256': digest(RUN/'protocol.json')}
    write(OUTPUT/'verification.json', report)
    (OUTPUT/'提交说明.md').write_text(
        f'2019年11月17日提交文件：submission_best_20191117.csv\n'
        f'格式 id,pctr，共{len(raw):,}行，按final_test原始行序，id为零起始行号。\n'
        f'兼容文件：submission_log_id_20191117.csv，格式log_id,pctr，保留重复log_id。\n\n'
        f'模型依据16号真实标签开发集选择，AUC为{protocol["development16_auc"]:.10f}。\n'
        f'使用冻结的新闻内容质量修正方案，树权重训练截至15号；基础网络与历史可使用截至16号数据。\n'
        f'17号反推标签仅用于此次评估，未用于训练、调参或选择提交。\n'
        f'反推标签AUC：{auc:.10f}；平台真实AUC未知。\n'
        f'新特征已从原始数据独立复算，最大预测重放误差{error:g}；原始数据未修改。\n', encoding='utf8')
    emit('verified_and_evaluated', **report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['predict', 'verify'], required=True)
    args = parser.parse_args()
    predict() if args.phase == 'predict' else verify()
