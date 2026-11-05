"""Export the real-day16 daily-quality winner before evaluating inferred17 labels."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import gc
import importlib.util
import inspect
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT/'analysis/day16_auc082_20261102'
PRIOR_EXPORT = ROOT/'analysis/final_best_recipe_quality_transfer_20261102'
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT/'analysis/final_best_recipe_20261102'))
spec = importlib.util.spec_from_file_location('frozen_quality_export', PRIOR_EXPORT/'export_best.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from catboost import Pool
from catboost_userday_residual import CATEGORIES, load_model
from rich_userday_activity import array_hash, correct
from fit_stacker import digest, write
from conditional_daily import read
from compose_quality_daily_transfer import predict as compose
from group_temperature import FIELDS, GROUPS, group_codes

RUN = HERE/'run_v1'
OUTPUT = ROOT/'output/best_recipe_daily_quality_transfer_20191117_20261102'
WINNER = EXP/'runs/compose_quality_daily_transfer_v1'
DAILY = EXP/'runs/quality_daily_catboost_v1'
MODEL_NAME = 'day16_poisson_d5_offset_trees1600'
LABEL_FILE = old.LABEL_FILE
LABEL_SHA = old.LABEL_SHA


def emit(event, **values):
    old.emit(event, **values)


def check_sources(raw=False):
    protocol = read(RUN/'protocol.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    if raw:
        for name, expected in protocol['raw_datasets_sha256'].items():
            assert digest(ROOT/'2026ml'/name) == expected, name
    return protocol


def freeze():
    assert not RUN.exists() and not OUTPUT.exists(), 'Preserve completed exports and failures'
    transfer = read(WINNER/'verification.json')
    daily = read(DAILY/'verification.json')
    previous = read(old.OUTPUT/'verification.json')
    assert transfer['status'] == daily['status'] == previous['status'] == 'passed'
    chosen = transfer['dates']['20191116']['best']
    count = daily['dates']['20191116']['best']
    assert chosen['name'] == 'task_between1.5_within1'
    assert count['name'] == MODEL_NAME and count['objective'] == 'poisson' and count['power'] == .125
    assert not previous['inferred_labels_used_for_training_tuning_feature_or_model_selection']
    audit = read(ROOT/'output/two_day_auc_targets_20261102/independent_target_audit.json')
    assert audit['audit_status'] == 'passed'
    assert abs(audit['dates']['20191116']['auc']-chosen['auc']) < 1e-12
    hashes = read(DAILY/'protocol.json')['raw_datasets_sha256']
    paths = [Path(__file__), HERE/'test_export.py', WINNER/'verification.json',
        WINNER/'protocol.json', DAILY/'verification.json', DAILY/'protocol.json',
        DAILY/'models_locked.json', DAILY/(MODEL_NAME+'.cbm'), DAILY/'input_contract16.json',
        DAILY/(MODEL_NAME+'_held_group_prediction.npy'), old.OUTPUT/'verification.json',
        PRIOR_EXPORT/'run_v2/protocol.json', EXP/'compose_quality_daily_transfer.py',
        EXP/'group_temperature.py', EXP/'rich_userday_activity.py']
    for protocol_path in [DAILY/'protocol.json', WINNER/'protocol.json', PRIOR_EXPORT/'run_v2/protocol.json']:
        parent = read(protocol_path)
        assert parent['raw_datasets_sha256'] == hashes
        for path, expected in parent['sources_sha256'].items():
            assert digest(path) == expected, path
            paths.append(Path(path))
    for day in [16, 17]:
        paths += [old.OUTPUT/f'day{day}_prediction.npy']
        paths += [PRIOR_EXPORT/f'run_v2/input_day{day}'/name for name in
            ['features.npy', 'daily_quality.npz', 'contract.json']]
    for folder in [WINNER, old.GROUPED, DAILY]:
        paths += [folder/'day16_prediction.npy']
    for core in [old.CORE, old.ALT_CORE]:
        paths += [core/'all_context.npz']
    for name, expected in hashes.items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    RUN.mkdir(parents=True)
    write(RUN/'protocol.json', dict(status='frozen_before_day17_prediction_and_inferred_label_evaluation',
        frozen_utc=datetime.now(timezone.utc).isoformat(), development16_auc=chosen['auc'],
        selection_basis='Highest independently verified published real-day16 recipe at freeze',
        selected_transfer=chosen, selected_daily=count, new_model_training=False,
        correction_training_max_date=20191115, ad_history_max_date=20191116,
        inherited_reference_training_max_date=20191116, same_day_supplied_news_feedback_used=True,
        development_dates_repeatedly_reused=True,
        inferred17_labels_used_for_training_tuning_feature_or_model_selection=False,
        inferred17_labels_previously_evaluated_on_older_frozen_exports=True,
        raw_datasets_sha256=hashes, sources_sha256={str(p):digest(p) for p in dict.fromkeys(paths)}))
    emit('recipe_frozen', development16_auc=chosen['auc'])


def query_features(day, verifying=False):
    """Materialize query rows only, preserving the fitted complete daily schema."""
    old.original.RUN = old.ALT_CORE if verifying else old.CORE
    pipeline = old.original.Pipeline(day)
    assert np.all(pipeline.counts[pipeline.query] == -1)
    native = read(DAILY/'input_contract16.json')
    native_names = native['source_contract']['native_contract']['native_contract']['feature_names']
    qkeys, keys, query = pipeline.qkeys.copy(), pipeline.keys.copy(), pipeline.query.copy()
    row_groups = pipeline.groups.copy()
    cats = pipeline.catframe.copy()
    categories = native['categories']
    assert categories == CATEGORIES+['user_id']
    x = np.empty((len(qkeys), len(native_names)), dtype='float32')
    offset, names = 0, []

    def append(values, feature_names):
        nonlocal offset
        assert values.shape[1] == len(feature_names)
        x[:, offset:offset+len(feature_names)] = values[query]
        offset += len(feature_names)
        names.extend(feature_names)

    append(pipeline.context, pipeline.context_names)
    append(pipeline.past, pipeline.past_names)
    append(pipeline.enhanced, pipeline.enhanced_names)
    values, news_names, changes = pipeline.news()
    append(values, ['news__'+n for n in news_names])
    append(changes, ['news_previous_delta__'+n for n in news_names]+
        ['news_personal_mean_delta__'+n for n in news_names]+['news_has_previous','news_log_previous_days'])
    del values, changes
    gc.collect()
    changes = old.personal_changes(pipeline.keys, pipeline.context)
    append(changes, ['context_previous_delta__'+n for n in pipeline.context_names]+
        ['context_personal_mean_delta__'+n for n in pipeline.context_names]+
        ['context_has_previous','context_log_previous_days'])
    del changes, pipeline
    gc.collect()
    news = pd.concat([pd.read_csv(ROOT/'2026ml'/name,
        usecols=['e_et','u_userId','i_cat','label','pro'], dtype='int32')
        for name in ['train_feeds.csv','test_feeds.csv']], ignore_index=True)
    topics = np.sort(news.loc[news.e_et < 20191116, 'i_cat'].unique()).tolist()
    namespace = dict(np=np, pd=pd, sparse=sparse, personal_changes=old.personal_changes, cutoff=day)
    code = inspect.getsource(old.topic_features).replace('20191116', 'cutoff')
    exec(compile(code, '<frozen_date_bound_topics>', 'exec'), namespace)
    topic, topic_names, _ = namespace['topic_features'](news, keys, topics)
    append(topic, topic_names)
    del topic, news
    assert offset == x.shape[1] and names == native_names
    gc.collect()
    fields = ['pt_d','label','user_id','spread_app_id','slot_id']
    ads = pd.read_csv(ROOT/'2026ml/train.csv', usecols=fields, dtype='int32')
    final = pd.read_csv(ROOT/'2026ml/final_test.csv', usecols=[n for n in fields if n != 'label'], dtype='int32')
    final['label'] = -1
    ads = pd.concat([ads, final], ignore_index=True)
    mass, audit = old.prior_mass(ads, keys, day)
    assert all(r['label_max_date'] is None or r['label_max_date'] < r['query_date'] for r in audit)
    del ads, final
    quality_file = PRIOR_EXPORT/f'run_v2/input_day{day%100}/daily_quality.npz'
    quality = np.load(quality_file)
    np.testing.assert_array_equal(qkeys, quality['keys'])
    frame = pd.DataFrame(x, columns=[f'x{i}' for i in range(x.shape[1])])
    cat_frame = pd.DataFrame({name:cats[name].to_numpy(dtype='int64') for name in CATEGORIES})
    cat_frame['user_id'] = qkeys[:,1].astype('int64')
    quality_names = native['source_contract']['quality_names']
    frame = pd.concat([frame, cat_frame,
        pd.DataFrame({'log_bayesian_click_mass':np.log(mass[query])}),
        pd.DataFrame(quality['values'], columns=quality_names)], axis=1)
    assert list(frame) == native['feature_names'] and len(quality_names) == 337
    assert np.isfinite(frame.to_numpy()).all()
    if day == 20191116:
        cached = np.load(EXP/'runs/news_quality_daily_v1/quality16.npz')
        keep = cached['keys'][:,0] == day
        np.testing.assert_array_equal(cached['keys'][keep], qkeys)
        np.testing.assert_array_equal(cached['values'][keep], quality['values'])
        assert array_hash(keys) == native['keys_sha256']
        assert array_hash(mass) == native['source_contract']['native_contract']['bayesian_mass_sha256']
    model = load_model(DAILY/(MODEL_NAME+'.cbm'))
    assert model.feature_names_ == list(frame)
    margin = model.predict(Pool(frame, cat_features=categories), prediction_type='RawFormulaVal', thread_count=2)
    predicted = np.exp((margin+np.log(mass[query])).clip(-15,8))
    if day == 20191116:
        np.testing.assert_allclose(predicted,
            np.load(DAILY/(MODEL_NAME+'_held_group_prediction.npy')), rtol=0, atol=2e-12)
    contract = dict(date=day, query_shape=list(frame.shape), frame_sha256=array_hash(frame.to_numpy(dtype='float32')),
        keys_sha256=array_hash(qkeys), row_groups_sha256=array_hash(row_groups),
        group_prediction_sha256=array_hash(predicted), bayesian_audit=audit,
        cached_quality_sha256=digest(quality_file), advertisement_query_targets_masked=True,
        frozen_news_vocabulary_cutoff=20191116, feature_count=len(frame.columns))
    emit('complete_daily_query_replayed', day=day, verifying=verifying, shape=contract['query_shape'])
    return qkeys, row_groups, predicted, contract


def grouped_anchor(day, verifying=False):
    """Replay the inherited grouped row-tree reference on actual query rows."""
    raw = pd.read_csv(ROOT/'2026ml'/('train.csv' if day == 20191116 else 'final_test.csv'),
        usecols=['pt_d']+FIELDS, dtype='int32')
    raw = raw.loc[raw.pt_d.eq(day), FIELDS].reset_index(drop=True)
    matrix = np.load(PRIOR_EXPORT/f'run_v2/input_day{day%100}/features.npy', mmap_mode='r')
    native = read(old.SOURCE/'input_contract16.json')
    forecasts = []
    for name in old.MODEL_NAMES[:2]:
        model = load_model(old.RICH/(name+'.cbm'))
        chunks = []
        batch = 7000 if verifying else 12000
        for start in range(0,len(raw),batch):
            values = np.asarray(matrix[start:start+batch])
            frame = old.frame_from_matrix(values, np.arange(len(values)), native['categorical_columns'])
            pool = Pool(frame, cat_features=['f'+str(i) for i in native['categorical_columns']])
            margin = model.predict(pool, prediction_type='RawFormulaVal', thread_count=2)
            chunks.append(expit(margin-np.log(5.)))
        forecasts.append(np.concatenate(chunks))
        del model, pool, frame, values
        gc.collect()
    protocol = read(PRIOR_EXPORT/'run_v2/protocol.json')
    anchor = old.normalized(np.load(old.BASE/f'day{day%100}_prediction.npy'), forecasts[1],
        protocol['selected']['rich']['weight'])
    record = protocol['selected']['grouped']
    mean = np.mean(forecasts, axis=0)
    if verifying:
        result = old.independent(raw, anchor, mean, record)
    else:
        groups = group_codes(raw, GROUPS[record['group']])
        result = old.grouped_predict(old.components(anchor, mean, groups), record['between'], record['within'])
    if day == 20191116:
        np.testing.assert_allclose(result, np.load(old.GROUPED/'day16_prediction.npy'), rtol=0, atol=2e-14)
    return raw, result


def independent_prediction(raw, current, anchor, predicted, row_groups, record):
    """Independent pandas reconstruction of Poisson mass and task decomposition."""
    table = pd.DataFrame({'group':row_groups, 'p':np.clip(anchor,1e-8,1-1e-8)})
    mass = table.groupby('group',sort=False).p.sum().reindex(np.arange(len(predicted)),fill_value=0).to_numpy()
    ratio = np.clip(np.maximum(predicted,1e-6)/mass.clip(1e-8),.05,20.)**.125
    odds = table.p.to_numpy()/(1-table.p.to_numpy())*ratio[row_groups]
    daily = odds/(1+odds)
    delta = logit(daily)-logit(anchor)
    task = raw[['task_id']].copy()
    task['delta'] = delta
    between = task.groupby('task_id',sort=False).delta.transform('mean').to_numpy()
    return expit(logit(current)+record['between']*between+record['within']*(delta-between))


def predict_day(day, verifying=False):
    protocol = read(RUN/'protocol.json')
    qkeys, row_groups, predicted, contract = query_features(day, verifying)
    raw, anchor = grouped_anchor(day, verifying)
    expected_groups = pd.Index(qkeys[:,1]).get_indexer(raw.user_id)
    np.testing.assert_array_equal(expected_groups, row_groups)
    daily = correct(anchor, predicted, row_groups, 'poisson', protocol['selected_daily']['power'])
    current = np.load(old.OUTPUT/f'day{day%100}_prediction.npy')
    values = dict(current=current, daily=daily, anchor=anchor)
    data = dict(frame=raw, values=values, logits={name:logit(p) for name,p in values.items()},
        groups={'task':group_codes(raw,GROUPS['task'])})
    p = compose(data, protocol['selected_transfer'], independent=verifying)
    independent = independent_prediction(raw,current,anchor,predicted,row_groups,protocol['selected_transfer'])
    assert np.max(np.abs(p-independent)) < 2e-14
    if day == 20191116:
        np.testing.assert_allclose(daily,np.load(DAILY/'day16_prediction.npy'),rtol=0,atol=2e-14)
        np.testing.assert_allclose(p,np.load(WINNER/'day16_prediction.npy'),rtol=0,atol=2e-14)
    contract['independent_pandas_maximum_error'] = float(np.max(np.abs(p-independent)))
    return p, contract


def predict():
    protocol = check_sources(raw=True)
    assert not OUTPUT.exists()
    contracts = {}
    for day in [20191116,20191117]:
        p, contract = predict_day(day)
        contracts[str(day)] = contract
        np.save(RUN/f'day{day%100}_prediction.npy',p)
    write(RUN/'input_contracts.json',contracts)
    OUTPUT.mkdir()
    raw = pd.read_csv(ROOT/'2026ml/final_test.csv',usecols=['pt_d','log_id'],dtype='int32')
    p = np.load(RUN/'day17_prediction.npy')
    assert len(p) == len(raw) == 404182 and raw.pt_d.eq(20191117).all()
    pd.DataFrame(dict(id=np.arange(len(p)),pctr=p)).to_csv(OUTPUT/'submission_best_20191117.csv',
        index=False,float_format='%.17g')
    pd.DataFrame(dict(log_id=raw.log_id,pctr=p)).to_csv(OUTPUT/'submission_log_id_20191117.csv',
        index=False,float_format='%.17g')
    for day in [16,17]:
        np.save(OUTPUT/f'day{day}_prediction.npy',np.load(RUN/f'day{day}_prediction.npy'))
    write(OUTPUT/'prediction_manifest.json',dict(status='locked_before_inferred17_label_evaluation',
        protocol_sha256=digest(RUN/'protocol.json'), input_contracts_sha256=digest(RUN/'input_contracts.json'),
        development16_auc=protocol['development16_auc'], inferred17_labels_used_for_prediction_or_selection=False,
        files_sha256={p.name:digest(p) for p in OUTPUT.iterdir() if p.is_file()}))
    emit('submission_locked',path=str(OUTPUT/'submission_best_20191117.csv'))


def verify():
    protocol = check_sources(raw=True)
    assert not (OUTPUT/'verification.json').exists()
    manifest = read(OUTPUT/'prediction_manifest.json')
    assert manifest['protocol_sha256'] == digest(RUN/'protocol.json')
    assert manifest['input_contracts_sha256'] == digest(RUN/'input_contracts.json')
    for name, expected in manifest['files_sha256'].items():
        assert digest(OUTPUT/name) == expected,name
    p, contract = predict_day(20191117,verifying=True)
    expected = read(RUN/'input_contracts.json')['20191117']
    for field in expected:
        if field != 'independent_pandas_maximum_error':
            assert contract[field] == expected[field],field
    error = float(np.max(np.abs(p-np.load(OUTPUT/'day17_prediction.npy'))))
    assert error < 2e-14
    raw = pd.read_csv(ROOT/'2026ml/final_test.csv',usecols=['pt_d','log_id'],dtype='int32')
    submission = pd.read_csv(OUTPUT/'submission_best_20191117.csv',float_precision='round_trip')
    alternate = pd.read_csv(OUTPUT/'submission_log_id_20191117.csv',float_precision='round_trip')
    assert list(submission) == ['id','pctr'] and list(alternate) == ['log_id','pctr']
    assert len(submission) == len(raw) == 404182
    np.testing.assert_array_equal(submission.id,np.arange(len(raw)))
    np.testing.assert_array_equal(alternate.log_id,raw.log_id)
    np.testing.assert_array_equal(submission.pctr,alternate.pctr)
    np.testing.assert_array_equal(submission.pctr,np.load(OUTPUT/'day17_prediction.npy'))
    assert np.isfinite(submission.pctr).all() and submission.pctr.between(0,1).all()
    actual = pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    auc16 = float(roc_auc_score(actual.loc[actual.pt_d.eq(20191116),'label'],np.load(OUTPUT/'day16_prediction.npy')))
    assert abs(auc16-protocol['development16_auc']) < 1e-12
    # Evaluation-only label access follows frozen selection, exported predictions, and independent replay.
    assert digest(LABEL_FILE) == LABEL_SHA
    inferred = pd.read_csv(LABEL_FILE)
    np.testing.assert_array_equal(inferred.id,submission.id)
    np.testing.assert_array_equal(inferred.log_id,raw.log_id)
    assert inferred.label.isin([0,1]).all()
    auc = float(roc_auc_score(inferred.label,submission.pctr))
    positives = int(inferred.label.sum())
    independent_auc = float((rankdata(submission.pctr)[inferred.label.eq(1)].sum()-positives*(positives+1)/2)/
        (positives*(len(inferred)-positives)))
    assert abs(auc-independent_auc) < 1e-12
    check_sources(raw=True)
    report = dict(status='passed',date=20191117,rows=len(raw),development16_auc=auc16,
        inferred_label_auc=auc,independent_rank_auc=independent_auc,inferred_positive_count=positives,
        labels_are_inferred_not_observed=True,platform_auc=None,
        inferred_labels_used_for_training_tuning_feature_or_model_selection=False,
        new_model_training=False,correction_training_max_date=20191115,ad_history_max_date=20191116,
        inherited_reference_training_max_date=20191116,development_dates_repeatedly_reused=True,
        complete_daily_features=expected['feature_count'],day16_winning_forecasts_replayed=True,
        day17_daily_features_rebuilt_twice=True,inherited_row_features_and_quality_previously_independently_verified=True,
        maximum_prediction_replay_error=error,csv_roundtrip_exact=True,original_row_order_verified=True,
        repeated_log_ids_preserved=int(raw.log_id.duplicated().sum()),original_datasets_unchanged=True,
        raw_datasets_sha256=protocol['raw_datasets_sha256'],inferred_label_sha256=LABEL_SHA,
        protocol_sha256=digest(RUN/'protocol.json'),submission_sha256=digest(OUTPUT/'submission_best_20191117.csv'))
    write(OUTPUT/'verification.json',report)
    (OUTPUT/'提交说明.md').write_text(
        f'2019年11月17日提交文件：submission_best_20191117.csv\n'
        f'格式 id,pctr；{len(raw):,}行，id为final_test原始行序的零起始行号。\n'
        f'冻结时最佳真实16号开发集AUC={auc16:.12f}；15/16号曾反复用于开发。\n'
        f'17号反推标签AUC={auc:.12f}；该标签不是官方真实标签，平台真实AUC未知。\n'
        '反推标签未用于训练、调参、特征选择或模型选择。\n'
        '已验证原始四份CSV未改变、行序、CSV精度与独立预测复算。\n',encoding='utf8')
    emit('verified_and_evaluated',**report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=['freeze','predict','verify'])
    args = parser.parse_args()
    {'freeze':freeze,'predict':predict,'verify':verify}[args.phase]()
