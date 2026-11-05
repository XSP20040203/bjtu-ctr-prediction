"""Transfer the currently verified day16 winner to17, then evaluate inferred labels.

Prediction and evaluation are separate CLI phases. All recipe choices come
from verified day16 records. Existing sources and submissions stay intact.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import gc
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT/'analysis/day16_auc082_20261102'
OLD = ROOT/'analysis/final_best_recipe_20261102'
RUN = HERE/'run_v1'
OUTPUT = ROOT/'output/best_recipe_updated_20191117_20261102'
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(OLD))
from predict_best import Pipeline, RUN as OLD_RUN, read, write, digest
from global_row_context import CAT, HISTORY, TARGET, histories, feature_names, build_multiscale_statistics
from catboost_global_rows import recipes, predict as cat_predict
from catboost_userday_residual import load_model
from catboost import Pool
from rolling_previous_app import correction
from fit_stacker import REFERENCE
import feed_behavior_userday as feed
from personal_context_residual import personal_changes

CATDIR = EXP/'runs/catboost_global_rows_v1'
ROWDIR = EXP/'runs/global_row_context_v1'
APPDIR = EXP/'runs/rolling_previous_app_v1'


def emit(event, **values):
    print(json.dumps(dict(event=event, **values), ensure_ascii=False), flush=True)


def check_sources():
    protocol = read(RUN/'protocol.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    return protocol


def freeze():
    if (RUN/'protocol.json').exists():
        return check_sources()
    cat = read(CATDIR/'verification.json')
    app = read(APPDIR/'verification.json')
    assert cat['status'] == app['status'] == 'passed'
    best = cat['best']
    record = next(r for r in read(CATDIR/'models_locked.json')['models'] if r['name'] == best['name'])
    raw_hashes = read(CATDIR/'protocol.json')['raw_datasets_sha256']
    model_path = CATDIR/(record['name']+'.cbm')
    assert digest(model_path) == record['model_sha256']
    paths = [Path(__file__), OLD/'predict_best.py', OLD/'run_v1/protocol.json',
             OLD_RUN/'all_context.npz', OLD_RUN/'day16_compose_verified_v1.npy',
             OLD_RUN/'day17_compose_verified_v1.npy',
             ROOT/'output/best_recipe_20191117_20261102/verification.json',
             ROWDIR/'input_contract.json', ROWDIR/'verification.json',
             CATDIR/'protocol.json', CATDIR/'verification.json', CATDIR/'models_locked.json', model_path,
             APPDIR/'protocol.json', APPDIR/'verification.json', APPDIR/'statistics_day16.npz',
             REFERENCE/'day15_prediction.npy', REFERENCE/'day16_prediction.npy',
             EXP/'global_row_context.py', EXP/'rolling_previous_app.py', EXP/'catboost_global_rows.py',
             EXP/'feed_behavior_userday.py', EXP/'previous_day_residual.py',
             ROOT/'analysis/auc085_next/prepare_multiscale_stats.py']
    for name, expected in raw_hashes.items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    RUN.mkdir(parents=True, exist_ok=True)
    protocol = {'status':'frozen_before_day17_inferred_label_evaluation',
                'selected_catboost':record, 'selected_blend':best['recipe'],
                'previous_app_power':app['dates']['20191116']['best']['power'],
                'previous_app_alpha':3.,
                'development16_auc':app['dates']['20191116']['auc'],
                'selection_basis':'Verified real day16 outcomes; day16 repeatedly reused for development',
                'day17_base':'Previously independently verified full-data v8 plus frozen correction pipeline',
                'catboost_training_max_date':20191115,
                'day17_features_and_previous_app_counts_max_ad_label_date':20191116,
                'same_day_supplied_news_feedback_allowed':True,
                'inferred_day17_labels_used_for_training_or_selection':False,
                'new_predictive_model_training':False,
                'raw_datasets_sha256':raw_hashes,
                'sources_sha256':{str(path):digest(path) for path in paths}}
    write(RUN/'protocol.json', protocol)
    return protocol


def daily(day, contract):
    pipeline = Pipeline(day)
    schema = read(EXP/'runs/feed_behavior_userday_v1/news_schema.json')
    code = inspect.getsource(feed.feed_features).replace('20191116', 'cutoff')
    namespace = {'np':np, 'pd':pd, 'READ':feed.READ, 'CATEGORIES':feed.CATEGORIES,
                 'DIVERSITY':feed.DIVERSITY, 'cutoff':day}
    exec(compile(code, '<frozen_news_transform>', 'exec'), namespace)
    feeds = pd.concat([pd.read_csv(ROOT/'2026ml'/filename, usecols=feed.READ,
                        dtype={'i_docId':str, 'i_s_sourceId':str})
                       for filename in ['train_feeds.csv', 'test_feeds.csv']], ignore_index=True)
    values, names, _ = namespace['feed_features'](feeds, pipeline.keys, schema)
    changes = personal_changes(pipeline.keys, values)
    all_names = pipeline.context_names + pipeline.past_names + pipeline.enhanced_names
    all_names += ['news__'+name for name in names]
    all_names += ['news_previous_delta__'+name for name in names]
    all_names += ['news_personal_mean_delta__'+name for name in names]
    all_names += ['news_has_previous', 'news_log_previous_days']
    columns = contract['selector_columns']
    selected_names = ['daily__'+all_names[i] for i in columns]
    assert selected_names == contract['feature_names'][len(CAT)+len(feature_names()):-12]
    matrix = np.column_stack([pipeline.context, pipeline.past, pipeline.enhanced, values, changes]).astype('float32')
    result = matrix[pipeline.query][:,columns][pipeline.groups]
    del pipeline, feeds, matrix
    gc.collect()
    return result


def build_features(directory):
    directory.mkdir(parents=True, exist_ok=False)
    contract = read(ROWDIR/'input_contract.json')
    train = pd.read_csv(ROOT/'2026ml/train.csv', usecols=CAT+['pt_d','label'], dtype='int32')
    final = pd.read_csv(ROOT/'2026ml/final_test.csv', usecols=CAT+['pt_d'], dtype='int32')
    final['label'] = -1
    raw = pd.concat([train, final], ignore_index=True)
    raw.loc[raw.pt_d >= 20191117, 'label'] = -1
    assert raw.loc[raw.pt_d == 20191117, 'label'].eq(-1).all()
    rows = np.flatnonzero(raw.pt_d.to_numpy() >= 20191116)
    stats = np.lib.format.open_memmap(directory/'past_rates.npy', mode='w+', dtype='float32',
                                      shape=(len(raw), len(feature_names())), fortran_order=True)
    stats, names = build_multiscale_statistics(raw, 20191117, output=stats,
                        progress=lambda i,n,f,s:emit('past_rate_group', group=i, total=n))
    assert CAT+names == contract['feature_names'][:len(CAT)+len(names)]
    x = np.lib.format.open_memmap(directory/'features.npy', mode='w+', dtype='float32',
                                 shape=(len(rows), len(contract['feature_names'])))
    for start in range(0,len(rows),20000):
        current = rows[start:start+20000]
        categorical = raw.iloc[current][CAT].to_numpy()
        np.testing.assert_array_equal(categorical.astype('float32').astype('int64'),categorical)
        x[start:start+len(current),:len(CAT)] = categorical
        x[start:start+len(current),len(CAT):len(CAT)+len(names)] = stats[current]
    del stats
    gc.collect()
    (directory/'past_rates.npy').unlink()
    dates = raw.pt_d.to_numpy()[rows]
    for day in [20191116,20191117]:
        selected = dates == day
        x[selected,len(CAT)+len(names):-12] = daily(day, contract)
        emit('daily_context_ready', day=day, rows=int(selected.sum()))
    global_to_local = np.full(len(raw), -1, dtype='int32')
    global_to_local[rows] = np.arange(len(rows))
    offset = 0
    for filename in ['train.csv','final_test.csv']:
        for chunk in pd.read_csv(ROOT/'2026ml'/filename, usecols=HISTORY+list(set(TARGET)),
                                 dtype={name:str for name in HISTORY}, chunksize=100000):
            local = global_to_local[offset:offset+len(chunk)]
            keep = local >= 0
            if np.any(keep):
                x[local[keep],-12:] = histories(chunk.loc[keep])
            offset += len(chunk)
    assert offset == len(raw)
    x.flush()
    for start in range(0,len(x),20000):
        assert np.isfinite(x[start:start+20000]).all()
    # Feature reconstruction must reproduce the already verified day16 input.
    original = np.load(ROWDIR/'independent_input/features.npy', mmap_mode='r')
    n16 = int(np.sum(dates == 20191116))
    assert n16 == 421983 and int(np.sum(dates == 20191117)) == 404182
    assert digest(ROWDIR/'independent_input/features.npy') == contract['features_sha256']
    for start in range(0,n16,20000):
        stop = min(n16,start+20000)
        np.testing.assert_array_equal(x[start:stop], original[-n16+start:-n16+stop] if stop<n16 else original[-n16+start:])
    np.save(directory/'dates.npy',dates)
    write(directory/'contract.json',{'shape':list(x.shape),'features_sha256':digest(directory/'features.npy'),
                                   'day16_exact_feature_replay':True,'day17_ad_outcomes_masked':True})
    emit('row_features_ready',shape=list(x.shape), day16_exact_replay=True)
    return x,dates,raw


def previous_stats(raw, day):
    train = raw.loc[raw.pt_d == day-1]
    assert len(train) and train.pt_d.max() < day
    expected = np.load(REFERENCE/f'day{(day-1)%100}_prediction.npy')
    np.testing.assert_array_equal(train.index,np.load(REFERENCE/f'day{(day-1)%100}_indices.npy'))
    expected = expected*train.label.sum()/expected.sum()
    codes, keys = pd.factorize(pd.MultiIndex.from_frame(train[['user_id','spread_app_id']]),sort=True)
    return {'keys':np.asarray(list(keys),dtype='int64'),
            'count':np.bincount(codes,weights=train.label,minlength=len(keys)),
            'mass':np.bincount(codes,weights=expected,minlength=len(keys))}


def prediction(x, dates, raw, protocol):
    record = protocol['selected_catboost']
    frame = pd.DataFrame(np.asarray(x),columns=[f'f{i}' for i in range(x.shape[1])])
    for i in range(len(CAT)):
        frame[f'f{i}'] = frame.iloc[:,i].to_numpy(dtype='int64')
    model = load_model(CATDIR/(record['name']+'.cbm'))
    result = {}
    for day in [20191116,20191117]:
        selected = dates == day
        pool = Pool(frame.loc[selected],cat_features=[f'f{i}' for i in range(len(CAT))])
        p = cat_predict(model,pool,record['positive_weight'])
        if day == 20191116:
            np.testing.assert_allclose(p,np.load(CATDIR/(record['name']+'_prediction.npy')),rtol=0,atol=2e-7)
        base = np.load(OLD_RUN/f'day{day%100}_compose_verified_v1.npy')
        combined = next(values for name,values in recipes(p,base) if name==protocol['selected_blend'])
        stats = previous_stats(raw,day)
        if day == 20191116:
            expected_stats = np.load(APPDIR/'statistics_day16.npz')
            for name in stats:
                np.testing.assert_array_equal(stats[name],expected_stats[name])
        query = raw.loc[raw.pt_d==day]
        final = expit(logit(combined.clip(1e-6,1-1e-6)) + protocol['previous_app_power']*correction(query,stats))
        result[day] = final
        emit('prediction_ready',day=day,rows=len(final))
    np.testing.assert_allclose(result[20191116],np.load(APPDIR/'day16_prediction.npy'),rtol=0,atol=2e-7)
    y16 = raw.loc[raw.pt_d==20191116,'label'].to_numpy()
    score = float(roc_auc_score(y16,result[20191116]))
    assert abs(score-protocol['development16_auc']) < 1e-12,(score,protocol['development16_auc'])
    return result


def predict():
    protocol = freeze()
    assert not (OUTPUT/'prediction_manifest.json').exists(), 'Preserving exported prediction'
    x,dates,raw = build_features(RUN/'input')
    results = prediction(x,dates,raw,protocol)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    final = results[20191117]
    original = pd.read_csv(ROOT/'2026ml/final_test.csv',usecols=['log_id','pt_d'])
    assert len(final)==len(original)==404182 and original.pt_d.eq(20191117).all()
    for day,p in results.items():
        np.save(OUTPUT/f'day{day%100}_prediction.npy',p)
    pd.DataFrame({'id':np.arange(len(final)),'pctr':final}).to_csv(OUTPUT/'submission_best_20191117.csv',index=False,float_format='%.10f')
    pd.DataFrame({'log_id':original.log_id,'pctr':final}).to_csv(OUTPUT/'submission_log_id_20191117.csv',index=False,float_format='%.10f')
    write(OUTPUT/'prediction_manifest.json',{'status':'prediction_frozen_before_label_evaluation',
          'rows':len(final),'development16_auc':protocol['development16_auc'],
          'inferred_labels_used_for_prediction_or_selection':False,
          'protocol_sha256':digest(RUN/'protocol.json'),
          'files_sha256':{name:digest(OUTPUT/name) for name in ['day17_prediction.npy','day16_prediction.npy','submission_best_20191117.csv','submission_log_id_20191117.csv']}})
    emit('submission_frozen',path=str(OUTPUT/'submission_best_20191117.csv'))


def rank_auc(y,p):
    positive=int(y.sum())
    return float((rankdata(p,method='average')[y==1].sum()-positive*(positive+1)/2)/(positive*(len(y)-positive)))


def verify():
    protocol = check_sources()
    manifest = read(OUTPUT/'prediction_manifest.json')
    assert digest(RUN/'protocol.json')==manifest['protocol_sha256']
    for name,h in manifest['files_sha256'].items():
        assert digest(OUTPUT/name)==h,name
    x,dates,raw = build_features(RUN/'independent_input')
    replay = prediction(x,dates,raw,protocol)[20191117]
    saved = np.load(OUTPUT/'day17_prediction.npy')
    replay_error = float(np.max(np.abs(saved-replay)))
    assert replay_error<2e-7,replay_error
    original = pd.read_csv(ROOT/'2026ml/final_test.csv',usecols=['log_id','pt_d'])
    submission = pd.read_csv(OUTPUT/'submission_best_20191117.csv')
    assignment = pd.read_csv(OUTPUT/'submission_log_id_20191117.csv')
    assert list(submission)==['id','pctr'] and list(assignment)==['log_id','pctr']
    assert len(submission)==len(assignment)==len(original)==404182
    np.testing.assert_array_equal(submission.id,np.arange(len(original)))
    np.testing.assert_array_equal(assignment.log_id,original.log_id)
    np.testing.assert_array_equal(assignment.pctr,submission.pctr)
    assert np.isfinite(submission.pctr).all() and submission.pctr.between(0,1).all()
    rounding=float(np.max(np.abs(submission.pctr.to_numpy()-saved)))
    assert rounding<5.01e-11
    check_sources()
    # Open inferred labels only after prediction, reconstruction and export checks.
    folder=ROOT/'analysis/eleven_score_strategy_20261102/imported_refinement'
    label_file=folder/'day17_candidate_labels_11scores.csv'
    label_manifest=read(folder/'output_manifest.json')
    expected=next(r['sha256'] for r in label_manifest['files'] if r['path']==label_file.name)
    assert digest(label_file)==expected
    assert read(folder/'independent_verification_log.json')['status']=='passed'
    labels=pd.read_csv(label_file)
    np.testing.assert_array_equal(labels.id,submission.id)
    np.testing.assert_array_equal(labels.log_id,original.log_id)
    assert labels.label.isin([0,1]).all()
    auc=float(roc_auc_score(labels.label,submission.pctr))
    assert abs(auc-rank_auc(labels.label.to_numpy(),submission.pctr.to_numpy()))<1e-12
    report={'status':'passed','date':20191117,'rows':len(submission),
            'development16_auc':protocol['development16_auc'],'inferred_label_auc':auc,
            'full_precision_inferred_auc':float(roc_auc_score(labels.label,saved)),
            'inferred_positive_count':int(labels.label.sum()),
            'inferred_label_path':str(label_file),'inferred_label_sha256':digest(label_file),
            'labels_are_inferred_not_observed':True,'platform_auc':None,
            'inferred_labels_used_for_training_or_selection':False,
            'day16_exact_feature_replay':True,'independent_day17_feature_rebuild':True,
            'maximum_prediction_replay_error':replay_error,'maximum_csv_rounding_error':rounding,
            'original_row_order_verified':True,'repeated_log_ids_preserved':int(original.log_id.duplicated().sum()),
            'original_datasets_unchanged':True,'submission_sha256':digest(OUTPUT/'submission_best_20191117.csv'),
            'protocol_sha256':digest(RUN/'protocol.json'),'verification_source_sha256':digest(__file__)}
    write(OUTPUT/'verification.json',report)
    description=(f'2019年11月17日提交文件：submission_best_20191117.csv\n'
        f'格式 id,pctr；{len(submission):,} 行，id为final_test原始顺序的零起始行号。\n'
        f'作业兼容文件：submission_log_id_20191117.csv，保留858条重复log_id。\n\n'
        f'当前最佳16号组合的开发集AUC：{protocol["development16_auc"]:.10f}。16号被反复用于模型选择。\n'
        f'17号使用已核验的全量v8基础预测、冻结的模型权重和最新已验证组合；历史特征及前日残差使用截至16号的真实广告标签。\n'
        f'未使用17号反推标签训练或选择模型。最新11次反馈反推标签上的AUC：{auc:.10f}。\n'
        f'这是反推标签下的评估值，平台真实AUC需上传后确认。\n\n'
        f'已在独立进程重建特征并复算，最大预测误差{replay_error:g}。原始数据哈希一致。\n')
    (OUTPUT/'提交说明.md').write_text(description,encoding='utf8')
    emit('verified',**report)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['predict','verify'],required=True)
    args=parser.parse_args()
    predict() if args.phase=='predict' else verify()
