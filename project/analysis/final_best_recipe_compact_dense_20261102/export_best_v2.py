"""Export the verified real16 winner, lock forecasts, then evaluate inferred17 labels."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
import argparse
from datetime import datetime, timezone
import gc
import importlib.util
import inspect
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EXP=ROOT/'analysis/day16_auc082_20261102'
PREVIOUS=ROOT/'analysis/final_best_recipe_daily_quality_transfer_20261102'
sys.path.insert(0,str(EXP))
spec=importlib.util.spec_from_file_location('frozen_daily_quality_export',PREVIOUS/'export_best.py')
previous=importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
old=previous.old
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit,logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from catboost import Pool
from catboost_userday_residual import CATEGORIES,load_model
from rich_userday_activity import array_hash,correct,lgb
from fit_stacker import digest,write
from conditional_daily import read,emit
from dense_context_rows import dense_codes
from compose_quality_daily_transfer import predict as compose
from group_temperature import GROUPS,group_codes,FIELDS
from within_user_ranking_core import within_prediction
from within_user_ranker import centered_unit

RUN=HERE/'run_v2'
OUTPUT=ROOT/'output/best_recipe_compact_dense_20191117_20261102'
PRIOR_EXPORT=previous.PRIOR_EXPORT
DAILY=previous.DAILY
DENSE=EXP/'runs/dense_context_rows_v1'
COMPACT=EXP/'runs/compact_quality_daily_catboost_v1'
DENSE_COMPOSE=EXP/'runs/compose_dense_context_transfer_v1'
BASE=EXP/'runs/compose_compact_dense_transfer_v1'
RANKER=EXP/'runs/within_user_ranker_v1'
NEWS_OUTPUT=ROOT/'output/best_recipe_news_quality_20191117_20261102'
LABEL_FILE=previous.LABEL_FILE
LABEL_SHA=previous.LABEL_SHA
DENSE_NAME='day16_context_numeric_trees1400'
COMPACT_NAME='day16_poisson_d7_top192_trees3200'


def query_frame(day, verifying=False):
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
    quality_phase = 'independent_day17' if verifying and day == 20191117 else f'input_day{day%100}'
    quality_file = PRIOR_EXPORT/'run_v2'/quality_phase/'daily_quality.npz'
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
    contract = dict(date=day, query_shape=list(frame.shape), frame_sha256=array_hash(frame.to_numpy(dtype='float32')),
        keys_sha256=array_hash(qkeys), row_groups_sha256=array_hash(row_groups),
        bayesian_audit=audit, cached_quality_sha256=digest(quality_file),
        advertisement_query_targets_masked=True, frozen_news_vocabulary_cutoff=20191116,
        feature_count=len(frame.columns))
    emit('updated_day17_complete_query_frame', day=day, verifying=verifying, shape=contract['query_shape'])
    return qkeys, row_groups, frame, mass[query], contract



def check_sources(raw=True):
    protocol=read(RUN/'protocol.json')
    for filename,expected in protocol['sources_sha256'].items():
        assert digest(filename)==expected,filename
    if raw:
        for filename,expected in protocol['raw_datasets_sha256'].items():
            assert digest(ROOT/'2026ml'/filename)==expected,filename
    return protocol


def freeze():
    assert not RUN.exists() and not OUTPUT.exists(),'Preserve earlier exports and failed runs'
    selected={}
    paths=[Path(__file__),HERE/'test_export_v2.py',HERE/'run_export_v2.ps1',HERE/'recovery_v2.json',HERE/'export_best.py',HERE/'run_v1/protocol.json']
    bindings={}
    hashes=read(BASE/'protocol.json')['raw_datasets_sha256']
    folders=dict(dense=DENSE,compact=COMPACT,dense_transfer=DENSE_COMPOSE,compact_transfer=BASE)
    rank=None
    if (RANKER/'verification.json').is_file():
        report=read(RANKER/'verification.json')
        assert report['status']=='passed'
        if report['dates']['20191116']['auc']>read(BASE/'verification.json')['dates']['20191116']['auc']:
            rank=report['dates']['20191116']['best']
            folders['ranking']=RANKER
    for name,folder in folders.items():
        report=read(folder/'verification.json')
        assert report['status']=='passed'
        assert not report['dates']['20191116']['evaluation_day_labels_fit']
        selected[name]=report['dates']['20191116']['best']
        paths += [folder/'protocol.json',folder/'verification.json',folder/'day16_prediction.npy']
    selected['ranking']=rank
    assert selected['dense']['name']==DENSE_NAME and selected['dense']['weight']==.1
    assert selected['compact']['name']==COMPACT_NAME and selected['compact']['power']==.125
    assert selected['dense_transfer']['name']=='user_between0_within1.5'
    assert selected['compact_transfer']['name']=='task_between1.5_within1'
    auc=(rank or selected['compact_transfer'])['auc']
    audit=read(ROOT/'output/two_day_auc_targets_20261102/independent_target_audit.json')
    assert audit['audit_status']=='passed' and abs(audit['dates']['20191116']['auc']-auc)<1e-12
    protocol_paths=[folder/'protocol.json' for folder in folders.values()]
    protocol_paths += [PREVIOUS/'run_v1/protocol.json',PRIOR_EXPORT/'run_v2/protocol.json']
    for path in protocol_paths:
        parent=read(path)
        assert parent['raw_datasets_sha256']==hashes
        paths.append(path)
        for filename,expected in parent['sources_sha256'].items():
            assert filename not in bindings or bindings[filename]==expected
            bindings[filename]=expected
    for folder in [previous.OUTPUT,NEWS_OUTPUT]:
        assert read(folder/'verification.json')['status']=='passed'
        paths += [folder/'verification.json']+[folder/f'day{day}_prediction.npy' for day in [16,17]]
    paths += [DENSE/(DENSE_NAME+'.txt'),COMPACT/(COMPACT_NAME+'.cbm'),
        DENSE/'models_locked.json',COMPACT/'models_locked.json',COMPACT/'input_contract16.json',
        DENSE/'contract_day16_context_numeric.json',
        DENSE/'input/day16_context_numeric/category_maps.npz',
        COMPACT/(COMPACT_NAME+'_held_group_prediction.npy'),
        PREVIOUS/'run_v1/input_contracts.json',EXP/'dense_context_rows.py',
        EXP/'compose_quality_daily_transfer.py',EXP/'within_user_ranking_core.py',EXP/'within_user_ranker.py']
    for phase in ['input_day16','input_day17','independent_day17']:
        paths += [PRIOR_EXPORT/'run_v2'/phase/name for name in ['features.npy','daily_quality.npz','contract.json']]
    if rank is not None:
        paths += [RANKER/'models_locked.json']
        rank_names=([f'day16_{obj}_trees1600' for obj in ['lambdarank','rank_xendcg']]
                    if rank['name']=='day16_mean_final_within_unit' else [rank['name']])
        paths += [RANKER/(name+'.txt') for name in rank_names]
    for path in dict.fromkeys(paths):
        filename=str(path)
        value=digest(path)
        assert filename not in bindings or bindings[filename]==value,filename
        bindings[filename]=value
    for filename,expected in bindings.items():
        assert digest(filename)==expected,filename
    for filename,expected in hashes.items():
        assert digest(ROOT/'2026ml'/filename)==expected,filename
    RUN.mkdir(parents=True)
    write(RUN/'protocol.json',dict(status='frozen_before_new_day17_prediction_and_inferred_label_evaluation',
        frozen_utc=datetime.now(timezone.utc).isoformat(),selected=selected,development16_auc=auc,
        selection_basis='Highest independently verified published real16 recipe at freeze time',
        new_model_training=False,correction_training_max_date=20191115,
        inherited_reference_training_max_date=20191116,ad_history_max_date=20191116,
        development_dates_repeatedly_reused=True,same_day_supplied_news_feedback_used=True,
        inferred17_labels_used_for_training_tuning_feature_or_model_selection=False,
        inferred17_labels_previously_evaluated_on_older_frozen_exports=True,
        raw_datasets_sha256=hashes,sources_sha256=bindings))
    emit('updated_day17_recipe_frozen',development16_auc=auc,ranking=rank)


def query_raw(day):
    frame=pd.read_csv(ROOT/'2026ml'/('train.csv' if day==20191116 else 'final_test.csv'),
        usecols=FIELDS+['pt_d','log_id'],dtype='int32')
    return frame.loc[frame.pt_d.eq(day)].reset_index(drop=True)


def matrix_path(day,verifying):
    phase='independent_day17' if verifying and day==20191117 else f'input_day{day%100}'
    directory=PRIOR_EXPORT/'run_v2'/phase
    contract=read(directory/'contract.json')
    assert digest(directory/'features.npy')==contract['matrix_sha256']
    assert contract['own_query_ad_labels_masked'] and contract['max_ad_history_date']<day
    return directory/'features.npy',contract


def dense_margins(day,verifying=False,ranking_names=()):
    path,contract=matrix_path(day,verifying)
    source=np.load(path,mmap_mode='r')
    native=read(DENSE/'contract_day16_context_numeric.json')
    assert native['max_training_date']<20191116
    vocab=np.load(DENSE/'input/day16_context_numeric/category_maps.npz')
    assert {key:array_hash(vocab[key]) for key in vocab.files}==native['vocabularies_sha256']
    models={DENSE_NAME:lgb.Booster(model_str=(DENSE/(DENSE_NAME+'.txt')).read_text(encoding='utf8'))}
    for name in ranking_names:
        models[name]=lgb.Booster(model_str=(RANKER/(name+'.txt')).read_text(encoding='utf8'))
    predictions={name:[] for name in models}
    batch=7000 if verifying else 14000
    for first in range(0,len(source),batch):
        block=np.array(source[first:first+batch,native['columns']],dtype='float32',copy=True)
        for key in vocab.files:
            column=int(key[1:])
            if verifying:
                # An independent exact-match dictionary, including explicit unseen-value handling.
                block[:,column]=pd.Index(vocab[key]).get_indexer(block[:,column].astype('int64'))
            else:
                block[:,column]=dense_codes(block[:,column],vocab[key])
        assert block.shape[1]==694 and np.isfinite(block).all()
        for name,model in models.items():
            predictions[name].append(model.predict(block,raw_score=True,num_threads=2))
    result={name:np.concatenate(values) for name,values in predictions.items()}
    emit('updated_day17_row_forecasts_ready',day=day,models=list(result),verifying=verifying)
    return result,contract


def transfer(raw,current,daily,anchor,record,independent=False):
    raw=raw[FIELDS]
    values=dict(current=current,daily=daily,anchor=anchor)
    data=dict(frame=raw,values=values,logits={name:logit(value) for name,value in values.items()},
        groups={record['group']:group_codes(raw,GROUPS[record['group']])})
    return compose(data,record,independent=independent)


def normalized(base,margin,weight):
    a=logit(base.clip(1e-6,1-1e-6))
    z=margin-np.log(5.)
    z=(z-z.mean())*a.std()/max(z.std(),1e-6)+a.mean()
    return expit((1-weight)*a+weight*z)


def forecast(day,verifying=False):
    protocol=read(RUN/'protocol.json')
    selected=protocol['selected']
    ranking=selected['ranking']
    rank_names=[]
    if ranking is not None:
        rank_names=([f'day16_{obj}_trees1600' for obj in ['lambdarank','rank_xendcg']]
            if ranking['name']=='day16_mean_final_within_unit' else [ranking['name']])
    margins,row_contract=dense_margins(day,verifying,rank_names)
    raw=query_raw(day)
    anchor_path=(old.SOURCE/'day16_prediction.npy' if day==20191116 else NEWS_OUTPUT/'day17_prediction.npy')
    anchor=np.load(anchor_path)
    current=np.load(previous.OUTPUT/f'day{day%100}_prediction.npy')
    assert len(raw)==len(anchor)==len(current)
    dense=normalized(anchor,margins[DENSE_NAME],selected['dense']['weight'])
    dense_transfer=transfer(raw,current,dense,anchor,selected['dense_transfer'],verifying)
    keys,groups,frame,mass,contract=query_frame(day,verifying)
    np.testing.assert_array_equal(groups,pd.Index(keys[:,1]).get_indexer(raw.user_id))
    model_record=next(item for item in read(COMPACT/'models_locked.json')['models'] if item['name']==COMPACT_NAME)
    columns=read(COMPACT/'input_contract16.json')['selected_columns']['poisson_d7_top192']
    assert columns==model_record['columns']
    model=load_model(COMPACT/(COMPACT_NAME+'.cbm'))
    assert model.feature_names_==columns
    margin=model.predict(Pool(frame[columns],cat_features=CATEGORIES+['user_id']),
        prediction_type='RawFormulaVal',thread_count=2)
    response=np.exp((margin+np.log(mass)).clip(-15,8))
    compact=correct(current,response,groups,'poisson',selected['compact']['power'])
    if verifying:
        table=pd.DataFrame(dict(group=groups,p=current.clip(1e-8,1-1e-8)))
        total=table.groupby('group',sort=False).p.transform('sum').to_numpy()
        ratio=np.clip(np.maximum(response[groups],1e-6)/np.maximum(total,1e-8),.05,20.)**selected['compact']['power']
        odds=table.p.to_numpy()/(1-table.p.to_numpy())*ratio
        independent=odds/(1+odds)
        np.testing.assert_allclose(compact,independent,rtol=0,atol=2e-14)
        compact=independent
    result=transfer(raw,dense_transfer,compact,current,selected['compact_transfer'],verifying)
    if day==20191116:
        for value,folder in [(dense,DENSE),(dense_transfer,DENSE_COMPOSE),(compact,COMPACT),(result,BASE)]:
            np.testing.assert_allclose(value,np.load(folder/'day16_prediction.npy'),rtol=0,atol=2e-14)
        np.testing.assert_allclose(response,np.load(COMPACT/(COMPACT_NAME+'_held_group_prediction.npy')),rtol=0,atol=2e-12)
    if ranking is not None:
        group=group_codes(raw[FIELDS],['user_id'])
        rank_margin=(np.mean([centered_unit(margins[name],group,verifying) for name in rank_names],axis=0)
                     if len(rank_names)>1 else margins[rank_names[0]])
        result=within_prediction(result,rank_margin,group,ranking['weight'],ranking['mode'],verifying)
        if day==20191116:
            np.testing.assert_allclose(result,np.load(RANKER/'day16_prediction.npy'),rtol=0,atol=2e-14)
    assert result.shape==(len(raw),) and np.isfinite(result).all() and np.all((result>0)&(result<1))
    contract.update(row_matrix_sha256=row_contract['matrix_sha256'],
        count_prediction_sha256=array_hash(response),row_margin_sha256={name:array_hash(value) for name,value in margins.items()},
        original_log_ids_sha256=array_hash(raw.log_id.to_numpy()))
    del frame,model,raw,margins
    gc.collect()
    emit('updated_day17_forecast_complete',day=day,verifying=verifying)
    return result,contract


def predict():
    protocol=check_sources()
    assert not OUTPUT.exists()
    contracts={}
    for day in [20191116,20191117]:
        p,contract=forecast(day)
        np.save(RUN/f'day{day%100}_prediction.npy',p)
        contracts[str(day)]=contract
    write(RUN/'input_contracts.json',contracts)
    raw=query_raw(20191117)
    p=np.load(RUN/'day17_prediction.npy')
    assert len(p)==404182
    OUTPUT.mkdir()
    pd.DataFrame(dict(id=np.arange(len(p)),pctr=p)).to_csv(OUTPUT/'submission_best_20191117.csv',index=False,float_format='%.17g')
    pd.DataFrame(dict(log_id=raw.log_id,pctr=p)).to_csv(OUTPUT/'submission_log_id_20191117.csv',index=False,float_format='%.17g')
    for day in [16,17]:
        np.save(OUTPUT/f'day{day}_prediction.npy',np.load(RUN/f'day{day}_prediction.npy'))
    write(OUTPUT/'prediction_manifest.json',dict(status='locked_before_inferred17_label_evaluation',
        protocol_sha256=digest(RUN/'protocol.json'),input_contracts_sha256=digest(RUN/'input_contracts.json'),
        development16_auc=protocol['development16_auc'],inferred17_labels_used_for_prediction_or_selection=False,
        files_sha256={p.name:digest(p) for p in OUTPUT.iterdir() if p.is_file()}))
    emit('updated_day17_submission_locked',path=str(OUTPUT/'submission_best_20191117.csv'))


def verify():
    protocol=check_sources()
    assert not (OUTPUT/'verification.json').exists()
    manifest=read(OUTPUT/'prediction_manifest.json')
    assert manifest['protocol_sha256']==digest(RUN/'protocol.json')
    assert manifest['input_contracts_sha256']==digest(RUN/'input_contracts.json')
    for filename,expected in manifest['files_sha256'].items():
        assert digest(OUTPUT/filename)==expected,filename
    replay,contract=forecast(20191117,True)
    expected=read(RUN/'input_contracts.json')['20191117']
    assert contract==expected
    error=float(np.max(np.abs(replay-np.load(OUTPUT/'day17_prediction.npy'))))
    assert error<2e-14
    raw=query_raw(20191117)
    submission=pd.read_csv(OUTPUT/'submission_best_20191117.csv',float_precision='round_trip')
    alternate=pd.read_csv(OUTPUT/'submission_log_id_20191117.csv',float_precision='round_trip')
    assert list(submission)==['id','pctr'] and list(alternate)==['log_id','pctr'] and len(submission)==len(raw)==404182
    np.testing.assert_array_equal(submission.id,np.arange(len(raw)))
    np.testing.assert_array_equal(alternate.log_id,raw.log_id)
    np.testing.assert_array_equal(submission.pctr,alternate.pctr)
    np.testing.assert_array_equal(submission.pctr,np.load(OUTPUT/'day17_prediction.npy'))
    actual=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    auc16=float(roc_auc_score(actual.loc[actual.pt_d.eq(20191116),'label'],np.load(OUTPUT/'day16_prediction.npy')))
    assert abs(auc16-protocol['development16_auc'])<1e-12
    # Evaluation-only label access starts after frozen selection, exported predictions, and independent replay.
    assert digest(LABEL_FILE)==LABEL_SHA
    inferred=pd.read_csv(LABEL_FILE)
    np.testing.assert_array_equal(inferred.id,submission.id)
    np.testing.assert_array_equal(inferred.log_id,raw.log_id)
    assert inferred.label.isin([0,1]).all()
    auc=float(roc_auc_score(inferred.label,submission.pctr))
    positives=int(inferred.label.sum())
    independent_auc=float((rankdata(submission.pctr)[inferred.label.eq(1)].sum()-positives*(positives+1)/2)/
        (positives*(len(inferred)-positives)))
    assert abs(auc-independent_auc)<1e-12
    check_sources()
    report=dict(status='passed',date=20191117,rows=len(raw),development16_auc=auc16,
        inferred_label_auc=auc,independent_rank_auc=independent_auc,inferred_positive_count=positives,
        labels_are_inferred_not_observed=True,platform_auc=None,
        inferred_labels_used_for_training_tuning_feature_or_model_selection=False,new_model_training=False,
        correction_training_max_date=20191115,inherited_reference_training_max_date=20191116,
        ad_history_max_date=20191116,development_dates_repeatedly_reused=True,
        day16_winning_forecasts_replayed=True,day17_daily_features_rebuilt_twice=True,
        inherited_row_features_independently_rebuilt_in_previous_verified_export=True,
        day17_dense_dictionary_and_group_corrections_independently_replayed=True,
        maximum_prediction_replay_error=error,csv_roundtrip_exact=True,original_row_order_verified=True,
        repeated_log_ids_preserved=int(raw.log_id.duplicated().sum()),original_datasets_unchanged=True,
        raw_datasets_sha256=protocol['raw_datasets_sha256'],inferred_label_sha256=LABEL_SHA,
        protocol_sha256=digest(RUN/'protocol.json'),submission_sha256=digest(OUTPUT/'submission_best_20191117.csv'))
    write(OUTPUT/'verification.json',report)
    (OUTPUT/'提交说明.md').write_text(
        f'2019年11月17日提交文件：submission_best_20191117.csv\n格式 id,pctr；{len(raw):,}行，id为final_test原始行序的零起始行号。\n'
        f'冻结时最佳真实16号开发集AUC={auc16:.12f}；15/16号曾反复用于开发。\n'
        f'17号反推标签AUC={auc:.12f}；该标签不是官方真实标签，平台真实AUC未知。\n'
        '反推标签未用于训练、调参、特征选择或模型选择。\n已验证原始四份CSV未改变、行序、CSV精度与独立预测复算。\n',encoding='utf8')
    emit('updated_day17_verified_and_evaluated',**report)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=['freeze','predict','verify'])
    {'freeze':freeze,'predict':predict,'verify':verify}[parser.parse_args().phase]()
