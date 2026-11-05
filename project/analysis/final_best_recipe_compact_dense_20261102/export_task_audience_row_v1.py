"""Export the verified real16 CatBoost rank correction, then evaluate inferred17."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import Pool
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import export_best_v2 as base
from catboost_userday_residual import load_model
from conditional_daily import read,emit
from fit_stacker import digest,write
from full_quality_row_catboost import full_frame,quality_rows
from rich_row_catboost import frame_hash
from rich_userday_activity import array_hash
from task_audience_row_ranker_v1 import prediction as row_prediction
from task_audience_row_core import aligned as align_audience
import export_task_audience_v1 as audience_export

HERE=Path(__file__).resolve().parent
RUN=HERE/'task_audience_row_run_v1'
OUTPUT=base.ROOT/'output/best_recipe_task_audience_row_20191117_20261103'
PARENT=base.ROOT/'output/best_recipe_task_audience_20191117_20261103'
PARENT_RUN=HERE/'task_audience_run_v1'
RANKER=base.EXP/'runs/task_audience_row_ranker_v1'
SOURCE=base.EXP/'runs/rich_augmented_rows_v1'


def check_sources():
    protocol=read(RUN/'protocol.json')
    for filename,expected in protocol['sources_sha256'].items():
        assert digest(filename)==expected,filename
    for filename,expected in protocol['raw_datasets_sha256'].items():
        assert digest(base.ROOT/'2026ml'/filename)==expected,filename
    return protocol


def freeze():
    assert not RUN.exists() and not OUTPUT.exists(),'Preserve all previous exports'
    report=read(RANKER/'verification.json')
    current=read(PARENT/'verification.json')
    assert report['status']==current['status']=='passed'
    selected=report['dates']['20191116']['best']
    if selected['auc']<=current['development16_auc']:
        emit('task_audience_row17_export_unnecessary',current_auc=current['development16_auc'],candidate_auc=selected['auc'])
        return False
    assert selected['name']!='base' and selected['mode'] in ['add_global','replace_global','add_within']
    assert not report['dates']['20191116']['evaluation_day_labels_fit']
    audit=read(base.ROOT/'output/two_day_auc_targets_20261102/independent_target_audit.json')
    assert audit['audit_status']=='passed'
    assert abs(audit['dates']['20191116']['auc']-selected['auc'])<1e-12
    parent=read(PARENT_RUN/'protocol.json')
    rank_protocol=read(RANKER/'protocol.json')
    assert parent['raw_datasets_sha256']==rank_protocol['raw_datasets_sha256']
    bindings=dict(parent['sources_sha256'])
    for filename,expected in rank_protocol['sources_sha256'].items():
        assert filename not in bindings or bindings[filename]==expected
        bindings[filename]=expected
    paths=[Path(__file__),HERE/'prepare_task_audience_row_export_v1.py',HERE/'export_cross_user_pair_v1.py',HERE/'export_task_audience_v1.py',HERE/'prepare_cross_user_pair_export_v1.py',HERE/'export_catboost_ranked_v1.py',PARENT_RUN/'protocol.json',PARENT/'verification.json',
        PARENT/'prediction_manifest.json',RANKER/'protocol.json',RANKER/'verification.json',
        RANKER/'models_locked.json',RANKER/'predictions_locked.json',
        RANKER/(selected['name']+'.cbm'),RANKER/'input_contract16.json',RANKER/'held_contract16.json',
        RANKER/(selected['name']+'_held_margin.npy'),RANKER/'day16_prediction.npy',
        SOURCE/'input_contract16.json',base.EXP/'full_quality_row_catboost.py',base.EXP/'rich_row_catboost.py']
    paths += [PARENT/f'day{day}_prediction.npy' for day in [16,17]]
    for phase in ['input_day16','input_day17','independent_day17']:
        paths += [base.PRIOR_EXPORT/'run_v2'/phase/name for name in ['features.npy','daily_quality.npz','contract.json']]
    for path in paths:
        filename=str(path)
        expected=digest(path)
        assert filename not in bindings or bindings[filename]==expected,filename
        bindings[filename]=expected
    for filename,expected in bindings.items():
        assert digest(filename)==expected,filename
    for filename,expected in parent['raw_datasets_sha256'].items():
        assert digest(base.ROOT/'2026ml'/filename)==expected,filename
    RUN.mkdir()
    write(RUN/'protocol.json',dict(status='frozen_before_task_audience_row17_forecast_and_inferred_label_evaluation',
        timestamp_utc=datetime.now(timezone.utc).isoformat(),selected=selected,
        development16_auc=selected['auc'],selection_basis='Highest independently verified published real16 model',
        development_dates_repeatedly_reused=True,new_model_training=False,
        correction_training_max_date=20191115,inherited_reference_training_max_date=20191116,
        inferred17_labels_used_for_training_tuning_feature_or_model_selection=False,
        inferred17_labels_previously_evaluated_on_older_frozen_exports=True,
        raw_datasets_sha256=parent['raw_datasets_sha256'],sources_sha256=bindings))
    emit('task_audience_row17_recipe_frozen',selected=selected)
    return True


def audience_for_raw(raw, day, independent):
    auxiliary, meta, audit = audience_export.inputs(day, independent)
    names = read(RANKER/'input_contract16.json')['audience_columns']
    values = align_audience(meta['keys'], auxiliary[names].to_numpy(dtype='float32'),
                            raw[['pt_d','task_id']].to_numpy(), day)
    contract = dict(audience_frame_sha256=audit['frame_sha256'],
                    selected_audience_columns=names, extra_sha256=array_hash(values))
    del auxiliary, meta
    gc.collect()
    return values, contract


def forecast(day,independent=False):
    protocol=read(RUN/'protocol.json')
    selected=protocol['selected']
    raw=base.query_raw(day)
    extra, audience_contract=audience_for_raw(raw,day,independent)
    path,row_contract=base.matrix_path(day,independent)
    matrix=np.load(path,mmap_mode='r')
    assert matrix.shape==(len(raw),744)
    native=read(SOURCE/'input_contract16.json')
    phase='independent_day17' if independent and day==20191117 else f'input_day{day%100}'
    quality=np.load(base.PRIOR_EXPORT/'run_v2'/phase/'daily_quality.npz')
    query=raw[['pt_d','user_id']].to_numpy()
    aligned=quality_rows(quality['keys'],quality['values'],query)
    frame,cats,keep=full_frame(matrix,np.arange(len(raw)),native['feature_names'],native['categorical_columns'],aligned)
    names=audience_contract['selected_audience_columns']
    frame=pd.concat([frame.reset_index(drop=True),pd.DataFrame(extra,columns=['task_audience__'+name for name in names])],axis=1)
    del extra
    gc.collect()
    assert frame.shape==(len(raw),1209)
    train_contract=read(RANKER/'input_contract16.json')
    assert cats==train_contract['categories'] and train_contract['max_training_date']==20191115
    frame_sha=frame_hash(frame)
    quality_sha=array_hash(aligned)
    query_sha=array_hash(query)
    if day==20191116:
        held=read(RANKER/'held_contract16.json')
        assert frame_sha==held['frame_sha256']
        assert quality_sha==held['quality_sha256'] and query_sha==held['query_keys_sha256']
    model_record=next(record for record in read(RANKER/'models_locked.json')['models'] if record['name']==selected['name'])
    model_path=RANKER/(selected['name']+'.cbm')
    assert digest(model_path)==model_record['model_sha256']
    model=load_model(model_path)
    assert model.feature_names_==list(frame)
    margin=model.predict(Pool(frame,cat_features=cats),prediction_type='RawFormulaVal',thread_count=2)
    if day==20191116:
        np.testing.assert_array_equal(margin,np.load(RANKER/(selected['name']+'_held_margin.npy')))
    groups=base.group_codes(raw[base.FIELDS],['user_id'])
    current=np.load(PARENT/f'day{day%100}_prediction.npy')
    prediction=row_prediction(current,margin,groups,selected['weight'],selected['mode'],independent)
    if day==20191116:
        np.testing.assert_allclose(prediction,np.load(RANKER/'day16_prediction.npy'),rtol=0,atol=2e-14)
    assert np.isfinite(prediction).all() and np.all((prediction>0)&(prediction<1))
    contract=dict(day=day,rows=len(raw),row_matrix_sha256=row_contract['matrix_sha256'],
        full_frame_sha256=frame_sha,quality_sha256=quality_sha,query_keys_sha256=query_sha,
        original_log_ids_sha256=array_hash(raw.log_id.to_numpy()),group_sha256=array_hash(groups),
        raw_model_margin_sha256=array_hash(margin),audience_contract=audience_contract,advertisement_prediction_date_labels_masked=True)
    del frame,matrix,aligned,quality,model,raw
    gc.collect()
    emit('task_audience_row17_forecast_complete',day=day,independent=independent)
    return prediction,contract


def predict():
    protocol=check_sources()
    assert not OUTPUT.exists()
    contracts={}
    for day in [20191116,20191117]:
        prediction,contract=forecast(day)
        np.save(RUN/f'day{day%100}_prediction.npy',prediction)
        contracts[str(day)]=contract
    write(RUN/'input_contracts.json',contracts)
    raw=base.query_raw(20191117)
    prediction=np.load(RUN/'day17_prediction.npy')
    assert len(raw)==len(prediction)==404182
    OUTPUT.mkdir()
    pd.DataFrame(dict(id=np.arange(len(raw)),pctr=prediction)).to_csv(OUTPUT/'submission_best_20191117.csv',index=False,float_format='%.17g')
    pd.DataFrame(dict(log_id=raw.log_id,pctr=prediction)).to_csv(OUTPUT/'submission_log_id_20191117.csv',index=False,float_format='%.17g')
    for day in [16,17]:
        np.save(OUTPUT/f'day{day}_prediction.npy',np.load(RUN/f'day{day}_prediction.npy'))
    write(OUTPUT/'prediction_manifest.json',dict(status='locked_before_inferred17_label_evaluation',
        protocol_sha256=digest(RUN/'protocol.json'),input_contracts_sha256=digest(RUN/'input_contracts.json'),
        inferred17_labels_used_for_prediction_or_selection=False,development16_auc=protocol['development16_auc'],
        files_sha256={str(path.name):digest(path) for path in OUTPUT.iterdir() if path.is_file()}))
    emit('task_audience_row17_submission_locked',path=str(OUTPUT/'submission_best_20191117.csv'))


def verify():
    protocol=check_sources()
    assert not (OUTPUT/'verification.json').exists()
    manifest=read(OUTPUT/'prediction_manifest.json')
    assert manifest['protocol_sha256']==digest(RUN/'protocol.json')
    assert manifest['input_contracts_sha256']==digest(RUN/'input_contracts.json')
    for filename,expected in manifest['files_sha256'].items():
        assert digest(OUTPUT/filename)==expected,filename
    prediction,contract=forecast(20191117,True)
    assert contract==read(RUN/'input_contracts.json')['20191117']
    error=float(np.max(np.abs(prediction-np.load(OUTPUT/'day17_prediction.npy'))))
    assert error<2e-14
    raw=base.query_raw(20191117)
    submission=pd.read_csv(OUTPUT/'submission_best_20191117.csv',float_precision='round_trip')
    alternate=pd.read_csv(OUTPUT/'submission_log_id_20191117.csv',float_precision='round_trip')
    assert list(submission)==['id','pctr'] and list(alternate)==['log_id','pctr']
    assert len(submission)==len(raw)==404182
    np.testing.assert_array_equal(submission.id,np.arange(len(raw)))
    np.testing.assert_array_equal(alternate.log_id,raw.log_id)
    np.testing.assert_array_equal(alternate.pctr,submission.pctr)
    np.testing.assert_array_equal(submission.pctr,np.load(OUTPUT/'day17_prediction.npy'))
    assert submission.pctr.between(0,1).all() and np.isfinite(submission.pctr).all()
    actual=pd.read_csv(base.ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    auc16=float(roc_auc_score(actual.loc[actual.pt_d.eq(20191116),'label'],np.load(OUTPUT/'day16_prediction.npy')))
    assert abs(auc16-protocol['development16_auc'])<1e-12
    # Open inferred labels only after model selection, locked output, and independent replay.
    assert digest(base.LABEL_FILE)==base.LABEL_SHA
    inferred=pd.read_csv(base.LABEL_FILE)
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
        selected_model=protocol['selected'],inferred_label_auc=auc,independent_rank_auc=independent_auc,
        inferred_positive_count=positives,labels_are_inferred_not_observed=True,platform_auc=None,new_model_training=False,
        inferred_labels_used_for_training_tuning_feature_or_model_selection=False,
        development_dates_repeatedly_reused=True,correction_training_max_date=20191115,
        inherited_reference_training_max_date=20191116,ad_history_max_date=20191116,
        day16_winning_forecasts_replayed=True,inherited_base_and_features_independently_verified=True,
        day17_full1209_feature_frame_and_model_margins_replayed_exactly=True,
        day17_group_correction_independently_replayed=True,maximum_prediction_replay_error=error,
        csv_roundtrip_exact=True,original_row_order_verified=True,
        repeated_log_ids_preserved=int(raw.log_id.duplicated().sum()),original_datasets_unchanged=True,
        raw_datasets_sha256=protocol['raw_datasets_sha256'],inferred_label_sha256=base.LABEL_SHA,
        protocol_sha256=digest(RUN/'protocol.json'),submission_sha256=digest(OUTPUT/'submission_best_20191117.csv'))
    write(OUTPUT/'verification.json',report)
    (OUTPUT/'提交说明.md').write_text(
        f'2019年11月17日提交文件：submission_best_20191117.csv\n格式 id,pctr；{len(raw):,}行，原始行序零起始行号。\n'
        f'依据真实16号开发集选择模型，AUC={auc16:.12f}；15/16号曾反复用于开发。\n'
        f'17号反推标签AUC={auc:.12f}；反推标签未用于训练、调参、特征选择或模型选择，平台真实AUC未知。\n'
        '已验证原始四份CSV不变、行序、CSV精度与独立预测复算。\n',encoding='utf8')
    write(HERE/'delivery_task_audience_row_v1.json',dict(request_complete=True,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),submission=str(OUTPUT/'submission_best_20191117.csv'),
        verification=str(OUTPUT/'verification.json'),rows=len(raw),selected_real16_recipe=protocol['selected'],
        development16_auc=auc16,inferred17_label_auc_evaluation_only=auc,
        inferred17_labels_used_for_selection=False,original_datasets_unchanged=True,
        submission_sha256=report['submission_sha256'],original_target_goal_remains_active=True))
    emit('task_audience_row17_verified_and_evaluated',**report)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=['freeze','predict','verify','all'])
    phase=parser.parse_args().phase
    if phase=='all':
        if freeze():
            predict()
            verify()
    else:
        {'freeze':freeze,'predict':predict,'verify':verify}[phase]()
