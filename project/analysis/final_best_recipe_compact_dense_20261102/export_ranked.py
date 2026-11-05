"""Apply a later real16 ranking winner to the already verified compact/dense export."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import export_best_v2 as base
from fit_stacker import digest,write
from conditional_daily import read,emit
from rich_userday_activity import array_hash
from within_user_ranking_core import within_prediction
from within_user_ranker import centered_unit

HERE=Path(__file__).resolve().parent
RUN=HERE/'ranked_run_v1'
OUTPUT=base.ROOT/'output/best_recipe_ranked_compact_dense_20191117_20261102'


def check_sources():
    protocol=read(RUN/'protocol.json')
    for filename,expected in protocol['sources_sha256'].items():
        assert digest(filename)==expected,filename
    for filename,expected in protocol['raw_datasets_sha256'].items():
        assert digest(base.ROOT/'2026ml'/filename)==expected,filename
    return protocol


def freeze():
    assert not RUN.exists() and not OUTPUT.exists()
    report=read(base.RANKER/'verification.json')
    current=read(base.OUTPUT/'verification.json')
    assert report['status']==current['status']=='passed'
    selected=report['dates']['20191116']['best']
    if selected['auc']<=current['development16_auc']:
        emit('ranking_export_unnecessary',ranking_auc=selected['auc'],current_auc=current['development16_auc'])
        return False
    assert selected['name']!='base'
    assert read(base.RUN/'protocol.json')['selected']['ranking'] is None
    names=([f'day16_{obj}_trees1600' for obj in ['lambdarank','rank_xendcg']]
        if selected['name']=='day16_mean_final_within_unit' else [selected['name']])
    audit=read(base.ROOT/'output/two_day_auc_targets_20261102/independent_target_audit.json')
    assert audit['audit_status']=='passed' and abs(audit['dates']['20191116']['auc']-selected['auc'])<1e-12
    parent=read(base.RUN/'protocol.json')
    paths=[Path(__file__),HERE/'run_ranked_export.ps1',HERE/'export_best_v2.py',base.RUN/'protocol.json',base.OUTPUT/'verification.json',
        base.OUTPUT/'prediction_manifest.json',base.RANKER/'protocol.json',base.RANKER/'verification.json',
        base.RANKER/'models_locked.json',base.RANKER/'day16_prediction.npy']
    paths += [base.RANKER/(name+'.txt') for name in names]
    paths += [base.OUTPUT/f'day{day}_prediction.npy' for day in [16,17]]
    bindings=dict(parent['sources_sha256'])
    for filename,expected in read(base.RANKER/'protocol.json')['sources_sha256'].items():
        assert filename not in bindings or bindings[filename]==expected
        bindings[filename]=expected
    for path in paths:
        bindings[str(path)]=digest(path)
    for filename,expected in bindings.items():
        assert digest(filename)==expected,filename
    for filename,expected in parent['raw_datasets_sha256'].items():
        assert digest(base.ROOT/'2026ml'/filename)==expected,filename
    RUN.mkdir()
    write(RUN/'protocol.json',dict(status='frozen_before_ranked17_forecast_and_label_evaluation',
        timestamp_utc=datetime.now(timezone.utc).isoformat(),selected=selected,names=names,
        development16_auc=selected['auc'],selection_basis='Highest verified published real16 model',
        development_dates_repeatedly_reused=True,new_model_training=False,
        inferred17_labels_used_for_training_tuning_feature_or_model_selection=False,
        inferred17_labels_previously_evaluated_on_older_frozen_exports=True,
        raw_datasets_sha256=parent['raw_datasets_sha256'],sources_sha256=bindings))
    emit('ranked17_recipe_frozen',selected=selected)
    return True


def forecast(day,independent=False):
    protocol=read(RUN/'protocol.json')
    raw=base.query_raw(day)
    group=base.group_codes(raw[base.FIELDS],['user_id'])
    margins,row_contract=base.dense_margins(day,independent,protocol['names'])
    names=protocol['names']
    margin=(np.mean([centered_unit(margins[name],group,independent) for name in names],axis=0)
        if len(names)>1 else margins[names[0]])
    selected=protocol['selected']
    current=np.load(base.OUTPUT/f'day{day%100}_prediction.npy')
    prediction=within_prediction(current,margin,group,selected['weight'],selected['mode'],independent)
    if day==20191116:
        np.testing.assert_allclose(prediction,np.load(base.RANKER/'day16_prediction.npy'),rtol=0,atol=2e-14)
    contract=dict(day=day,rows=len(raw),row_matrix_sha256=row_contract['matrix_sha256'],
        original_log_ids_sha256=array_hash(raw.log_id.to_numpy()),group_sha256=array_hash(group),
        margin_sha256={name:array_hash(margins[name]) for name in names})
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
        files_sha256={path.name:digest(path) for path in OUTPUT.iterdir() if path.is_file()}))
    emit('ranked17_submission_locked',path=str(OUTPUT/'submission_best_20191117.csv'))


def verify():
    protocol=check_sources()
    assert not (OUTPUT/'verification.json').exists()
    manifest=read(OUTPUT/'prediction_manifest.json')
    assert manifest['protocol_sha256']==digest(RUN/'protocol.json')
    assert manifest['input_contracts_sha256']==digest(RUN/'input_contracts.json')
    for filename,expected in manifest['files_sha256'].items():
        assert digest(OUTPUT/filename)==expected
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
    # Evaluation-only inferred label access starts after independent prediction replay.
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
        inferred_label_auc=auc,independent_rank_auc=independent_auc,inferred_positive_count=positives,
        labels_are_inferred_not_observed=True,platform_auc=None,new_model_training=False,
        inferred_labels_used_for_training_tuning_feature_or_model_selection=False,
        development_dates_repeatedly_reused=True,correction_training_max_date=20191115,
        inherited_reference_training_max_date=20191116,ad_history_max_date=20191116,
        day16_winning_forecasts_replayed=True,inherited_base_and_features_independently_verified=True,
        day17_dictionary_rank_forecasts_and_group_correction_independently_replayed=True,
        maximum_prediction_replay_error=error,csv_roundtrip_exact=True,original_row_order_verified=True,
        repeated_log_ids_preserved=int(raw.log_id.duplicated().sum()),original_datasets_unchanged=True,
        raw_datasets_sha256=protocol['raw_datasets_sha256'],inferred_label_sha256=base.LABEL_SHA,
        protocol_sha256=digest(RUN/'protocol.json'),submission_sha256=digest(OUTPUT/'submission_best_20191117.csv'))
    write(OUTPUT/'verification.json',report)
    (OUTPUT/'提交说明.md').write_text(
        f'2019年11月17日提交文件：submission_best_20191117.csv\n格式 id,pctr；{len(raw):,}行，原始行序零起始行号。\n'
        f'依据真实16号开发集选择模型，AUC={auc16:.12f}；15/16号曾反复用于开发。\n'
        f'17号反推标签AUC={auc:.12f}；反推标签未用于训练、调参或模型选择，平台真实AUC未知。\n'
        '已验证原始数据不变、行序、CSV精度与独立预测复算。\n',encoding='utf8')
    emit('ranked17_verified_and_evaluated',**report)


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
