"""Select a verified daily-rate recipe on real16 and export its day17 predictions."""
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
from daily_auc_pair_math import daily_prediction,rate_baseline
from rich_row_catboost import frame_hash
from rich_userday_activity import array_hash

HERE=Path(__file__).resolve().parent
RUN=HERE/'daily_rate_best_run_v1'
OUTPUT=base.ROOT/'output/best_recipe_daily_rate_20191117_20261102'
PARENT=base.ROOT/'output/best_recipe_catboost_ranked_20191117_20261102'
PARENT_RUN=HERE/'catboost_ranked_run_v1'
TRIALS=[base.EXP/'runs'/name for name in ['daily_auc_pair_ranker_v1','daily_binomial_rate_v1']]


def check_sources():
    protocol=read(RUN/'protocol.json')
    for filename,expected in protocol['sources_sha256'].items():
        assert digest(filename)==expected,filename
    for filename,expected in protocol['raw_datasets_sha256'].items():
        assert digest(base.ROOT/'2026ml'/filename)==expected,filename
    return protocol


def freeze():
    assert not RUN.exists() and not OUTPUT.exists(),'Preserve previous exports'
    current=read(PARENT/'verification.json')
    assert current['status']=='passed'
    candidates=[]
    for folder in TRIALS:
        path=folder/'verification.json'
        if not path.is_file():
            continue
        report=read(path)
        assert report['status']=='passed'
        result=report['dates']['20191116']
        assert not result['evaluation_day_labels_fit']
        candidates.append((result['auc'],folder,result['best']))
    if not candidates or max(item[0] for item in candidates)<=current['development16_auc']:
        emit('daily_rate17_export_unnecessary',current_auc=current['development16_auc'])
        return False
    auc,folder,selected=max(candidates,key=lambda item:item[0])
    assert selected['name']!='base' and selected['mode'] in ['residual_add','rate_replace','count_mass']
    audit=read(base.ROOT/'output/two_day_auc_targets_20261102/independent_target_audit.json')
    assert audit['audit_status']=='passed' and abs(audit['dates']['20191116']['auc']-auc)<1e-12
    parent=read(PARENT_RUN/'protocol.json')
    native=read(folder/'protocol.json')
    assert parent['raw_datasets_sha256']==native['raw_datasets_sha256']
    assert Path(native['baseline_run']).name=='catboost_within_ranker_v2'
    bindings=dict(parent['sources_sha256'])
    for filename,expected in native['sources_sha256'].items():
        assert filename not in bindings or bindings[filename]==expected,filename
        bindings[filename]=expected
    paths=[Path(__file__),PARENT_RUN/'protocol.json',PARENT/'verification.json',PARENT/'prediction_manifest.json',
        folder/'verification.json',folder/'protocol.json',folder/'models_locked.json',folder/'predictions_locked.json',
        folder/(selected['name']+'.cbm'),folder/(selected['name']+'_held_margin.npy'),
        folder/'day16_prediction.npy',folder/'input_day16/contract.json',
        folder/'input_day16/frame.pkl',folder/'input_day16/metadata.npz',folder/'input_day16/files_locked.json',
        base.EXP/'daily_auc_pair_math.py',base.EXP/'rich_row_catboost.py']
    paths += [PARENT/f'day{day}_prediction.npy' for day in [16,17]]
    for candidate_folder in TRIALS:
        if (candidate_folder/'verification.json').is_file():
            paths.append(candidate_folder/'verification.json')
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
    write(RUN/'protocol.json',dict(status='frozen_before_daily_rate17_predictions_and_inferred_label_evaluation',
        timestamp_utc=datetime.now(timezone.utc).isoformat(),selected=selected,chosen_source=str(folder),
        development16_auc=auc,selection_basis='Highest independently verified published real16 recipe',
        development_dates_repeatedly_reused=True,new_model_training=False,
        correction_training_max_date=20191115,inherited_reference_training_max_date=20191116,
        inferred17_labels_used_for_training_tuning_feature_or_model_selection=False,
        inferred17_labels_previously_evaluated_on_older_frozen_exports=True,
        raw_datasets_sha256=parent['raw_datasets_sha256'],sources_sha256=bindings))
    emit('daily_rate17_recipe_frozen',selected=selected,source=folder.name)
    return True


def forecast(day,independent=False):
    protocol=read(RUN/'protocol.json')
    selected=protocol['selected']
    folder=Path(protocol['chosen_source'])
    native=read(folder/'input_day16/contract.json')
    keys,groups,frame,mass,query_contract=base.query_frame(day,independent)
    raw=base.query_raw(day)
    np.testing.assert_array_equal(groups,pd.Index(keys[:,1]).get_indexer(raw.user_id))
    sizes=np.bincount(groups,minlength=len(keys))
    assert len(frame)==len(keys) and np.all(sizes>0)
    model_record=next(record for record in read(folder/'models_locked.json')['models'] if record['name']==selected['name'])
    assert model_record['day']==16
    path=folder/(selected['name']+'.cbm')
    assert digest(path)==model_record['model_sha256']
    columns=model_record['columns']
    assert sum(name.startswith('quality__') for name in columns)==337
    offset=rate_baseline(mass,sizes)
    if day==20191116:
        data=np.load(folder/'input_day16/metadata.npz')
        held=np.flatnonzero(data['keys'][:,0]==day)
        np.testing.assert_array_equal(keys,data['keys'][held])
        np.testing.assert_array_equal(sizes,data['sizes'][held])
        np.testing.assert_array_equal(offset,data['baseline'][held])
        cached=pd.read_pickle(folder/'input_day16/frame.pkl')
        pd.testing.assert_frame_equal(frame[columns].reset_index(drop=True),cached.iloc[held][columns].reset_index(drop=True),
            check_dtype=False,check_exact=True)
        del cached,data
    model=load_model(path)
    assert model.feature_names_==columns
    margin=model.predict(Pool(frame[columns],cat_features=native['categories']),prediction_type='RawFormulaVal',thread_count=2)
    if day==20191116:
        np.testing.assert_array_equal(margin,np.load(folder/(selected['name']+'_held_margin.npy')))
    current=np.load(PARENT/f'day{day%100}_prediction.npy')
    prediction=daily_prediction(current,margin,margin+offset,sizes,groups,selected['weight'],selected['mode'],independent)
    if day==20191116:
        np.testing.assert_allclose(prediction,np.load(folder/'day16_prediction.npy'),rtol=0,atol=2e-14)
    assert len(prediction)==len(raw) and np.isfinite(prediction).all() and np.all((prediction>0)&(prediction<1))
    contract=dict(day=day,rows=len(raw),keys_sha256=array_hash(keys),groups_sha256=array_hash(groups),
        sizes_sha256=array_hash(sizes),baseline_sha256=array_hash(offset),
        selected_feature_names=columns,selected_frame_sha256=frame_hash(frame[columns]),
        raw_model_margin_sha256=array_hash(margin),original_log_ids_sha256=array_hash(raw.log_id.to_numpy()),
        own_prediction_date_ad_labels_masked=True)
    del frame,model,raw
    gc.collect()
    emit('daily_rate17_forecast_complete',day=day,independent=independent)
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
        development16_auc=protocol['development16_auc'],inferred17_labels_used_for_prediction_or_selection=False,
        files_sha256={path.name:digest(path) for path in OUTPUT.iterdir() if path.is_file()}))
    emit('daily_rate17_submission_locked',path=str(OUTPUT/'submission_best_20191117.csv'))


def verify():
    protocol=check_sources()
    assert not (OUTPUT/'verification.json').exists()
    manifest=read(OUTPUT/'prediction_manifest.json')
    assert manifest['protocol_sha256']==digest(RUN/'protocol.json')
    assert manifest['input_contracts_sha256']==digest(RUN/'input_contracts.json')
    for filename,expected in manifest['files_sha256'].items():
        assert digest(OUTPUT/filename)==expected,filename
    replay,contract=forecast(20191117,True)
    assert contract==read(RUN/'input_contracts.json')['20191117']
    error=float(np.max(np.abs(replay-np.load(OUTPUT/'day17_prediction.npy'))))
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
    assert np.isfinite(submission.pctr).all() and submission.pctr.between(0,1).all()
    actual=pd.read_csv(base.ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    auc16=float(roc_auc_score(actual.loc[actual.pt_d.eq(20191116),'label'],np.load(OUTPUT/'day16_prediction.npy')))
    assert abs(auc16-protocol['development16_auc'])<1e-12
    # Only evaluation after model selection, exported file locks, and independent replay.
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
        chosen_source=protocol['chosen_source'],selected_model=protocol['selected'],
        inferred_label_auc=auc,independent_rank_auc=independent_auc,inferred_positive_count=positives,
        labels_are_inferred_not_observed=True,platform_auc=None,new_model_training=False,
        inferred_labels_used_for_training_tuning_feature_or_model_selection=False,
        development_dates_repeatedly_reused=True,correction_training_max_date=20191115,
        inherited_reference_training_max_date=20191116,ad_history_max_date=20191116,
        day16_winning_forecasts_replayed=True,day17_daily_features_rebuilt_twice=True,
        selected_features_and_raw_model_margins_replayed_exactly=True,
        between_user_correction_independently_replayed=True,maximum_prediction_replay_error=error,
        csv_roundtrip_exact=True,original_row_order_verified=True,
        repeated_log_ids_preserved=int(raw.log_id.duplicated().sum()),original_datasets_unchanged=True,
        raw_datasets_sha256=protocol['raw_datasets_sha256'],inferred_label_sha256=base.LABEL_SHA,
        protocol_sha256=digest(RUN/'protocol.json'),submission_sha256=digest(OUTPUT/'submission_best_20191117.csv'))
    write(OUTPUT/'verification.json',report)
    (OUTPUT/'提交说明.md').write_text(
        f'2019年11月17日提交文件：submission_best_20191117.csv\n格式 id,pctr；{len(raw):,}行，原始行序零起始行号。\n'
        f'按真实16号开发集选择模型，AUC={auc16:.12f}；15/16号曾反复用于开发。\n'
        f'17号反推标签AUC={auc:.12f}，仅用于最终评估，未用于训练、调参、特征选择或模型选择。\n'
        '平台真实AUC待提交。原始四份CSV、行序、CSV精度与独立预测复算已核验。\n',encoding='utf8')
    write(HERE/'delivery_daily_rate_best_v1.json',dict(request_complete=True,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),submission=str(OUTPUT/'submission_best_20191117.csv'),
        verification=str(OUTPUT/'verification.json'),rows=len(raw),selected_real16_recipe=protocol['selected'],
        development16_auc=auc16,inferred17_label_auc_evaluation_only=auc,
        inferred17_labels_used_for_selection=False,original_datasets_unchanged=True,
        submission_sha256=report['submission_sha256'],original_target_goal_remains_active=True))
    emit('daily_rate17_verified_and_evaluated',**report)


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
