"""Replay frozen models exactly and bound independent reduction effects on tied AUC."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='2'
import gc
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.metrics import roc_auc_score
import catboost_within_ranker_v2 as frozen
from catboost_userday_residual import load_model
from conditional_daily import read,emit
from fit_stacker import HERE,ROOT,digest,write
from rich_userday_activity import array_hash
from within_user_ranker import centered_unit

RECOVERY=HERE/'runs/catboost_ranker_precision_verification_v2'
AGGREGATE_ATOL=2e-13
PROBABILITY_ATOL=2e-14


def possible_pair_change_bound(y,p,error):
    """Only pairs separated by at most twice the score perturbation can change."""
    if error==0.:
        return 0.
    negatives=np.sort(p[y==0])
    positives=p[y==1]
    low=np.nextafter(positives-2*error,-np.inf)
    high=np.nextafter(positives+2*error,np.inf)
    counts=np.searchsorted(negatives,high,side='right')-np.searchsorted(negatives,low,side='left')
    return float(counts.sum()/(len(negatives)*len(positives)))


def main():
    assert not RECOVERY.exists(),'Preserve previous verification attempts'
    assert not (frozen.OUT/'verification.json').exists()
    protocol=frozen.check_sources()
    lock=read(frozen.OUT/'predictions_locked.json')
    models=read(frozen.OUT/'models_locked.json')
    original=read(frozen.OUT/'result.json')
    assert lock['protocol_sha256']==digest(frozen.OUT/'protocol.json')
    assert lock['models_lock_sha256']==digest(frozen.OUT/'models_locked.json')
    paths=[Path(__file__),HERE/'verify_catboost_ranker_precision_v1.py',
        HERE/'runs/catboost_ranker_precision_verification_v1/protocol.json',
        frozen.OUT/'protocol.json',frozen.OUT/'models_locked.json',
        frozen.OUT/'predictions_locked.json',frozen.OUT/'result.json']
    paths+=list(frozen.OUT.glob('*.npy'))+list(frozen.OUT.glob('*.cbm'))
    paths+=[frozen.OUT/f'{part}_contract{day}.json' for day in [15,16] for part in ['input','held']]
    bindings={str(path):digest(path) for path in paths}
    RECOVERY.mkdir()
    write(RECOVERY/'protocol.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        frozen_run=str(frozen.OUT),sources_sha256=bindings,
        original_verifier_failure='Bitwise equality was required for independent pandas vs numpy reductions',
        first_supplement_failure='A blanket 1e-12 AUC tolerance does not account for tied primary scores',
        raw_model_margins_and_original_candidates_require_bitwise_equality=True,
        aggregate_absolute_tolerance=AGGREGATE_ATOL,probability_absolute_tolerance=PROBABILITY_ATOL,
        independent_auc_bound='Fraction of positive-negative pairs separated by no more than twice the measured maximum probability error',
        require_same_independent_winning_recipe=True,training_or_reselection=False,inferred17_labels_read=False))
    summaries={}
    for day in [15,16]:
        train,cats,info,audit=frozen.inputs(day,'train',True)
        assert audit==read(frozen.OUT/f'input_contract{day}.json')
        del train,info
        gc.collect()
        frame,cats,info,audit=frozen.inputs(day,'held',True)
        assert audit==read(frozen.OUT/f'held_contract{day}.json')
        np.testing.assert_array_equal(info['held_rows'],np.load(frozen.OUT/f'day{day}_indices.npy'))
        pool=Pool(frame,cat_features=cats)
        del frame
        primary,independent={},{}
        primary_last,independent_last=[],[]
        for record in models['models']:
            if record['day']!=day:
                continue
            path=frozen.OUT/(record['name']+'.cbm')
            assert digest(path)==record['model_sha256']
            assert record['input_contract_sha256']==digest(frozen.OUT/f'input_contract{day}.json')
            value=load_model(path).predict(pool,prediction_type='RawFormulaVal',thread_count=2)
            np.testing.assert_array_equal(value,np.load(frozen.OUT/(record['name']+'_held_margin.npy')))
            assert array_hash(value)==lock['dates'][str(day)]['forecasts_sha256'][record['name']]
            primary[record['name']]=independent[record['name']]=value
            if record['trees']==max(frozen.CHECKPOINTS):
                primary_last.append(centered_unit(value,info['groups'],False))
                independent_last.append(centered_unit(value,info['groups'],True))
        assert len(primary)==6 and len(primary_last)==2
        name=f'day{day}_mean_final_within_unit'
        primary[name]=np.mean(primary_last,axis=0)
        independent[name]=np.mean(independent_last,axis=0)
        np.testing.assert_array_equal(primary[name],np.load(frozen.OUT/(name+'_held_margin.npy')))
        assert array_hash(primary[name])==lock['dates'][str(day)]['forecasts_sha256'][name]
        aggregate_error=float(np.max(np.abs(primary[name]-independent[name])))
        assert aggregate_error<AGGREGATE_ATOL
        actual=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
        y=actual.loc[actual.pt_d.eq(20191100+day),'label'].to_numpy()
        del actual
        records=[]
        expected_records=original['dates'][str(20191100+day)]['records']
        best_independent=None
        for (record,p),(other,q) in zip(frozen.candidates(day,primary,info['groups']),
                                     frozen.candidates(day,independent,info['groups'],True)):
            assert record==other
            assert array_hash(p)==lock['dates'][str(day)]['candidates_sha256'][frozen.candidate_key(record)]
            probability_error=float(np.max(np.abs(p-q)))
            assert probability_error<PROBABILITY_ATOL
            primary_auc=float(roc_auc_score(y,p))
            independent_auc=float(roc_auc_score(y,q))
            expected=expected_records[len(records)]
            assert all(expected[key]==value for key,value in record.items())
            assert abs(primary_auc-expected['auc'])<1e-12
            auc_error=abs(primary_auc-independent_auc)
            bound=possible_pair_change_bound(y,p,probability_error)
            assert auc_error<=bound+1e-12
            detail=dict(record,primary_auc=primary_auc,independent_auc=independent_auc,
                maximum_probability_error=probability_error,auc_error=auc_error,pair_order_auc_error_bound=bound)
            records.append(detail)
            if best_independent is None or independent_auc>best_independent['independent_auc']:
                best_independent=detail
        assert len(records)==113
        winner=original['dates'][str(20191100+day)]['best']
        assert all(best_independent[key]==winner[key] for key in ['name','mode','weight'])
        summaries[str(20191100+day)]=dict(raw_model_margins_exact=True,original_numpy_aggregate_exact=True,
            maximum_independent_aggregate_error=aggregate_error,
            maximum_independent_candidate_error=max(r['maximum_probability_error'] for r in records),
            maximum_independent_candidate_auc_error=max(r['auc_error'] for r in records),
            independent_winning_recipe_unchanged=True,independent_winning_auc=best_independent['independent_auc'],
            candidates=len(records),training_and_held_frames_exact=True,full337_quality_rebuilt_from_raw=True,
            independent_candidate_records=records)
        write(RECOVERY/f'day{day}_replay.json',summaries[str(20191100+day)])
        emit('catboost_ranker_precision_replay_passed',day=day,
            **{key:value for key,value in summaries[str(20191100+day)].items() if key!='independent_candidate_records'})
        del pool,primary,independent,primary_last,independent_last,info
        frozen.QUALITY_REPLAY.pop(day,None)
        gc.collect()
    for path,expected in bindings.items():
        assert digest(path)==expected,path
    frozen.check_sources()
    write(RECOVERY/'verification.json',dict(status='passed',dates=summaries,
        original_frozen_sources_models_forecasts_and_selection_preserved=True,
        inferred17_labels_read=False,raw_datasets_sha256=protocol['raw_datasets_sha256']))
    frozen.score(True)
    report=read(frozen.OUT/'verification.json')
    report['supplemental_precision_verification']=str(RECOVERY/'verification.json')
    report['supplemental_precision_verification_sha256']=digest(RECOVERY/'verification.json')
    report['independent_aggregate_uses_documented_float64_tolerance']=True
    report['independent_auc_discrepancies_bounded_by_score_ties']=True
    write(frozen.OUT/'verification.json',report)
    emit('catboost_ranker_supplemental_verification_complete',aucs={day:r['auc'] for day,r in report['dates'].items()})


if __name__=='__main__':
    main()
