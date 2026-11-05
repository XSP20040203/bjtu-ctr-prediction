"""Full-cohort chronological row rankers with locked both-date model selection."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='3'
import argparse
import gc
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from fit_stacker import HERE,ROOT,digest,write
from conditional_daily import read,emit
from rich_userday_activity import array_hash,lgb
from verify_stacker import rank_sum_auc
import within_user_ranking_core as core

OUT=HERE/'runs/within_user_ranker_v1'
BASE=HERE/'runs/compose_compact_dense_transfer_v1'
OBJECTIVES=['lambdarank','rank_xendcg']
CHECKPOINTS=[200,600,1600]
WEIGHTS=[.025,.05,.1,.2,.35,.5,.75,1.]
MODES=['add_within','replace_within']
PREDICTION_BATCH=40000


def checked_layout(rows,dates,users,log_ids,cutoff):
    rows=np.asarray(rows,dtype='int64')
    dates,users,log_ids=map(np.asarray,(dates,users,log_ids))
    if np.any(np.diff(rows)<=0) or np.any(dates[rows]>cutoff):
        raise ValueError('Sorted unique source rows must contain no later advertisement dates')
    held=np.flatnonzero(dates==cutoff)
    positions=np.flatnonzero(dates[rows]==cutoff)
    if not np.array_equal(rows[positions],held):
        raise ValueError('Every original held advertisement row is required in its original order')
    groups,_=pd.factorize(users[held],sort=True)
    assert np.all(groups>=0)
    audit=dict(held_rows_sha256=array_hash(held),held_positions_sha256=array_hash(positions),
        held_users_sha256=array_hash(users[held]),held_log_ids_sha256=array_hash(log_ids[held]),
        held_groups_sha256=array_hash(groups),rows=len(held))
    return positions,held,groups,audit


def layout(day):
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','user_id','log_id','label'],dtype='int32')
    dates=raw.pt_d.to_numpy()
    labels=raw.label.to_numpy(copy=True)
    cutoff=20191100+day
    labels[dates>=cutoff]=-1
    chosen,weights=(core.rows15 if day==15 else core.rows16).selected_training(labels,dates)
    source=read(core.SOURCE/f'input_contract{day}.json')
    assert array_hash(chosen)==source['training_rows_sha256']
    assert array_hash(labels[chosen])==source['training_labels_sha256']
    assert array_hash(weights)==source['weights_sha256']
    rows=np.sort(np.r_[chosen,np.flatnonzero(dates==cutoff)])
    positions,held,groups,audit=checked_layout(rows,dates,raw.user_id.to_numpy(),raw.log_id.to_numpy(),cutoff)
    np.testing.assert_array_equal(held,np.load(BASE/f'day{day}_indices.npy'))
    assert len(held)=={15:424780,16:421983}[day]
    return positions,held,groups,audit


def check_sources():
    protocol=read(OUT/'protocol.json')
    assert protocol['objectives']==OBJECTIVES and protocol['checkpoints']==CHECKPOINTS
    assert protocol['weights']==WEIGHTS and protocol['modes']==MODES and protocol['params']==core.PARAMS
    for path,expected in protocol['sources_sha256'].items():
        assert digest(path)==expected,path
    for name,expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name)==expected,name
    return protocol


def freeze():
    assert not OUT.exists(),'Preserve completed fits and failed runs'
    assert read(core.PREFLIGHT/'verification.json')['status']=='passed'
    assert read(BASE/'verification.json')['status']=='passed'
    hashes=read(BASE/'protocol.json')['raw_datasets_sha256']
    paths=[Path(__file__),HERE/'test_within_user_ranker.py',HERE/'run_within_user_ranker.ps1',
        HERE/'within_user_ranking_core.py',HERE/'test_within_user_ranking_core.py',
        HERE/'fit_stacker.py',HERE/'conditional_daily.py',HERE/'rich_userday_activity.py',
        HERE/'verify_stacker.py',core.PREFLIGHT/'protocol.json',core.PREFLIGHT/'verification.json']
    for folder in [core.DENSE,BASE]:
        assert read(folder/'verification.json')['status']=='passed'
        parent=read(folder/'protocol.json')
        assert parent['raw_datasets_sha256']==hashes
        paths += [folder/'protocol.json',folder/'verification.json']
        for path,expected in parent['sources_sha256'].items():
            assert digest(path)==expected,path
            paths.append(Path(path))
    for path,expected in read(core.PREFLIGHT/'protocol.json')['sources_sha256'].items():
        assert digest(path)==expected,path
        paths.append(Path(path))
    for day in [15,16]:
        paths += [BASE/f'day{day}_indices.npy',BASE/f'day{day}_prediction.npy',
            core.SOURCE/f'input_contract{day}.json',core.DENSE/f'contract_day{day}_context_numeric.json']
        for phase in ['input','independent']:
            paths.append(core.DENSE/f'{phase}/day{day}_context_numeric/features.npy')
    for name,expected in hashes.items():
        assert digest(ROOT/'2026ml'/name)==expected,name
    OUT.mkdir()
    write(OUT/'protocol.json',dict(dates=[20191115,20191116],objectives=OBJECTIVES,
        checkpoints=CHECKPOINTS,weights=WEIGHTS,modes=MODES,params=core.PARAMS,
        prediction_batch_size=PREDICTION_BATCH,lightgbm_version=lgb.__version__,
        training='Complete mixed-label user/day queries within the frozen strictly-prior advertisement sample',
        sampling='All earlier positives and fixed one-third earlier negatives; query ranking without inverse row weights',
        feature_source='Independently raw-rebuilt frozen dense/context matrices with prior-only category dictionaries',
        normalization='Held unlabeled user groups and prediction margins only; no held ad labels',
        selection='All both-date models and candidate hashes lock before scoring; repeatedly reused development dates',
        evaluation_day_labels_fit=False,inferred17_labels_read=False,
        raw_datasets_sha256=hashes,sources_sha256={str(p):digest(p) for p in dict.fromkeys(paths)}))
    emit('within_user_ranker_protocol_frozen',objectives=OBJECTIVES,checkpoints=CHECKPOINTS)


def training_values(matrix,order):
    values=np.empty((len(order),matrix.shape[1]),dtype='float32')
    for first in range(0,len(order),5000):
        values[first:first+5000]=matrix[order[first:first+5000]]
    return values


def training_values_hash(matrix,order):
    value=hashlib.sha256()
    for first in range(0,len(order),5000):
        block=np.ascontiguousarray(matrix[order[first:first+5000]],dtype='float32')
        value.update(block.tobytes())
    return value.hexdigest()


def fit():
    freeze()
    locks=[]
    for day in [15,16]:
        matrix,order,sizes,target,keys,audit=core.prior_input(day)
        positions,rows,groups,held_audit=layout(day)
        values=training_values(matrix,order)
        assert np.isfinite(values).all()
        contract=dict(prior=audit,held=held_audit,training_values_sha256=array_hash(values),
            training_shape=list(values.shape),largest_query=int(sizes.max()))
        write(OUT/f'input_contract{day}.json',contract)
        np.save(OUT/f'day{day}_indices.npy',rows)
        emit('within_user_ranker_training_ready',day=day,shape=list(values.shape),
            queries=len(sizes),max_training_date=audit['max_training_date'])
        dataset=lgb.Dataset(values,label=target,group=sizes,categorical_feature=audit['categorical_columns'],
            params=core.PARAMS,free_raw_data=True)
        dataset.construct()
        del values,matrix
        gc.collect()
        for objective in OBJECTIVES:
            def progress_callback(env):
                if (env.iteration+1)%400==0:
                    emit('within_user_ranker_training_iteration',day=day,objective=objective,iteration=env.iteration+1)
            model=lgb.train(dict(core.PARAMS,objective=objective),dataset,
                num_boost_round=max(CHECKPOINTS),callbacks=[progress_callback])
            for trees in CHECKPOINTS:
                name=f'day{day}_{objective}_trees{trees}'
                path=OUT/(name+'.txt')
                path.write_text(model.model_to_string(num_iteration=trees),encoding='utf8')
                locks.append(dict(day=day,objective=objective,trees=trees,name=name,
                    actual_trees=min(trees,model.current_iteration()),
                    model_sha256=digest(path),input_contract_sha256=digest(OUT/f'input_contract{day}.json')))
            emit('within_user_ranker_arm_complete',day=day,objective=objective)
            del model
            gc.collect()
        del dataset,order,sizes,target,keys,groups
        gc.collect()
    write(OUT/'models_locked.json',dict(protocol_sha256=digest(OUT/'protocol.json'),models=locks))
    check_sources()


def centered_unit(values,groups,independent=False):
    if independent:
        table=pd.DataFrame(dict(group=groups,value=values))
        center=table.groupby('group',sort=False).value.transform('mean').to_numpy()
    else:
        center=(np.bincount(groups,weights=values)/np.bincount(groups).clip(1))[groups]
    residual=values-center
    return residual/max(residual.std(),1e-6)


def forecasts(day,verifying=False):
    matrix,order,sizes,target,keys,audit=core.prior_input(day,verifying)
    positions,rows,groups,held_audit=layout(day)
    contract=read(OUT/f'input_contract{day}.json')
    assert audit==contract['prior'] and held_audit==contract['held']
    if verifying:
        assert training_values_hash(matrix,order)==contract['training_values_sha256']
    locked=read(OUT/'models_locked.json')
    assert locked['protocol_sha256']==digest(OUT/'protocol.json')
    result={}
    last=[]
    for record in locked['models']:
        if record['day']!=day:
            continue
        path=OUT/(record['name']+'.txt')
        assert digest(path)==record['model_sha256']
        assert digest(OUT/f'input_contract{day}.json')==record['input_contract_sha256']
        model=lgb.Booster(model_str=path.read_text(encoding='utf8'))
        pieces=[]
        for first in range(0,len(positions),PREDICTION_BATCH):
            block=np.asarray(matrix[positions[first:first+PREDICTION_BATCH]])
            pieces.append(model.predict(block,raw_score=True,num_threads=3))
        value=np.concatenate(pieces).astype('float64')
        assert np.isfinite(value).all()
        result[record['name']]=value
        if record['trees']==max(CHECKPOINTS):
            unit=centered_unit(value,groups)
            independent=centered_unit(value,groups,True)
            assert np.max(np.abs(unit-independent))<2e-12
            last.append(unit)
        del model,block
    assert len(last)==len(OBJECTIVES)
    result[f'day{day}_mean_final_within_unit']=np.mean(last,axis=0)
    for name,value in result.items():
        path=OUT/(name+'_held_margin.npy')
        if verifying:
            np.testing.assert_array_equal(value,np.load(path))
        else:
            assert not path.exists()
            np.save(path,value)
    del matrix,order,sizes,target,keys
    gc.collect()
    emit('within_user_ranker_forecasts_ready',day=day,forecasts=len(result),verifying=verifying)
    return result,rows,groups


def candidate_key(record):
    return f'{record["name"]}__{record["mode"]}__weight{record["weight"]:g}'


def candidates(day,values,groups,independent=False):
    base=np.load(BASE/f'day{day}_prediction.npy')
    yield dict(name='base',mode='base',weight=0.),base
    for name,margin in values.items():
        for mode in MODES:
            for weight in WEIGHTS:
                p=core.within_prediction(base,margin,groups,weight,mode,independent)
                assert np.isfinite(p).all() and np.all((p>0)&(p<1))
                yield dict(name=name,mode=mode,weight=weight),p


def score_locked(verifying=False):
    check_sources()
    locked=read(OUT/'predictions_locked.json')
    assert locked['protocol_sha256']==digest(OUT/'protocol.json')
    assert locked['models_lock_sha256']==digest(OUT/'models_locked.json')
    actual=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    results={}
    for day in [15,16]:
        _,rows,groups,_=layout(day)
        np.testing.assert_array_equal(rows,np.flatnonzero(actual.pt_d.to_numpy()==20191100+day))
        y=actual.label.to_numpy()[rows]
        assert len(y)=={15:424780,16:421983}[day] and int(y.sum())=={15:3523,16:3724}[day]
        values={name:np.load(OUT/(name+'_held_margin.npy')) for name in locked['dates'][str(day)]['forecasts_sha256']}
        best,winner,records=None,None,[]
        maximum_error=0.
        base=np.load(BASE/f'day{day}_prediction.npy')
        for record,p in candidates(day,values,groups):
            assert array_hash(p)==locked['dates'][str(day)]['candidates_sha256'][candidate_key(record)]
            if verifying and record['name']!='base':
                replay=core.within_prediction(base,values[record['name']],groups,record['weight'],record['mode'],True)
                error=float(np.max(np.abs(p-replay)))
                assert error<2e-14
                maximum_error=max(maximum_error,error)
            record=dict(record,auc=rank_sum_auc(y,p))
            records.append(record)
            if best is None or record['auc']>best['auc']:
                best,winner=record,p.copy()
        assert abs(roc_auc_score(y,winner)-best['auc'])<1e-12
        if verifying:
            np.testing.assert_array_equal(winner,np.load(OUT/f'day{day}_prediction.npy'))
            assert best==read(OUT/'result.json')['dates'][str(20191100+day)]['best']
        else:
            np.save(OUT/f'day{day}_prediction.npy',winner)
        results[str(20191100+day)]=dict(auc=best['auc'],best=best,records=records,rows=len(y),clicks=int(y.sum()),
            evaluation_day_labels_fit=False,held_labels_used_to_select_fixed_recipe=True,
            both_date_models_and_candidate_hashes_locked_before_scoring=True,
            inherited_features_independently_rebuilt_from_raw=True,new_query_groups_rebuilt_from_raw=verifying,
            selected_training_matrix_rebuilt_from_independent_cache=verifying,
            models_and_forecast_hashes_replayed_exactly=verifying,maximum_independent_pandas_error=maximum_error,
            target_met=best['auc']>={15:.815,16:.82}[day])
        emit('within_user_ranker_scored',day=day,best=best,verifying=verifying)
    check_sources()
    write(OUT/('verification.json' if verifying else 'result.json'),dict(status='passed' if verifying else 'evaluated',
        dates=results,original_datasets_unchanged=True,both_targets_met=all(r['target_met'] for r in results.values())))


def evaluate():
    check_sources()
    assert not (OUT/'predictions_locked.json').exists(),'Preserve locked candidates and evaluations'
    dates={}
    for day in [15,16]:
        values,rows,groups=forecasts(day)
        np.testing.assert_array_equal(rows,np.load(OUT/f'day{day}_indices.npy'))
        dates[str(day)]=dict(forecasts_sha256={name:array_hash(value) for name,value in values.items()},
            candidates_sha256={candidate_key(record):array_hash(p) for record,p in candidates(day,values,groups)})
        emit('within_user_ranker_candidates_locked',day=day,candidates=len(dates[str(day)]['candidates_sha256']))
    write(OUT/'predictions_locked.json',dict(protocol_sha256=digest(OUT/'protocol.json'),
        models_lock_sha256=digest(OUT/'models_locked.json'),dates=dates,held_labels_opened_for_scoring=False))
    score_locked(False)


def verify():
    check_sources()
    assert not (OUT/'verification.json').exists(),'Preserve completed independent verification'
    locked=read(OUT/'predictions_locked.json')
    assert locked['protocol_sha256']==digest(OUT/'protocol.json')
    assert locked['models_lock_sha256']==digest(OUT/'models_locked.json')
    for day in [15,16]:
        values,rows,groups=forecasts(day,True)
        np.testing.assert_array_equal(rows,np.load(OUT/f'day{day}_indices.npy'))
        for name,value in values.items():
            assert array_hash(value)==locked['dates'][str(day)]['forecasts_sha256'][name]
        for record,p in candidates(day,values,groups):
            assert array_hash(p)==locked['dates'][str(day)]['candidates_sha256'][candidate_key(record)]
    score_locked(True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=['fit','evaluate','verify'])
    {'fit':fit,'evaluate':evaluate,'verify':verify}[parser.parse_args().phase]()
