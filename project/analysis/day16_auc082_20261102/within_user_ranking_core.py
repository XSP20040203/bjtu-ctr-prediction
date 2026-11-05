"""Causal query grouping and row-ranking arithmetic for the next full-cohort fits."""
from pathlib import Path
import gc
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from fit_stacker import HERE,ROOT,digest,write
from conditional_daily import read,emit
from rich_userday_activity import array_hash,lgb
import global_row_context as rows16
import global_row_context15 as rows15

DENSE=HERE/'runs/dense_context_rows_v1'
SOURCE=HERE/'runs/rich_augmented_rows_v1'
PREFLIGHT=HERE/'runs/within_user_ranking_preflight_v1'
PARAMS=dict(learning_rate=.025,num_leaves=31,min_data_in_leaf=200,lambda_l2=40.,
    cat_l2=100.,cat_smooth=100.,max_bin=63,feature_fraction=.8,
    num_threads=3,seed=151644,verbosity=-1,deterministic=True,force_col_wise=True,
    feature_pre_filter=False,metric='None',label_gain=[0,1],
    lambdarank_truncation_level=1000,lambdarank_norm=True)


def ranking_order(rows,dates,users,labels,cutoff):
    """Keep complete mixed-label queries within the given prior training sample."""
    rows=np.asarray(rows,dtype='int64')
    dates,users,labels=map(np.asarray,(dates,users,labels))
    if not (len(dates)==len(users)==len(labels)) or np.any(labels[dates>=cutoff]!=-1):
        raise ValueError('Held and future advertisement labels must be masked')
    prior=np.flatnonzero(dates[rows]<cutoff)
    if not np.isin(labels[rows[prior]],[0,1]).all():
        raise ValueError('Prior query targets must be observed binary labels')
    order=prior[np.lexsort((rows[prior],users[rows[prior]],dates[rows[prior]]))]
    keys=np.column_stack([dates[rows[order]],users[rows[order]]])
    _,groups=np.unique(keys,axis=0,return_inverse=True)
    sizes=np.bincount(groups)
    positives=np.bincount(groups,weights=labels[rows[order]])
    selected=(positives>0)&(positives<sizes)
    order=order[selected[groups]]
    keys=np.column_stack([dates[rows[order]],users[rows[order]]])
    query_keys,groups=np.unique(keys,axis=0,return_inverse=True)
    sizes=np.bincount(groups).astype('int32')
    target=labels[rows[order]].astype('int32')
    if not len(sizes) or int(sizes.sum())!=len(order) or np.any(np.diff(groups)<0):
        raise ValueError('Nonempty, contiguous, complete mixed-label queries are required')
    sums=np.bincount(groups,weights=target)
    assert np.all((sums>0)&(sums<sizes)) and np.all(query_keys[:,0]<cutoff)
    return order,sizes,target,query_keys


def within_prediction(base,margin,groups,weight,mode,independent=False):
    if mode not in ['add_within','replace_within']:
        raise ValueError('Undeclared within-user ranking formula')
    z=logit(np.clip(np.asarray(base,dtype='float64'),1e-8,1-1e-8))
    margin=np.asarray(margin,dtype='float64')
    if independent:
        table=pd.DataFrame(dict(group=groups,z=z,margin=margin))
        mean_z=table.groupby('group',sort=False).z.transform('mean').to_numpy()
        mean_margin=table.groupby('group',sort=False).margin.transform('mean').to_numpy()
    else:
        sizes=np.bincount(groups).clip(1)
        mean_z=(np.bincount(groups,weights=z)/sizes)[groups]
        mean_margin=(np.bincount(groups,weights=margin)/sizes)[groups]
    existing=z-mean_z
    ranked=margin-mean_margin
    ranked=ranked*existing.std()/max(ranked.std(),1e-6)
    delta=ranked if mode=='add_within' else ranked-existing
    return expit(np.clip(z+weight*delta,-35,20))


def prior_input(day,verifying=False):
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','user_id','label'],dtype='int32')
    dates,users,labels=(raw[name].to_numpy(copy=True) for name in ['pt_d','user_id','label'])
    cutoff=20191100+day
    labels[dates>=cutoff]=-1
    del raw
    chosen,weights=(rows15 if day==15 else rows16).selected_training(labels,dates)
    held=np.flatnonzero(dates==cutoff)
    rows=np.sort(np.r_[chosen,held])
    source=read(SOURCE/f'input_contract{day}.json')
    assert array_hash(chosen)==source['training_rows_sha256']
    assert array_hash(labels[chosen])==source['training_labels_sha256']
    assert array_hash(weights)==source['weights_sha256']
    contract=read(DENSE/f'contract_day{day}_context_numeric.json')
    directory=DENSE/('independent' if verifying else 'input')/f'day{day}_context_numeric'
    assert digest(directory/'features.npy')==contract['matrix_sha256']
    x=np.load(directory/'features.npy',mmap_mode='r')
    assert len(x)==len(rows) and contract['max_training_date']<cutoff
    order,sizes,target,keys=ranking_order(rows,dates,users,labels,cutoff)
    audit=dict(date=cutoff,source_matrix_sha256=contract['matrix_sha256'],
        feature_names=contract['feature_names'],categorical_columns=contract['categorical_columns'],
        training_order_sha256=array_hash(order),query_sizes_sha256=array_hash(sizes),
        training_targets_sha256=array_hash(target),query_keys_sha256=array_hash(keys),
        max_training_date=int(keys[:,0].max()),training_rows=len(order),queries=len(sizes),
        sampling='Inherited all earlier positives and fixed one-third earlier negatives',
        mixed_prior_queries_only=True,held_advertisement_labels_fit=False)
    return x,order,sizes,target,keys,audit


def preflight():
    assert not PREFLIGHT.exists(),'Preserve prior ranking preflights and failures'
    assert read(DENSE/'verification.json')['status']=='passed'
    hashes=read(DENSE/'protocol.json')['raw_datasets_sha256']
    for name,expected in hashes.items():
        assert digest(ROOT/'2026ml'/name)==expected,name
    x,order,sizes,target,keys,audit=prior_input(15)
    selected_queries=min(400,len(sizes))
    count=int(sizes[:selected_queries].sum())
    # Select complete queries within the frozen prior sample; no held target is used.
    values=np.array(x[order[:count]],dtype='float32',copy=True)
    del x
    paths=[Path(__file__),HERE/'test_within_user_ranking_core.py',HERE/'global_row_context.py',
        HERE/'global_row_context15.py',HERE/'fit_stacker.py',HERE/'conditional_daily.py',
        HERE/'rich_userday_activity.py',SOURCE/'input_contract15.json',
        DENSE/'verification.json',DENSE/'protocol.json',
        DENSE/'contract_day15_context_numeric.json',DENSE/'input/day15_context_numeric/features.npy']
    PREFLIGHT.mkdir()
    write(PREFLIGHT/'protocol.json',dict(params=PARAMS,objectives=['lambdarank','rank_xendcg'],
        date=20191115,source_contract=audit,raw_datasets_sha256=hashes,
        sources_sha256={str(p):digest(p) for p in paths},inferred17_labels_read=False))
    results=[]
    for objective in ['lambdarank','rank_xendcg']:
        params=dict(PARAMS,objective=objective,min_data_in_leaf=20)
        dataset=lgb.Dataset(values,label=target[:count],group=sizes[:selected_queries],
            categorical_feature=audit['categorical_columns'],params=params,free_raw_data=True)
        model=lgb.train(params,dataset,num_boost_round=30)
        path=PREFLIGHT/(objective+'.txt')
        path.write_text(model.model_to_string(),encoding='utf8')
        restored=lgb.Booster(model_str=path.read_text(encoding='utf8'))
        p=model.predict(values,raw_score=True,num_threads=3)
        replay=restored.predict(values,raw_score=True,num_threads=3)
        error=float(np.max(np.abs(p-replay)))
        assert error==0. and np.isfinite(p).all()
        results.append(dict(objective=objective,model_sha256=digest(path),maximum_serialization_error=error))
        del dataset,model,restored
        gc.collect()
    report=dict(status='passed',rows=count,queries=selected_queries,features=values.shape[1],
        max_training_date=int(keys[:selected_queries,0].max()),label_cutoff_exclusive=20191115,
        queries_complete_within_frozen_prior_sample=True,results=results,full_date_models_fitted=False,
        held_auc_evaluated=False,original_datasets_unchanged=True,lightgbm_version=lgb.__version__)
    for name,expected in hashes.items():
        assert digest(ROOT/'2026ml'/name)==expected,name
    write(PREFLIGHT/'verification.json',report)
    emit('within_user_ranking_preflight_passed',**report)


if __name__=='__main__':
    preflight()
