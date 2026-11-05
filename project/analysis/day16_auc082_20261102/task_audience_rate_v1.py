"""Predict task/day CTR from its full audience, using strictly earlier ad labels."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='2'
import argparse
import gc
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from catboost import CatBoost,Pool
from sklearn.metrics import roc_auc_score
from task_audience_core import FIELDS,build
from news_quality_features import READ,build_quality
from fit_stacker import HERE,ROOT,digest,write
from conditional_daily import read,emit
from rich_row_catboost import frame_hash
from rich_userday_activity import array_hash
from catboost_userday_residual import save_model,load_model
from verify_stacker import rank_sum_auc

OUT=HERE/'runs/task_audience_rate_v1'
PREFLIGHT=HERE/'runs/task_audience_preflight_v1'
QUALITY=HERE/'runs/news_quality_daily_v1'
BASE={15:HERE/'runs/firstday_task_ranker_v1',16:HERE/'runs/cold_task_ranker_v1'}
ARMS=[dict(name='audience_d6',depth=6),dict(name='audience_d8',depth=8)]
CHECKS=[200,600,1600]
WEIGHTS=[.025,.05,.1,.2,.35,.5]
MODES=['replace_task','add_task']
PARAMS=dict(loss_function='CrossEntropy',learning_rate=.03,l2_leaf_reg=100.,
    random_strength=1.,bootstrap_type='Bernoulli',subsample=.8,border_count=32,
    one_hot_max_size=16,max_ctr_complexity=1,thread_count=2,random_seed=151649,
    task_type='GPU',devices='0',gpu_ram_part=.5,allow_writing_files=False,verbose=200)


def dependencies():
    paths=[Path(__file__),HERE/'task_audience_core.py',HERE/'test_task_audience_core.py',
        HERE/'run_task_audience_rate_v1.ps1',HERE/'news_quality_features.py',
        HERE/'fit_stacker.py',HERE/'conditional_daily.py',HERE/'rich_row_catboost.py',
        HERE/'rich_userday_activity.py',HERE/'catboost_userday_residual.py',HERE/'verify_stacker.py']
    assert read(QUALITY/'verification.json')['status']=='passed'
    paths += [QUALITY/'verification.json',QUALITY/'protocol.json']
    for day in [15,16]:
        assert read(BASE[day]/'verification.json')['status']=='passed'
        assert not read(BASE[day]/'verification.json')['dates'][str(20191100+day)]['evaluation_day_labels_fit']
        paths += [QUALITY/f'quality{day}.npz',QUALITY/f'input_contract{day}.json',
            BASE[day]/'protocol.json',BASE[day]/'verification.json',
            BASE[day]/f'day{day}_indices.npy',BASE[day]/f'day{day}_prediction.npy']
    return list(dict.fromkeys(paths))


def check(directory=OUT):
    protocol=read(directory/'protocol.json')
    for filename,expected in protocol['sources_sha256'].items():
        assert digest(filename)==expected,filename
    for filename,expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/filename)==expected,filename
    return protocol


def inputs(day,independent=False):
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=FIELDS+['label'],dtype='int32')
    labels=raw.label.to_numpy(copy=True)
    cutoff=20191100+day
    labels[raw.pt_d.to_numpy()>=cutoff]=-1
    quality=np.load(QUALITY/f'quality{day}.npz')
    values,keys=quality['values'],quality['keys']
    original=read(QUALITY/f'input_contract{day}.json')
    assert array_hash(values)==original['quality_sha256']
    if independent:
        news=pd.read_csv(ROOT/'2026ml/train_feeds.csv',usecols=READ,dtype={'i_docId':str,'i_s_sourceId':str})
        values,names=build_quality(news,keys,cutoff)
        np.testing.assert_array_equal(values,quality['values'])
        assert names==original['quality_names']
        del news
    frame,cats,meta=build(raw[FIELDS],labels,keys,values,cutoff,'sparse' if independent else 'bincount')
    assert values.shape[1]==337
    np.testing.assert_array_equal(meta['rows'],np.load(BASE[day]/f'day{day}_indices.npy'))
    assert np.all(meta['counts'][meta['keys'][:,0]>=cutoff]==-1)
    contract=dict(day=cutoff,shape=list(frame.shape),feature_names=list(frame),categories=cats,
        frame_sha256=frame_hash(frame),metadata_sha256={k:array_hash(v) for k,v in meta.items()},
        maximum_training_date=int(meta['keys'][meta['keys'][:,0]<cutoff,0].max()),
        full_advertisement_population_used=True,training_subsampling=False,
        held_counts_masked=True,all337_quality_inputs_aggregated=True)
    emit('task_audience_input_ready',day=day,independent=independent,shape=contract['shape'])
    gc.collect()
    return frame,meta,contract


def save_inputs(folder,frame,meta,contract):
    assert not folder.exists()
    folder.mkdir()
    frame.to_pickle(folder/'frame.pkl')
    np.savez(folder/'metadata.npz',**meta)
    write(folder/'contract.json',contract)
    write(folder/'files_locked.json',{name:digest(folder/name) for name in ['frame.pkl','metadata.npz','contract.json']})


def load_inputs(folder):
    for filename,expected in read(folder/'files_locked.json').items():
        assert digest(folder/filename)==expected
    frame=pd.read_pickle(folder/'frame.pkl')
    saved=np.load(folder/'metadata.npz')
    meta={k:saved[k] for k in saved.files}
    contract=read(folder/'contract.json')
    assert frame_hash(frame)==contract['frame_sha256']
    assert {k:array_hash(v) for k,v in meta.items()}==contract['metadata_sha256']
    return frame,meta,contract


def training_pool(frame,meta,contract):
    indices=np.flatnonzero(meta['keys'][:,0]<contract['day'])
    counts=meta['counts'][indices]
    sizes=meta['sizes'][indices]
    assert np.all((counts>=0)&(counts<=sizes))
    return Pool(frame.iloc[indices],label=counts/sizes,weight=sizes,
                cat_features=contract['categories'])


def preflight():
    assert not PREFLIGHT.exists() and not OUT.exists(),'Preserve previous experiments'
    paths=dependencies()
    PREFLIGHT.mkdir()
    write(PREFLIGHT/'protocol.json',dict(params=PARAMS,arms=ARMS,
        sources_sha256={str(p):digest(p) for p in paths},
        raw_datasets_sha256=read(QUALITY/'protocol.json')['raw_datasets_sha256']))
    check(PREFLIGHT)
    frame,meta,contract=inputs(16)
    save_inputs(PREFLIGHT/'input_day16',frame,meta,contract)
    other,other_meta,other_contract=inputs(16,True)
    assert contract==other_contract
    pd.testing.assert_frame_equal(frame,other)
    for name in meta:np.testing.assert_array_equal(meta[name],other_meta[name])
    del other,other_meta
    gc.collect()
    pool=training_pool(frame,meta,contract)
    sample=Pool(frame.iloc[:1000],cat_features=contract['categories'])
    results=[]
    for arm in ARMS:
        model=CatBoost(dict(PARAMS,depth=arm['depth'],iterations=20,verbose=False))
        model.fit(pool)
        before=model.predict(sample,prediction_type='RawFormulaVal',thread_count=2)
        path=PREFLIGHT/(arm['name']+'.cbm')
        save_model(model,path)
        after=load_model(path).predict(sample,prediction_type='RawFormulaVal',thread_count=2)
        np.testing.assert_array_equal(before,after)
        results.append(dict(arm,serialization_maximum_error=0.))
    check(PREFLIGHT)
    write(PREFLIGHT/'verification.json',dict(status='passed',shape=contract['shape'],
        source_code_sha256=digest(__file__),results=results,
        full_prior_cohort_capacity_verified=True,sparse_independent_feature_replay_exact=True,
        held_labels_evaluated=False,inferred17_labels_read=False))
    emit('task_audience_preflight_passed',shape=contract['shape'])


def fit():
    assert not OUT.exists(),'Preserve previous full fits'
    smoke=read(PREFLIGHT/'verification.json')
    assert smoke['status']=='passed' and smoke['source_code_sha256']==digest(__file__)
    parent=check(PREFLIGHT)
    paths=dependencies()+[PREFLIGHT/'protocol.json',PREFLIGHT/'verification.json',PREFLIGHT/'input_day16/files_locked.json']
    OUT.mkdir()
    write(OUT/'protocol.json',dict(params=PARAMS,arms=ARMS,checkpoints=CHECKS,weights=WEIGHTS,modes=MODES,
        objective='Full-population task/day exposure-weighted binomial CTR',
        features='Current task audience quality means and variances, demographics, strictly earlier user/task/advertiser/app response',
        selection='Both-date models and all candidates locked before scoring; repeatedly reused development dates',
        dates=[20191115,20191116],inferred17_labels_read=False,own_held_ad_labels_fit=False,
        sources_sha256={str(p):digest(p) for p in paths},raw_datasets_sha256=parent['raw_datasets_sha256']))
    models=[]
    for day in [15,16]:
        if day==16:
            shutil.copytree(PREFLIGHT/'input_day16',OUT/'input_day16')
            frame,meta,contract=load_inputs(OUT/'input_day16')
        else:
            frame,meta,contract=inputs(day)
            save_inputs(OUT/f'input_day{day}',frame,meta,contract)
        np.save(OUT/f'day{day}_indices.npy',meta['rows'])
        pool=training_pool(frame,meta,contract)
        for arm in ARMS:
            emit('task_audience_fit_started',day=day,arm=arm['name'])
            model=CatBoost(dict(PARAMS,depth=arm['depth'],iterations=max(CHECKS)))
            model.fit(pool)
            for trees in CHECKS:
                name=f'day{day}_{arm["name"]}_trees{trees}'
                saved=model.copy();saved.shrink(trees)
                path=OUT/(name+'.cbm');save_model(saved,path)
                models.append(dict(arm,day=day,trees=trees,name=name,model_sha256=digest(path),
                    input_contract_sha256=digest(OUT/f'input_day{day}/contract.json')))
                del saved
            del model
            gc.collect()
        del frame,pool,meta
        gc.collect()
    write(OUT/'models_locked.json',dict(protocol_sha256=digest(OUT/'protocol.json'),models=models))
    check()


def margins(day,frame,meta,contract,independent=False):
    result={}
    held=meta['held_groups']
    pool=Pool(frame.iloc[held],cat_features=contract['categories'])
    locked=read(OUT/'models_locked.json')
    assert locked['protocol_sha256']==digest(OUT/'protocol.json')
    for record in locked['models']:
        if record['day']!=day:continue
        path=OUT/(record['name']+'.cbm')
        assert digest(path)==record['model_sha256']
        assert digest(OUT/f'input_day{day}/contract.json')==record['input_contract_sha256']
        value=load_model(path).predict(pool,prediction_type='RawFormulaVal',thread_count=2)
        assert value.shape==held.shape and np.isfinite(value).all()
        if independent:
            np.testing.assert_array_equal(value,np.load(OUT/(record['name']+'_margin.npy')))
        else:np.save(OUT/(record['name']+'_margin.npy'),value)
        result[record['name']]=value
    return result


def prediction(base,margin,meta,weight,mode,independent=False):
    if not weight:return base.copy()
    groups=meta['row_groups'];sizes=meta['sizes'][meta['held_groups']]
    mean=np.bincount(groups,weights=base,minlength=len(sizes))/sizes
    if mode=='replace_task':delta=margin-logit(np.clip(mean,1e-8,1-1e-8))
    else:delta=margin-np.average(margin,weights=sizes)
    delta=np.clip(delta,-3,3)
    if independent:
        odds=base/(1-base)*np.exp(weight*delta[groups])
        return odds/(1+odds)
    return expit(logit(base)+weight*delta[groups])


def candidates(day,values,meta,independent=False):
    base=np.load(BASE[day]/f'day{day}_prediction.npy')
    yield dict(name='base',mode='base',weight=0.),base.copy()
    for name,margin in values.items():
        for mode in MODES:
            for weight in WEIGHTS:
                record=dict(name=name,mode=mode,weight=weight)
                yield record,prediction(base,margin,meta,weight,mode,independent)


def candidate_key(record):return f'{record["name"]}_{record["mode"]}_{record["weight"]:g}'


def evaluate():
    check()
    locked={}
    for day in [15,16]:
        frame,meta,contract=load_inputs(OUT/f'input_day{day}')
        values=margins(day,frame,meta,contract)
        locked[str(day)]={candidate_key(r):array_hash(p) for r,p in candidates(day,values,meta)}
    write(OUT/'predictions_locked.json',dict(protocol_sha256=digest(OUT/'protocol.json'),
        candidates_sha256=locked,all_models_locked_before_scoring=True,held_labels_read=False))
    emit('task_audience_candidates_locked')


def verify():
    check()
    assert not (OUT/'verification.json').exists()
    locked=read(OUT/'predictions_locked.json')
    assert locked['protocol_sha256']==digest(OUT/'protocol.json')
    predictions={}
    for day in [15,16]:
        frame,meta,contract=inputs(day,True)
        assert contract==read(OUT/f'input_day{day}/contract.json')
        values=margins(day,frame,meta,contract,True)
        normal=list(candidates(day,values,meta))
        error=0.
        for (record,p),(_,q) in zip(normal,candidates(day,values,meta,True)):
            assert array_hash(p)==locked['candidates_sha256'][str(day)][candidate_key(record)]
            error=max(error,float(np.max(np.abs(p-q))))
        assert error<2e-14
        predictions[day]=(normal,meta['rows'],error)
        del frame,values,meta
        gc.collect()
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    reports={}
    for day in [15,16]:
        normal,rows,error=predictions[day]
        np.testing.assert_array_equal(rows,np.flatnonzero(raw.pt_d.to_numpy()==20191100+day))
        y=raw.label.to_numpy()[rows]
        assert len(y)=={15:424780,16:421983}[day] and y.sum()=={15:3523,16:3724}[day]
        records=[dict(r,auc=rank_sum_auc(y,p)) for r,p in normal]
        best=max(records,key=lambda r:r['auc'])
        p=next(p for r,p in normal if candidate_key(r)==candidate_key(best))
        assert abs(roc_auc_score(y,p)-best['auc'])<1e-12
        np.save(OUT/f'day{day}_prediction.npy',p)
        reports[str(20191100+day)]=dict(auc=best['auc'],best=best,records=records,
            rows=len(y),clicks=int(y.sum()),evaluation_day_labels_fit=False,
            held_labels_used_to_select_fixed_recipe=True,independent_sparse_and_raw_news_replay=True,
            all12_checkpoint_predictions_replayed_exactly=True,maximum_independent_prediction_error=error,
            target_met=best['auc']>={15:.815,16:.82}[day])
        emit('task_audience_scored',day=day,best=best)
    check()
    write(OUT/'verification.json',dict(status='passed',dates=reports,original_datasets_unchanged=True,
        inferred17_labels_read=False,both_targets_met=all(r['target_met'] for r in reports.values())))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=['preflight','fit','evaluate','verify'])
    dict(preflight=preflight,fit=fit,evaluate=evaluate,verify=verify)[parser.parse_args().phase]()
