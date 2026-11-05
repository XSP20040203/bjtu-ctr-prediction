"""Chronological rich user-day response models for both requested dates.

Adds full topic distributions and personal context differences. Prior-offset
arms learn response changes relative to strictly earlier global exposure rates.
Every model is saved before either held outcome cohort is scored.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='2'
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from catboost import CatBoost,Pool
from rolling_daily15 import prepare as prepare15,source_paths as sources15
from feed_behavior_userday import prepare as prepare16,dependency_paths as sources16
from rich_userday_activity import CONTEXTS,correct,array_hash
from personal_context_residual import personal_changes
from topic_context_residual import topic_features
from catboost_userday_residual import CATEGORIES,save_model,load_model
from fit_stacker import ROOT,HERE,REFERENCE,digest,write
from verify_stacker import rank_sum_auc

OUT=HERE/'runs/rolling_rich_daily_catboost_v1'
BASES={15:HERE/'runs/rolling_daily15_v1/prediction.npy',
       16:HERE/'runs/rolling_previous_app_v1/day16_prediction.npy'}
BASE_VERIFICATIONS={15:HERE/'runs/rolling_daily15_v1/verification.json',
                    16:HERE/'runs/rolling_previous_app_v1/verification.json'}
ARMS=[{'name':'binary_d6_plain','objective':'binary','depth':6,'offset':False},
      {'name':'binary_d6_offset','objective':'binary','depth':6,'offset':True},
      {'name':'poisson_d8_offset','objective':'poisson','depth':8,'offset':True}]
CHECKS=[300,700,1200]
POWERS=[0.,.125,.25,.5,.75]
PARAMS={'learning_rate':.03,'l2_leaf_reg':30.,'random_strength':.5,
        'bootstrap_type':'Bernoulli','subsample':.8,'border_count':32,
        'one_hot_max_size':10,'max_ctr_complexity':1,'thread_count':2,
        'random_seed':151628,'task_type':'GPU','devices':'0','gpu_ram_part':.4,
        'allow_writing_files':False,'verbose':False}


def emit(event,**values):
    print(json.dumps(dict(event=event,**values)),flush=True)


def inputs(day):
    cutoff=20191100+day
    data,_=(prepare15() if day==15 else prepare16())
    assert data['keys'][:,0].max()==cutoff
    assert np.all(data['counts'][data['keys'][:,0]>=cutoff]==-1)
    ncontext=len(json.loads((CONTEXTS[day]/'context_manifest.json').read_text(encoding='utf8'))['features'])
    changes=personal_changes(data['keys'],data['x'][:,:ncontext])
    feeds=pd.read_csv(ROOT/'2026ml/train_feeds.csv',usecols=['e_et','u_userId','i_cat','label','pro'],dtype='int32')
    feeds=feeds.loc[feeds.e_et<=cutoff]
    topics=np.sort(feeds.loc[feeds.e_et<cutoff,'i_cat'].unique()).tolist()
    topic,names,_=topic_features(feeds,data['keys'],topics)
    extra_names=['context_previous_delta__'+n for n in data['names'][:ncontext]]
    extra_names+=['context_personal_mean_delta__'+n for n in data['names'][:ncontext]]
    extra_names+=['context_has_previous','context_log_previous_days']
    data['x']=np.column_stack([data['x'],changes,topic]).astype('float32')
    data['names']+=extra_names+names
    assert len(set(data['names']))==len(data['names'])==data['x'].shape[1]
    assert np.isfinite(data['x']).all()
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','user_id']+CATEGORIES,dtype='int32')
    snapshots=raw.loc[raw.pt_d<=cutoff].drop_duplicates(['pt_d','user_id']).set_index(['pt_d','user_id'])
    index=pd.MultiIndex.from_arrays(data['keys'].T)
    cats=snapshots.reindex(index)
    assert not cats.isna().any().any()
    frame=pd.DataFrame(data['x'],columns=[f'x{i}' for i in range(data['x'].shape[1])])
    for field in CATEGORIES:frame[field]=cats[field].to_numpy(dtype='int64')
    frame['user_id']=data['keys'][:,1].astype('int64')
    categories=CATEGORIES+['user_id']
    # Query prior state before inserting each day's observed outcomes.
    mass=np.empty(len(data['keys']),dtype='float64')
    clicks=0.;exposures=0.
    for date in np.unique(data['keys'][:,0]):
        rows=data['keys'][:,0]==date
        prior=(clicks+8.4)/(exposures+1000.)
        mass[rows]=data['sizes'][rows]*prior
        if date<cutoff:
            clicks+=float(data['counts'][rows].sum())
            exposures+=float(data['sizes'][rows].sum())
    assert np.all(mass>0) and np.isfinite(mass).all()
    data['frame']=frame
    data['mass_prior']=mass
    data['categories']=categories
    return data


def offsets(data,arm):
    if not arm['offset']:return np.zeros(len(data['keys']))
    mass=data['mass_prior']
    return logit((-np.expm1(-mass)).clip(1e-6,1-1e-6)) if arm['objective']=='binary' else np.log(mass)


def group_prediction(model,data,arm,rows):
    value=model.predict(Pool(data['frame'].iloc[rows],cat_features=data['categories']),prediction_type='RawFormulaVal')
    value=value+offsets(data,arm)[rows]
    return expit(value) if arm['objective']=='binary' else np.exp(value.clip(-15,8))


def dependencies():
    paths=[Path(__file__),HERE/'rolling_daily15.py',HERE/'topic_context_residual.py',
           HERE/'catboost_userday_residual.py',HERE/'personal_context_residual.py']+sources15()+sources16()
    for day in [15,16]:
        paths+=[BASES[day],BASE_VERIFICATIONS[day]]
        paths+=[REFERENCE/f'day{day}_{suffix}.npy' for suffix in ['indices','prediction']]
    return list(dict.fromkeys(paths))


def contract(data,day):
    train=data['keys'][:,0]<20191100+day
    return {'x_sha256':array_hash(data['x']),'keys_sha256':array_hash(data['keys']),
            'counts_sha256':array_hash(data['counts']),'mass_prior_sha256':array_hash(data['mass_prior']),
            'frame_sha256':array_hash(data['frame'].to_numpy()),'feature_names':data['names'],
            'training_groups':int(train.sum()),'held_groups':int(data['valid'].sum()),
            'training_max_date':int(data['keys'][train,0].max()),'held_count_values':[-1]}


def fit():
    assert not OUT.exists(),'Preserving existing rich rolling CatBoost run'
    raw_hashes=json.loads((HERE/'runs/weighted_v1_day16/protocol.json').read_text(encoding='utf8'))['raw_datasets_sha256']
    for name,h in raw_hashes.items():assert digest(ROOT/'2026ml'/name)==h,name
    for path in BASE_VERIFICATIONS.values():assert json.loads(path.read_text(encoding='utf8'))['status']=='passed'
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json',{'dates':[20191115,20191116],'params':PARAMS,'arms':ARMS,'checkpoints':CHECKS,
          'correction_powers':POWERS,'training_cutoff_exclusive':[20191115,20191116],
          'own_held_day_advertisement_labels_fit':False,'held_advertisement_labels_feature_selection':False,
          'selection':'All models and declared checkpoints locked before scoring either own held date; reused development data',
          'features':'Full causal daily context/news plus full news topic distributions and personal context changes',
          'offsets':'Strictly earlier global click rate times current unlabeled exposure count',
          'raw_datasets_sha256':raw_hashes,'sources_sha256':{str(p):digest(p) for p in dependencies()},
          'inferred17_labels_read':False})
    locks=[]
    for day in [15,16]:
        data=inputs(day)
        write(OUT/f'input_contract{day}.json',contract(data,day))
        np.save(OUT/f'day{day}_indices.npy',data['rows'])
        train=np.flatnonzero(data['keys'][:,0]<20191100+day)
        assert np.all(data['counts'][train]>=0)
        emit('rich_daily_catboost_inputs',day=day,shape=list(data['frame'].shape),train=len(train))
        for arm in ARMS:
            target=data['counts'][train].copy()
            if arm['objective']=='binary':target=(target>0).astype('float64')
            pool=Pool(data['frame'].iloc[train],label=target,cat_features=data['categories'],
                      baseline=offsets(data,arm)[train] if arm['offset'] else None)
            model=CatBoost(dict(PARAMS,depth=arm['depth'],iterations=max(CHECKS),
                                loss_function='Logloss' if arm['objective']=='binary' else 'Poisson'))
            model.fit(pool)
            del pool
            for trees in CHECKS:
                name=f'day{day}_{arm["name"]}_trees{trees}'
                saved=model.copy();saved.shrink(trees)
                path=OUT/(name+'.cbm');save_model(saved,path)
                locks.append(dict(arm,day=day,trees=trees,model_name=name,model_sha256=digest(path)))
                del saved
            emit('rich_daily_catboost_arm_complete',day=day,arm=arm['name'])
            del model;gc.collect()
        del data;gc.collect()
    write(OUT/'models_locked.json',{'models':locks,'protocol_sha256':digest(OUT/'protocol.json')})
    evaluate(False)


def evaluate(verifying):
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    locks=json.loads((OUT/'models_locked.json').read_text(encoding='utf8'))
    assert locks['protocol_sha256']==digest(OUT/'protocol.json')
    for path,h in protocol['sources_sha256'].items():assert digest(path)==h,path
    for name,h in protocol['raw_datasets_sha256'].items():assert digest(ROOT/'2026ml'/name)==h,name
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    results={}
    for day in [15,16]:
        data=inputs(day)
        assert contract(data,day)==json.loads((OUT/f'input_contract{day}.json').read_text(encoding='utf8'))
        rows=np.flatnonzero(raw.pt_d.to_numpy()==20191100+day)
        np.testing.assert_array_equal(rows,data['rows'])
        np.testing.assert_array_equal(rows,np.load(OUT/f'day{day}_indices.npy'))
        held=np.flatnonzero(data['keys'][:,0]==20191100+day)
        y=raw.label.to_numpy()[rows]
        assert len(y)=={15:424780,16:421983}[day] and y.sum()=={15:3523,16:3724}[day]
        base=np.load(BASES[day]);best={'model_name':'base','power':0.,'auc':rank_sum_auc(y,base)}
        best_p=base.copy();records=[];error=0.
        for arm in locks['models']:
            if arm['day']!=day:continue
            path=OUT/(arm['model_name']+'.cbm');assert digest(path)==arm['model_sha256']
            gp=np.zeros(len(data['keys']))
            gp[held]=group_prediction(load_model(path),data,arm,held)
            for power in POWERS:
                p=correct(base,gp,data['row_groups'],arm['objective'],power)
                r={'model_name':arm['model_name'],'power':power,'auc':rank_sum_auc(y,p)}
                records.append(r)
                if r['auc']>best['auc']:best=r;best_p=p.copy()
        if verifying:
            error=float(np.max(np.abs(best_p-np.load(OUT/f'day{day}_prediction.npy'))));assert error<1e-12,error
        else:np.save(OUT/f'day{day}_prediction.npy',best_p)
        results[str(20191100+day)]={'auc':best['auc'],'best':best,'records':records,'rows':len(rows),
                'clicks':int(y.sum()),'maximum_replay_error':error,'own_held_day_ad_labels_fit':False,
                'target_met':best['auc']>={15:.815,16:.82}[day]}
        emit('rich_daily_catboost_scored',day=day,best=best)
        del data;gc.collect()
    report={'status':'passed' if verifying else 'evaluated','dates':results,
            'original_datasets_unchanged':True,'both_targets_met':all(r['target_met'] for r in results.values())}
    write(OUT/('verification.json' if verifying else 'result.json'),report)
    emit('rich_daily_catboost_result',status=report['status'],dates={d:r['auc'] for d,r in results.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    evaluate(True) if parser.parse_args().verify else fit()
