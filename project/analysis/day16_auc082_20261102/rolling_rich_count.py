"""Fit full rich daily count models with two GPU memory configurations."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoost,Pool
from rolling_rich_daily_catboost import inputs,contract,offsets,group_prediction,dependencies
from catboost_userday_residual import save_model,load_model
from rich_userday_activity import correct
from fit_stacker import ROOT,HERE,digest,write
from verify_stacker import rank_sum_auc

OUT=HERE/'runs/rolling_rich_count_v1'
BASES={15:HERE/'runs/catboost_global_rows15_v1/prediction.npy',
       16:HERE/'runs/rolling_previous_app_v1/day16_prediction.npy'}
VERIFICATIONS={15:HERE/'runs/catboost_global_rows15_v1/verification.json',
               16:HERE/'runs/rolling_previous_app_v1/verification.json'}
ARMS=[{'name':'poisson_d6_bins32','depth':6,'border_count':32,'objective':'poisson','offset':True},
      {'name':'poisson_d8_bins8','depth':8,'border_count':8,'objective':'poisson','offset':True}]
CHECKS=[300,700,1200]
POWERS=[0.,.125,.25,.5,.75]
PARAMS={'loss_function':'Poisson','learning_rate':.03,'l2_leaf_reg':30.,'random_strength':.5,
        'bootstrap_type':'Bernoulli','subsample':.8,'one_hot_max_size':10,'max_ctr_complexity':1,
        'thread_count':2,'random_seed':151629,'task_type':'GPU','devices':'0',
        'gpu_ram_part':.65,'allow_writing_files':False,'verbose':False}


def emit(event,**values):print(json.dumps(dict(event=event,**values)),flush=True)


def fit():
    assert not OUT.exists(),'Preserving rich count experiment'
    raw_hashes=json.loads((HERE/'runs/weighted_v1_day16/protocol.json').read_text(encoding='utf8'))['raw_datasets_sha256']
    for name,h in raw_hashes.items():assert digest(ROOT/'2026ml'/name)==h,name
    for p in VERIFICATIONS.values():assert json.loads(p.read_text(encoding='utf8'))['status']=='passed'
    paths=list(dict.fromkeys([Path(__file__),HERE/'rolling_rich_daily_catboost.py']+dependencies()
                              +list(BASES.values())+list(VERIFICATIONS.values())))
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json',{'dates':[20191115,20191116],'params':PARAMS,'arms':ARMS,
          'checkpoints':CHECKS,'powers':POWERS,'training_cutoffs_exclusive':[20191115,20191116],
          'own_held_day_ad_labels_fit':False,'held_ad_labels_feature_selection':False,
          'features':'Same full 5007+ rich daily columns; no numerical column removal',
          'memory_change':'Depth6 retains32 bins; depth8 uses8 bins to reduce GPU histogram allocation',
          'selection':'All saved models lock before held scoring; repeatedly reused development dates',
          'raw_datasets_sha256':raw_hashes,'sources_sha256':{str(p):digest(p) for p in paths},
          'inferred17_labels_read':False})
    locks=[]
    for day in [15,16]:
        data=inputs(day)
        write(OUT/f'input_contract{day}.json',contract(data,day))
        np.save(OUT/f'day{day}_indices.npy',data['rows'])
        train=np.flatnonzero(data['keys'][:,0]<20191100+day)
        assert np.all(data['counts'][train]>=0)
        emit('rich_count_ready',day=day,shape=list(data['frame'].shape),train=len(train))
        for arm in ARMS:
            pool=Pool(data['frame'].iloc[train],label=data['counts'][train],
                      cat_features=data['categories'],baseline=offsets(data,arm)[train])
            model=CatBoost(dict(PARAMS,depth=arm['depth'],border_count=arm['border_count'],iterations=max(CHECKS)))
            model.fit(pool)
            del pool
            for trees in CHECKS:
                name=f'day{day}_{arm["name"]}_trees{trees}'
                saved=model.copy();saved.shrink(trees)
                path=OUT/(name+'.cbm');save_model(saved,path)
                locks.append(dict(arm,day=day,trees=trees,model_name=name,model_sha256=digest(path)))
                del saved
            emit('rich_count_arm_complete',day=day,arm=arm['name'])
            del model;gc.collect()
        del data;gc.collect()
    write(OUT/'models_locked.json',{'models':locks,'protocol_sha256':digest(OUT/'protocol.json')})
    evaluate(False)


def evaluate(verifying):
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    locked=json.loads((OUT/'models_locked.json').read_text(encoding='utf8'))
    assert locked['protocol_sha256']==digest(OUT/'protocol.json')
    for p,h in protocol['sources_sha256'].items():assert digest(p)==h,p
    for name,h in protocol['raw_datasets_sha256'].items():assert digest(ROOT/'2026ml'/name)==h,name
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    results={}
    for day in [15,16]:
        data=inputs(day)
        assert contract(data,day)==json.loads((OUT/f'input_contract{day}.json').read_text(encoding='utf8'))
        rows=np.flatnonzero(raw.pt_d.to_numpy()==20191100+day)
        np.testing.assert_array_equal(rows,data['rows'])
        np.testing.assert_array_equal(rows,np.load(OUT/f'day{day}_indices.npy'))
        y=raw.label.to_numpy()[rows]
        assert len(y)=={15:424780,16:421983}[day] and y.sum()=={15:3523,16:3724}[day]
        held=np.flatnonzero(data['valid'])
        base=np.load(BASES[day]);best={'model_name':'base','power':0.,'auc':rank_sum_auc(y,base)}
        best_p=base.copy();records=[]
        for arm in locked['models']:
            if arm['day']!=day:continue
            path=OUT/(arm['model_name']+'.cbm');assert digest(path)==arm['model_sha256']
            gp=np.zeros(len(data['keys']));gp[held]=group_prediction(load_model(path),data,arm,held)
            for power in POWERS:
                p=correct(base,gp,data['row_groups'],'poisson',power)
                r={'model_name':arm['model_name'],'power':power,'auc':rank_sum_auc(y,p)}
                records.append(r)
                if r['auc']>best['auc']:best=r;best_p=p.copy()
        error=0.
        if verifying:
            error=float(np.max(np.abs(best_p-np.load(OUT/f'day{day}_prediction.npy'))));assert error<1e-12,error
        else:np.save(OUT/f'day{day}_prediction.npy',best_p)
        results[str(20191100+day)]={'auc':best['auc'],'best':best,'records':records,'rows':len(rows),
              'clicks':int(y.sum()),'maximum_replay_error':error,'own_held_day_ad_labels_fit':False,
              'target_met':best['auc']>={15:.815,16:.82}[day]}
        emit('rich_count_scored',day=day,best=best)
        del data;gc.collect()
    report={'status':'passed' if verifying else 'evaluated','dates':results,'original_datasets_unchanged':True,
            'both_targets_met':all(r['target_met'] for r in results.values())}
    write(OUT/('verification.json' if verifying else 'result.json'),report)
    emit('rich_count_result',status=report['status'],dates={d:r['auc'] for d,r in results.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    evaluate(True) if parser.parse_args().verify else fit()
