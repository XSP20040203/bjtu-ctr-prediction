"""Full-context prior-offset binary daily model on CPU for complete16."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import argparse
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

OUT=HERE/'runs/rich_binary16_cpu_v1'
BASE=HERE/'runs/rolling_previous_app_v1'
ARM={'objective':'binary','offset':True}
CHECKS=[300,700,1200]
POWERS=[0.,.125,.25,.5,.75]
PARAMS={'loss_function':'Logloss','learning_rate':.03,'depth':6,'l2_leaf_reg':30.,
        'random_strength':.5,'bootstrap_type':'Bernoulli','subsample':.8,'rsm':.7,
        'border_count':32,'one_hot_max_size':10,'max_ctr_complexity':1,'thread_count':4,
        'random_seed':151630,'task_type':'CPU','allow_writing_files':False,'verbose':False}


def fit():
    assert not OUT.exists(),'Preserving CPU rich binary16 run'
    expected=json.loads((HERE/'runs/weighted_v1_day16/protocol.json').read_text(encoding='utf8'))['raw_datasets_sha256']
    for name,h in expected.items():assert digest(ROOT/'2026ml'/name)==h,name
    assert json.loads((BASE/'verification.json').read_text(encoding='utf8'))['status']=='passed'
    paths=list(dict.fromkeys([Path(__file__),HERE/'rolling_rich_daily_catboost.py']+dependencies()+[
                       BASE/'day16_prediction.npy',BASE/'verification.json',BASE/'day16_indices.npy']))
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json',{'day':20191116,'params':PARAMS,'arm':ARM,'checkpoints':CHECKS,'powers':POWERS,
          'evaluation16_labels_fit':False,'max_training_date':20191115,'development16_model_selection':True,
          'selection':'All CPU endpoints locked before held16 scoring; same full-rich feature definitions',
          'raw_datasets_sha256':expected,'sources_sha256':{str(p):digest(p) for p in paths},'inferred17_labels_read':False})
    data=inputs(16)
    write(OUT/'input_contract.json',contract(data,16))
    np.save(OUT/'valid_indices.npy',data['rows'])
    train=np.flatnonzero(data['keys'][:,0]<20191116)
    assert data['keys'][train,0].max()==20191115 and np.all(data['counts'][train]>=0)
    target=(data['counts'][train]>0).astype('float64')
    print(json.dumps({'event':'rich_binary16_cpu_ready','shape':list(data['frame'].shape),'train':len(train)}),flush=True)
    model=CatBoost(dict(PARAMS,iterations=max(CHECKS)))
    model.fit(Pool(data['frame'].iloc[train],label=target,cat_features=data['categories'],baseline=offsets(data,ARM)[train]))
    locks=[]
    for trees in CHECKS:
        saved=model.copy();saved.shrink(trees)
        name=f'binary_offset_trees{trees}';path=OUT/(name+'.cbm');save_model(saved,path)
        locks.append({'name':name,'trees':trees,'model_sha256':digest(path)})
    write(OUT/'models_locked.json',{'models':locks,'protocol_sha256':digest(OUT/'protocol.json')})
    evaluate(False,data)


def evaluate(verifying,data=None):
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    lock=json.loads((OUT/'models_locked.json').read_text(encoding='utf8'))
    assert lock['protocol_sha256']==digest(OUT/'protocol.json')
    for p,h in protocol['sources_sha256'].items():assert digest(p)==h,p
    for name,h in protocol['raw_datasets_sha256'].items():assert digest(ROOT/'2026ml'/name)==h,name
    if data is None:data=inputs(16)
    assert contract(data,16)==json.loads((OUT/'input_contract.json').read_text(encoding='utf8'))
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    rows=np.flatnonzero(raw.pt_d.to_numpy()==20191116)
    np.testing.assert_array_equal(rows,data['rows'])
    np.testing.assert_array_equal(rows,np.load(OUT/'valid_indices.npy'))
    np.testing.assert_array_equal(rows,np.load(BASE/'day16_indices.npy'))
    y=raw.label.to_numpy()[rows];assert len(y)==421983 and y.sum()==3724
    held=np.flatnonzero(data['valid'])
    base=np.load(BASE/'day16_prediction.npy');best={'name':'base','power':0.,'auc':rank_sum_auc(y,base)}
    best_p=base.copy();records=[]
    for r in lock['models']:
        path=OUT/(r['name']+'.cbm');assert digest(path)==r['model_sha256']
        gp=np.zeros(len(data['keys']));gp[held]=group_prediction(load_model(path),data,ARM,held)
        for power in POWERS:
            p=correct(base,gp,data['row_groups'],'binary',power)
            record={'name':r['name'],'power':power,'auc':rank_sum_auc(y,p)}
            records.append(record)
            if record['auc']>best['auc']:best=record;best_p=p.copy()
    error=0.
    if verifying:
        error=float(np.max(np.abs(best_p-np.load(OUT/'prediction.npy'))));assert error<1e-12,error
    else:np.save(OUT/'prediction.npy',best_p)
    report={'status':'passed' if verifying else 'evaluated','auc':best['auc'],'best':best,'records':records,
            'rows':len(rows),'clicks':int(y.sum()),'maximum_replay_error':error,'evaluation16_labels_fit':False,
            'development16_model_selection':True,'original_datasets_unchanged':True,'target_met':best['auc']>=.82}
    write(OUT/('verification.json' if verifying else 'result.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    evaluate(True) if parser.parse_args().verify else fit()
