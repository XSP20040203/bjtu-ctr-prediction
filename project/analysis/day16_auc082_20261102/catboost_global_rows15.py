"""Full row CatBoost on independently reconstructed strict-past features."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[k]='2'
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from catboost import CatBoost,Pool
from global_row_context15 import selected_training,CAT
from catboost_userday_residual import save_model,load_model
from fit_stacker import ROOT,HERE,digest,write
from verify_stacker import rank_sum_auc
from rich_userday_activity import array_hash

OUT=HERE/'runs/catboost_global_rows15_v1'
SOURCE=HERE/'runs/global_row_context15_v1'
BASE=HERE/'runs/rolling_daily15_v1'
ARMS=[{'name':'depth6_weight1','depth':6,'positive_weight':1.},
      {'name':'depth6_weight5','depth':6,'positive_weight':5.},
      {'name':'depth8_weight5','depth':8,'positive_weight':5.}]
CHECKS=[400,800,1200]
PARAMS={'loss_function':'Logloss','learning_rate':.035,'l2_leaf_reg':30.,
        'random_strength':.5,'bootstrap_type':'Bernoulli','subsample':.8,
        'border_count':64,'one_hot_max_size':16,'max_ctr_complexity':1,
        'thread_count':2,'random_seed':8243,'task_type':'GPU','devices':'0',
        'gpu_ram_part':.4,'allow_writing_files':False,'verbose':False}


def inputs(independent=False):
    contract=json.loads((SOURCE/'input_contract.json').read_text(encoding='utf8'))
    path=SOURCE/('independent_input' if independent else 'input')/'features.npy'
    if digest(path)!=contract['features_sha256']:
        raise AssertionError('Independently verified feature matrix changed')
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    y,dates=raw.label.to_numpy().copy(),raw.pt_d.to_numpy()
    chosen,weights=selected_training(y,dates)
    valid=np.flatnonzero(dates==20191115)
    rows=np.sort(np.r_[chosen,valid])
    train=np.flatnonzero(dates[rows]<20191115)
    held=np.flatnonzero(dates[rows]==20191115)
    assert array_hash(rows)==contract['rows_sha256']
    assert array_hash(chosen)==contract['train_rows_sha256']
    assert array_hash(y[chosen])==contract['training_labels_sha256']
    assert array_hash(weights)==contract['weights_sha256']
    y[dates>=20191115]=-1
    assert np.all(y[rows[held]]==-1)
    x=np.load(path,mmap_mode='r')
    assert x.shape==(len(rows),len(contract['feature_names']))
    frame=pd.DataFrame(np.asarray(x),columns=[f'f{i}' for i in range(x.shape[1])])
    for i in range(len(CAT)):
        values=frame.iloc[:,i].to_numpy()
        np.testing.assert_array_equal(values,values.astype('int64'))
        frame[f'f{i}']=values.astype('int64')
    cats=[f'f{i}' for i in range(len(CAT))]
    return frame,train,held,y[rows[train]],weights,rows[held],cats


def recipes(p,base):
    yield 'base',base
    yield 'single',p
    a,b=logit(base.clip(1e-6,1-1e-6)),logit(p.clip(1e-6,1-1e-6))
    b=(b-b.mean())*(a.std()/max(b.std(),1e-6))+a.mean()
    for power in [.05,.1,.25]:
        yield f'probability_{power:g}',(1-power)*base+power*p
        yield f'logit_{power:g}',expit((1-power)*a+power*b)


def predict(model,pool,weight):
    return expit(model.predict(pool,prediction_type='RawFormulaVal')-np.log(weight))


def fit():
    if OUT.exists():raise FileExistsError('Preserving row CatBoost experiment')
    feature_protocol=json.loads((SOURCE/'protocol.json').read_text(encoding='utf8'))
    feature_contract=json.loads((SOURCE/'input_contract.json').read_text(encoding='utf8'))
    assert feature_protocol['evaluation15_labels_fit'] is False
    assert feature_contract['max_fit_date']<20191115
    assert json.loads((BASE/'verification.json').read_text(encoding='utf8'))['status']=='passed'
    raw_hashes=json.loads((HERE/'runs/weighted_v1_day16/protocol.json').read_text(encoding='utf8'))['raw_datasets_sha256']
    for name,h in raw_hashes.items():assert digest(ROOT/'2026ml'/name)==h,name
    paths=[Path(__file__),HERE/'global_row_context15.py',HERE/'catboost_userday_residual.py',
           SOURCE/'input_contract.json',SOURCE/'protocol.json',BASE/'prediction.npy',BASE/'verification.json']
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json',{'params':PARAMS,'arms':ARMS,'checkpoints':CHECKS,
          'sources_sha256':{str(p):digest(p) for p in paths},'raw_datasets_sha256':raw_hashes,
          'training':'All pre15 positives; one third of pre15 negatives, inverse inclusion weights',
          'independent_feature_verification_pending_at_fit':True,
          'evaluation15_labels_fit':False,'development15_model_selection':True,
          'selection':'All fixed tree endpoints locked before held16 labels; declared single and blend recipes',
          'feature_source':'Previously independently rebuilt strict-past matrix; 96 daily features selected by pre16 model',
          'inferred17_labels_read':False})
    frame,train,held,y,weights,rows,cats=inputs()
    np.save(OUT/'valid_indices.npy',rows)
    write(OUT/'training_contract.json',{'training_rows':len(train),'training_clicks':int(y.sum()),
          'max_training_date':20191114,'training_labels_sha256':array_hash(y),
          'rows_sha256':array_hash(rows),'held_targets_masked':True,'feature_shape':list(frame.shape)})
    print(json.dumps({'event':'row_catboost_ready','rows':len(train),'width':frame.shape[1]}),flush=True)
    locks=[]
    for arm in ARMS:
        w=weights*np.where(y==1,arm['positive_weight'],1.)
        pool=Pool(frame.iloc[train],label=y,weight=w,cat_features=cats)
        model=CatBoost(dict(PARAMS,depth=arm['depth'],iterations=max(CHECKS)))
        model.fit(pool)
        del pool
        gc.collect()
        for trees in CHECKS:
            name=arm['name']+f'_trees{trees}'
            saved=model.copy()
            saved.shrink(trees)
            path=OUT/(name+'.cbm')
            save_model(saved,path)
            locks.append(dict(arm,name=name,trees=trees,model_sha256=digest(path)))
            del saved
        print(json.dumps({'event':'row_catboost_arm_complete','name':arm['name']}),flush=True)
        del model
        gc.collect()
    write(OUT/'models_locked.json',{'models':locks,'protocol_sha256':digest(OUT/'protocol.json')})
    evaluate(False,frame,held,rows,cats)


def evaluate(verifying,frame=None,held=None,rows=None,cats=None):
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    lock=json.loads((OUT/'models_locked.json').read_text(encoding='utf8'))
    assert lock['protocol_sha256']==digest(OUT/'protocol.json')
    for path,h in protocol['sources_sha256'].items():assert digest(path)==h,path
    for name,h in protocol['raw_datasets_sha256'].items():assert digest(ROOT/'2026ml'/name)==h,name
    if verifying:
        assert json.loads((SOURCE/'verification.json').read_text(encoding='utf8'))['status']=='passed'
        frame,_,held,_,_,rows,cats=inputs(True)
    pool=Pool(frame.iloc[held],cat_features=cats)
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    np.testing.assert_array_equal(rows,np.flatnonzero(raw.pt_d.to_numpy()==20191115))
    np.testing.assert_array_equal(rows,np.load(OUT/'valid_indices.npy'))
    y=raw.label.to_numpy()[rows]
    assert len(y)==424780 and y.sum()==3523
    base=np.load(BASE/'prediction.npy')
    records=[]
    best_p=base.copy();best={'name':'base','recipe':'base','auc':rank_sum_auc(y,base)}
    max_error=0.
    for r in lock['models']:
        path=OUT/(r['name']+'.cbm')
        assert digest(path)==r['model_sha256']
        p=predict(load_model(path),pool,r['positive_weight'])
        if verifying:
            err=float(np.max(np.abs(p-np.load(OUT/(r['name']+'_prediction.npy')))))
            assert err<2e-7,(r['name'],err)
            max_error=max(max_error,err)
        else:np.save(OUT/(r['name']+'_prediction.npy'),p)
        for recipe,q in recipes(p,base):
            score=rank_sum_auc(y,q)
            record={'name':r['name'],'recipe':recipe,'auc':score}
            records.append(record)
            if score>best['auc']:best=record;best_p=q.copy()
        print(json.dumps({'event':'row_catboost_scored','name':r['name'],'current_best':best}),flush=True)
    if verifying:
        err=float(np.max(np.abs(best_p-np.load(OUT/'prediction.npy'))))
        assert err<2e-7,err
        max_error=max(max_error,err)
    else:np.save(OUT/'prediction.npy',best_p)
    result={'status':'passed' if verifying else 'evaluated','auc':best['auc'],'best':best,'records':records,
            'rows':len(y),'clicks':int(y.sum()),'maximum_replay_error':max_error,
            'original_datasets_unchanged':True,'evaluation15_labels_fit':False,'development15_model_selection':True,
            'independent_feature_matrix_used':verifying,'target_met':best['auc']>=.815}
    write(OUT/('verification.json' if verifying else 'result.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify:evaluate(True)
    elif (OUT/'result.json').exists():print('Preserving completed training; independent verification follows',flush=True)
    else:fit()
