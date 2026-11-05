"""Chronological signed-hash logistic crosses as a complementary row model."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[name]='2'
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import save_npz,load_npz
from scipy.special import expit,logit
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from fit_stacker import ROOT,HERE,digest,write
from conditional_daily import read,emit
from rich_userday_activity import array_hash
from rich_row_catboost import selection
from verify_stacker import rank_sum_auc
from sparse_cross_core_v1 import FIELDS,CHANNELS,WIDTH,encode_chunks,encode,margin

OUT=HERE/'runs/sparse_cross_ranker_v1'
BASE=HERE/'runs/compose_unordered_masks_v1'
SOURCE=HERE/'runs/rich_augmented_rows_v1'
ARMS=[dict(name='alpha3e6',alpha=3e-6),dict(name='alpha1e5',alpha=1e-5),dict(name='alpha3e5',alpha=3e-5)]
CHECKPOINTS=[2,6,14]
WEIGHTS=[.025,.05,.1,.2,.35,.5,.75,1.]
MODES=['add_global','replace_global']


def sparse_hash(x):
    return dict(data=array_hash(x.data),indices=array_hash(x.indices),indptr=array_hash(x.indptr),shape=list(x.shape))


def inputs(day,partition,independent=False):
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=FIELDS+['pt_d','label'],
                    dtype={name:str for name in CHANNELS})
    dates=raw.pt_d.to_numpy(dtype='int32')
    safe=raw.label.to_numpy(dtype='int32',copy=True)
    safe[dates>=20191100+day]=-1
    chosen,weights,held_rows,_,_,target=selection(day,safe,dates)
    native=read(SOURCE/f'input_contract{day}.json')
    assert array_hash(chosen)==native['training_rows_sha256']
    assert array_hash(target)==native['training_labels_sha256']
    rows=chosen if partition=='train' else held_rows
    assert np.all(dates[chosen]<20191100+day)
    frame=raw.iloc[rows]
    x=encode_chunks(frame)
    probe=encode(frame.iloc[:1000],independent=True)
    assert (x[:1000]!=probe).nnz==0
    if independent:
        # Re-encode the entire matrix in a different chunk layout, including all held rows.
        replay=encode_chunks(frame,chunk_size=17003)
        assert sparse_hash(x)==sparse_hash(replay)
        del replay
    report=dict(shape=list(x.shape),nonzeros=int(x.nnz),matrix_sha256=sparse_hash(x),
        rows_sha256=array_hash(rows),training_rows_sha256=array_hash(chosen),
        training_targets_sha256=array_hash(target),max_training_date=int(dates[chosen].max()),
        independent_hash_probe_rows=1000,hash_width=WIDTH,labels_used_in_encoder=False)
    del raw,frame
    gc.collect()
    emit('sparse_cross_input_ready',day=day,partition=partition,independent=independent,
         shape=report['shape'],nonzeros=report['nonzeros'])
    return x,rows,target,report


def dependencies():
    paths=[Path(__file__),HERE/'sparse_cross_core_v1.py',HERE/'test_sparse_cross_core_v1.py',
        HERE/'audit_sparse_cross_core_v1.py',HERE/'sparse_cross_core_preflight_v1.json',
        HERE/'run_sparse_cross_ranker_v1.ps1',HERE/'start_sparse_cross_queue_v1.ps1',
        HERE/'rich_row_catboost.py',HERE/'global_row_context.py',HERE/'global_row_context15.py',
        HERE/'fit_stacker.py',HERE/'conditional_daily.py',HERE/'verify_stacker.py',HERE/'rich_userday_activity.py']
    audit=read(HERE/'sparse_cross_core_preflight_v1.json')
    assert audit['status']=='passed' and audit['source_sha256']==digest(HERE/'sparse_cross_core_v1.py')
    for directory in [BASE,SOURCE]:
        assert read(directory/'verification.json')['status']=='passed'
        for name,expected in read(directory/'protocol.json')['sources_sha256'].items():
            assert digest(name)==expected,name
        paths += [directory/'verification.json',directory/'protocol.json']
    for day in [15,16]:
        paths += [BASE/f'day{day}_prediction.npy',BASE/f'day{day}_indices.npy',SOURCE/f'input_contract{day}.json']
    return list(dict.fromkeys(paths))


def check():
    protocol=read(OUT/'protocol.json')
    for name,expected in protocol['sources_sha256'].items():
        assert digest(name)==expected,name
    for name,expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name)==expected,name
    return protocol


def fit():
    assert not OUT.exists(),'Preserve previous runs'
    paths=dependencies()
    OUT.mkdir()
    audit=read(HERE/'sparse_cross_core_preflight_v1.json')
    write(OUT/'protocol.json',dict(arms=ARMS,checkpoints=CHECKPOINTS,weights=WEIGHTS,modes=MODES,
        feature_definition='Fixed signed-hash scalar fields, user/ad crosses, profile bags and bag/app/slot crosses',
        training='All strictly earlier positives and fixed one-third negatives, positive weight3; averaged SGD logistic',
        optimizer=dict(loss='log_loss',learning_rate='constant',eta0=.01,shuffle=True,seed=151655),
        evaluation_day_labels_fit=False,inferred17_labels_read=False,
        selection='Both-date models and predictions locked before held scoring; reused development dates',
        raw_datasets_sha256=audit['raw_datasets_sha256'],sources_sha256={str(path):digest(path) for path in paths}))
    locked=[]
    for day in [15,16]:
        x,rows,y,report=inputs(day,'train')
        write(OUT/f'train_contract{day}.json',report)
        sample_weight=np.where(y==1,3.,1.)
        np.save(OUT/f'day{day}_training_indices.npy',rows)
        for arm in ARMS:
            model=SGDClassifier(loss='log_loss',alpha=arm['alpha'],learning_rate='constant',eta0=.01,
                shuffle=True,average=len(y)//2,random_state=np.random.RandomState(151655))
            for epoch in range(1,max(CHECKPOINTS)+1):
                model.partial_fit(x,y,classes=np.array([0,1]),sample_weight=sample_weight)
                if epoch in CHECKPOINTS:
                    name=f'day{day}_{arm["name"]}_epoch{epoch}'
                    path=OUT/(name+'.npz')
                    coefficients=model.coef_.reshape(-1).astype('float64')
                    intercept=float(model.intercept_[0])
                    np.savez_compressed(path,coefficients=coefficients,intercept=intercept)
                    a=margin(x[:1000],coefficients,intercept)
                    b=margin(x[:1000],coefficients,intercept,True)
                    assert np.max(np.abs(a-b))<1e-10
                    locked.append(dict(arm,day=day,epoch=epoch,name=name,model_sha256=digest(path)))
                    emit('sparse_cross_checkpoint',day=day,arm=arm['name'],epoch=epoch)
            del model
        del x,rows,y
        gc.collect()
    write(OUT/'models_locked.json',dict(protocol_sha256=digest(OUT/'protocol.json'),models=locked))
    check()


def forecasts(day,independent=False):
    x,rows,_,report=inputs(day,'held',independent)
    contract=OUT/f'held_contract{day}.json'
    if independent:
        assert report==read(contract)
        np.testing.assert_array_equal(rows,np.load(OUT/f'day{day}_indices.npy'))
    else:
        write(contract,report)
        np.save(OUT/f'day{day}_indices.npy',rows)
    values={}
    for item in read(OUT/'models_locked.json')['models']:
        if item['day']!=day:continue
        path=OUT/(item['name']+'.npz')
        assert digest(path)==item['model_sha256']
        with np.load(path) as model:
            value=margin(x,model['coefficients'],float(model['intercept']),independent)
        assert np.isfinite(value).all()
        prediction=OUT/(item['name']+'_held_margin.npy')
        if independent:
            np.testing.assert_allclose(value,np.load(prediction),rtol=0,atol=1e-10)
        else:
            np.save(prediction,value)
        values[item['name']]=value
    del x
    gc.collect()
    return values,rows


def candidates(day,values):
    base=np.load(BASE/f'day{day}_prediction.npy')
    yield dict(name='base',mode='base',weight=0.),base
    z=logit(np.clip(base,1e-8,1-1e-8))
    for name,value in values.items():
        scaled=(value-value.mean())/max(float(value.std()),1e-6)*z.std()
        for mode in MODES:
            delta=scaled if mode=='add_global' else scaled+z.mean()-z
            for weight in WEIGHTS:
                yield dict(name=name,mode=mode,weight=weight),expit(z+weight*delta)


def key(record):return f'{record["name"]}__{record["mode"]}__{record["weight"]:g}'


def evaluate():
    check()
    locked={}
    for day in [15,16]:
        values,rows=forecasts(day)
        locked[str(day)]={key(r):array_hash(p) for r,p in candidates(day,values)}
    write(OUT/'predictions_locked.json',dict(protocol_sha256=digest(OUT/'protocol.json'),dates=locked,
        both_date_models_locked=True,held_labels_opened_for_scoring=False))
    emit('sparse_cross_predictions_locked')


def verify():
    check()
    locks=read(OUT/'predictions_locked.json')
    assert locks['protocol_sha256']==digest(OUT/'protocol.json')
    verified={}
    for day in [15,16]:
        x,_,_,report=inputs(day,'train',True)
        assert report==read(OUT/f'train_contract{day}.json')
        del x
        values,rows=forecasts(day,True)
        # Floating-point summation order differs independently, so compare to locked primary margins.
        primary={name:np.load(OUT/(name+'_held_margin.npy')) for name in values}
        for r,p in candidates(day,primary):
            assert array_hash(p)==locks['dates'][str(day)][key(r)]
        for (r,p),(other,q) in zip(candidates(day,primary),candidates(day,values)):
            assert r==other
            np.testing.assert_allclose(p,q,rtol=0,atol=1e-10)
        verified[day]=(primary,rows)
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    reports={}
    for day in [15,16]:
        values,rows=verified[day]
        np.testing.assert_array_equal(rows,np.flatnonzero(raw.pt_d.to_numpy()==20191100+day))
        y=raw.label.to_numpy()[rows]
        assert len(y)=={15:424780,16:421983}[day] and y.sum()=={15:3523,16:3724}[day]
        records=[];best_value=-1
        for r,p in candidates(day,values):
            value=rank_sum_auc(y,p);record=dict(r,auc=value);records.append(record)
            if value>best_value:best_value=value;best=record;best_prediction=p.copy()
        assert abs(roc_auc_score(y,best_prediction)-best_value)<1e-12
        np.save(OUT/f'day{day}_prediction.npy',best_prediction)
        reports[str(20191100+day)]=dict(auc=best_value,best=best,records=records,rows=len(y),clicks=int(y.sum()),
            evaluation_day_labels_fit=False,held_labels_used_to_select_fixed_recipe=True,
            all_both_date_candidates_locked_before_scoring=True,independent_encoding_and_margins_replayed=True,
            target_met=best_value>={15:.815,16:.82}[day])
        emit('sparse_cross_scored',day=day,best=best)
    check()
    write(OUT/'verification.json',dict(status='passed',dates=reports,original_datasets_unchanged=True,
        both_targets_met=all(r['target_met'] for r in reports.values())))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=['fit','evaluate','verify'])
    {'fit':fit,'evaluate':evaluate,'verify':verify}[parser.parse_args().phase]()
