"""Previous-date observed/expected user-app errors, separately for15 and16."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.stats import rankdata
from fit_stacker import ROOT,HERE,REFERENCE,digest,write
from previous_day_residual import posterior_error

OUT=HERE/'runs/rolling_previous_app_v1'
BASES={15:HERE/'runs/rolling_daily15_v1',16:HERE/'runs/catboost_global_rows_v1'}
POWERS=[0.,.125,.25,.5]
ALPHA=3.


def auc(y,p):
    n=int(y.sum());return float((rankdata(p,method='average')[y==1].sum()-n*(n+1)/2)/(n*(len(y)-n)))


def statistics(raw,day):
    prior_day=20191100+day-1
    train=raw.loc[raw.pt_d==prior_day]
    assert not train.empty and int(train.pt_d.max())<20191100+day
    if day==15:
        expected=np.ones(len(train),dtype='float64')
    else:
        expected=np.load(REFERENCE/'day15_prediction.npy')
        np.testing.assert_array_equal(np.flatnonzero(raw.pt_d.to_numpy()==prior_day),np.load(REFERENCE/'day15_indices.npy'))
    expected=expected*train.label.sum()/expected.sum()
    codes,keys=pd.factorize(pd.MultiIndex.from_frame(train[['user_id','spread_app_id']]),sort=True)
    count=np.bincount(codes,weights=train.label,minlength=len(keys))
    mass=np.bincount(codes,weights=expected,minlength=len(keys))
    return {'keys':np.asarray(list(keys),dtype='int64'),'count':count,'mass':mass}


def correction(frame,stats):
    position=pd.MultiIndex.from_arrays(stats['keys'].T).get_indexer(pd.MultiIndex.from_frame(frame[['user_id','spread_app_id']]))
    output=np.zeros(len(frame));known=position>=0
    value=posterior_error(stats['count'],stats['mass'],ALPHA)
    output[known]=value[position[known]]
    return output


def raw_data():
    return pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label','user_id','spread_app_id'],dtype='int32')


def fit():
    if OUT.exists():raise FileExistsError('Preserving rolling app residuals')
    expected=json.loads((HERE/'runs/weighted_v1_day16/protocol.json').read_text(encoding='utf8'))['raw_datasets_sha256']
    for name,h in expected.items():assert digest(ROOT/'2026ml'/name)==h,name
    paths=[Path(__file__),HERE/'previous_day_residual.py',REFERENCE/'day15_prediction.npy',REFERENCE/'day15_indices.npy']
    for base in BASES.values():
        assert json.loads((base/'verification.json').read_text(encoding='utf8'))['status']=='passed'
        paths += [base/'verification.json',base/'prediction.npy',base/'valid_indices.npy']
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json',{'dates':[20191115,20191116],'group':['user_id','spread_app_id'],
          'alpha':ALPHA,'powers':POWERS,'fit_dates':{'20191115':20191114,'20191116':20191115},
          'expected_mass15':'Previous-day exposure volume times previous-day overall CTR',
          'expected_mass16':'Chronological15 forecasts, rescaled to previous15 observed total',
          'sources_sha256':{str(p):digest(p) for p in paths},'raw_datasets_sha256':expected,
          'own_evaluation_day_labels_fit':False,'dates_reused_for_development':True})
    raw=raw_data();locks=[]
    for day in [15,16]:
        stats=statistics(raw,day);path=OUT/f'statistics_day{day}.npz'
        np.savez_compressed(path,**stats)
        locks.append({'day':day,'fit_max_date':20191100+day-1,'model_sha256':digest(path)})
    write(OUT/'models_locked.json',{'models':locks,'protocol_sha256':digest(OUT/'protocol.json')})
    evaluate(False)


def evaluate(verifying):
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    locked=json.loads((OUT/'models_locked.json').read_text(encoding='utf8'))
    assert locked['protocol_sha256']==digest(OUT/'protocol.json')
    for p,h in protocol['sources_sha256'].items():assert digest(p)==h,p
    for name,h in protocol['raw_datasets_sha256'].items():assert digest(ROOT/'2026ml'/name)==h,name
    raw=raw_data();results={}
    for record in locked['models']:
        day=record['day'];path=OUT/f'statistics_day{day}.npz'
        assert digest(path)==record['model_sha256']
        stats=dict(np.load(path))
        if verifying:
            rebuilt=statistics(raw,day)
            for k in stats:np.testing.assert_array_equal(rebuilt[k],stats[k])
        rows=np.flatnonzero(raw.pt_d.to_numpy()==20191100+day)
        np.testing.assert_array_equal(rows,np.load(BASES[day]/'valid_indices.npy'))
        y=raw.label.to_numpy()[rows]
        assert len(y)=={15:424780,16:421983}[day] and y.sum()=={15:3523,16:3724}[day]
        base=np.load(BASES[day]/'prediction.npy');delta=correction(raw.iloc[rows],stats)
        predictions=[base.copy() if p==0 else expit(logit(base.clip(1e-6,1-1e-6))+p*delta) for p in POWERS]
        scores=[{'power':p,'auc':auc(y,q)} for p,q in zip(POWERS,predictions)]
        best=int(np.argmax([r['auc'] for r in scores]));p=predictions[best]
        error=0.
        if verifying:
            error=float(np.max(np.abs(p-np.load(OUT/f'day{day}_prediction.npy'))));assert error<1e-12
        else:
            np.save(OUT/f'day{day}_prediction.npy',p);np.save(OUT/f'day{day}_indices.npy',rows)
        results[str(20191100+day)]={'auc':scores[best]['auc'],'best':scores[best],'records':scores,
                   'rows':len(rows),'clicks':int(y.sum()),'maximum_replay_error':error,
                   'evaluation_day_labels_fit':False,'target_met':scores[best]['auc']>={15:.815,16:.82}[day]}
    result={'status':'passed' if verifying else 'evaluated','dates':results,'original_datasets_unchanged':True,
            'both_targets_met':all(r['target_met'] for r in results.values())}
    write(OUT/('verification.json' if verifying else 'result.json'),result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    evaluate(True) if parser.parse_args().verify else fit()
