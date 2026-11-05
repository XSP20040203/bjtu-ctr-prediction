"""Learn past-only daily news representations and forecast both held ad dates.

Document/source/topic identities add content information beyond coarse category
counts. TF-IDF and SVD fit only earlier user-days; held news feedback is supplied
offline input. Advertisement outcomes on each held date never enter fitting.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='2'
import argparse
import gc
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from rolling_daily15 import prepare as prepare15,source_paths as sources15
from feed_behavior_userday import prepare as prepare16,dependency_paths as sources16
from personal_context_residual import personal_changes
from rich_userday_activity import array_hash,correct
from fit_stacker import ROOT,HERE,REFERENCE,digest,write
from rich_group_residual import lgb
from verify_stacker import rank_sum_auc

OUT=HERE/'runs/rolling_news_latent_v1'
BASES={15:HERE/'runs/catboost_global_rows15_v1/prediction.npy',
       16:HERE/'runs/rolling_previous_app_v1/day16_prediction.npy'}
VERIFICATIONS={15:HERE/'runs/catboost_global_rows15_v1/verification.json',
               16:HERE/'runs/rolling_previous_app_v1/verification.json'}
WIDTH=32768
COMPONENTS=48
POWERS=[0.,.125,.25,.5,.75]
CHECKS=[300,700]
PARAMS={'learning_rate':.025,'min_data_in_leaf':100,'lambda_l2':20.,'feature_fraction':.75,
        'max_bin':31,'num_threads':2,'seed':185163,'verbosity':-1,'deterministic':True,
        'force_col_wise':True,'feature_pre_filter':False}


def emit(event,**values):print(json.dumps(dict(event=event,**values)),flush=True)


def raw_features(day):
    data,_=(prepare15() if day==15 else prepare16())
    cutoff=20191100+day
    assert data['keys'][:,0].max()==cutoff
    assert np.all(data['counts'][data['keys'][:,0]>=cutoff]==-1)
    fields=['i_docId','i_s_sourceId','i_cat']
    news=pd.read_csv(ROOT/'2026ml/train_feeds.csv',usecols=['e_et','u_userId','label','pro']+fields,
                     dtype={'i_docId':str,'i_s_sourceId':str})
    index=pd.MultiIndex.from_arrays(data['keys'].T)
    groups=index.get_indexer(pd.MultiIndex.from_frame(news[['e_et','u_userId']]))
    keep=(groups>=0)&(news.e_et.to_numpy()<=cutoff)
    news=news.loc[keep];groups=groups[keep]
    assert news.label.isin([-1,1]).all()
    clicked=news.label.to_numpy()==1
    read=clicked*np.clip(news.pro.to_numpy(dtype='float32'),0,100)/100.
    matrices=[]
    for channel,weights in enumerate([np.ones(len(news),dtype='float32'),clicked.astype('float32'),read]):
        for field in fields:
            codes=(pd.util.hash_pandas_object(news[field],index=False).to_numpy(dtype='uint64')%WIDTH).astype('int32')
            nonzero=weights>0
            matrix=sparse.csr_matrix((weights[nonzero],(groups[nonzero],codes[nonzero])),
                                    shape=(len(data['keys']),WIDTH),dtype='float32')
            matrix.sum_duplicates();matrix.sort_indices()
            matrices.append(matrix)
    matrix=sparse.hstack(matrices,format='csr',dtype='float32')
    matrix.sum_duplicates();matrix.sort_indices()
    emit('news_sparse_ready',day=day,shape=list(matrix.shape),nonzero=matrix.nnz)
    return data,matrix


def csr_hash(x):
    return {'data':array_hash(x.data),'indices':array_hash(x.indices),'indptr':array_hash(x.indptr),'shape':list(x.shape)}


def represent(data,matrix,day,fitting):
    train=np.flatnonzero(data['keys'][:,0]<20191100+day)
    path=OUT/f'news_representation{day}.joblib'
    if fitting:
        tfidf=TfidfTransformer(norm='l2',sublinear_tf=True)
        tfidf.fit(matrix[train])
        encoded=tfidf.transform(matrix).astype('float32')
        svd=TruncatedSVD(n_components=COMPONENTS,n_iter=4,random_state=151600+day)
        svd.fit(encoded[train])
        selector=HERE/('runs/rolling_daily15_v1/binary_l15_trees700.txt' if day==15 else
                      'runs/feed_behavior_userday_v1/binary_15.txt')
        selected=np.argsort(-lgb.Booster(model_str=selector.read_text(encoding='utf8')).feature_importance('gain'),kind='stable')[:96]
        numeric=data['x'][:,selected]
        center=numeric[train].mean(0).astype('float32')
        scale=np.maximum(numeric[train].std(0),.05).astype('float32')
        joblib.dump({'tfidf':tfidf,'svd':svd,'selected':selected,'center':center,'scale':scale,
                     'selector_sha256':digest(selector)},path)
        emit('news_representation_fit',day=day,train_days_max=int(data['keys'][train,0].max()),
             variance=float(svd.explained_variance_ratio_.sum()))
    else:
        model=joblib.load(path)
        tfidf,svd,selected,center,scale=[model[k] for k in ['tfidf','svd','selected','center','scale']]
        encoded=tfidf.transform(matrix).astype('float32')
    latent=svd.transform(encoded).astype('float32')
    changes=personal_changes(data['keys'],latent)
    x=np.column_stack([data['x'],latent,changes]).astype('float32')
    normalized=np.clip((data['x'][:,selected]-center)/scale,-8,8).astype('float32')
    normalized/=np.maximum(np.linalg.norm(normalized,axis=1,keepdims=True),1.)
    linear=sparse.hstack([3.*encoded,sparse.csr_matrix(normalized)],format='csr',dtype='float32')
    assert np.isfinite(x).all() and np.isfinite(linear.data).all()
    return x,linear


def input_contract(data,matrix,x,linear,day):
    training=data['keys'][:,0]<20191100+day
    return {'keys_sha256':array_hash(data['keys']),'counts_sha256':array_hash(data['counts']),
            'sparse_hashes':csr_hash(matrix),'x_sha256':array_hash(x),'linear_hashes':csr_hash(linear),
            'training_groups':int(training.sum()),'max_training_date':int(data['keys'][training,0].max()),
            'held_groups':int(data['valid'].sum()),'held_outcome_values':[-1]}


def fit():
    assert not OUT.exists(),'Preserving latent news experiment'
    expected=json.loads((HERE/'runs/weighted_v1_day16/protocol.json').read_text(encoding='utf8'))['raw_datasets_sha256']
    for name,h in expected.items():assert digest(ROOT/'2026ml'/name)==h,name
    for path in VERIFICATIONS.values():assert json.loads(path.read_text(encoding='utf8'))['status']=='passed'
    paths=list(dict.fromkeys([Path(__file__),HERE/'rolling_daily15.py',HERE/'personal_context_residual.py']+sources15()+sources16()
               +list(BASES.values())+list(VERIFICATIONS.values())))
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json',{'dates':[20191115,20191116],'hash_width_per_channel':WIDTH,'svd_components':COMPONENTS,
          'trees':CHECKS,'powers':POWERS,'tree_params':PARAMS,'linear_alphas':[.0001,.001],
          'training_cutoff_exclusive':[20191115,20191116],'own_held_day_ad_labels_fit':False,
          'representation_fitting':'TF-IDF and SVD only on strictly earlier user-days',
          'inputs':'Supplied same-day news exposures/click/read feedback; hashed document/source/topic IDs',
          'selection':'Models locked before own held outcomes; both dates repeatedly reused for development',
          'sources_sha256':{str(p):digest(p) for p in paths},'raw_datasets_sha256':expected,'inferred17_labels_read':False})
    locks=[]
    for day in [15,16]:
        data,matrix=raw_features(day)
        x,linear=represent(data,matrix,day,True)
        write(OUT/f'input_contract{day}.json',input_contract(data,matrix,x,linear,day))
        np.save(OUT/f'day{day}_indices.npy',data['rows'])
        train=np.flatnonzero(data['keys'][:,0]<20191100+day)
        counts=data['counts'][train]
        assert np.all(counts>=0)
        for objective in ['binary','poisson']:
            target=(counts>0).astype('float64') if objective=='binary' else counts.copy()
            def checkpoint(env):
                trees=env.iteration+1
                if trees in CHECKS:
                    name=f'day{day}_{objective}_trees{trees}'
                    path=OUT/(name+'.txt');path.write_text(env.model.model_to_string(),encoding='utf8')
                    locks.append({'day':day,'name':name,'kind':'tree','objective':objective,'model_sha256':digest(path)})
            lgb.train(dict(PARAMS,objective=objective,metric='None',num_leaves=15),
                      lgb.Dataset(x[train],label=target,params=PARAMS),num_boost_round=max(CHECKS),callbacks=[checkpoint])
            emit('news_tree_fit',day=day,objective=objective)
        for alpha in [.0001,.001]:
            name=f'day{day}_linear_alpha{alpha:g}'
            model=SGDClassifier(loss='log_loss',penalty='l2',alpha=alpha,random_state=15163,
                                max_iter=1000,tol=1e-5,average=True,n_jobs=2)
            model.fit(linear[train],(counts>0).astype('int32'))
            path=OUT/(name+'.joblib');joblib.dump(model,path)
            locks.append({'day':day,'name':name,'kind':'linear','objective':'binary','model_sha256':digest(path)})
            emit('news_linear_fit',day=day,alpha=alpha)
        locks.append({'day':day,'name':f'news_representation{day}','kind':'representation',
                      'model_sha256':digest(OUT/f'news_representation{day}.joblib')})
        del data,matrix,x,linear;gc.collect()
    write(OUT/'models_locked.json',{'models':locks,'protocol_sha256':digest(OUT/'protocol.json')})
    evaluate(False)


def evaluate(verifying):
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    lock=json.loads((OUT/'models_locked.json').read_text(encoding='utf8'))
    assert lock['protocol_sha256']==digest(OUT/'protocol.json')
    for p,h in protocol['sources_sha256'].items():assert digest(p)==h,p
    for name,h in protocol['raw_datasets_sha256'].items():assert digest(ROOT/'2026ml'/name)==h,name
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    results={}
    for day in [15,16]:
        representation=next(r for r in lock['models'] if r['day']==day and r['kind']=='representation')
        assert digest(OUT/(representation['name']+'.joblib'))==representation['model_sha256']
        data,matrix=raw_features(day)
        x,linear=represent(data,matrix,day,False)
        assert input_contract(data,matrix,x,linear,day)==json.loads((OUT/f'input_contract{day}.json').read_text(encoding='utf8'))
        rows=np.flatnonzero(raw.pt_d.to_numpy()==20191100+day)
        np.testing.assert_array_equal(rows,np.load(OUT/f'day{day}_indices.npy'))
        np.testing.assert_array_equal(rows,data['rows'])
        y=raw.label.to_numpy()[rows]
        assert len(y)=={15:424780,16:421983}[day] and y.sum()=={15:3523,16:3724}[day]
        held=np.flatnonzero(data['valid'])
        base=np.load(BASES[day]);best={'name':'base','power':0.,'auc':rank_sum_auc(y,base)}
        best_p=base.copy();records=[]
        for r in lock['models']:
            if r['day']!=day or r['kind']=='representation':continue
            path=OUT/(r['name']+('.txt' if r['kind']=='tree' else '.joblib'))
            assert digest(path)==r['model_sha256']
            gp=np.zeros(len(data['keys']))
            if r['kind']=='tree':
                gp[held]=lgb.Booster(model_str=path.read_text(encoding='utf8')).predict(x[held])
            else:gp[held]=joblib.load(path).predict_proba(linear[held])[:,1]
            for power in POWERS:
                p=correct(base,gp,data['row_groups'],r['objective'],power)
                record={'name':r['name'],'power':power,'auc':rank_sum_auc(y,p)}
                records.append(record)
                if record['auc']>best['auc']:best=record;best_p=p.copy()
        error=0.
        if verifying:
            error=float(np.max(np.abs(best_p-np.load(OUT/f'day{day}_prediction.npy'))));assert error<1e-12,error
        else:np.save(OUT/f'day{day}_prediction.npy',best_p)
        results[str(20191100+day)]={'auc':best['auc'],'best':best,'records':records,'rows':len(rows),
              'clicks':int(y.sum()),'maximum_replay_error':error,'own_held_day_ad_labels_fit':False,
              'target_met':best['auc']>={15:.815,16:.82}[day]}
        emit('news_latent_scored',day=day,best=best)
        del data,matrix,x,linear;gc.collect()
    report={'status':'passed' if verifying else 'evaluated','dates':results,'original_datasets_unchanged':True,
            'both_targets_met':all(r['target_met'] for r in results.values())}
    write(OUT/('verification.json' if verifying else 'result.json'),report)
    emit('news_latent_result',status=report['status'],dates={d:r['auc'] for d,r in results.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    evaluate(True) if parser.parse_args().verify else fit()
