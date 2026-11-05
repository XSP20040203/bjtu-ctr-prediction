"""Strict-past query rankers with full categorical and337 content-quality inputs."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='2'
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRanker,Pool
from sklearn.metrics import roc_auc_score
from fit_stacker import HERE,ROOT,digest,write
from conditional_daily import read,emit
from rich_userday_activity import array_hash
from rich_row_catboost import frame_hash
from catboost_userday_residual import save_model,load_model
from full_quality_row_catboost import full_frame,quality_rows
from news_quality_features import READ,build_quality
from verify_stacker import rank_sum_auc
import within_user_ranking_core as core
from within_user_ranker import checked_layout,centered_unit

SOURCE=HERE/'runs/rich_augmented_rows_v1'
QUALITY=HERE/'runs/news_quality_daily_v1'
BASE=HERE/'runs/within_user_ranker_v1'
OUT=HERE/'runs/catboost_within_ranker_v2'
PREFLIGHT=HERE/'runs/catboost_within_ranker_preflight_v2'
CHECKPOINTS=[200,600,1600]
ARMS=[dict(name='pairlogit_d6',loss_function='PairLogit:max_pairs=256',depth=6),
      dict(name='yetirank_d6',loss_function='YetiRank',depth=6)]
WEIGHTS=[.025,.05,.1,.2,.35,.5,.75,1.]
MODES=['add_within','replace_within']
PARAMS=dict(learning_rate=.025,l2_leaf_reg=60.,random_strength=1.,
    bootstrap_type='Bernoulli',subsample=.8,border_count=32,one_hot_max_size=16,
    max_ctr_complexity=1,thread_count=2,random_seed=151645,task_type='GPU',devices='0',
    gpu_ram_part=.55,allow_writing_files=False,verbose=200)
QUALITY_REPLAY={}


def query_ids(sizes,target):
    sizes=np.asarray(sizes)
    target=np.asarray(target)
    if sizes.ndim!=1 or not len(sizes) or np.any(sizes<=1) or not np.array_equal(sizes,sizes.astype('int64')):
        raise ValueError('Complete nontrivial query sizes are required')
    if sizes.sum()!=len(target) or target.ndim!=1 or not np.isin(target,[0,1]).all():
        raise ValueError('Observed binary targets must cover exactly the complete queries')
    group=np.repeat(np.arange(len(sizes),dtype='int64'),sizes.astype('int64'))
    positives=np.bincount(group,weights=target)
    if np.any((positives<=0)|(positives>=sizes)):
        raise ValueError('Every training query must contain both observed classes')
    return group


def metadata(day):
    cutoff=20191100+day
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','user_id','log_id','label'],dtype='int32')
    dates=raw.pt_d.to_numpy()
    labels=raw.label.to_numpy(copy=True)
    labels[dates>=cutoff]=-1
    raw.loc[dates>=cutoff,'label']=-1
    chosen,weights=(core.rows15 if day==15 else core.rows16).selected_training(labels,dates)
    source=read(SOURCE/f'input_contract{day}.json')
    assert array_hash(chosen)==source['training_rows_sha256']
    assert array_hash(labels[chosen])==source['training_labels_sha256']
    assert array_hash(weights)==source['weights_sha256']
    rows=np.sort(np.r_[chosen,np.flatnonzero(dates==cutoff)])
    order,sizes,target,keys=core.ranking_order(rows,dates,raw.user_id.to_numpy(),labels,cutoff)
    held,held_rows,groups,held_audit=checked_layout(rows,dates,raw.user_id.to_numpy(),raw.log_id.to_numpy(),cutoff)
    np.testing.assert_array_equal(held_rows,np.load(BASE/f'day{day}_indices.npy'))
    return dict(raw=raw,rows=rows,order=order,sizes=sizes,target=target,keys=keys,
        held=held,held_rows=held_rows,groups=groups,held_audit=held_audit,
        source=source,training_max_date=int(keys[:,0].max()))


def inputs(day,partition,independent=False):
    info=metadata(day)
    positions=info['order'] if partition=='train' else info['held']
    query=info['raw'].iloc[info['rows'][positions]][['pt_d','user_id']].to_numpy()
    if partition=='train':
        assert np.all(query[:,0]<20191100+day)
        ids=query_ids(info['sizes'],info['target'])
        assert np.all(np.diff(ids)>=0)
    else:
        assert np.all(query[:,0]==20191100+day)
    cached=np.load(QUALITY/f'quality{day}.npz')
    keys,values=cached['keys'],cached['values']
    native=read(QUALITY/f'input_contract{day}.json')
    assert array_hash(values)==native['quality_sha256']
    assert array_hash(keys)==native['native_contract']['native_contract']['keys_sha256']
    if independent:
        if day not in QUALITY_REPLAY:
            news=pd.read_csv(ROOT/'2026ml/train_feeds.csv',usecols=READ,dtype={'i_docId':str,'i_s_sourceId':str})
            rebuilt,names=build_quality(news,keys,20191100+day)
            QUALITY_REPLAY[day]=(rebuilt,names)
            del news
            gc.collect()
        rebuilt,names=QUALITY_REPLAY[day]
        np.testing.assert_array_equal(rebuilt,values)
        assert names==native['quality_names']
        values=rebuilt
    quality=quality_rows(keys,values,query)
    directory=SOURCE/(f'independent_day{day}' if independent else f'input_day{day}')
    path=directory/'features.npy'
    assert digest(path)==info['source']['matrix_sha256']
    matrix=np.load(path,mmap_mode='r')
    frame,cats,keep=full_frame(matrix,positions,info['source']['feature_names'],info['source']['categorical_columns'],quality)
    assert frame.shape==(len(positions),1017)
    audit=dict(shape=list(frame.shape),frame_sha256=frame_hash(frame),source_matrix_sha256=info['source']['matrix_sha256'],
        quality_sha256=array_hash(quality),query_keys_sha256=array_hash(query),categories=cats,
        max_training_date=info['training_max_date'],training_order_sha256=array_hash(info['order']),
        query_sizes_sha256=array_hash(info['sizes']),training_target_sha256=array_hash(info['target']),
        training_query_keys_sha256=array_hash(info['keys']),held=info['held_audit'],all337_quality_columns=True,
        own_held_ad_labels_fit=False)
    del matrix,values,quality,cached,info['raw']
    gc.collect()
    emit('catboost_within_ranker_input_ready',day=day,partition=partition,independent=independent,shape=audit['shape'])
    return frame,cats,info,audit


def dependencies():
    paths=[Path(__file__),HERE/'catboost_ranker_recovery_v2.json',HERE/'catboost_ranker_api_smoke_v2.json',HERE/'test_catboost_within_ranker_v2.py',HERE/'run_catboost_within_ranker_v2.ps1',
        HERE/'within_user_ranking_core.py',HERE/'within_user_ranker.py',HERE/'test_within_user_ranking_core.py',
        HERE/'full_quality_row_catboost.py',HERE/'test_full_quality_row_catboost.py',HERE/'rich_row_catboost.py',
        HERE/'news_quality_features.py',HERE/'catboost_userday_residual.py',HERE/'fit_stacker.py',
        HERE/'conditional_daily.py',HERE/'rich_userday_activity.py',HERE/'verify_stacker.py',
        HERE/'global_row_context.py',HERE/'global_row_context15.py']
    for folder in [SOURCE,QUALITY,BASE]:
        report=read(folder/'verification.json')
        assert report['status']=='passed'
        paths += [folder/'verification.json',folder/'protocol.json']
    for day in [15,16]:
        paths += [SOURCE/f'input_contract{day}.json',QUALITY/f'input_contract{day}.json',QUALITY/f'quality{day}.npz',
            SOURCE/f'input_day{day}/features.npy',SOURCE/f'independent_day{day}/features.npy',
            BASE/f'day{day}_indices.npy',BASE/f'day{day}_prediction.npy']
    return list(dict.fromkeys(paths))


def preflight():
    assert not PREFLIGHT.exists(),'Preserve completed and failed GPU preflights'
    hashes=read(BASE/'protocol.json')['raw_datasets_sha256']
    for name,expected in hashes.items():
        assert digest(ROOT/'2026ml'/name)==expected,name
    PREFLIGHT.mkdir()
    write(PREFLIGHT/'protocol.json',dict(params=PARAMS,arms=ARMS,raw_datasets_sha256=hashes,
        sources_sha256={str(path):digest(path) for path in dependencies()},held_auc_evaluated=False))
    frame,cats,info,audit=inputs(16,'train')
    pool=Pool(frame,label=info['target'],group_id=query_ids(info['sizes'],info['target']),cat_features=cats)
    results=[]
    for arm in ARMS:
        model=CatBoostRanker(**dict(PARAMS,**{k:v for k,v in arm.items() if k!='name'},iterations=20,verbose=False))
        model.fit(pool)
        sample=Pool(frame.iloc[:2000],cat_features=cats)
        before=model.predict(sample,thread_count=2)
        path=PREFLIGHT/(arm['name']+'.cbm')
        save_model(model,path)
        after=load_model(path).predict(sample,prediction_type='RawFormulaVal',thread_count=2)
        error=float(np.max(np.abs(before-after)))
        assert error==0.
        results.append(dict(arm,serialization_maximum_error=error,model_sha256=digest(path)))
        del model,sample
        gc.collect()
    write(PREFLIGHT/'verification.json',dict(status='passed',source_code_sha256=digest(__file__),audit=audit,
        results=results,largest_full_prior_training_cohort_used=True,held_auc_evaluated=False,
        own_held_labels_fit=False,inferred17_labels_read=False))
    emit('catboost_within_ranker_preflight_passed',shape=audit['shape'],queries=len(info['sizes']),results=results)


def check_sources():
    protocol=read(OUT/'protocol.json')
    for path,expected in protocol['sources_sha256'].items():
        assert digest(path)==expected,path
    for name,expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name)==expected,name
    return protocol


def fit():
    assert not OUT.exists(),'Preserve completed and failed full fits'
    smoke=read(PREFLIGHT/'verification.json')
    assert smoke['status']=='passed' and smoke['source_code_sha256']==digest(__file__)
    frozen=read(PREFLIGHT/'protocol.json')
    for path,expected in frozen['sources_sha256'].items():
        assert digest(path)==expected,path
    for name,expected in frozen['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name)==expected,name
    paths=dependencies()+[PREFLIGHT/'protocol.json',PREFLIGHT/'verification.json']
    OUT.mkdir()
    write(OUT/'protocol.json',dict(params=PARAMS,arms=ARMS,checkpoints=CHECKPOINTS,weights=WEIGHTS,modes=MODES,
        training='Complete mixed-label user/day queries in the frozen strictly-prior advertisement sample',
        features='680 nonquality source row columns plus all337 content-quality columns; native categorical inputs',
        sampling='All prior positives and fixed one-third negatives; no inverse row weights in ranking',
        selection='All both-date model and candidate hashes lock before scoring; repeatedly reused development dates',
        own_held_ad_labels_fit=False,inferred17_labels_read=False,
        raw_datasets_sha256=frozen['raw_datasets_sha256'],sources_sha256={str(path):digest(path) for path in paths}))
    locks=[]
    for day in [15,16]:
        frame,cats,info,audit=inputs(day,'train')
        write(OUT/f'input_contract{day}.json',audit)
        np.save(OUT/f'day{day}_indices.npy',info['held_rows'])
        pool=Pool(frame,label=info['target'],group_id=query_ids(info['sizes'],info['target']),cat_features=cats)
        del frame
        gc.collect()
        for arm in ARMS:
            emit('catboost_within_ranker_fit_started',day=day,arm=arm['name'],queries=len(info['sizes']))
            model=CatBoostRanker(**dict(PARAMS,**{k:v for k,v in arm.items() if k!='name'},iterations=max(CHECKPOINTS)))
            model.fit(pool)
            for trees in CHECKPOINTS:
                name=f'day{day}_{arm["name"]}_trees{trees}'
                saved=model.copy()
                saved.shrink(trees)
                path=OUT/(name+'.cbm')
                save_model(saved,path)
                locks.append(dict(arm,day=day,trees=trees,name=name,model_sha256=digest(path),
                    input_contract_sha256=digest(OUT/f'input_contract{day}.json')))
                del saved
            del model
            gc.collect()
            emit('catboost_within_ranker_fit_complete',day=day,arm=arm['name'])
        del pool,info
        gc.collect()
    write(OUT/'models_locked.json',dict(protocol_sha256=digest(OUT/'protocol.json'),models=locks))
    check_sources()


def forecasts(day,independent=False):
    train,cats,info,training_audit=inputs(day,'train',independent)
    assert training_audit==read(OUT/f'input_contract{day}.json')
    del train,info
    gc.collect()
    frame,cats,info,audit=inputs(day,'held',independent)
    if independent:
        assert audit==read(OUT/f'held_contract{day}.json')
    else:
        write(OUT/f'held_contract{day}.json',audit)
    np.testing.assert_array_equal(info['held_rows'],np.load(OUT/f'day{day}_indices.npy'))
    pool=Pool(frame,cat_features=cats)
    del frame
    locked=read(OUT/'models_locked.json')
    assert locked['protocol_sha256']==digest(OUT/'protocol.json')
    result,last={},[]
    for record in locked['models']:
        if record['day']!=day:
            continue
        path=OUT/(record['name']+'.cbm')
        assert digest(path)==record['model_sha256']
        assert record['input_contract_sha256']==digest(OUT/f'input_contract{day}.json')
        value=load_model(path).predict(pool,prediction_type='RawFormulaVal',thread_count=2)
        assert np.isfinite(value).all()
        result[record['name']]=value
        if record['trees']==max(CHECKPOINTS):
            last.append(centered_unit(value,info['groups'],independent))
    assert len(last)==len(ARMS)
    result[f'day{day}_mean_final_within_unit']=np.mean(last,axis=0)
    for name,value in result.items():
        path=OUT/(name+'_held_margin.npy')
        if independent:
            np.testing.assert_array_equal(value,np.load(path))
        else:
            assert not path.exists()
            np.save(path,value)
    QUALITY_REPLAY.pop(day,None)
    gc.collect()
    emit('catboost_within_ranker_forecasts_ready',day=day,independent=independent)
    return result,info['held_rows'],info['groups']


def candidate_key(record):
    return f'{record["name"]}__{record["mode"]}__weight{record["weight"]:g}'


def candidates(day,values,groups,independent=False):
    base=np.load(BASE/f'day{day}_prediction.npy')
    yield dict(name='base',mode='base',weight=0.),base
    for name,margin in values.items():
        for mode in MODES:
            for weight in WEIGHTS:
                p=core.within_prediction(base,margin,groups,weight,mode,independent)
                yield dict(name=name,mode=mode,weight=weight),p


def score(independent=False):
    locked=read(OUT/'predictions_locked.json')
    actual=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    results={}
    for day in [15,16]:
        info=metadata(day)
        rows,groups=info['held_rows'],info['groups']
        np.testing.assert_array_equal(rows,np.flatnonzero(actual.pt_d.to_numpy()==20191100+day))
        y=actual.label.to_numpy()[rows]
        assert len(y)=={15:424780,16:421983}[day] and y.sum()=={15:3523,16:3724}[day]
        values={name:np.load(OUT/(name+'_held_margin.npy')) for name in locked['dates'][str(day)]['forecasts_sha256']}
        best,winner,records=None,None,[]
        maximum=0.
        base=np.load(BASE/f'day{day}_prediction.npy')
        for record,p in candidates(day,values,groups):
            assert array_hash(p)==locked['dates'][str(day)]['candidates_sha256'][candidate_key(record)]
            if independent and record['name']!='base':
                replay=core.within_prediction(base,values[record['name']],groups,record['weight'],record['mode'],True)
                error=float(np.max(np.abs(p-replay)))
                assert error<2e-14
                maximum=max(maximum,error)
            record=dict(record,auc=rank_sum_auc(y,p))
            records.append(record)
            if best is None or record['auc']>best['auc']:
                best,winner=record,p.copy()
        assert abs(roc_auc_score(y,winner)-best['auc'])<1e-12
        if independent:
            np.testing.assert_array_equal(winner,np.load(OUT/f'day{day}_prediction.npy'))
            assert best==read(OUT/'result.json')['dates'][str(20191100+day)]['best']
        else:
            np.save(OUT/f'day{day}_prediction.npy',winner)
        results[str(20191100+day)]=dict(auc=best['auc'],best=best,records=records,rows=len(y),clicks=int(y.sum()),
            evaluation_day_labels_fit=False,held_labels_used_to_select_fixed_recipe=True,
            all_both_date_models_and_candidate_hashes_locked_before_scoring=True,
            full337_quality_features_rebuilt_from_raw=independent,training_and_held_inputs_replayed=independent,
            saved_models_and_margin_hashes_replayed_exactly=independent,maximum_independent_pandas_error=maximum,
            target_met=best['auc']>={15:.815,16:.82}[day])
        emit('catboost_within_ranker_scored',day=day,best=best,independent=independent)
    check_sources()
    write(OUT/('verification.json' if independent else 'result.json'),dict(status='passed' if independent else 'evaluated',
        dates=results,original_datasets_unchanged=True,both_targets_met=all(item['target_met'] for item in results.values())))


def evaluate():
    check_sources()
    assert not (OUT/'predictions_locked.json').exists()
    dates={}
    for day in [15,16]:
        values,rows,groups=forecasts(day)
        dates[str(day)]=dict(forecasts_sha256={name:array_hash(value) for name,value in values.items()},
            candidates_sha256={candidate_key(record):array_hash(p) for record,p in candidates(day,values,groups)})
    write(OUT/'predictions_locked.json',dict(protocol_sha256=digest(OUT/'protocol.json'),
        models_lock_sha256=digest(OUT/'models_locked.json'),dates=dates,held_labels_opened_for_scoring=False))
    score()


def verify():
    check_sources()
    assert not (OUT/'verification.json').exists()
    locked=read(OUT/'predictions_locked.json')
    assert locked['protocol_sha256']==digest(OUT/'protocol.json')
    assert locked['models_lock_sha256']==digest(OUT/'models_locked.json')
    for day in [15,16]:
        values,rows,groups=forecasts(day,True)
        for name,value in values.items():
            assert array_hash(value)==locked['dates'][str(day)]['forecasts_sha256'][name]
        for record,p in candidates(day,values,groups):
            assert array_hash(p)==locked['dates'][str(day)]['candidates_sha256'][candidate_key(record)]
    score(True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=['preflight','fit','evaluate','verify'])
    {'preflight':preflight,'fit':fit,'evaluate':evaluate,'verify':verify}[parser.parse_args().phase]()
