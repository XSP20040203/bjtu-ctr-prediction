"""Full-quality explicit daily pair models with frozen chronological evaluation."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='2'
import argparse
import gc
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from catboost import CatBoostRanker,Pool
from sklearn.metrics import roc_auc_score
from compact_quality_daily_catboost import inputs as selected_inputs
from daily_auc_pair_core import sample_pairs
from daily_auc_pair_math import rate_baseline,daily_prediction
from catboost_userday_residual import save_model,load_model
from rich_row_catboost import frame_hash
from rich_userday_activity import array_hash
from fit_stacker import HERE,ROOT,digest,write
from conditional_daily import read,emit
from verify_stacker import rank_sum_auc

OUT=HERE/'runs/daily_auc_pair_ranker_v1'
PREFLIGHT=HERE/'runs/daily_auc_pair_ranker_preflight_v1'
API=HERE/'runs/daily_auc_pair_api_preflight_v2'
SOURCE=HERE/'runs/compact_quality_daily_catboost_v1'
BASE=HERE/'runs/catboost_within_ranker_v2'
ARMS=[dict(name='pair_d6_top384',depth=6,selector='poisson_d6_top384'),
      dict(name='pair_d7_top192',depth=7,selector='poisson_d7_top192')]
CHECKS=[200,600,1600]
WEIGHTS=[.025,.05,.1,.2,.35,.5,.75,1.]
POWERS=[.0625,.125,.25,.5]
PARAMS=dict(loss_function='PairLogit',learning_rate=.025,l2_leaf_reg=100.,random_strength=2.,
    bootstrap_type='Bernoulli',subsample=.8,border_count=32,one_hot_max_size=16,
    max_ctr_complexity=1,thread_count=2,random_seed=151646,task_type='GPU',devices='0',
    gpu_ram_part=.5,allow_writing_files=False,verbose=200)
META=['keys','sizes','counts','row_groups','rows','bayesian_mass']
PAIRS_PER_CLICK=48


def dependencies():
    paths=[Path(__file__),HERE/'run_daily_auc_pair_ranker_v1.ps1',
        HERE/'daily_auc_pair_core.py',HERE/'daily_auc_pair_math.py',
        HERE/'test_daily_auc_pair_core.py',HERE/'test_daily_auc_pair_math.py',
        HERE/'compact_quality_daily_catboost.py',HERE/'quality_daily_catboost.py',
        HERE/'news_quality_daily.py',HERE/'news_quality_features.py',HERE/'bayesian_daily_offset.py',
        HERE/'catboost_userday_residual.py',HERE/'rich_row_catboost.py',HERE/'rich_userday_activity.py',
        HERE/'fit_stacker.py',HERE/'conditional_daily.py',HERE/'verify_stacker.py',
        API/'protocol.json',API/'verification.json']
    for folder in [SOURCE,BASE]:
        assert read(folder/'verification.json')['status']=='passed'
        paths += [folder/'verification.json',folder/'protocol.json']
        parent=read(folder/'protocol.json')
        for filename,expected in parent['sources_sha256'].items():
            assert digest(filename)==expected,filename
            paths.append(Path(filename))
    for day in [15,16]:
        paths += [SOURCE/f'input_contract{day}.json',BASE/f'day{day}_prediction.npy',BASE/f'day{day}_indices.npy']
    return list(dict.fromkeys(paths))


def build_inputs(day):
    data,selector=selected_inputs(day)
    assert selector==read(SOURCE/f'input_contract{day}.json')
    choices={arm['name']:selector['selected_columns'][arm['selector']] for arm in ARMS}
    keep=set().union(*map(set,choices.values()))
    frame=data['frame'][[name for name in data['frame'] if name in keep]].copy()
    meta={name:np.asarray(data[name]).copy() for name in META}
    offset=rate_baseline(meta['bayesian_mass'],meta['sizes'])
    meta['baseline']=offset
    cutoff=20191100+day
    training,pairs,groups=sample_pairs(meta['keys'],meta['sizes'],meta['counts'],cutoff,
        PAIRS_PER_CLICK,151646)
    assert np.all(meta['keys'][training,0]<cutoff)
    assert np.all(meta['counts'][meta['keys'][:,0]>=cutoff]==-1)
    assert np.all(meta['keys'][meta['row_groups'],0]==cutoff)
    assert sum(name.startswith('quality__') for name in frame)==337
    contract=dict(day=cutoff,native_selector_contract=selector,shape=list(frame.shape),
        frame_sha256=frame_hash(frame),feature_names=list(frame),categories=data['categories'],
        selected_columns=choices,metadata_sha256={name:array_hash(value) for name,value in meta.items()},
        training_indices_sha256=array_hash(training),explicit_pairs_sha256=array_hash(pairs),
        date_groups_sha256=array_hash(groups),pair_count=len(pairs),training_groups=len(training),
        maximum_training_date=int(meta['keys'][training,0].max()),
        held_counts_masked=True,all337_quality_columns=True,selection_uses_held_labels=False)
    del data
    gc.collect()
    emit('daily_auc_pairs_input_ready',day=day,shape=contract['shape'],pairs=len(pairs),training_groups=len(training))
    return frame,meta,contract


def save_inputs(directory,frame,meta,contract):
    assert not directory.exists()
    directory.mkdir()
    frame.to_pickle(directory/'frame.pkl')
    np.savez(directory/'metadata.npz',**meta)
    write(directory/'contract.json',contract)
    write(directory/'files_locked.json',{name:digest(directory/name) for name in ['frame.pkl','metadata.npz','contract.json']})


def load_inputs(directory):
    for filename,expected in read(directory/'files_locked.json').items():
        assert digest(directory/filename)==expected,filename
    frame=pd.read_pickle(directory/'frame.pkl')
    saved=np.load(directory/'metadata.npz')
    meta={name:saved[name] for name in saved.files}
    contract=read(directory/'contract.json')
    assert frame_hash(frame)==contract['frame_sha256'] and list(frame)==contract['feature_names']
    assert {name:array_hash(value) for name,value in meta.items()}==contract['metadata_sha256']
    return frame,meta,contract


def training_pool(frame,meta,contract,arm):
    training,pairs,groups=sample_pairs(meta['keys'],meta['sizes'],meta['counts'],contract['day'],PAIRS_PER_CLICK,151646)
    assert array_hash(training)==contract['training_indices_sha256']
    assert array_hash(pairs)==contract['explicit_pairs_sha256']
    assert array_hash(groups)==contract['date_groups_sha256']
    target=meta['counts'][training]/meta['sizes'][training]
    columns=contract['selected_columns'][arm['name']]
    pool=Pool(frame.iloc[training][columns],label=target,group_id=groups,pairs=pairs,
        baseline=meta['baseline'][training],cat_features=contract['categories'])
    np.testing.assert_array_equal(pool.get_baseline().reshape(-1),meta['baseline'][training])
    return pool,columns,training


def check_sources():
    protocol=read(OUT/'protocol.json')
    for filename,expected in protocol['sources_sha256'].items():
        assert digest(filename)==expected,filename
    for filename,expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/filename)==expected,filename
    return protocol


def preflight():
    assert not PREFLIGHT.exists(),'Preserve previous preflights'
    assert read(API/'verification.json')['status']=='passed'
    paths=dependencies()
    hashes=read(BASE/'protocol.json')['raw_datasets_sha256']
    for filename,expected in hashes.items():
        assert digest(ROOT/'2026ml'/filename)==expected,filename
    PREFLIGHT.mkdir()
    write(PREFLIGHT/'protocol.json',dict(params=PARAMS,arms=ARMS,pairs_per_click=PAIRS_PER_CLICK,
        raw_datasets_sha256=hashes,sources_sha256={str(path):digest(path) for path in paths},
        no_held_auc_evaluation=True,inferred17_labels_read=False))
    frame,meta,contract=build_inputs(16)
    save_inputs(PREFLIGHT/'input_day16',frame,meta,contract)
    results=[]
    for arm in ARMS:
        pool,columns,training=training_pool(frame,meta,contract,arm)
        model=CatBoostRanker(**dict(PARAMS,depth=arm['depth'],iterations=20,verbose=False))
        model.fit(pool)
        sample=Pool(frame.iloc[training[:2000]][columns],cat_features=contract['categories'])
        prediction=model.predict(sample,thread_count=2)
        path=PREFLIGHT/(arm['name']+'.cbm')
        save_model(model,path)
        replay=load_model(path).predict(sample,prediction_type='RawFormulaVal',thread_count=2)
        np.testing.assert_array_equal(prediction,replay)
        results.append(dict(arm,shape=[len(training),len(columns)],serialization_maximum_error=0.))
        del pool,model,sample
        gc.collect()
    write(PREFLIGHT/'verification.json',dict(status='passed',source_code_sha256=digest(__file__),
        results=results,largest_full_prior_cohort=True,contract_sha256=digest(PREFLIGHT/'input_day16/contract.json'),
        inferred17_labels_read=False,held_auc_evaluated=False))
    emit('daily_auc_pairs_preflight_passed',results=results)


def fit():
    assert not OUT.exists(),'Preserve all previous fits'
    smoke=read(PREFLIGHT/'verification.json')
    assert smoke['status']=='passed' and smoke['source_code_sha256']==digest(__file__)
    parent=read(PREFLIGHT/'protocol.json')
    for filename,expected in parent['sources_sha256'].items():
        assert digest(filename)==expected,filename
    for filename,expected in parent['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/filename)==expected,filename
    paths=dependencies()+[PREFLIGHT/'protocol.json',PREFLIGHT/'verification.json',PREFLIGHT/'input_day16/files_locked.json']
    OUT.mkdir()
    write(OUT/'protocol.json',dict(params=PARAMS,arms=ARMS,checkpoints=CHECKS,weights=WEIGHTS,powers=POWERS,
        baseline_run=str(BASE),pairs_per_click=PAIRS_PER_CLICK,dates=[20191115,20191116],
        objective='Explicit click vs nonclick exposure pairs within earlier advertisement dates',
        baseline='Prior Bayesian per-row CTR; float32 Pool representation; add exactly once at prediction',
        feature_selection='Frozen strictly-prior Poisson model importance; all337 quality columns and native categories',
        selection='Both-date models and candidates lock before held scoring; repeatedly reused development dates',
        own_held_ad_labels_fit=False,inferred17_labels_read=False,
        raw_datasets_sha256=parent['raw_datasets_sha256'],sources_sha256={str(path):digest(path) for path in paths}))
    locks=[]
    for day in [15,16]:
        directory=OUT/f'input_day{day}'
        if day==16:
            shutil.copytree(PREFLIGHT/'input_day16',directory)
            frame,meta,contract=load_inputs(directory)
        else:
            frame,meta,contract=build_inputs(day)
            save_inputs(directory,frame,meta,contract)
        np.save(OUT/f'day{day}_indices.npy',meta['rows'])
        for arm in ARMS:
            pool,columns,training=training_pool(frame,meta,contract,arm)
            emit('daily_auc_pairs_fit_started',day=day,arm=arm['name'],shape=[len(training),len(columns)])
            model=CatBoostRanker(**dict(PARAMS,depth=arm['depth'],iterations=max(CHECKS)))
            model.fit(pool)
            for trees in CHECKS:
                name=f'day{day}_{arm["name"]}_trees{trees}'
                path=OUT/(name+'.cbm')
                saved=model.copy()
                saved.shrink(trees)
                save_model(saved,path)
                locks.append(dict(arm,day=day,trees=trees,name=name,columns=columns,
                    model_sha256=digest(path),input_contract_sha256=digest(directory/'contract.json')))
                del saved
            emit('daily_auc_pairs_fit_complete',day=day,arm=arm['name'])
            del pool,model
            gc.collect()
        del frame,meta
        gc.collect()
    write(OUT/'models_locked.json',dict(protocol_sha256=digest(OUT/'protocol.json'),models=locks))
    check_sources()


def forecasts(day,frame,meta,contract,independent=False):
    held=np.flatnonzero(meta['keys'][:,0]==20191100+day)
    assert np.all(meta['counts'][held]==-1)
    locked=read(OUT/'models_locked.json')
    assert locked['protocol_sha256']==digest(OUT/'protocol.json')
    result={}
    for record in locked['models']:
        if record['day']!=day:
            continue
        assert record['input_contract_sha256']==digest(OUT/f'input_day{day}/contract.json')
        assert record['columns']==contract['selected_columns'][record['name'].split('_trees')[0][6:]]
        path=OUT/(record['name']+'.cbm')
        assert digest(path)==record['model_sha256']
        model=load_model(path)
        assert model.feature_names_==record['columns']
        margin=model.predict(Pool(frame.iloc[held][record['columns']],cat_features=contract['categories']),
            prediction_type='RawFormulaVal',thread_count=2)
        assert np.isfinite(margin).all()
        path=OUT/(record['name']+'_held_margin.npy')
        if independent:
            np.testing.assert_array_equal(margin,np.load(path))
        else:
            assert not path.exists()
            np.save(path,margin)
        result[record['name']]=margin
    assert len(result)==6
    return result


def candidates(day,meta,values,independent=False):
    base=np.load(BASE/f'day{day}_prediction.npy')
    np.testing.assert_array_equal(meta['rows'],np.load(BASE/f'day{day}_indices.npy'))
    held=np.flatnonzero(meta['keys'][:,0]==20191100+day)
    yield dict(name='base',mode='base',weight=0.),base
    for name,margin in values.items():
        residual=np.zeros(len(meta['keys']))
        residual[held]=margin
        total=residual+meta['baseline']
        recipes=[(mode,weight) for mode in ['residual_add','rate_replace'] for weight in WEIGHTS]
        recipes += [('count_mass',power) for power in POWERS]
        for mode,weight in recipes:
            prediction=daily_prediction(base,residual,total,meta['sizes'],meta['row_groups'],weight,mode,independent)
            yield dict(name=name,mode=mode,weight=weight),prediction


def candidate_key(record):
    return f'{record["name"]}__{record["mode"]}__{record["weight"]:g}'


def score(independent=False):
    lock=read(OUT/'predictions_locked.json')
    actual=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    reports={}
    for day in [15,16]:
        frame,meta,contract=load_inputs(OUT/f'input_day{day}')
        del frame
        rows=np.flatnonzero(actual.pt_d.to_numpy()==20191100+day)
        np.testing.assert_array_equal(rows,meta['rows'])
        y=actual.label.to_numpy()[rows]
        assert len(y)=={15:424780,16:421983}[day] and y.sum()=={15:3523,16:3724}[day]
        values={name:np.load(OUT/(name+'_held_margin.npy')) for name in lock['dates'][str(day)]['forecasts_sha256']}
        best,winner,records=None,None,[]
        maximum=0.
        for record,p in candidates(day,meta,values):
            assert array_hash(p)==lock['dates'][str(day)]['candidates_sha256'][candidate_key(record)]
            if independent and record['name']!='base':
                residual=np.zeros(len(meta['keys']))
                held=np.flatnonzero(meta['keys'][:,0]==20191100+day)
                residual[held]=values[record['name']]
                q=daily_prediction(np.load(BASE/f'day{day}_prediction.npy'),residual,residual+meta['baseline'],
                    meta['sizes'],meta['row_groups'],record['weight'],record['mode'],True)
                error=float(np.max(np.abs(p-q)))
                assert error<2e-14
                maximum=max(maximum,error)
            record=dict(record,auc=rank_sum_auc(y,p))
            records.append(record)
            if best is None or record['auc']>best['auc']:
                best,winner=record,p.copy()
        assert len(records)==121 and abs(roc_auc_score(y,winner)-best['auc'])<1e-12
        if independent:
            np.testing.assert_array_equal(winner,np.load(OUT/f'day{day}_prediction.npy'))
            assert best==read(OUT/'result.json')['dates'][str(20191100+day)]['best']
        else:
            np.save(OUT/f'day{day}_prediction.npy',winner)
        reports[str(20191100+day)]=dict(auc=best['auc'],best=best,records=records,rows=len(y),clicks=int(y.sum()),
            evaluation_day_labels_fit=False,held_labels_used_to_select_fixed_recipe=True,
            all_both_date_models_and_candidate_hashes_locked_before_scoring=True,
            prior_feature_selection_and_all337_quality_rebuilt_from_raw=independent,
            saved_models_and_margins_replayed_exactly=independent,maximum_independent_pandas_error=maximum,
            target_met=best['auc']>={15:.815,16:.82}[day])
        emit('daily_auc_pairs_scored',day=day,best=best,independent=independent)
    check_sources()
    write(OUT/('verification.json' if independent else 'result.json'),dict(status='passed' if independent else 'evaluated',
        dates=reports,original_datasets_unchanged=True,both_targets_met=all(r['target_met'] for r in reports.values())))


def evaluate():
    check_sources()
    assert not (OUT/'predictions_locked.json').exists()
    dates={}
    for day in [15,16]:
        frame,meta,contract=load_inputs(OUT/f'input_day{day}')
        values=forecasts(day,frame,meta,contract)
        dates[str(day)]=dict(forecasts_sha256={name:array_hash(value) for name,value in values.items()},
            candidates_sha256={candidate_key(record):array_hash(p) for record,p in candidates(day,meta,values)})
        del frame,meta,values
        gc.collect()
    write(OUT/'predictions_locked.json',dict(protocol_sha256=digest(OUT/'protocol.json'),
        models_lock_sha256=digest(OUT/'models_locked.json'),dates=dates,held_labels_opened_for_scoring=False))
    score()


def verify():
    check_sources()
    assert not (OUT/'verification.json').exists()
    lock=read(OUT/'predictions_locked.json')
    assert lock['protocol_sha256']==digest(OUT/'protocol.json')
    assert lock['models_lock_sha256']==digest(OUT/'models_locked.json')
    for day in [15,16]:
        frame,meta,contract=build_inputs(day)
        assert contract==read(OUT/f'input_day{day}/contract.json')
        values=forecasts(day,frame,meta,contract,True)
        for name,margin in values.items():
            assert array_hash(margin)==lock['dates'][str(day)]['forecasts_sha256'][name]
        for record,p in candidates(day,meta,values):
            assert array_hash(p)==lock['dates'][str(day)]['candidates_sha256'][candidate_key(record)]
        del frame,meta,values
        gc.collect()
    score(True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=['preflight','fit','evaluate','verify'])
    {'preflight':preflight,'fit':fit,'evaluate':evaluate,'verify':verify}[parser.parse_args().phase]()
