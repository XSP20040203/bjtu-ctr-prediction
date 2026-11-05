"""Complete15 evaluation of causal daily-response models trained strictly pre15."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rich_userday_activity import load_day,correct,array_hash
from extended_history_residual import enhanced_history
from personal_context_residual import personal_changes
from feed_behavior_userday import feed_features,READ,CATEGORIES
from fit_stacker import HERE,ROOT,REFERENCE,digest,write
from verify_stacker import rank_sum_auc
from rich_group_residual import lgb

OUT=HERE/'runs/rolling_daily15_v1'
EXPORT=ROOT/'output/two_day_auc_targets_20261102'
CHECKS=[300,700]
POWERS=[0.,.125,.25,.5,.75,1.]
PARAMS={'learning_rate':.025,'min_data_in_leaf':100,'lambda_l2':20.,
        'feature_fraction':.75,'max_bin':31,'num_threads':2,'seed':15162,
        'verbosity':-1,'deterministic':True,'force_col_wise':True,'feature_pre_filter':False}


def prepare():
    data=load_day(15,True,evaluation_labels=False)
    assert np.max(data['keys'][:,0])==20191115
    assert np.all(data['counts'][data['keys'][:,0]>=20191115]==-1)
    feeds=pd.read_csv(ROOT/'2026ml/train_feeds.csv',usecols=READ,dtype={'i_docId':str,'i_s_sourceId':str})
    feeds=feeds.loc[feeds.e_et<=20191115]
    earlier=feeds.e_et<20191115
    schema={name:[int(v) for v in feeds.loc[earlier,name].value_counts().index[:16]] for name in CATEGORIES}
    values,names,_=feed_features(feeds,data['keys'],schema)
    nc=personal_changes(data['keys'],values)
    history,hn=enhanced_history(data['keys'],data['sizes'],data['counts'],20191115)
    # Feature selection uses no advertisement outcome on either held date.
    data['x']=np.column_stack([data['x'],history,values,nc]).astype('float32')
    data['names']+=hn+['news__'+n for n in names]
    data['names']+=['news_previous_delta__'+n for n in names]+['news_personal_mean_delta__'+n for n in names]+['news_has_previous','news_log_previous_days']
    assert data['x'].shape[1]==len(data['names']) and np.isfinite(data['x']).all()
    return data,schema


def source_paths():
    return [Path(__file__),HERE/'rich_userday_activity.py',HERE/'extended_history_residual.py',
            HERE/'personal_context_residual.py',HERE/'feed_behavior_userday.py',
            ROOT/'analysis/userday_aux_20261101/runs/protected_day_aux_v1_day15_seed42/context.npz',
            ROOT/'analysis/userday_aux_20261101/runs/protected_day_aux_v1_day15_seed42/context_manifest.json',
            REFERENCE/'day15_prediction.npy',REFERENCE/'day15_indices.npy']


def fit():
    if OUT.exists():raise FileExistsError('Preserving genuine day15 run')
    expected=json.loads((HERE/'runs/weighted_v1_day16/protocol.json').read_text(encoding='utf8'))['raw_datasets_sha256']
    for name,h in expected.items():assert digest(ROOT/'2026ml'/name)==h,name
    OUT.mkdir(parents=True)
    write(OUT/'protocol.json',{'day':20191115,'required_auc':.815,'training_cutoff_exclusive':20191115,
          'params':PARAMS,'objectives':['binary','poisson'],'leaves':[15,31],
          'checkpoints':CHECKS,'powers':POWERS,'evaluation15_labels_fit':False,
          'held_label_feature_selection':False,'schema':'All context15 columns, past outcomes, enhanced history, pre15-news category schema, causal news deltas',
          'selection':'Fixed checkpoints locked before complete15 outcome evaluation;15 is development data',
          'raw_datasets_sha256':expected,'sources_sha256':{str(p):digest(p) for p in source_paths()}})
    data,schema=prepare()
    train=np.flatnonzero(data['keys'][:,0]<20191115)
    assert np.all(data['counts'][train]>=0)
    write(OUT/'input_contract.json',{'x_sha256':array_hash(data['x']),'keys_sha256':array_hash(data['keys']),
          'counts_sha256':array_hash(data['counts']),'feature_names':data['names'],'schema':schema,
          'train_groups':len(train),'max_training_date':int(data['keys'][train,0].max()),
          'held_groups':int(data['valid'].sum()),'held_count_values':[-1]})
    np.save(OUT/'valid_indices.npy',data['rows'])
    print(json.dumps({'event':'rolling15_ready','shape':list(data['x'].shape),'training_groups':len(train)}),flush=True)
    locks=[]
    for objective in ['binary','poisson']:
        y=data['counts'][train].copy()
        if objective=='binary':y=(y>0).astype('float64')
        dataset=lgb.Dataset(data['x'][train],label=y,free_raw_data=True,params=PARAMS)
        for leaves in [15,31]:
            def checkpoint(env):
                trees=env.iteration+1
                if trees in CHECKS:
                    name=f'{objective}_l{leaves}_trees{trees}'
                    path=OUT/(name+'.txt');path.write_text(env.model.model_to_string(),encoding='utf8')
                    locks.append({'name':name,'objective':objective,'trees':trees,'leaves':leaves,'model_sha256':digest(path)})
                    print(json.dumps({'event':'rolling15_checkpoint','name':name}),flush=True)
            lgb.train(dict(PARAMS,objective=objective,metric='None',num_leaves=leaves),dataset,
                      num_boost_round=max(CHECKS),callbacks=[checkpoint])
    write(OUT/'models_locked.json',{'models':locks,'protocol_sha256':digest(OUT/'protocol.json')})
    evaluate(False,data)


def evaluate(verifying,data=None):
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    locks=json.loads((OUT/'models_locked.json').read_text(encoding='utf8'))
    assert locks['protocol_sha256']==digest(OUT/'protocol.json')
    for p,h in protocol['sources_sha256'].items():assert digest(p)==h,p
    for name,h in protocol['raw_datasets_sha256'].items():assert digest(ROOT/'2026ml'/name)==h,name
    if verifying:data,_=prepare()
    contract=json.loads((OUT/'input_contract.json').read_text(encoding='utf8'))
    for n in ['x','keys','counts']:assert array_hash(data[n])==contract[n+'_sha256'],n
    np.testing.assert_array_equal(data['rows'],np.load(OUT/'valid_indices.npy'))
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label','log_id','user_id'],dtype='int32')
    rows=np.flatnonzero(raw.pt_d.to_numpy()==20191115)
    np.testing.assert_array_equal(data['rows'],rows)
    y=raw.label.to_numpy()[rows]
    base=np.load(REFERENCE/'day15_prediction.npy')
    best={'name':'reference','power':0.,'auc':rank_sum_auc(y,base)}
    best_p=base.copy();records=[];error=0.
    for r in locks['models']:
        path=OUT/(r['name']+'.txt');assert digest(path)==r['model_sha256']
        b=lgb.Booster(model_str=path.read_text(encoding='utf8'))
        gp=np.zeros(len(data['keys']),dtype='float64');gp[data['valid']]=b.predict(data['x'][data['valid']])
        for power in POWERS:
            p=correct(base,gp,data['row_groups'],r['objective'],power)
            record={'name':r['name'],'power':power,'auc':rank_sum_auc(y,p)}
            records.append(record)
            if record['auc']>best['auc']:best=record;best_p=p.copy()
    if verifying:
        error=float(np.max(np.abs(best_p-np.load(OUT/'prediction.npy'))));assert error<1e-12
    else:np.save(OUT/'prediction.npy',best_p)
    result={'status':'passed' if verifying else 'evaluated','day':20191115,'auc':best['auc'],
            'reference_auc':rank_sum_auc(y,base),'best':best,'records':records,'rows':len(rows),'clicks':int(y.sum()),
            'maximum_replay_error':error,'original_datasets_unchanged':True,
            'evaluation15_labels_fit':False,'held_label_feature_selection':False,
            'development15_model_selection':True,'target_auc':.815,'target_met':best['auc']>=.815}
    write(OUT/('verification.json' if verifying else 'result.json'),result)
    if verifying:
        EXPORT.mkdir(parents=True,exist_ok=True)
        frame=pd.DataFrame({'row_index':rows,'log_id':raw.log_id.to_numpy()[rows],
                            'user_id':raw.user_id.to_numpy()[rows],'label':y,'pctr':best_p})
        path=EXPORT/'day15_predictions.csv';frame.to_csv(path,index=False,float_format='%.17g')
        replay=pd.read_csv(path,float_precision='round_trip')
        np.testing.assert_array_equal(replay.pctr,best_p)
        assert abs(rank_sum_auc(y,replay.pctr.to_numpy())-best['auc'])<1e-12
        write(EXPORT/'day15_verification.json',dict(result,submission_sha256=digest(path)))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    evaluate(True) if parser.parse_args().verify else fit()
