"""Export the frozen day16 winning recipe on actual day17 inputs.

Use the already verified full-data v8 reference, frozen pre16 correction
models, and the frozen weighted/ranking neural checkpoints. No inferred
day17 outcome is opened here. Validation labels are used only for replay.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT/'analysis/day16_auc082_20261102'
RUN = HERE/'run_v1'
OUTPUT = ROOT/'output/best_recipe_20191117_20261102'
sys.path.insert(0, str(EXP))
from train_weighted import predict, configure
from features import FoldTransform, standardize_architecture, read, write, digest, emit
from source_profiles import load_profiles
from prepare_exposure_features import build_exposure_features, SINGLE
from prepare_snapshot_features import build_features as snapshot_features
from day_context import HISTOGRAM_FIELDS, HISTORY_FIELDS, past_exposure_features
from protected_day_model import ProtectedUserDayCTR
from rich_userday_activity import past_outcomes, correct
from extended_history_residual import enhanced_history
from personal_context_residual import personal_changes
from rich_group_residual import aggregate, prediction as residual_probability, lgb
from catboost_userday_residual import CATEGORIES, load_model, predict as cat_residual_probability
from catboost_daily_history import probability as daily_probability
from group_score_calibration import group_features, transform as calibration_design, FIELDS
import feed_behavior_userday as feed
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.special import expit, logit


def save_np(name, value):
    np.save(RUN/(name+'.npy'), value)


def joined(columns, **kwargs):
    return pd.concat([pd.read_csv(ROOT/'2026ml'/f, usecols=columns, **kwargs)
                      for f in ['train.csv', 'final_test.csv']], ignore_index=True)


def raw_context(raw, categories, numeric, names, meta, history):
    excluded = [i for i, n in enumerate(names) if n.startswith('target_stat__')]
    allowed = [i for i in range(len(names)) if i not in excluded]
    keys, groups = np.unique(raw[['pt_d','user_id']].to_numpy(dtype='int64'), axis=0, return_inverse=True)
    sizes = np.bincount(groups).astype('int32')
    membership = sparse.csr_matrix((1./sizes[groups], (groups,np.arange(len(groups)))), shape=(len(keys),len(raw)))
    means = np.empty((len(keys),len(allowed)), dtype='float32')
    for start in range(0,len(allowed),16):
        means[:,start:start+16] = membership @ numeric[:,allowed[start:start+16]]
    maximum = np.full_like(means,-np.inf)
    for start in range(0,len(raw),20000):
        np.maximum.at(maximum,groups[start:start+20000],numeric[start:start+20000,allowed])
    pieces = [means,maximum,np.log1p(sizes).astype('float32')[:,None]]
    for field in HISTOGRAM_FIELDS:
        column = meta['categorical_fields'].index(field)
        width = meta['vocabulary_sizes'][meta['group_for_field'][field]]
        pieces.append(sparse.csr_matrix((1./sizes[groups],(groups,categories[:,column])),shape=(len(keys),width)).toarray().astype('float32'))
    history_mean = (membership @ history).astype('float32')
    history_max = np.full((len(keys),len(HISTORY_FIELDS)),-np.inf,dtype='float32')
    np.maximum.at(history_max,groups,history)
    pieces.extend([history_mean,history_max,(np.log1p(sizes)-history_mean[:,3]).astype('float32')[:,None]])
    return keys,groups.astype('int32'),sizes,np.column_stack(pieces).astype('float32')


def prepare():
    if (RUN/'inputs_ready.json').exists():
        emit('using_saved_inputs')
        return
    RUN.mkdir(parents=True,exist_ok=True)
    expected = read(EXP/'runs/weighted_v1_day16/protocol.json')['raw_datasets_sha256']
    for name,h in expected.items():
        assert digest(ROOT/'2026ml'/name)==h, name
    assert read(EXP/'runs/compose_verified_v1/verification.json')['status']=='passed'
    protocol = {
        'status':'frozen_before_day17_evaluation','selected_development_auc':0.8088122948138387,
        'selected_recipe':'compose_verified_v1; rank_delta_1 plus 0.5 daily-history delta',
        'reference17':'Already verified full-data v8 reference (trained through November16)',
        'new_correction_models':'Existing frozen pre16 checkpoints; no new fitting',
        'feature_transform':'Frozen day16 vocabulary, target-stat models, scalers; day17 raw inputs and prior observed histories',
        'history_cutoff17':20191117,'inferred_day17_labels_used_for_prediction':False,
        'new_model_training':False,'raw_datasets_sha256':expected,'source_sha256':digest(__file__),
    }
    write(RUN/'protocol.json',protocol)
    transform = FoldTransform(20191116)
    transform.profiles = load_profiles(ROOT/'analysis/auc085')
    # The historical function selects dates earlier than each request.
    transform.past_feeds = pd.read_csv(ROOT/'2026ml/train_feeds.csv',usecols=['u_userId','e_et','label','pro','i_cat'],dtype='int32')
    arch = read(ROOT/'analysis/architecture_goal_20261029/cache/day16_oof/manifest.json')
    names = arch['numeric_fields'][:166]
    meta = transform.manifest
    raw = joined(['pt_d','user_id','task_id'],dtype='int32')
    selected = np.flatnonzero(raw.pt_d.to_numpy()>=20191116)
    dates = raw.pt_d.to_numpy()[selected]
    categories = np.empty((len(selected),27),dtype='int32')
    sequences = np.empty((len(selected),12,5),dtype='int32')
    numeric = np.empty((len(selected),166),dtype='float32')
    offset=0
    for filename in ['train.csv','final_test.csv']:
        for chunk in pd.read_csv(ROOT/'2026ml'/filename,usecols=transform.read_columns,
                                 dtype={f:str for f in transform.raw_sequences},chunksize=60000):
            chunk=chunk.loc[chunk.pt_d>=20191116]
            if not len(chunk): continue
            c,s,v=transform.transform(chunk)
            end=offset+len(chunk)
            categories[offset:end]=c
            sequences[offset:end]=s
            numeric[offset:end,:75]=standardize_architecture(v['signed'],names[:75],arch)
            offset=end
            emit('encode_final_neural_inputs',rows=offset,total=len(selected))
    assert offset==len(selected)
    del transform,c,s,v,chunk
    gc.collect()
    exposure,exposure_names=build_exposure_features(raw.assign(**{f:joined([f],dtype='int32')[f] for f in SINGLE if f not in raw}))
    assert exposure_names==names[75:118]
    numeric[:,75:118]=standardize_architecture(exposure[selected],exposure_names,arch)
    del exposure
    gc.collect()
    fields=['user_id','pt_d','task_id','adv_prim_id','spread_app_id']+[f'ad_{a}_list_v00{i}' for a in ['click','close'] for i in [1,2,3]]
    snapshot_raw=joined(fields,dtype={f:'category' if 'list' in f else 'int32' for f in fields})
    snapshot,snapshot_names=snapshot_features(snapshot_raw)
    assert snapshot_names==names[118:166]
    numeric[:,118:166]=standardize_architecture(snapshot[selected],snapshot_names,arch)
    del snapshot,snapshot_raw
    gc.collect()
    history=past_exposure_features(raw)[selected]
    keys,groups,sizes,values=raw_context(raw.iloc[selected],categories,numeric,names,meta,history)
    cm=read(EXP/'runs/weighted_v1_day16/context_manifest.json')
    values=np.clip((values-np.array(cm['scaler_mean']))/np.array(cm['scaler_std']),-8,8).astype('float32')
    old=np.load(EXP/'runs/weighted_v1_day16/context.npz')
    mask16=keys[:,0]==20191116
    np.testing.assert_array_equal(keys[mask16],old['keys'][old['keys'][:,0]==20191116])
    max_context_error=float(np.max(np.abs(values[mask16]-old['values'][old['keys'][:,0]==20191116])))
    assert max_context_error<2e-5, max_context_error
    # Retain the exact validated day16 context for replay; use new17 context with frozen scaler.
    values[mask16]=old['values'][old['keys'][:,0]==20191116]
    np.savez_compressed(RUN/'query_context.npz',keys=keys,groups=groups,sizes=sizes,values=values)
    np.savez_compressed(RUN/'all_context.npz',keys=np.concatenate([old['keys'],keys[~mask16]]),
                        sizes=np.concatenate([old['sizes'],sizes[~mask16]]),
                        values=np.concatenate([old['values'],values[~mask16]]))
    for name,a in [('categories',categories),('sequences',sequences),('numeric',numeric),('dates',dates),('global_rows',selected)]:save_np(name,a)
    write(RUN/'feature_manifest.json',{'meta':meta,'numeric_fields':names,'maximum_context16_reconstruction_error':max_context_error,
          'scaler':'Frozen pre16','matrix_shapes':{'categories':list(categories.shape),'numeric':list(numeric.shape)}})
    emit('features_ready',context_reconstruction_error=max_context_error)
    write(RUN/'inputs_ready.json',{'status':'ready','source_sha256':digest(__file__)})


def neural():
    if (RUN/'neural_ready.json').exists(): return
    meta=read(RUN/'feature_manifest.json')['meta']
    context=np.load(RUN/'query_context.npz')
    dates=np.load(RUN/'dates.npy')
    data={name:np.load(RUN/(name+'.npy'),mmap_mode='r') for name in ['categories','sequences','numeric']}
    data['day_rows']=context['groups']
    configure(42)
    tensors=torch.as_tensor(context['values'],device='cuda')
    results={}
    paths={'weighted':EXP/'runs/weighted_v1_day16/positive_weight_5/ema_best.pt',
           'rank':EXP/'runs/rank_finetune_v1_day16/t1_bce025/best.pt'}
    expected={'weighted':EXP/'runs/weighted_v1_day16/positive_weight_5/ema_best_prediction.npy',
              'rank':EXP/'runs/rank_finetune_v1_day16/t1_bce025/raw_prediction.npy'}
    for name,path in paths.items():
        model=ProtectedUserDayCTR(meta,.04,tensors,.1,use_context=True).cuda()
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        p=predict(model,data,np.arange(len(dates)),5.)
        err=float(np.max(np.abs(p[dates==20191116]-np.load(expected[name]))))
        assert err<2e-6,(name,err)
        save_np(name+'_raw',p)
        results[name]={'checkpoint_sha256':digest(path),'day16_maximum_replay_error':err}
        emit('neural_complete',name=name,day16_maximum_error=err)
        del model
        gc.collect();torch.cuda.empty_cache()
    write(RUN/'neural_ready.json',results)


class Pipeline:
    def __init__(self,day):
        self.day=day
        self.raw=joined(['pt_d','user_id','task_id','spread_app_id','slot_id','adv_prim_id']+CATEGORIES,dtype='int32')
        self.rows=np.flatnonzero(self.raw.pt_d.to_numpy()==day)
        self.frame=self.raw.iloc[self.rows].reset_index(drop=True)
        ac=np.load(RUN/'all_context.npz')
        mask=ac['keys'][:,0]<=day
        self.keys=ac['keys'][mask]; self.sizes=ac['sizes'][mask]; self.context=ac['values'][mask]
        y=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['label'],dtype='int8').label.to_numpy()
        labels=np.r_[y,np.full(len(self.raw)-len(y),-1,dtype='int8')]
        positions=pd.MultiIndex.from_arrays(self.keys.T).get_indexer(pd.MultiIndex.from_frame(self.raw[['pt_d','user_id']]))
        tr=self.raw.pt_d.to_numpy()<day
        self.counts=np.full(len(self.keys),-1.,dtype='float64')
        hist=self.keys[:,0]<day
        self.counts[hist]=np.bincount(positions[tr],weights=labels[tr],minlength=len(self.keys))[hist]
        self.query=self.keys[:,0]==day
        self.qkeys=self.keys[self.query]
        self.groups=pd.Index(self.qkeys[:,1]).get_indexer(self.frame.user_id)
        assert np.all(self.groups>=0)
        self.past,self.past_names=past_outcomes(self.keys,self.sizes,self.counts,day)
        self.enhanced,self.enhanced_names=enhanced_history(self.keys,self.sizes,self.counts,day)
        self.context_names=read(EXP/'runs/weighted_v1_day16/context_manifest.json')['features']
        self.common_names=self.context_names+self.past_names+['base_log_size','base_log_mass','base_mean_logit','base_max_logit']
        self.catframe=self.frame.drop_duplicates('user_id').set_index('user_id').reindex(self.qkeys[:,1])[CATEGORIES]
        assert not self.catframe.isna().any().any()
        self.corrections={}

    def common(self,base,extended=False,personal=False):
        mass,any_p,block=aggregate(base,self.groups,len(self.qkeys))
        arrays=[self.context[self.query],self.past[self.query],block]
        names=list(self.common_names)
        if extended:
            arrays.append(self.enhanced[self.query]);names+=self.enhanced_names
        if personal:
            arrays.append(personal_changes(self.keys,self.context)[self.query])
            names += ['previous_delta__'+s for s in self.context_names]+['personal_mean_delta__'+s for s in self.context_names]+['has_personal_context','log_personal_context_days']
        return {'x':np.column_stack(arrays).astype('float32'),'keys':self.qkeys,'groups':self.groups,
                'base':base,'mass':mass,'any':any_p,'count':np.full(len(self.qkeys),-1.),'names':names}

    def booster(self,folder,name):
        path=EXP/'runs'/folder/(name+'.txt')
        locks=read(EXP/'runs'/folder/'models_locked.json')['models']
        expected=next(r['model_sha256'] for r in locks if r['name']==name)
        assert digest(path)==expected
        return lgb.Booster(model_str=path.read_text(encoding='utf8'))

    def record(self,name,p):
        assert len(p)==len(self.rows) and np.isfinite(p).all() and np.all((p>=0)&(p<=1)),name
        save_np(f'day{self.day%100}_{name}',p)
        if self.day==20191116:
            filename='best_prediction.npy' if name=='weighted_calibration_transfer_v1' else 'prediction.npy'
            expected=np.load(EXP/'runs'/name/filename)
            error=float(np.max(np.abs(p-expected)))
            assert error<2e-5,(name,error)
            self.corrections[name]=error
        emit('recipe_stage',day=self.day,stage=name)
        return p

    def subgroup(self,field,base,selected):
        reference=self.common(self.reference)
        codes,unique=pd.factorize(pd.MultiIndex.from_frame(self.frame[['user_id',field]]),sort=True)
        users=np.array([k[0] for k in unique]);ids=np.array([k[1] for k in unique])
        ui=pd.Index(self.qkeys[:,1]).get_indexer(users)
        sizes=np.bincount(codes)
        user_size=np.bincount(self.groups)
        mass,any_p,block=aggregate(base,codes,len(unique))
        umass,_,_=aggregate(base,self.groups,len(self.qkeys))
        app_count=self.frame.groupby(['user_id',field],sort=True).size().groupby(level=0).size().reindex(users).to_numpy()
        extra=np.column_stack([np.log1p(app_count),sizes/user_size[ui],mass/umass[ui].clip(1e-8)])
        # Freeze category codes from original training+development inputs. New IDs get missing code -1.
        old_ids=pd.read_csv(ROOT/'2026ml/train.csv',usecols=[field],dtype='int32')[field]
        category=pd.Index(np.sort(old_ids.unique())).get_indexer(ids)
        return {'x':np.column_stack([reference['x'][ui][:,selected],block,extra,category]).astype('float32'),'keys':users,
                'groups':codes,'mass':mass,'any':any_p,'count':np.full(len(unique),-1.),'base':base}

    def news(self):
        schema=read(EXP/'runs/feed_behavior_userday_v1/news_schema.json')
        # Identical transformation with only the explicit query-date cutoff advanced.
        import inspect
        code=inspect.getsource(feed.feed_features).replace('20191116','cutoff')
        namespace={'np':np,'pd':pd,'READ':feed.READ,'CATEGORIES':feed.CATEGORIES,'DIVERSITY':feed.DIVERSITY,'cutoff':self.day}
        exec(compile(code,str(HERE/'date_bound_news_features'), 'exec'),namespace)
        feeds=pd.concat([pd.read_csv(ROOT/'2026ml'/f,usecols=feed.READ,dtype={'i_docId':str,'i_s_sourceId':str}) for f in ['train_feeds.csv','test_feeds.csv']],ignore_index=True)
        values,names,_=namespace['feed_features'](feeds,self.keys,schema)
        return values,names,personal_changes(self.keys,values)

    def run(self):
        dates=np.load(RUN/'dates.npy')
        weighted=np.load(RUN/'weighted_raw.npy')[dates==self.day]
        rank=np.load(RUN/'rank_raw.npy')[dates==self.day]
        self.reference=np.load(EXP.parents[0]/'userday_aux_20261101/reference_day_state_protected_seed42_v1/day16_prediction.npy') if self.day==20191116 else np.load(ROOT/'output/best_nextday_daystate_v8/ensemble_final_prediction.npy')
        base=.75*self.reference+.25*weighted
        model=np.load(EXP/'runs/group_score_calibration_v1/model.npz')
        gx,_=group_features(self.frame[FIELDS],base)
        calibrated=self.record('weighted_calibration_transfer_v1',expit(calibration_design(gx,base,model['center'],model['scale'])@model['beta']))
        rx=np.column_stack([self.context[self.query],self.past[self.query]]).astype('float32')
        gp=self.booster('rich_userday_activity_v1','context_history_binary_15').predict(rx)
        rich=self.record('rich_userday_activity_v1',correct(calibrated,gp,self.groups,'binary',.25))
        common=self.common(rich)
        pair=self.record('pairwise_group_boost_v1',expit(logit(rich.clip(1e-6,1-1e-6))+np.clip(self.booster('pairwise_group_boost_v1','t1_l7').predict(common['x'],raw_score=True),-3,3)[self.groups]))
        # Frozen meta model uses reference context with the chosen alternative base in its four score columns.
        common=self.common(calibrated)
        extended=self.common(calibrated,extended=True)
        cols=[]
        meta_protocol=read(EXP/'runs/group_meta_calibration_v1/protocol.json')
        selected=read(EXP/'runs/subgroup_residual_v1/protocol.json')['selected_context_columns']
        margin=logit(calibrated.clip(1e-6,1-1e-6))
        for s in meta_protocol['sources']:
            r=s['record'];kind=s['kind']
            d=self.subgroup(r['field'],calibrated,selected) if kind=='subgroup' else extended if kind=='extended' else common
            b=self.booster(Path(s['directory']).name,r['name'])
            if kind=='pair':p=expit(margin+r['power']*np.clip(b.predict(d['x'],raw_score=True),-3,3)[d['groups']])
            else:p=correct(calibrated,residual_probability(b,d,r['objective']),d['groups'],r['objective'],r['power'])
            cols.append(logit(p.clip(1e-6,1-1e-6))-margin)
        models=np.load(EXP/'runs/group_meta_calibration_v1/models.npz')
        prefix='logistic_0.01_'
        z=np.column_stack([np.ones(len(margin)),margin,np.clip((np.column_stack(cols)-models[prefix+'center'])/models[prefix+'scale'],-8,8)])
        meta_p=self.record('group_meta_calibration_v1',expit(z@models[prefix+'beta']))
        common=self.common(meta_p,extended=True)
        common['frame']=pd.DataFrame(common['x'],columns=[f'x{i}' for i in range(common['x'].shape[1])])
        for f in CATEGORIES:common['frame'][f]=self.catframe[f].to_numpy(dtype='int64')
        cb=load_model(EXP/'runs/catboost_userday_residual_v1/poisson_d6.cbm')
        cat_p=self.record('catboost_userday_residual_v1',correct(meta_p,cat_residual_probability(cb,common,'poisson'),self.groups,'poisson',.75))
        personal_d=self.common(pair,extended=True,personal=True)
        personal=self.record('personal_context_residual_v1',correct(pair,residual_probability(self.booster('personal_context_residual_v1','binary_7'),personal_d,'binary'),self.groups,'binary',.75))
        nv,nn,nc=self.news()
        news_cols=[i for i,n in enumerate(nn) if ('news__'+n).startswith('news_')]
        # context_rank appends every column named news_* in feed.prepare, including its historical deltas.
        full_news=np.column_stack([nv,nc]).astype('float32')
        context_d=self.common(cat_p,extended=True,personal=True)
        context_d['x']=np.column_stack([context_d['x'],full_news[self.query]]).astype('float32')
        cr=self.record('context_rank_boost_v1',expit(logit(cat_p.clip(1e-6,1-1e-6))+np.clip(self.booster('context_rank_boost_v1','t0.5_l15').predict(context_d['x'],raw_score=True),-3,3)[self.groups]))
        from previous_day_residual import apply_model,make_statistics
        if self.day==20191116:stats=dict(np.load(EXP/'runs/previous_day_residual_v1/statistics.npz'))
        else:
            from fit_stacker import REFERENCE
            earlier=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label','user_id','task_id','spread_app_id','slot_id','adv_prim_id'],dtype='int32')
            earlier=earlier.loc[earlier.pt_d==20191116].copy();earlier['pt_d']=20191115
            stats=make_statistics(earlier,np.load(REFERENCE/'day16_prediction.npy'))
            np.savez_compressed(RUN/'previous16_statistics.npz',**stats)
        previous=self.record('previous_day_residual_v1',expit(logit(cr.clip(1e-8,1-1e-8))+.5*apply_model(self.frame,stats,'user_app',3.)))
        daily_x=np.column_stack([self.context,self.past,self.enhanced,nv,nc]).astype('float32')[self.query]
        daily_frame=pd.DataFrame(daily_x,columns=[f'x{i}' for i in range(daily_x.shape[1])])
        for f in CATEGORIES:daily_frame[f]=self.catframe[f].to_numpy(dtype='int64')
        daily_frame['user_id']=self.qkeys[:,1]
        lock=read(EXP/'runs/catboost_daily_history_v1/models_locked.json')
        rec=next(r for r in lock['models'] if r['name']=='identity1_poisson')
        cb=load_model(EXP/'runs/catboost_daily_history_v1/identity1_poisson.cbm')
        daily=self.record('catboost_daily_history_v1',correct(personal,daily_probability(cb,daily_frame[rec['columns']],'poisson',rec['categories']),self.groups,'poisson',.125))
        rank_p=self.record('rank_finetune_v1_day16',expit(logit(previous.clip(1e-6,1-1e-6))+logit(rank.clip(1e-6,1-1e-6))-logit(weighted.clip(1e-6,1-1e-6))))
        final=self.record('compose_verified_v1',expit(logit(rank_p.clip(1e-6,1-1e-6))+.5*(logit(daily.clip(1e-6,1-1e-6))-logit(previous.clip(1e-6,1-1e-6)))))
        write(RUN/f'day{self.day%100}_pipeline_verification.json',{'status':'passed','component_maximum_replay_errors':self.corrections,'rows':len(final)})
        return final


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','neural','predict','all'],default='all')
    args=parser.parse_args()
    if args.phase in ['prepare','all']:prepare()
    if args.phase in ['neural','all']:neural()
    if args.phase in ['predict','all']:
        replay=Pipeline(20191116);replay.run();del replay;gc.collect()
        final=Pipeline(20191117).run()
        OUTPUT.mkdir(parents=True,exist_ok=True)
        np.save(OUTPUT/'prediction.npy',final)
        raw=pd.read_csv(ROOT/'2026ml/final_test.csv',usecols=['log_id','pt_d'])
        assert len(raw)==404182 and (raw.pt_d==20191117).all()
        pd.DataFrame({'id':np.arange(len(final)),'pctr':final}).to_csv(OUTPUT/'submission_best_20191117.csv',index=False,float_format='%.10f')
        pd.DataFrame({'log_id':raw.log_id,'pctr':final}).to_csv(OUTPUT/'submission_log_id_20191117.csv',index=False,float_format='%.10f')
        write(OUTPUT/'prediction_manifest.json',{'status':'prediction_frozen_before_label_evaluation','rows':len(final),
              'submission_sha256':digest(OUTPUT/'submission_best_20191117.csv'),'prediction_sha256':digest(OUTPUT/'prediction.npy'),
              'inferred_labels_used_for_prediction':False,'development16_auc':0.8088122948138387,
              'day17_reference':'full-data v8','day17_corrections':'frozen pre16; previous-day statistics advanced to observed16',
              'protocol_sha256':digest(RUN/'protocol.json'),'source_sha256':digest(__file__)})
        emit('submission_ready',path=str(OUTPUT/'submission_best_20191117.csv'))

if __name__=='__main__':main()
