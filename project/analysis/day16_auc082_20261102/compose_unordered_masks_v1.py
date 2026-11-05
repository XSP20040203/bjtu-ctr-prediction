"""Mask fixed snapshot-model blends using label-free same-date input conditions."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from fit_stacker import ROOT,HERE,digest,write
from conditional_daily import read,emit
from rich_userday_activity import array_hash
from verify_stacker import rank_sum_auc
from unordered_snapshot_core_v1 import FIELDS,CHANNELS,METRICS,build

SOURCE=HERE/'runs/unordered_snapshot_ranker_v1'
BASE=HERE/'runs/compact_full_sample_ranker_v1'
OUT=HERE/'runs/compose_unordered_masks_v1'
WEIGHTS=[.05,.1,.2,.35,.5,.75,1.]
MASK_NAMES=['changed_any','changed_click','added_click','unique_click']


def masked_blend(base,margin,mask,weight,independent=False):
    base=np.asarray(base,dtype='float64')
    margin=np.asarray(margin,dtype='float64')
    mask=np.asarray(mask,dtype=bool)
    if base.shape!=margin.shape or base.shape!=mask.shape or not 0<=weight<=1:
        raise ValueError('Aligned arrays and a convex weight are required')
    z=logit(np.clip(base,1e-8,1-1e-8))
    target=(margin-margin.mean())/max(float(margin.std()),1e-6)*z.std()+z.mean()
    delta=weight*(target-z)
    if independent:
        odds=np.exp(z)*np.exp(delta)
        adjusted=odds/(1+odds)
    else:
        adjusted=expit(z+delta)
    return np.where(mask,adjusted,base)


def masks_for_day(day,independent=False):
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=FIELDS,
        dtype={name:str for name in FIELDS if 'list' in name})
    rows=np.flatnonzero(raw.pt_d.to_numpy()==20191100+day)
    values,names=build(raw.iloc[rows],independent=independent)
    contract=read(SOURCE/f'held_contract{day}.json')
    assert array_hash(values)==contract['unordered_features_sha256']
    np.testing.assert_array_equal(rows,np.load(SOURCE/f'day{day}_indices.npy'))
    width=len(METRICS)
    click=values[:,:3*width]
    masks=dict(changed_any=np.any(values!=0,axis=1),changed_click=np.any(click!=0,axis=1),
        added_click=np.any(click[:,[width*i+4 for i in range(3)]]>0,axis=1),
        unique_click=np.any(click[:,[width*i+8 for i in range(3)]]>0,axis=1))
    assert list(masks)==MASK_NAMES
    return rows,masks


def dependencies():
    paths=[Path(__file__),HERE/'test_compose_unordered_masks_v1.py',
        HERE/'run_compose_unordered_masks_v1.ps1',HERE/'start_compose_unordered_masks_queue_v1.ps1',
        HERE/'unordered_snapshot_core_v1.py',HERE/'fit_stacker.py',HERE/'conditional_daily.py',
        HERE/'rich_userday_activity.py',HERE/'verify_stacker.py']
    for directory in [SOURCE,BASE]:
        assert read(directory/'verification.json')['status']=='passed'
        for filename,expected in read(directory/'protocol.json')['sources_sha256'].items():
            assert digest(filename)==expected,filename
        paths += [directory/'verification.json',directory/'protocol.json']
        for day in [15,16]:
            paths += [directory/f'day{day}_prediction.npy',directory/f'day{day}_indices.npy']
    locked=read(SOURCE/'models_locked.json')
    paths += [SOURCE/'models_locked.json']
    for record in locked['models']:
        path=SOURCE/(record['name']+'_held_margin.npy')
        paths.append(path)
    paths += [SOURCE/f'held_contract{day}.json' for day in [15,16]]
    return list(dict.fromkeys(paths))


def check():
    protocol=read(OUT/'protocol.json')
    for filename,expected in protocol['sources_sha256'].items():
        assert digest(filename)==expected,filename
    for filename,expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/filename)==expected,filename
    return protocol


def candidates(day,masks,independent=False):
    base=np.load(BASE/f'day{day}_prediction.npy')
    current=np.load(SOURCE/f'day{day}_prediction.npy')
    yield dict(name='best_snapshot_model',mask='all',weight=0.),current
    yield dict(name='compact_parent',mask='all',weight=0.),base
    for record in read(SOURCE/'models_locked.json')['models']:
        if record['day']!=day:
            continue
        margin=np.load(SOURCE/(record['name']+'_held_margin.npy'))
        for mask_name,mask in masks.items():
            for weight in WEIGHTS:
                yield dict(name=record['name'],mask=mask_name,weight=weight),masked_blend(base,margin,mask,weight,independent)


def key(record):
    return f'{record["name"]}__{record["mask"]}__{record["weight"]:g}'


def prepare():
    assert not OUT.exists(),'Preserve previous runs'
    paths=dependencies()
    OUT.mkdir()
    write(OUT/'protocol.json',dict(weights=WEIGHTS,masks=MASK_NAMES,
        training='No additional model fitting; fixed, independently replayed parent models',
        mask_definition='Label-free current-date input state changes; selected from prior14 development only',
        current_day_advertisement_labels_used_to_build_masks=False,inferred17_labels_read=False,
        selection='Both-date candidate hashes locked before scoring; repeatedly reused development dates',
        sources_sha256={str(path):digest(path) for path in paths},
        raw_datasets_sha256=read(SOURCE/'protocol.json')['raw_datasets_sha256']))
    records={}
    for day in [15,16]:
        rows,masks=masks_for_day(day)
        np.save(OUT/f'day{day}_indices.npy',rows)
        np.savez_compressed(OUT/f'day{day}_masks.npz',**masks)
        records[str(day)]=dict(rows_sha256=array_hash(rows),masks_sha256={n:array_hash(m) for n,m in masks.items()},
            mask_counts={n:int(m.sum()) for n,m in masks.items()},
            predictions={key(r):array_hash(p) for r,p in candidates(day,masks)})
    write(OUT/'predictions_locked.json',dict(protocol_sha256=digest(OUT/'protocol.json'),dates=records,
        both_date_models_verified=True,held_labels_opened_for_scoring=False))
    check()
    emit('unordered_mask_candidates_locked',counts={d:r['mask_counts'] for d,r in records.items()})


def verify():
    check()
    locks=read(OUT/'predictions_locked.json')
    assert locks['protocol_sha256']==digest(OUT/'protocol.json')
    verified_masks={}
    for day in [15,16]:
        rows,masks=masks_for_day(day,True)
        saved=locks['dates'][str(day)]
        assert array_hash(rows)==saved['rows_sha256']
        for name,mask in masks.items():
            assert array_hash(mask)==saved['masks_sha256'][name]
            with np.load(OUT/f'day{day}_masks.npz') as cached:
                np.testing.assert_array_equal(mask,cached[name])
        for r,p in candidates(day,masks):
            assert array_hash(p)==saved['predictions'][key(r)]
        verified_masks[day]=(rows,masks)
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    reports={}
    for day in [15,16]:
        rows,masks=verified_masks[day]
        np.testing.assert_array_equal(rows,np.flatnonzero(raw.pt_d.to_numpy()==20191100+day))
        y=raw.label.to_numpy()[rows]
        assert len(rows)=={15:424780,16:421983}[day] and y.sum()=={15:3523,16:3724}[day]
        records=[]
        error=0.
        best_value=-1.
        for (r,p),(other,q) in zip(candidates(day,masks),candidates(day,masks,True)):
            assert r==other and np.isfinite(p).all() and np.all((p>0)&(p<1))
            local=float(np.max(np.abs(p-q)))
            assert local<2e-14
            error=max(error,local)
            auc=rank_sum_auc(y,p)
            record=dict(r,auc=auc)
            records.append(record)
            if auc>best_value:
                best_value=auc; best=record; best_prediction=p.copy()
        assert abs(roc_auc_score(y,best_prediction)-best_value)<1e-12
        np.save(OUT/f'day{day}_prediction.npy',best_prediction)
        reports[str(20191100+day)]=dict(auc=best_value,best=best,records=records,
            rows=len(rows),clicks=int(y.sum()),evaluation_day_labels_fit=False,
            held_labels_used_to_select_fixed_recipe=True,mask_counts={n:int(m.sum()) for n,m in masks.items()},
            masks_rebuilt_independently=True,parent_model_replay_verification_required=True,
            all_both_date_candidates_locked_before_scoring=True,maximum_independent_prediction_error=error,
            target_met=best_value>={15:.815,16:.82}[day])
        emit('unordered_mask_scored',day=day,best=best)
    check()
    write(OUT/'verification.json',dict(status='passed',dates=reports,original_datasets_unchanged=True,
        both_targets_met=all(r['target_met'] for r in reports.values())))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=['prepare','verify'])
    {'prepare':prepare,'verify':verify}[parser.parse_args().phase]()
