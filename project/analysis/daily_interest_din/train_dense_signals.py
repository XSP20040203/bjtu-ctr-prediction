"""Chronological DCN with supplied daily exposure and history-change inputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'model_comparison'))
sys.path.insert(0,str(HERE.parent/'user_personalization'))
from bounded_id_models import BoundedIDDCN
from train_models import load_arrays
from evaluation import metrics,full_diagnostics


def read(p):return json.loads(p.read_text(encoding='utf-8'))

def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def main(args):
    out=HERE/'dense_results'/args.tag
    if out.exists():raise FileExistsError('Preserving dense signal experiment')
    cache=HERE.parent/'user_personalization/cache_day15_stats' if args.day==20191115 else HERE.parent/'auc085/cache_rich_stats_time16'
    base=load_arrays(cache);manifest=read(cache/'manifest.json');train=base['train_indices'];valid=base['valid_indices']
    if args.recent_days:
        cutoff=int((pd.Timestamp(str(args.day))-pd.Timedelta(days=args.recent_days)).strftime('%Y%m%d'))
        train=train[base['date'][train]>=cutoff]
    strict=np.load(HERE/f'cache_day{str(args.day)[-2:]}'/'numeric.npy',mmap_mode='r')
    files=[HERE/'exposure_features.npy',HERE/'snapshot_features.npy']
    if args.generic_features:
        meta=read(args.generic_features.with_suffix('.json'))
        historical_only=meta.get('advertisement_validation_labels_used') is False and meta.get('validation_day')==args.day
        if meta.get('advertisement_labels_loaded') is not False and not historical_only:
            raise ValueError('Extra dense inputs require no current or validation advertisement labels')
        files.append(args.generic_features)
    extras=[np.load(p,mmap_mode='r') for p in files]
    if any(len(values)!=len(strict) for values in extras):raise ValueError('Extra dense rows do not align')
    names=list(manifest['numeric_fields'])
    for p in files:names+=read(p.with_suffix('.json'))['features']
    out.mkdir(parents=True)
    numeric=open_memmap(out/'numeric.npy',mode='w+',dtype='float32',shape=(len(strict),len(names)))
    for start in range(0,len(strict),50000):numeric[start:start+50000,:75]=strict[start:start+50000]
    offset=75;scalers={}
    for source in extras:
        for column in range(source.shape[1]):
            values=np.asarray(source[:,column]);mean=float(values[train].mean());std=max(float(values[train].std()),1e-6)
            numeric[:,offset]=(values-mean)/std;scalers[names[offset]]={'mean':mean,'std':std};offset+=1
    numeric.flush()
    torch.set_num_threads(6);torch.manual_seed(args.seed);np.random.seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    options={'numeric_dim':len(names),'embedding_dim':8,'hidden_dims':[128,64],'dropout':.1,'unknown_dropout':.01,'cross_layers':2,'group_norm_caps':{'user_id':.2,'ad_task':.2,'adv_id':.2}}
    if not math.isfinite(args.pos_weight) or args.pos_weight<=0:raise ValueError('Positive weight must be finite and positive')
    prior=float(base['label'][train].mean());weighted_prior=args.pos_weight*prior/(1-prior+args.pos_weight*prior)
    model=BoundedIDDCN(manifest['categorical_fields'],manifest['sequence_fields'],manifest['group_for_field'],manifest['vocabulary_sizes'],initial_ctr=weighted_prior,**options).to(device)
    positive_weight=torch.tensor(args.pos_weight,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-5)
    y=base['label'][valid];batchsize=2048;batches=math.ceil(len(train)/batchsize);evalsteps={math.ceil(batches*x) for x in [.25,.5,.75,1.]}
    def inputs(rows):
        return torch.as_tensor(base['categories'][rows].astype(np.int64),device=device),torch.as_tensor(base['sequences'][rows].astype(np.int64),device=device),torch.as_tensor(np.array(numeric[rows]),device=device)
    def predict():
        model.eval()
        with torch.no_grad():return np.concatenate([torch.sigmoid(model(*inputs(valid[s:s+4096]))-math.log(args.pos_weight)).cpu().numpy() for s in range(0,len(valid),4096)])
    best=-1.;bad=0;history=[];record=None;started=time.perf_counter();stop=False
    for epoch in range(1,5):
        order=np.random.default_rng(args.seed+epoch).permutation(train);model.train();total=0.;seen=0
        for step,start in enumerate(range(0,len(order),batchsize),1):
            rows=order[start:start+batchsize];target=torch.as_tensor(base['label'][rows].astype(np.float32),device=device)
            optimizer.zero_grad(set_to_none=True);loss=nn.functional.binary_cross_entropy_with_logits(model(*inputs(rows)),target,pos_weight=positive_weight)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite dense objective')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();total+=float(loss.detach())*len(rows);seen+=len(rows)
            if step not in evalsteps:continue
            p=predict();score=metrics(y,p);entry={'fractional_epoch':epoch-1+step/batches,'online_train_loss':total/seen,**score};history.append(entry)
            pd.DataFrame(history).to_csv(out/'history.csv',index=False);print(json.dumps({'tag':args.tag,**entry}),flush=True)
            if score['auc']>best+1e-5:
                best,bad=score['auc'],0
                torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'manifest':manifest,'options':options,'numeric_fields':names,'extra_scalers':scalers,'day':args.day,'pos_weight':args.pos_weight},out/'model.pt')
                np.save(out/'validation_prediction.npy',p);record={**score,'best_fractional_epoch':entry['fractional_epoch']}
            else:bad+=1
            model.train()
            if bad>=4:stop=True;break
        if stop:break
    checkpoint=torch.load(out/'model.pt',map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['state_dict']);replay=predict();saved=np.load(out/'validation_prediction.npy');error=float(np.max(np.abs(replay-saved)))
    if error>2e-7:raise AssertionError('Dense replay mismatch')
    ucol=manifest['categorical_fields'].index('user_id');tcol=manifest['categorical_fields'].index('task_id');users=base['categories'][valid,ucol];clicks=np.bincount(base['categories'][train,ucol],weights=base['label'][train],minlength=manifest['vocabulary_sizes']['user_id']);new=~np.isin(base['categories'][valid,tcol],np.unique(base['categories'][train,tcol]))
    record.update(status='complete_and_verified',day=args.day,tag=args.tag,seed=args.seed,cache=str(cache.resolve()),train_rows=len(train),recent_days=args.recent_days,numeric_dim=len(names),extra_scalers=scalers,options=options,learning_rate=args.learning_rate,pos_weight=args.pos_weight,generic_features=str(args.generic_features) if args.generic_features else None,maximum_reload_error=error,seconds=time.perf_counter()-started,model_sha256=hashlib.sha256((out/'model.pt').read_bytes()).hexdigest(),diagnostics=full_diagnostics(y,replay,users,clicks,new),final_labels_used=False)
    write(out/'metrics.json',record);print('DENSE_COMPLETE '+json.dumps(record),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--day',type=int,choices=[20191115,20191116],required=True);parser.add_argument('--tag',required=True)
    parser.add_argument('--generic-features',type=Path)
    parser.add_argument('--pos-weight',type=float,default=1.)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--recent-days',type=int,default=0);parser.add_argument('--learning-rate',type=float,default=.001);main(parser.parse_args())
