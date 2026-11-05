"""Fine-grained input-only same-day exposure density and unique-count statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SPECS=[('user_id',f) for f in ['creat_type_cd','inter_type_cd','device_size','net_type','adv_prim_id','hispace_app_tags','app_score','u_refreshTimes','u_feedLifeCycle']]
SPECS += [('user_id',a,b) for a,b in [('task_id','slot_id'),('adv_id','slot_id'),('spread_app_id','slot_id'),('app_second_class','slot_id'),('adv_prim_id','slot_id'),('task_id','creat_type_cd'),('task_id','device_size'),('task_id','net_type'),('adv_id','creat_type_cd'),('spread_app_id','creat_type_cd'),('spread_app_id','inter_type_cd'),('creat_type_cd','slot_id'),('inter_type_cd','slot_id'),('device_size','slot_id'),('net_type','slot_id')]]
SPECS += [(a,b) for a,b in [('task_id','creat_type_cd'),('task_id','inter_type_cd'),('task_id','age'),('task_id','sex'),('task_id','device_size'),('task_id','net_type'),('adv_id','creat_type_cd'),('adv_id','inter_type_cd'),('adv_id','age'),('adv_id','sex'),('adv_id','device_size'),('slot_id','creat_type_cd'),('slot_id','inter_type_cd'),('slot_id','device_size'),('slot_id','net_type'),('spread_app_id','creat_type_cd'),('spread_app_id','app_second_class'),('app_second_class','creat_type_cd')]]
SPECS += [('user_id',p,c) for p in ['ad_click_list_v001','ad_click_list_v003','u_newsCatInterestsST'] for c in ['task_id','spread_app_id','slot_id','creat_type_cd']]
SPECS += [('task_id','slot_id','creat_type_cd'),('adv_id','slot_id','creat_type_cd'),('user_id','task_id','slot_id','creat_type_cd'),('user_id','spread_app_id','slot_id','creat_type_cd')]


def build_extended_counts(frame):
    data=frame.reset_index(drop=True)
    cols={}
    user_count=data.groupby(['pt_d','user_id'],sort=False).user_id.transform('size').to_numpy(dtype=np.float32)
    task_count=data.groupby(['pt_d','task_id'],sort=False).task_id.transform('size').to_numpy(dtype=np.float32)
    for fields in SPECS:
        count=data.groupby(['pt_d']+list(fields),sort=False,observed=True).user_id.transform('size').to_numpy(dtype=np.float32)
        prefix='fine_exposure__'+'__'.join(fields)
        cols[prefix+'__logcount']=np.log1p(count)
        if 'user_id' in fields:cols[prefix+'__user_fraction']=count/np.maximum(user_count,1)
        if 'task_id' in fields:cols[prefix+'__task_fraction']=count/np.maximum(task_count,1)
    for first in ['user_id','task_id','adv_id','spread_app_id','slot_id']:
        for second in ['creat_type_cd','inter_type_cd','device_size','net_type']:
            cols[f'fine_diversity__{first}__{second}']=np.log1p(data.groupby(['pt_d',first],sort=False)[second].transform('nunique').to_numpy(dtype=np.float32))
    return np.column_stack(list(cols.values())).astype(np.float32),list(cols)


def main(args):
    if args.output.exists():raise FileExistsError('Preserving fine exposure features')
    fields=sorted({f for s in SPECS for f in s}|{'pt_d','user_id','task_id','adv_id','spread_app_id','slot_id','creat_type_cd','inter_type_cd','device_size','net_type'})
    data=pd.read_csv(ROOT/'2026ml/train.csv',usecols=fields,dtype={f:str for f in fields if 'list' in f or f=='u_newsCatInterestsST'})
    values,names=build_extended_counts(data)
    np.save(args.output,values)
    meta={'shape':list(values.shape),'features':names,'advertisement_labels_loaded':False,
          'rule':'Fine categorical exposure density within provided full day; no label or cross-date aggregation',
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k!='features'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'extended_counts.npy');main(parser.parse_args())
