"""Daily input volume changes and history-pattern counts; advertisement labels unused."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FIELDS=['user_id','task_id','adv_id','adv_prim_id','spread_app_id','app_second_class','slot_id']
PATTERNS=['ad_click_list_v001','ad_click_list_v002','ad_click_list_v003','u_newsCatInterestsST']
SPECS=[(f,) for f in FIELDS]+[('user_id',f) for f in ['task_id','adv_id','spread_app_id','app_second_class','slot_id']]+[('task_id','slot_id'),('adv_id','slot_id')]+[('user_id',p) for p in PATTERNS]+[('user_id','task_id','slot_id'),('user_id','spread_app_id','slot_id'),('user_id','ad_click_list_v003','slot_id')]


def build_context_volume(frame):
    data=frame.reset_index(drop=True)
    columns={}
    for fields in SPECS:
        daily=data.groupby(['pt_d']+list(fields),sort=False,observed=True).size().rename('count').reset_index()
        # Prefix sums over date groups use no outcomes and never include later dates.
        daily=daily.sort_values('pt_d',kind='stable')
        grouped=daily.groupby(list(fields),sort=False,observed=True)
        daily['past_count']=grouped['count'].cumsum()-daily['count']
        daily['active_days']=grouped.cumcount()
        daily['last_count']=grouped['count'].shift(1).fillna(0)
        base=data[['pt_d']+list(fields)].merge(daily,on=['pt_d']+list(fields),how='left',sort=False,validate='many_to_one')
        prefix='input_volume__'+'__'.join(fields)
        columns[prefix+'__today_logcount']=np.log1p(base['count'].to_numpy(dtype=np.float32))
        columns[prefix+'__past_logcount']=np.log1p(base.past_count.to_numpy(dtype=np.float32))
        columns[prefix+'__last_logcount']=np.log1p(base.last_count.to_numpy(dtype=np.float32))
        average=base.past_count.to_numpy(dtype=np.float32)/np.maximum(base.active_days.to_numpy(dtype=np.float32),1)
        columns[prefix+'__logratio_average']=np.log((base['count'].to_numpy(dtype=np.float32)+1)/(average+1))
        columns[prefix+'__logratio_last']=np.log((base['count'].to_numpy(dtype=np.float32)+1)/(base.last_count.to_numpy(dtype=np.float32)+1))
        columns[prefix+'__new_combination']=(base.active_days.to_numpy()==0).astype(np.float32)
    return np.column_stack(list(columns.values())).astype(np.float32),list(columns)


def main(args):
    if args.output.exists():raise FileExistsError('Preserving context volume features')
    data=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d']+FIELDS+PATTERNS,dtype={p:str for p in PATTERNS})
    result,names=build_context_volume(data)
    if not np.isfinite(result).all():raise AssertionError('Invalid volume features')
    args.output.parent.mkdir(parents=True,exist_ok=True);np.save(args.output,result)
    meta={'shape':list(result.shape),'features':names,'advertisement_labels_loaded':False,
          'rule':'Provided current-day exposures plus strictly earlier input counts; no outcome labels',
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k!='features'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'context_volume.npy');main(parser.parse_args())
