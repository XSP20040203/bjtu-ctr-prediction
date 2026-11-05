"""Strict historical response interactions conditioned on stable context and day input bins."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
from interest_data import build_past_statistics

BASE_SPECS=[['creat_type_cd'],['inter_type_cd'],['user_id','creat_type_cd'],['user_id','inter_type_cd'],
 ['creat_type_cd','slot_id'],['inter_type_cd','slot_id'],['creat_type_cd','spread_app_id'],['creat_type_cd','app_second_class'],
 ['user_id','creat_type_cd','slot_id'],['user_id','spread_app_id','creat_type_cd'],
 ['age','creat_type_cd','slot_id'],['sex','creat_type_cd','slot_id'],['device_name','creat_type_cd'],
 ['spread_app_id','slot_id','creat_type_cd'],['adv_prim_id','creat_type_cd'],['task_id','creat_type_cd']]


def build_context_statistics(frame,day):
    data=frame.copy()
    boundaries=np.array([0,1,2,3,5,10,20,50,100],dtype=np.float32)
    for field in ['spread_app_id','task_id','slot_id']:
        count=data.groupby(['pt_d','user_id',field],sort=False).label.transform('size').to_numpy()
        data['today_countbin_'+field]=np.searchsorted(boundaries,count,side='left').astype(np.int32)
    specs=[(fields,100.) for fields in BASE_SPECS]
    for field in ['spread_app_id','task_id','slot_id']:
        b='today_countbin_'+field
        specs += [([b,'creat_type_cd'],200.),([b,'creat_type_cd','slot_id'],200.),([b,'inter_type_cd'],200.)]
    values=build_past_statistics(data,day,specs)
    names=['past_context__'+'__'.join(fields)+'__'+s for fields,_ in specs for s in ['log_count','relative_logit']]
    return values,names


def main(args):
    if args.output.exists():raise FileExistsError('Preserving context response features')
    fields=sorted({f for x in BASE_SPECS for f in x}|{'pt_d','label','user_id','spread_app_id','task_id','slot_id'})
    data=pd.read_csv(ROOT/'2026ml/train.csv',usecols=fields,dtype='int32')
    values,names=build_context_statistics(data,args.day);np.save(args.output,values)
    meta={'shape':list(values.shape),'features':names,'advertisement_labels_loaded':True,'advertisement_validation_labels_used':False,
         'validation_day':args.day,'rule':'Response rates strictly earlier dates; bins based only on provided day exposures',
         'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k!='features'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--day',type=int,choices=[20191115,20191116],required=True);parser.add_argument('--output',type=Path,required=True);main(parser.parse_args())
