"""Historical daily click budgets normalized by today's supplied exposure volume."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SPECS=[['user_id'],['user_id','slot_id'],['user_id','spread_app_id'],['user_id','app_second_class'],
       ['task_id'],['adv_id'],['slot_id'],['adv_prim_id'],['spread_app_id']]


def build_click_budget(frame,validation_day):
    data=frame.reset_index(drop=True);dates=data.pt_d.to_numpy();columns={}
    ordinals=pd.to_datetime(data.pt_d.astype(str),format='%Y%m%d').to_numpy(dtype='datetime64[D]').astype(np.int64)
    for fields in SPECS:
        prefix='past_daily_budget__'+'__'.join(fields)
        current=data.groupby(['pt_d']+fields,sort=False).label.transform('size').to_numpy(dtype=np.float32)
        for suffix in ['log_past_clicks','log_past_exposures','log_active_days','positive_day_fraction','clicks_per_active_day',
                       'log_predicted_daily_ctr','last_day_clicks','last_day_ctr','recent3_ctr']:
            columns[prefix+'__'+suffix]=np.zeros(len(data),dtype=np.float32)
        for day in sorted(np.unique(dates[dates<=validation_day])):
            positions=np.flatnonzero(dates==day);past=data.loc[dates<day]
            if past.empty:continue
            daily=past.groupby(fields+['pt_d'],sort=False).label.agg(['size','sum']).reset_index()
            daily['positive']=(daily['sum']>0).astype(np.float32)
            stats=daily.groupby(fields,sort=False).agg(clicks=('sum','sum'),exposures=('size','sum'),active=('pt_d','size'),positive=('positive','sum'))
            query=data.iloc[positions]
            keys=pd.Index(query[fields[0]]) if len(fields)==1 else pd.MultiIndex.from_frame(query[fields])
            aligned=stats.reindex(keys).fillna(0)
            # Shared daily count prior is fitted on historical group-days only.
            daily_prior=(daily['sum'].sum()+10*.4)/(len(daily)+10)
            budget=(aligned.clicks.to_numpy()+5*daily_prior)/(aligned.active.to_numpy()+5)
            prob=np.clip(budget/np.maximum(current[positions],1),1e-6,.5)
            prefix_values=[np.log1p(aligned.clicks),np.log1p(aligned.exposures),np.log1p(aligned.active),
                           (aligned.positive+1)/(aligned.active+5),budget,np.log(prob/(1-prob))]
            for suffix,value in zip(['log_past_clicks','log_past_exposures','log_active_days','positive_day_fraction','clicks_per_active_day','log_predicted_daily_ctr'],prefix_values):
                columns[prefix+'__'+suffix][positions]=np.asarray(value,dtype=np.float32)
            ordinal=ordinals[positions[0]]
            for days,suffix in [(1,'last_day'),(3,'recent3')]:
                mask=(dates<day)&(ordinals>=ordinal-days)
                recent=data.loc[mask].groupby(fields,sort=False).label.agg(['size','sum']).reindex(keys).fillna(0)
                prior=(data.loc[mask,'label'].sum()+1000*.0084)/(mask.sum()+1000)
                rate=(recent['sum'].to_numpy()+100*prior)/(recent['size'].to_numpy()+100)
                columns[prefix+'__'+suffix+'_ctr'][positions]=rate
                if days==1:columns[prefix+'__last_day_clicks'][positions]=np.log1p(recent['sum'].to_numpy())
    return np.column_stack(list(columns.values())).astype(np.float32),list(columns)


def main(args):
    if args.output.exists():raise FileExistsError('Preserving click-budget features')
    fields=sorted({f for fields in SPECS for f in fields}|{'pt_d','label'})
    data=pd.read_csv(ROOT/'2026ml/train.csv',usecols=fields,dtype='int32')
    result,names=build_click_budget(data,args.day)
    if not np.isfinite(result).all():raise AssertionError('Nonfinite historical daily budget')
    args.output.parent.mkdir(parents=True,exist_ok=True);np.save(args.output,result)
    meta={'shape':list(result.shape),'features':names,'advertisement_labels_loaded':True,
          'advertisement_validation_labels_used':False,'validation_day':args.day,
          'rule':'Labels strictly before each row date; daily expected historical count divided by current input-only exposure count',
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in meta.items() if k!='features'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--day',type=int,choices=[20191115,20191116],required=True)
    parser.add_argument('--output',type=Path,required=True);main(parser.parse_args())
