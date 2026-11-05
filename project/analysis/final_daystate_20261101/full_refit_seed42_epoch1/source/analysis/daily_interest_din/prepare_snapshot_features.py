"""Current-day supplied ad-history unions, novelty and candidate matches; no labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def build_features(frame):
    """All rows for one supplied user-day snapshot are available as unlabeled input."""
    data=frame.reset_index(drop=True)
    dates=data.pt_d.to_numpy();users=data.user_id.to_numpy()
    names=[]
    for action in ["click","close"]:
        for index in [1,2,3]:
            prefix=f"snapshot_{action}_{index}"
            names.extend(prefix+"__"+x for x in ["today_union_size","new_token_count","candidate_in_union","candidate_new",
                         "candidate_snapshot_fraction","candidate_in_past_union","row_novel_fraction","candidate_in_other_snapshot"])
    result=np.zeros((len(data),len(names)),dtype=np.float32)
    previous={field:defaultdict(set) for field in [f"ad_{a}_list_v00{i}" for a in ["click","close"] for i in [1,2,3]]}
    cache={}
    for day in sorted(np.unique(dates)):
        rows=np.flatnonzero(dates==day);current=data.iloc[rows]
        for group,(action,index,target) in enumerate([(a,i,t) for a in ["click","close"] for i,t in [(1,"task_id"),(2,"adv_prim_id"),(3,"spread_app_id")]]):
            field=f"ad_{action}_list_v00{index}";today={};frequencies={};number={}
            unique=current[["user_id",field]].drop_duplicates()
            for user,values in unique.itertuples(index=False,name=None):
                text=str(values)
                if text not in cache:cache[text]=set(int(x) for x in text.split('^') if x not in ['', 'nan', 'None'])
                tokens=cache[text]
                today.setdefault(user,set()).update(tokens)
                frequencies.setdefault(user,Counter()).update(tokens)
                number[user]=number.get(user,0)+1
            for local,(user,target_value,text) in enumerate(current[["user_id",target,field]].itertuples(index=False,name=None)):
                tokens=cache[str(text)];all_today=today[user];past=previous[field][user];new=all_today-past
                count=frequencies[user][target_value];in_row=target_value in tokens
                result[rows[local],group*8:(group+1)*8]=[np.log1p(len(all_today)),np.log1p(len(new)),target_value in all_today,
                       target_value in new,count/max(number[user],1),target_value in past,len(tokens-past)/max(len(tokens),1),
                       count-int(in_row)>0]
            for user,tokens in today.items():previous[field][user].update(tokens)
    if not np.isfinite(result).all():raise AssertionError("Invalid snapshot input")
    return result,names


def main(args):
    if args.output.exists():raise FileExistsError("Preserving existing snapshot features")
    fields=["user_id","pt_d","task_id","adv_prim_id","spread_app_id"]+[f"ad_{a}_list_v00{i}" for a in ["click","close"] for i in [1,2,3]]
    data=pd.read_csv(ROOT/"2026ml/train.csv",usecols=fields,dtype={f:str for f in fields if 'list' in f})
    # Includes only provided raw input fields; no advertisement labels are loaded.
    result,names=build_features(data)
    args.output.parent.mkdir(parents=True,exist_ok=True);np.save(args.output,result)
    meta={"shape":list(result.shape),"features":names,"advertisement_labels_loaded":False,
          "rule":"Current supplied ad-history snapshots unioned within user/date; past unions strictly earlier days; all-day offline input assumption",
          "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'snapshot_features.npy');main(parser.parse_args())
