"""Unlabeled within-day exposure volume, concentration and support features."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SINGLE=["user_id","task_id","adv_id","adv_prim_id","spread_app_id","app_second_class","slot_id"]
PAIR=[("user_id",f) for f in SINGLE[1:]]+[("task_id","slot_id"),("adv_id","slot_id"),("spread_app_id","slot_id")]


def build_exposure_features(frame):
    data=frame.reset_index(drop=True)
    # Selecting these columns explicitly prevents accidental dependence on any target column.
    data=data[["pt_d"]+SINGLE]
    columns={};sizes={}
    for field in SINGLE:
        count=data.groupby(["pt_d",field],sort=False)[field].transform("size").to_numpy(dtype=np.float32)
        sizes[field]=count;columns["day_exposure__"+field+"__log_count"]=np.log1p(count)
    for first,second in PAIR:
        count=data.groupby(["pt_d",first,second],sort=False)[first].transform("size").to_numpy(dtype=np.float32)
        prefix="day_exposure__"+first+"__"+second
        columns[prefix+"__log_count"]=np.log1p(count)
        columns[prefix+"__fraction_first"]=count/np.maximum(sizes[first],1)
        columns[prefix+"__fraction_second"]=count/np.maximum(sizes[second],1)
    for field in SINGLE[1:]:
        count=data.groupby(["pt_d","user_id"],sort=False)[field].transform("nunique").to_numpy(dtype=np.float32)
        columns["day_exposure__user_unique_"+field]=np.log1p(count)
    for first,second in [("task_id","user_id"),("adv_id","user_id"),("slot_id","user_id")]:
        count=data.groupby(["pt_d",first],sort=False)[second].transform("nunique").to_numpy(dtype=np.float32)
        columns["day_exposure__"+first+"_unique_"+second]=np.log1p(count)
    result=np.column_stack(list(columns.values())).astype(np.float32)
    if not np.isfinite(result).all():raise AssertionError("Nonfinite day exposure features")
    return result,list(columns)


def main(args):
    if args.output.exists():raise FileExistsError("Preserving existing daily exposure features")
    source=ROOT/"2026ml/train.csv"
    data=pd.read_csv(source,usecols=["pt_d"]+SINGLE,dtype="int32")
    result,names=build_exposure_features(data)
    args.output.parent.mkdir(parents=True,exist_ok=True);np.save(args.output,result)
    meta={"shape":list(result.shape),"features":names,"advertisement_labels_loaded":False,
          "rule":"All provided input exposures within each day; no labels or later dates; sample itself included in exposure count",
          "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'exposure_features.npy');main(parser.parse_args())
