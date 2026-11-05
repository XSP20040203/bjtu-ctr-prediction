"""Build chronological numeric replacements and compact daily news interests."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent/"auc085"))
from target_statistics import SPECS
from interest_data import build_past_statistics, build_topic_store, request_keys


def digest(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):value.update(block)
    return value.hexdigest()


def main(args):
    if args.output_dir.exists():raise FileExistsError(f"Preserving features: {args.output_dir}")
    started=time.perf_counter()
    cache=HERE.parent/"user_personalization/cache_day15_stats" if args.day==20191115 else HERE.parent/"auc085/cache_rich_stats_time16"
    manifest=json.loads((cache/"manifest.json").read_text(encoding="utf-8"))
    vocabulary=json.loads((cache/"vocabs.json").read_text(encoding="utf-8"))["news_interest"]
    fields=sorted({f for spec,_ in SPECS for f in spec}|{"pt_d","user_id","label"})
    ads=pd.read_csv(ROOT/"2026ml/train.csv",usecols=fields,dtype="int32")
    print("PAST_STATISTICS start",flush=True)
    raw=build_past_statistics(ads,args.day,SPECS)
    train=np.load(cache/"train_indices.npy");valid=np.load(cache/"valid_indices.npy")
    expected=["target_stat__"+"__".join(fields)+"__"+suffix for fields,_ in SPECS for suffix in ["log_count","relative_logit"]]
    if expected!=manifest["numeric_fields"][39:]:raise ValueError("Statistics feature order mismatch")
    args.output_dir.mkdir(parents=True)
    numeric=open_memmap(args.output_dir/"numeric.npy",mode="w+",dtype="float32",shape=(len(ads),75))
    original=np.load(cache/"numeric.npy",mmap_mode="r")
    for start in range(0,len(ads),100000):numeric[start:start+100000,:39]=original[start:start+100000,:39]
    scalers={}
    for col,name in enumerate(expected):
        mean=float(raw[train,col].mean());std=max(float(raw[train,col].std()),1e-6)
        numeric[:,39+col]=(raw[:,col]-mean)/std
        scalers[name]={"mean":mean,"std":std}
    numeric[ads.pt_d.to_numpy()>args.day]=0
    numeric.flush();del raw
    feeds=pd.read_csv(ROOT/"2026ml/train_feeds.csv",usecols=["u_userId","e_et","i_cat","label","pro","u_newsCatDislike"],dtype={"u_newsCatDislike":str})
    feeds=feeds.loc[feeds.e_et<=args.day].copy()
    requests=ads.loc[ads.pt_d<=args.day,["user_id","pt_d"]].drop_duplicates()
    print("DAILY_TOPICS start",flush=True)
    store=build_topic_store(feeds,requests,vocabulary,history_days=3)
    indices=np.zeros(len(ads),dtype=np.int32)
    eligible=ads.pt_d.to_numpy()<=args.day
    indices[eligible]=np.searchsorted(store["keys"],request_keys(ads.loc[eligible,"user_id"],ads.loc[eligible,"pt_d"]))
    if np.any(indices[train]==0) or np.any(indices[valid]==0):raise AssertionError("Missing request")
    for name in ["keys","topic_mass","interest_context"]:np.save(args.output_dir/(name+".npy"),store[name])
    np.save(args.output_dir/"topic_weights.npy",store["topic_weights"].astype(args.topic_dtype))
    np.save(args.output_dir/"request_index.npy",indices)
    np.testing.assert_array_equal(numeric[train[:100],:39],original[train[:100],:39])
    sums=store["topic_weights"].sum(-1)
    if not np.allclose(sums[store["topic_mass"]>0],1,atol=1e-5):raise AssertionError("Topic normalization error")
    result={"status":"complete","validation_day":args.day,"base_cache":str(cache.resolve()),"rows":len(ads),
            "train_rows":len(train),"valid_rows":len(valid),"requests":len(store["keys"]),"n_topics":store["n_topics"],
            "context_dim":store["context_dim"],"context_names":store["context_names"],"topic_vocabulary":vocabulary,
            "timeaxis":["long3day","today"],"channels":["exposure","click","progress","dislike"],
            "topic_dtype":args.topic_dtype,"history_days":3,"numeric_fields":manifest["numeric_fields"],"numeric_scaler":scalers,
            "ad_statistics":"Every row uses strictly earlier advertisement dates; fixed initial .0084 prior mass1000; outer validation labels excluded",
            "source_rule":"Source e_et at request day and previous3 natural days; full supplied same-day feedback permitted; unordered within day",
            "dislike_rule":"Only u_newsCatDislike nonzero tokens; nonclick does not imply dislike; 0 excluded with missing flag",
            "future_ad_rows":"Excluded from training/validation, numeric=0 and request_index=0",
            "input_sha256":{name:digest(ROOT/"2026ml"/name) for name in ["train.csv","train_feeds.csv"]},
            "source_sha256":{str(p.resolve()):digest(p) for p in [Path(__file__),HERE/"interest_data.py"]},
            "preparation_seconds":time.perf_counter()-started}
    (args.output_dir/"manifest.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("FEATURES_COMPLETE "+json.dumps({k:result[k] for k in ["validation_day","rows","requests","n_topics","context_dim","preparation_seconds"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--day",type=int,choices=[20191115,20191116],required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--topic-dtype",choices=["float32","float16"],default="float32")
    main(parser.parse_args())
