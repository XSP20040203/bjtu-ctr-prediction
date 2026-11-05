"""Tree baseline using strict-past ad statistics and complete daily topic distributions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score

HERE=Path(__file__).resolve().parent


def main(args):
    out=HERE/"tree_results"/args.tag
    if out.exists():raise FileExistsError("Preserving existing tree result")
    cache=HERE.parent/"user_personalization/cache_day15_stats" if args.day==20191115 else HERE.parent/"auc085/cache_rich_stats_time16"
    features=HERE/f"cache_day{str(args.day)[-2:]}"
    m=json.loads((cache/"manifest.json").read_text(encoding="utf-8"));f=json.loads((features/"manifest.json").read_text(encoding="utf-8"))
    train=np.load(cache/"train_indices.npy");valid=np.load(cache/"valid_indices.npy");dates=np.load(cache/"date.npy",mmap_mode="r")
    if args.recent_days:
        cutoff=int((pd.Timestamp(str(args.day))-pd.Timedelta(days=args.recent_days)).strftime("%Y%m%d"))
        train=train[dates[train]>=cutoff]
    labels=np.load(cache/"label.npy",mmap_mode="r")
    categories=np.load(cache/"categories.npy",mmap_mode="r")
    sequences=np.load(cache/"sequences.npy",mmap_mode="r")
    numeric=np.load(features/"numeric.npy",mmap_mode="r");requests=np.load(features/"request_index.npy",mmap_mode="r")
    topics=np.load(features/"topic_weights.npy",mmap_mode="r");context=np.load(features/"interest_context.npy",mmap_mode="r")
    selected=np.r_[train,valid];ntrain=len(train);columns={};cat_names=[]
    for col,name in enumerate(m["categorical_fields"]):
        if args.no_identity and name in ['user_id','task_id','adv_id']:
            continue
        columns[name]=pd.Categorical(categories[selected,col]);cat_names.append(name)
    for col,name in enumerate(m["numeric_fields"]):columns[name]=numeric[selected,col]
    for col,name in enumerate(m["sequence_fields"]):
        if args.no_sequence_ids:
            continue
        for position in range(5):
            feature=name+"_token"+str(position);columns[feature]=pd.Categorical(sequences[selected,col,position]);cat_names.append(feature)
    if args.snapshot_features:
        snapshot=np.load(args.snapshot_features,mmap_mode="r")
        meta=json.loads(args.snapshot_features.with_suffix('.json').read_text(encoding='utf-8'))
        if meta.get('advertisement_labels_loaded') is not False:raise ValueError('Snapshot inputs require explicit no-label provenance')
        if len(snapshot)!=len(labels):raise ValueError('Snapshot row mismatch')
        for index,name in enumerate(meta['features']):columns[name]=snapshot[selected,index]
    if args.exposure_features:
        exposure=np.load(args.exposure_features,mmap_mode="r")
        meta=json.loads(args.exposure_features.with_suffix('.json').read_text(encoding='utf-8'))
        if meta.get('advertisement_labels_loaded') is not False or len(exposure)!=len(labels):
            raise ValueError('Daily exposure inputs require aligned rows and no-label provenance')
        for index,name in enumerate(meta['features']):columns[name]=exposure[selected,index]
    if args.generic_features:
        generic=np.load(args.generic_features,mmap_mode="r")
        meta=json.loads(args.generic_features.with_suffix('.json').read_text(encoding='utf-8'))
        historical_only=meta.get('advertisement_validation_labels_used') is False and meta.get('validation_day')==args.day
        if (meta.get('advertisement_labels_loaded') is not False and not historical_only) or len(generic)!=len(labels):
            raise ValueError('Extra inputs require aligned rows and no current/validation advertisement target use')
        for index,name in enumerate(meta['features']):columns[name]=np.asarray(generic[selected,index],dtype=np.float32)
    if args.topics:
        req=requests[selected]
        for axis in range(2):
            for channel in range(4):
                # Choose common source dimensions on training requests only to fit local RAM.
                train_requests=np.unique(requests[train])
                relevance=np.asarray(topics[train_requests,axis,channel]).sum(axis=0)
                relevance[:2]=0
                chosen=np.argsort(relevance)[::-1][:args.top_topics]
                for topic in chosen:
                    if relevance[topic]<=0:continue
                    columns[f"topic_{axis}_{channel}_{topic}"]=np.asarray(topics[req,axis,channel,topic])
            for index in range(context.shape[-1]):columns[f"context_{axis}_{index}"]=context[req,axis,index]
    data=pd.DataFrame(columns);del columns
    print(f"CHRONO_TREE_FEATURES rows={len(data)} fields={len(data.columns)} train={ntrain}",flush=True)
    fit=Pool(data.iloc[:ntrain],labels[train],cat_features=cat_names);validation=Pool(data.iloc[ntrain:],labels[valid],cat_features=cat_names)
    del data
    params=dict(iterations=args.iterations,depth=args.depth,learning_rate=.04,l2_leaf_reg=20,
                loss_function="Logloss",eval_metric="AUC",random_seed=42,task_type="GPU",
                border_count=64,one_hot_max_size=12,max_ctr_complexity=args.ctr_complexity,gpu_ram_part=.60,
                gpu_cat_features_storage="CpuPinnedMemory",pinned_memory_size="512mb",thread_count=6,allow_writing_files=False)
    out.mkdir(parents=True);started=time.perf_counter()
    model=CatBoostClassifier(**params);model.fit(fit,eval_set=validation,early_stopping_rounds=100,verbose=50,use_best_model=True)
    probability=model.predict_proba(validation)[:,1];np.save(out/"validation_prediction.npy",probability);model.save_model(str(out/"model.cbm"))
    restored=CatBoostClassifier();restored.load_model(str(out/"model.cbm"));replay=restored.predict_proba(validation)[:,1]
    error=float(np.max(np.abs(replay-probability)))
    if error>1e-10:raise AssertionError("Tree replay error")
    result={"status":"complete_and_verified","day":args.day,"tag":args.tag,"train_rows":len(train),"valid_rows":len(valid),
            "auc":float(roc_auc_score(labels[valid],probability)),"logloss":float(log_loss(labels[valid],probability,labels=[0,1])),
            "average_precision":float(average_precision_score(labels[valid],probability)),"features":len(model.feature_names_),
            "trees":model.tree_count_,"params":params,"no_identity":args.no_identity,"no_sequence_ids":args.no_sequence_ids,"recent_days":args.recent_days,"topics":args.topics,"top_topics_per_role":args.top_topics,"snapshot_features":str(args.snapshot_features) if args.snapshot_features else None,
            "exposure_features":str(args.exposure_features) if args.exposure_features else None,
            "generic_features":str(args.generic_features) if args.generic_features else None,
            "cache":str(cache.resolve()),"feature_cache":str(features.resolve()),"maximum_reload_error":error,
            "model_sha256":hashlib.sha256((out/"model.cbm").read_bytes()).hexdigest(),"seconds":time.perf_counter()-started,
            "advertisement_stats":"strict dates before each row","source_rule":"provided same-day plus previous3 days",
            "validation":"whole future day, development checkpoint selection","final_labels_used":False}
    (out/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    pd.DataFrame({"feature":model.feature_names_,"importance":model.get_feature_importance()}).sort_values("importance",ascending=False).to_csv(out/"importance.csv",index=False)
    print("CHRONO_TREE_COMPLETE "+json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--day",type=int,choices=[20191115,20191116],required=True)
    parser.add_argument("--tag",required=True);parser.add_argument("--topics",action="store_true")
    parser.add_argument("--recent-days",type=int,default=0);parser.add_argument("--depth",type=int,default=6)
    parser.add_argument("--snapshot-features",type=Path)
    parser.add_argument("--exposure-features",type=Path)
    parser.add_argument("--generic-features",type=Path)
    parser.add_argument("--no-identity",action="store_true")
    parser.add_argument("--no-sequence-ids",action="store_true")
    parser.add_argument("--ctr-complexity",type=int,choices=[1,2],default=1)
    parser.add_argument("--iterations",type=int,default=1800);parser.add_argument("--top-topics",type=int,default=16);main(parser.parse_args())
