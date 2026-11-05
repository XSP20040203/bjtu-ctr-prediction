"""Preserve complete supplied news-topic distributions for daily ad correction."""
import os
for _key in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:
    os.environ[_key]="2"
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from personal_context_residual import matrix as personal_matrix,personal_changes
from rich_userday_activity import load_day,correct
from rich_group_residual import (ROOT,HERE,REFERENCE,digest,write,lgb,array_hash,
                                 rank_sum_auc,target_offset,prediction,aggregate)

OUT=HERE / "runs/topic_context_residual_v1"
LEAVES=[3,7,15]
POWERS=[0.,.25,.5,.75,1.]
PARAMS={"learning_rate":.025,"min_data_in_leaf":150,"lambda_l2":20.,"feature_fraction":.65,
        "max_bin":31,"num_threads":2,"seed":42,"verbosity":-1,"deterministic":True,
        "force_col_wise":True,"boost_from_average":False}
SOURCES={"reference":None,
         "pairwise":HERE / "runs/pairwise_group_boost_v1/prediction.npy",
         "personal_context":HERE / "runs/personal_context_residual_v1/prediction.npy",
         "neural_userday":HERE / "runs/neural_userday_v1/prediction.npy",
         "context_rank":HERE / "runs/context_rank_boost_v1/prediction.npy",
         "catboost_daily":HERE / "runs/catboost_daily_history_v1/prediction.npy"}


def topic_features(feeds,keys,topics=None):
    if set(feeds.columns)!={"e_et","u_userId","i_cat","label","pro"}:
        raise ValueError("Only declared news inputs allowed")
    if not feeds.label.isin([-1,1]).all():
        raise ValueError("News labels must be -1/1")
    if topics is None:
        topics=np.sort(feeds.loc[feeds.e_et<20191116,"i_cat"].unique()).tolist()
    vocabulary={int(value):i for i,value in enumerate(topics)}
    index=pd.MultiIndex.from_arrays([keys[:,0],keys[:,1]])
    groups=index.get_indexer(pd.MultiIndex.from_frame(feeds[["e_et","u_userId"]]))
    codes=feeds.i_cat.map(vocabulary).fillna(len(topics)).to_numpy(dtype="int32")
    keep=(groups>=0)&(feeds.e_et.to_numpy()<=20191116)
    groups,codes=groups[keep],codes[keep]
    selected=feeds.loc[keep]
    clicked=selected.label.to_numpy()==1
    read=np.clip(selected.pro.to_numpy(dtype=float),0,100)/100*clicked
    proportions,logs,names=[],[],[]
    width=len(topics)+1
    for channel,weights in [("exposure",np.ones(len(groups))),("click",clicked.astype(float)),("read",read)]:
        matrix=sparse.csr_matrix((weights,(groups,codes)),shape=(len(keys),width),dtype="float32").toarray()
        total=matrix.sum(1)
        share=matrix/np.maximum(total[:,None],1e-8)
        proportions.append(share)
        names += [f"news_topic_{channel}_{topic}" for topic in topics+["unknown"]]
        if channel!="read":
            logs.append(np.log1p(matrix))
    values=np.column_stack(proportions).astype("float32")
    changes=personal_changes(keys,values)
    values=np.column_stack([values,*logs,changes]).astype("float32")
    names += [f"news_topic_{channel}_{topic}_log_count" for channel in ["exposure","click"] for topic in topics+["unknown"]]
    names += [f"news_topic_previous_delta_{i}" for i in range(3*width)]
    names += [f"news_topic_mean_delta_{i}" for i in range(3*width)]
    names += ["news_topic_has_history","news_topic_log_history_days"]
    if len(names)!=values.shape[1] or not np.isfinite(values).all():
        raise AssertionError("Invalid topic context")
    return values,names,topics


def matrix(day,topics=None,include_labels=False):
    data=personal_matrix(day,evaluation_labels=include_labels)
    context=load_day(16,False,evaluation_labels=False)
    feeds=pd.read_csv(ROOT / "2026ml/train_feeds.csv",usecols=["e_et","u_userId","i_cat","label","pro"],dtype="int32")
    values,names,topics=topic_features(feeds,context["keys"],topics)
    chosen=context["keys"][:,0]==20191100+day
    np.testing.assert_array_equal(data["keys"],context["keys"][chosen])
    data["x"]=np.column_stack([data["x"],values[chosen]]).astype("float32")
    data["names"]+=names
    return data,topics


def dependencies():
    return [Path(__file__),HERE / "personal_context_residual.py",HERE / "extended_history_residual.py",
            HERE / "rich_userday_activity.py",HERE / "rich_group_residual.py",HERE / "fit_stacker.py",
            HERE / "verify_stacker.py",HERE / "runs/weighted_v1_day16/context.npz",
            HERE / "runs/weighted_v1_day16/context_manifest.json"]+[
                p for p in SOURCES.values() if p is not None]+[
                REFERENCE / f"day{d}_{s}.npy" for d in [15,16] for s in ["indices","prediction"]]


def main():
    if OUT.exists():
        raise FileExistsError("Preserving topic-context experiment")
    OUT.mkdir(parents=True)
    protocol={"params":PARAMS,"leaves":LEAVES,"powers":POWERS,"objectives":["binary","poisson"],
              "max_rounds":500,"patience":60,"fit_date":20191115,"evaluation_date":20191116,
              "inputs_sha256":{str(p):digest(p) for p in dependencies()},
              "raw_datasets_sha256":{p.name:digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
              "selection":"15 user3fold chooses iterations and powers; six models locked before16 outcomes compared.",
              "topics":"All observed pre16 news category IDs; held unknown bucket; proportions, counts, personal earlier-day deviations.",
              "evaluation16_labels_fit":False,"same_day_news_feedback":True,"development16_model_selection":True,
              "sources":list(SOURCES)}
    write(OUT / "protocol.json",protocol)
    data,topics=matrix(15,include_labels=True)
    write(OUT / "topics.json",topics)
    print(json.dumps({"event":"topic_context_ready","shape":list(data["x"].shape)}),flush=True)
    folds=((data["keys"][:,1].astype("uint64")*2654435761)%3).astype("int32")
    records,locks=[],[]
    for objective in protocol["objectives"]:
        target,offset=target_offset(data,objective)
        params=dict(PARAMS,objective=objective,metric="binary_logloss" if objective=="binary" else "poisson")
        for leaves in LEAVES:
            rounds,oof=[],np.empty(len(target))
            for fold in range(3):
                tr,va=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
                train=lgb.Dataset(data["x"][tr],label=target[tr],init_score=offset[tr])
                valid=lgb.Dataset(data["x"][va],label=target[va],init_score=offset[va],reference=train)
                model=lgb.train(dict(params,num_leaves=leaves),train,num_boost_round=500,valid_sets=[valid],
                                callbacks=[lgb.early_stopping(60,verbose=False)])
                rounds.append(model.best_iteration)
                oof[va]=prediction(model,data,objective,va)
            scores=[{"power":p,"auc":rank_sum_auc(data["y"],correct(data["base"],oof,data["groups"],objective,p))} for p in POWERS]
            best=max(scores,key=lambda r:r["auc"])
            r={"name":f"{objective}_{leaves}","objective":objective,"leaves":leaves,"rounds":rounds,
               "refit_rounds":max(1,int(np.median(rounds))),"power":best["power"],"oof15_auc":best["auc"],"scores":scores}
            records.append(r)
            np.save(OUT / (r["name"]+"_oof.npy"),oof)
            write(OUT / "cv.json",records)
            print(json.dumps(r),flush=True)
    for r in records:
        target,offset=target_offset(data,r["objective"])
        model=lgb.train(dict(PARAMS,objective=r["objective"],num_leaves=r["leaves"]),
                        lgb.Dataset(data["x"],label=target,init_score=offset),num_boost_round=r["refit_rounds"])
        path=OUT / (r["name"]+".txt")
        path.write_text(model.model_to_string(),encoding="utf8")
        locks.append(dict(r,model_sha256=digest(path)))
    write(OUT / "models_locked.json",{"models":locks,"protocol_sha256":digest(OUT / "protocol.json"),
          "topics_sha256":digest(OUT / "topics.json"),"training_x_sha256":array_hash(data["x"]),
          "training_counts_sha256":array_hash(data["count"]),"training_keys_sha256":array_hash(data["keys"])})
    del data
    gc.collect()
    evaluate(False)


def evaluate(verifying):
    protocol=json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    lock=json.loads((OUT / "models_locked.json").read_text(encoding="utf8"))
    if digest(OUT / "protocol.json")!=lock["protocol_sha256"] or digest(OUT / "topics.json")!=lock["topics_sha256"]:
        raise AssertionError("Topic model configuration changed")
    for p,h in protocol["inputs_sha256"].items():
        if digest(p)!=h:
            raise AssertionError("Topic input dependency changed")
    for name,h in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name)!=h:
            raise AssertionError("Original data changed")
    topics=json.loads((OUT / "topics.json").read_text(encoding="utf8"))
    if verifying:
        data,_=matrix(15,topics)
        for field,key in [("x","training_x_sha256"),("count","training_counts_sha256"),("keys","training_keys_sha256")]:
            if array_hash(data[field])!=lock[key]:
                raise AssertionError("Topic training reconstruction differs")
        del data
        gc.collect()
    data,_=matrix(16,topics,include_labels=True)
    if len(data["y"])!=421983 or int(data["y"].sum())!=3724 or np.any(data["count"]!=-1):
        raise AssertionError("Topic full16 boundary failed")
    records,predictions=[],[]
    for source,path in SOURCES.items():
        if path is not None:
            data["base"]=np.load(path)
            data["mass"],data["any"],block=aggregate(data["base"],data["groups"],len(data["keys"]))
            pos=data["names"].index("base_log_size")
            data["x"][:,pos:pos+4]=block
        for r in lock["models"]:
            path=OUT / (r["name"]+".txt")
            if digest(path)!=r["model_sha256"]:
                raise AssertionError("Topic model changed")
            model=lgb.Booster(model_str=path.read_text(encoding="utf8"))
            gp=prediction(model,data,r["objective"])
            p=correct(data["base"],gp,data["groups"],r["objective"],r["power"])
            records.append({"name":r["name"],"source":source,"power":r["power"],"oof15_auc":r["oof15_auc"],"auc":rank_sum_auc(data["y"],p)})
            predictions.append(p)
    best=int(np.argmax([r["auc"] for r in records]))
    p=predictions[best]
    if verifying:
        error=float(np.max(np.abs(p-np.load(OUT / "prediction.npy"))))
        if error>1e-12:
            raise AssertionError("Topic independent replay differs")
        np.testing.assert_array_equal(data["rows"],np.load(OUT / "valid_indices.npy"))
    else:
        error=0.
        np.save(OUT / "prediction.npy",p);np.save(OUT / "valid_indices.npy",data["rows"])
    result={"status":"passed" if verifying else "evaluated","auc":records[best]["auc"],"best":records[best],
            "records":records,"rows":len(p),"clicks":3724,"maximum_replay_error":error,
            "original_datasets_unchanged":True,"evaluation16_labels_fit":False,"development16_model_selection":True,
            "target_met":records[best]["auc"]>=.82}
    write(OUT / ("verification.json" if verifying else "result.json"),result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--verify",action="store_true")
    evaluate(True) if parser.parse_args().verify else main()
