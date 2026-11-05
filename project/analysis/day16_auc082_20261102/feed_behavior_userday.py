"""Daily browser/event distributions from supplied news data, not ad outcomes.

The existing offline snapshot protocol allows same-day news feedback. All
advertisement labels used for fitting still strictly precede November16.
"""
import os
for _key in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:
    os.environ[_key]="2"
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from extended_history_residual import enhanced_history
from personal_context_residual import personal_changes
from rich_group_residual import ROOT,HERE,REFERENCE,digest,write,lgb,rank_sum_auc
from rich_userday_activity import load_day,array_hash,correct

OUT=HERE / "runs/feed_behavior_userday_v1"
CATEGORIES=["u_phonePrice","u_browserLifeCycle","u_browserMode","i_dtype",
            "i_upTimes","i_dislikeTimes","e_ch","e_m","e_po","e_rn","e_section"]
DIVERSITY=["i_docId","i_s_sourceId","i_cat","i_regionEntity","e_pl"]
READ=["u_userId","e_et","label","cillabel","pro"]+CATEGORIES+DIVERSITY
LEAVES=[7,15,31]
POWERS=[0.,.125,.25,.5,.75,1.]
PARAMS={"learning_rate":.025,"min_data_in_leaf":100,"lambda_l2":20.,
        "feature_fraction":.75,"max_bin":31,"num_threads":2,"seed":42,
        "verbosity":-1,"deterministic":True,"force_col_wise":True}
SOURCES={"reference":None,
         "weighted_transfer":HERE / "runs/weighted_calibration_transfer_v1/best_prediction.npy",
         "pairwise":HERE / "runs/pairwise_group_boost_v1/prediction.npy",
         "extended_history":HERE / "runs/extended_history_residual_v1/prediction.npy",
         "catboost":HERE / "runs/catboost_userday_residual_v1/prediction.npy",
         "hurdle":HERE / "runs/hurdle_userday_v1/prediction.npy",
         "personal_context":HERE / "runs/personal_context_residual_v1/prediction.npy"}


def feed_features(feeds,keys,schema=None):
    if set(feeds.columns)!=set(READ):
        raise ValueError("Only declared news fields are allowed")
    if not feeds.label.isin([-1,1]).all():
        raise ValueError("Expected the separate news source's -1/1 labels")
    index=pd.MultiIndex.from_arrays([keys[:,0],keys[:,1]])
    g=index.get_indexer(pd.MultiIndex.from_frame(feeds[["e_et","u_userId"]]))
    keep=(g>=0)&(feeds.e_et.to_numpy()<=20191116)
    frame=feeds.loc[keep].copy()
    g=g[keep]
    if schema is None:
        training=frame.e_et.to_numpy()<20191116
        schema={name:[int(v) for v in frame.loc[training,name].value_counts().index[:16]] for name in CATEGORIES}
    n=len(keys)
    size=np.bincount(g,minlength=n)
    clicked=frame.label.to_numpy()==1
    clicks=np.bincount(g,weights=clicked,minlength=n)
    like=frame.cillabel.to_numpy()==1
    likes=np.bincount(g,weights=like,minlength=n)
    pieces=[np.column_stack([np.log1p(size),np.log1p(clicks),np.log1p(likes),
                             (clicks+.5)/(size+5),(likes+.5)/(size+5),size==0])]
    names=["log_news_count","log_news_clicks","log_news_likes","news_ctr","news_like_rate","news_missing"]
    for name in ["pro","e_pl"]:
        values=frame[name].to_numpy(dtype="float64")
        temp=pd.DataFrame({"group":g,"value":values})
        stats=temp.groupby("group").value.agg(["mean","std","min","max"])
        stats["p25"]=temp.groupby("group").value.quantile(.25)
        stats["p75"]=temp.groupby("group").value.quantile(.75)
        stats=stats.reindex(np.arange(n)).fillna(0.)
        pieces.append(np.arcsinh(stats.to_numpy()))
        names += [name+"_"+c for c in stats.columns]
        clicked_mean=np.bincount(g,weights=values*clicked,minlength=n)/clicks.clip(1)
        pieces.append(np.arcsinh(clicked_mean)[:,None])
        names.append(name+"_clicked_mean")
    for name in DIVERSITY:
        pairs=pd.DataFrame({"group":g,"value":frame[name].to_numpy()}).groupby(["group","value"]).size()
        pg=pairs.index.get_level_values(0).to_numpy()
        count=pairs.to_numpy()
        probability=count/size[pg].clip(1)
        unique=np.bincount(pg,minlength=n)
        entropy=np.bincount(pg,weights=-probability*np.log(probability),minlength=n)
        maximum=np.zeros(n)
        np.maximum.at(maximum,pg,count)
        pieces.append(np.column_stack([np.log1p(unique),unique/size.clip(1),entropy,maximum/size.clip(1)]))
        names += [name+"_"+s for s in ["log_unique","unique_fraction","entropy","top_share"]]
    for name in CATEGORIES:
        values=frame[name].to_numpy()
        for value in schema[name]+[None]:
            matches=(values==value) if value is not None else ~np.isin(values,schema[name])
            total=np.bincount(g,weights=matches,minlength=n)/size.clip(1)
            positive=np.bincount(g,weights=matches&clicked,minlength=n)/clicks.clip(1)
            pieces.append(np.column_stack([total,positive]))
            names += [f"{name}_{value}_share",f"{name}_{value}_clicked_share"]
    result=np.column_stack(pieces).astype("float32")
    if not np.isfinite(result).all():
        raise AssertionError("Nonfinite news features")
    return result,names,schema


def prepare(schema=None):
    data=load_day(16,True,evaluation_labels=False)
    feeds=pd.read_csv(ROOT / "2026ml/train_feeds.csv",usecols=READ,
                      dtype={"i_docId":str,"i_s_sourceId":str})
    values,names,schema=feed_features(feeds,data["keys"],schema)
    changes=personal_changes(data["keys"],values)
    history,hist_names=enhanced_history(data["keys"],data["sizes"],data["counts"],20191116)
    data["x"]=np.column_stack([data["x"],history,values,changes]).astype("float32")
    data["names"] += hist_names+["news__"+s for s in names]
    data["names"] += ["news_previous_delta__"+s for s in names]
    data["names"] += ["news_personal_mean_delta__"+s for s in names]
    data["names"] += ["news_has_previous","news_log_previous_days"]
    return data,schema


def dependency_paths():
    return [Path(__file__),HERE / "extended_history_residual.py",HERE / "personal_context_residual.py",
            HERE / "rich_group_residual.py",HERE / "rich_userday_activity.py",HERE / "fit_stacker.py",
            HERE / "verify_stacker.py",HERE / "runs/weighted_v1_day16/context.npz",
            HERE / "runs/weighted_v1_day16/context_manifest.json"]+[
                p for p in SOURCES.values() if p is not None]+[
                REFERENCE / f"day{d}_{s}.npy" for d in [15,16] for s in ["indices","prediction"]]


def main():
    if OUT.exists():
        raise FileExistsError("Preserving feed-behavior experiment")
    OUT.mkdir(parents=True)
    protocol={"params":PARAMS,"leaves":LEAVES,"powers":POWERS,"objectives":["binary","poisson"],
              "max_rounds":700,"patience":70,"refit_cutoff":20191116,
              "inputs_sha256":{str(p):digest(p) for p in dependency_paths()},
              "raw_datasets_sha256":{p.name:digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
              "selection":"Pre15 ad outcomes train;15 selects rounds/power; refit all strictly pre16. Six models locked before16 labels.",
              "news_protocol":"Supplied same-day news feedback is permitted under existing offline snapshot protocol; no future news rows. News schema uses only pre16 news input frequencies.",
              "evaluation16_labels_fit":False,"development16_model_selection":True,"sources":list(SOURCES)}
    write(OUT / "protocol.json",protocol)
    data,schema=prepare()
    print(json.dumps({"event":"feed_behavior_features_ready","shape":list(data["x"].shape)}),flush=True)
    write(OUT / "news_schema.json",schema)
    train=data["keys"][:,0]<20191115
    valid=data["keys"][:,0]==20191115
    refit=data["keys"][:,0]<20191116
    groups15=pd.MultiIndex.from_arrays([data["keys"][valid,0],data["keys"][valid,1]])
    raw=pd.read_csv(ROOT / "2026ml/train.csv",usecols=["pt_d","user_id","label"],dtype="int32")
    rows15=np.flatnonzero(raw.pt_d.to_numpy()==20191115)
    np.testing.assert_array_equal(rows15,np.load(REFERENCE / "day15_indices.npy"))
    g15=groups15.get_indexer(pd.MultiIndex.from_frame(raw.iloc[rows15][["pt_d","user_id"]]))
    base15=np.load(REFERENCE / "day15_prediction.npy")
    y15=raw.label.to_numpy()[rows15]
    del raw
    records,locks=[],[]
    for objective in protocol["objectives"]:
        target=data["counts"].copy()
        if objective=="binary":
            target[refit]=(target[refit]>0).astype(float)
        params=dict(PARAMS,objective=objective,metric="binary_logloss" if objective=="binary" else "poisson")
        for leaves in LEAVES:
            training=lgb.Dataset(data["x"][train],label=target[train])
            validation=lgb.Dataset(data["x"][valid],label=target[valid],reference=training)
            model=lgb.train(dict(params,num_leaves=leaves),training,num_boost_round=700,valid_sets=[validation],
                            callbacks=[lgb.early_stopping(70,verbose=False)])
            gp=model.predict(data["x"][valid])
            scores=[{"power":p,"auc":rank_sum_auc(y15,correct(base15,gp,g15,objective,p))} for p in POWERS]
            best=max(scores,key=lambda r:r["auc"])
            record={"name":f"{objective}_{leaves}","objective":objective,"leaves":leaves,
                    "rounds":model.best_iteration,"power":best["power"],"validation15_auc":best["auc"],"scores":scores}
            records.append(record)
            np.save(OUT / (record["name"]+"_prediction15.npy"),gp)
            write(OUT / "selection15.json",records)
            final=lgb.train(dict(params,num_leaves=leaves),lgb.Dataset(data["x"][refit],label=target[refit]),
                            num_boost_round=max(1,record["rounds"]))
            path=OUT / (record["name"]+".txt")
            path.write_text(final.model_to_string(),encoding="utf8")
            locks.append(dict(record,model_sha256=digest(path)))
            print(json.dumps(record),flush=True)
    write(OUT / "models_locked.json",{"models":locks,"protocol_sha256":digest(OUT / "protocol.json"),
          "schema_sha256":digest(OUT / "news_schema.json"),"features_sha256":array_hash(data["x"]),
          "target_counts_sha256":array_hash(data["counts"]),"keys_sha256":array_hash(data["keys"]),
          "feature_count":data["x"].shape[1],"max_training_date":int(data["keys"][refit,0].max())})
    del data
    gc.collect()
    evaluate(False)


def evaluate(verifying):
    protocol=json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    lock=json.loads((OUT / "models_locked.json").read_text(encoding="utf8"))
    if digest(OUT / "protocol.json")!=lock["protocol_sha256"] or digest(OUT / "news_schema.json")!=lock["schema_sha256"]:
        raise AssertionError("Locked source protocol changed")
    for p,h in protocol["inputs_sha256"].items():
        if digest(p)!=h:
            raise AssertionError("Feed dependency changed")
    for name,h in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name)!=h:
            raise AssertionError("Original data changed")
    schema=json.loads((OUT / "news_schema.json").read_text(encoding="utf8"))
    data,_=prepare(schema)
    print(json.dumps({"event":"feed_behavior_reconstructed","verify":verifying,"shape":list(data["x"].shape)}),flush=True)
    for field,key in [("x","features_sha256"),("counts","target_counts_sha256"),("keys","keys_sha256")]:
        if array_hash(data[field])!=lock[key]:
            raise AssertionError("Feed input reconstruction differs")
    held=data["keys"][:,0]==20191116
    if np.any(data["counts"][held]!=-1) or lock["max_training_date"]>=20191116:
        raise AssertionError("Advertisement label boundary failed")
    raw=pd.read_csv(ROOT / "2026ml/train.csv",usecols=["pt_d","user_id","label"],dtype="int32")
    rows=np.flatnonzero(raw.pt_d.to_numpy()==20191116)
    np.testing.assert_array_equal(rows,np.load(REFERENCE / "day16_indices.npy"))
    index=pd.MultiIndex.from_arrays([data["keys"][held,0],data["keys"][held,1]])
    groups=index.get_indexer(pd.MultiIndex.from_frame(raw.iloc[rows][["pt_d","user_id"]]))
    y=raw.label.to_numpy()[rows]
    if len(y)!=421983 or int(y.sum())!=3724 or np.any(groups<0):
        raise AssertionError("Full16 cohort changed")
    records,predictions=[],[]
    for r in lock["models"]:
        path=OUT / (r["name"]+".txt")
        if digest(path)!=r["model_sha256"]:
            raise AssertionError("Feed model changed")
        model=lgb.Booster(model_str=path.read_text(encoding="utf8"))
        gp=model.predict(data["x"][held])
        for source,source_path in SOURCES.items():
            base=np.load(REFERENCE / "day16_prediction.npy" if source_path is None else source_path)
            p=correct(base,gp,groups,r["objective"],r["power"])
            records.append({"name":r["name"],"source":source,"power":r["power"],
                            "validation15_auc":r["validation15_auc"],"auc":rank_sum_auc(y,p)})
            predictions.append(p)
    best=int(np.argmax([r["auc"] for r in records]))
    p=predictions[best]
    if verifying:
        error=float(np.max(np.abs(p-np.load(OUT / "prediction.npy"))))
        if error>1e-12:
            raise AssertionError("Feed model replay differs")
        np.testing.assert_array_equal(rows,np.load(OUT / "valid_indices.npy"))
    else:
        error=0.
        np.save(OUT / "prediction.npy",p)
        np.save(OUT / "valid_indices.npy",rows)
    result={"status":"passed" if verifying else "evaluated","auc":records[best]["auc"],"best":records[best],
            "records":records,"rows":len(p),"clicks":3724,"maximum_replay_error":error,
            "original_datasets_unchanged":True,"evaluation16_labels_fit":False,
            "same_day_news_feedback":True,"development16_model_selection":True,"target_met":records[best]["auc"]>=.82}
    write(OUT / ("verification.json" if verifying else "result.json"),result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--verify",action="store_true")
    evaluate(True) if parser.parse_args().verify else main()
