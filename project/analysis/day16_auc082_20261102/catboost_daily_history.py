"""Learn repeated-user daily response from all earlier days and news inputs."""
import os
for _key in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:
    os.environ[_key]="2"
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from catboost import CatBoost,Pool
from feed_behavior_userday import prepare,dependency_paths,SOURCES
from catboost_userday_residual import CATEGORIES,save_model,load_model
from neural_userday import row_mapping
from rich_group_residual import ROOT,HERE,REFERENCE,digest,write,rank_sum_auc
from rich_userday_activity import array_hash,correct

OUT=HERE / "runs/catboost_daily_history_v1"
POWERS=[0.,.125,.25,.5,.75,1.]
PARAMS={"learning_rate":.03,"depth":6,"l2_leaf_reg":30.,"random_strength":.5,
        "bootstrap_type":"Bernoulli","subsample":.8,"border_count":32,
        "one_hot_max_size":10,"max_ctr_complexity":1,"thread_count":2,
        "random_seed":42,"task_type":"GPU","devices":"0","gpu_ram_part":.45,
        "allow_writing_files":False,"verbose":False}
SOURCES={**SOURCES,"neural_userday":HERE / "runs/neural_userday_v1/prediction.npy"}


def features(schema=None):
    data,schema=prepare(schema)
    raw=pd.read_csv(ROOT / "2026ml/train.csv",usecols=["pt_d","user_id"]+CATEGORIES,dtype="int32")
    raw=raw.drop_duplicates(["pt_d","user_id"],keep="first").set_index(["pt_d","user_id"])
    index=pd.MultiIndex.from_arrays([data["keys"][:,0],data["keys"][:,1]])
    cats=raw.reindex(index)
    if cats.isna().any().any():
        raise AssertionError("Daily category alignment failed")
    frame=pd.DataFrame(data["x"],columns=[f"x{i}" for i in range(data["x"].shape[1])])
    for field in CATEGORIES:
        frame[field]=cats[field].to_numpy(dtype="int64")
    frame["user_id"]=data["keys"][:,1].astype("int64")
    data["frame"]=frame
    return data,schema


def probability(model,frame,objective,categories):
    raw=model.predict(Pool(frame,cat_features=categories),prediction_type="RawFormulaVal")
    return expit(raw) if objective=="binary" else np.exp(np.clip(raw,-15,8))


def dependencies():
    return list(dict.fromkeys([Path(__file__),HERE / "catboost_userday_residual.py",
             HERE / "feed_behavior_userday.py",HERE / "neural_userday.py"]+dependency_paths()+[
                p for p in SOURCES.values() if p is not None]))


def main():
    if OUT.exists():
        raise FileExistsError("Preserving daily categorical history experiment")
    OUT.mkdir(parents=True)
    protocol={"params":PARAMS,"powers":POWERS,"objectives":["binary","poisson"],
              "identity_scopes":[False,True],"max_iterations":1200,"patience":100,
              "inputs_sha256":{str(p):digest(p) for p in dependencies()},
              "raw_datasets_sha256":{p.name:digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
              "selection":"Train pre15 daily responses;15 selects iterations and power. Refit strictly pre16; all four models locked before16 labels.",
              "raw_categories":"First supplied snapshot per user/day; no within-day order claim. Optional user_id categorical uses only pre16 outcomes.",
              "group_weight":"One training example per observed user/day",
              "evaluation16_labels_fit":False,"same_day_news_feedback":True,
              "development16_model_selection":True,"sources":list(SOURCES)}
    write(OUT / "protocol.json",protocol)
    data,schema=features()
    write(OUT / "news_schema.json",schema)
    train=np.flatnonzero(data["keys"][:,0]<20191115)
    valid=np.flatnonzero(data["keys"][:,0]==20191115)
    refit=np.flatnonzero(data["keys"][:,0]<20191116)
    _,groups,y=row_mapping(data,15)
    base=np.load(REFERENCE / "day15_prediction.npy")
    print(json.dumps({"event":"daily_categorical_ready","shape":list(data["frame"].shape)}),flush=True)
    records,locks=[],[]
    for identity in [False,True]:
        cats=CATEGORIES+(["user_id"] if identity else [])
        columns=[c for c in data["frame"].columns if c!="user_id" or identity]
        frame=data["frame"][columns]
        for objective in ["binary","poisson"]:
            target=data["counts"].copy()
            if objective=="binary":
                target[refit]=target[refit]>0
            loss="Logloss" if objective=="binary" else "Poisson"
            model=CatBoost(dict(PARAMS,iterations=1200,loss_function=loss,eval_metric=loss))
            model.fit(Pool(frame.iloc[train],label=target[train],cat_features=cats),
                      eval_set=Pool(frame.iloc[valid],label=target[valid],cat_features=cats),
                      early_stopping_rounds=100,use_best_model=True)
            gp=probability(model,frame.iloc[valid],objective,cats)
            scores=[{"power":p,"auc":rank_sum_auc(y,correct(base,gp,groups,objective,p))} for p in POWERS]
            best=max(scores,key=lambda r:r["auc"])
            name=f"identity{int(identity)}_{objective}"
            record={"name":name,"identity":identity,"objective":objective,"iterations":model.get_best_iteration()+1,
                    "power":best["power"],"validation15_auc":best["auc"],"scores":scores}
            records.append(record)
            np.save(OUT / (name+"_prediction15.npy"),gp)
            write(OUT / "selection15.json",records)
            print(json.dumps(record),flush=True)
            del model
            final=CatBoost(dict(PARAMS,iterations=record["iterations"],loss_function=loss))
            final.fit(Pool(frame.iloc[refit],label=target[refit],cat_features=cats))
            path=OUT / (name+".cbm")
            save_model(final,path)
            locks.append(dict(record,model_sha256=digest(path),columns=columns,categories=cats))
            del final
            gc.collect()
    write(OUT / "models_locked.json",{"models":locks,"protocol_sha256":digest(OUT / "protocol.json"),
          "schema_sha256":digest(OUT / "news_schema.json"),"features_sha256":array_hash(data["frame"].to_numpy()),
          "counts_sha256":array_hash(data["counts"]),"keys_sha256":array_hash(data["keys"])})
    del data,frame
    gc.collect()
    evaluate(False)


def evaluate(verifying):
    protocol=json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    lock=json.loads((OUT / "models_locked.json").read_text(encoding="utf8"))
    if digest(OUT / "protocol.json")!=lock["protocol_sha256"] or digest(OUT / "news_schema.json")!=lock["schema_sha256"]:
        raise AssertionError("Daily CatBoost protocol changed")
    for p,h in protocol["inputs_sha256"].items():
        if digest(p)!=h:
            raise AssertionError("Bound dependency changed")
    for name,h in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name)!=h:
            raise AssertionError("Original data changed")
    data,_=features(json.loads((OUT / "news_schema.json").read_text(encoding="utf8")))
    for actual,expected in [(data["frame"].to_numpy(),lock["features_sha256"]),
                            (data["counts"],lock["counts_sha256"]),(data["keys"],lock["keys_sha256"])]:
        if array_hash(actual)!=expected:
            raise AssertionError("Daily CatBoost input reconstruction differs")
    held=np.flatnonzero(data["keys"][:,0]==20191116)
    if np.any(data["counts"][held]!=-1):
        raise AssertionError("Held daily labels are not masked")
    rows,groups,y=row_mapping(data,16)
    if len(rows)!=421983 or int(y.sum())!=3724:
        raise AssertionError("Full16 cohort changed")
    records,predictions=[],[]
    for r in lock["models"]:
        path=OUT / (r["name"]+".cbm")
        if digest(path)!=r["model_sha256"]:
            raise AssertionError("Daily CatBoost model changed")
        model=load_model(path)
        gp=probability(model,data["frame"].iloc[held][r["columns"]],r["objective"],r["categories"])
        for source,path in SOURCES.items():
            base=np.load(REFERENCE / "day16_prediction.npy" if path is None else path)
            p=correct(base,gp,groups,r["objective"],r["power"])
            records.append({"name":r["name"],"source":source,"power":r["power"],"validation15_auc":r["validation15_auc"],"auc":rank_sum_auc(y,p)})
            predictions.append(p)
    best=int(np.argmax([r["auc"] for r in records]))
    p=predictions[best]
    if verifying:
        error=float(np.max(np.abs(p-np.load(OUT / "prediction.npy"))))
        if error>1e-12:
            raise AssertionError("Daily CatBoost replay differs")
        np.testing.assert_array_equal(rows,np.load(OUT / "valid_indices.npy"))
    else:
        error=0.
        np.save(OUT / "prediction.npy",p);np.save(OUT / "valid_indices.npy",rows)
    result={"status":"passed" if verifying else "evaluated","auc":records[best]["auc"],"best":records[best],
            "records":records,"rows":len(p),"clicks":3724,"maximum_replay_error":error,
            "original_datasets_unchanged":True,"evaluation16_labels_fit":False,"development16_model_selection":True,
            "target_met":records[best]["auc"]>=.82}
    write(OUT / ("verification.json" if verifying else "result.json"),result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--verify",action="store_true")
    evaluate(True) if parser.parse_args().verify else main()
