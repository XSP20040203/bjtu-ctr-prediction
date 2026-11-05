"""Learn row-level ad/context interactions using strictly pre16 ad outcomes."""
import os
for _key in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:
    os.environ[_key]="2"
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from context_rank_boost import matrix as group_matrix
from rich_group_residual import ROOT,HERE,REFERENCE,digest,write,lgb,array_hash,rank_sum_auc

OUT=HERE / "runs/row_context_residual_v1"
CATEGORIES=["age","sex","residence","city","city_rank","series_dev","series_group",
            "device_dev","device_name","device_size","net_type","creat_type_cd","adv_prim_id",
            "inter_type_cd","slot_id","spread_app_id","hispace_app_tags","app_second_class",
            "app_score","u_refreshTimes","u_feedLifeCycle","task_id","adv_id"]
LEAVES=[7,15,31]
POWERS=[0.,.25,.5,.75,1.]
PARAMS={"learning_rate":.025,"min_data_in_leaf":1200,"lambda_l2":40.,"cat_l2":40.,
        "cat_smooth":80.,"max_bin":31,"max_cat_threshold":32,"feature_fraction":.8,
        "num_threads":2,"seed":42,"verbosity":-1,"deterministic":True,"force_col_wise":True,
        "boost_from_average":False,"objective":"binary","metric":"auc"}
SELECTOR=HERE / "runs/context_rank_boost_v1/t0.5_l15.txt"
SOURCES={"reference":None,
         "context_rank":HERE / "runs/context_rank_boost_v1/prediction.npy",
         "previous_day":HERE / "runs/previous_day_residual_v1/prediction.npy",
         "context_scale":HERE / "runs/context_scale_rank_v1/prediction.npy",
         "topic_context":HERE / "runs/topic_context_residual_v1/prediction.npy",
         "catboost_daily":HERE / "runs/catboost_daily_history_v1/prediction.npy"}


def select_columns():
    model=lgb.Booster(model_str=SELECTOR.read_text(encoding="utf8"))
    importance=model.feature_importance(importance_type="gain")
    return np.argsort(-importance,kind="stable")[:100].astype("int32")


def matrix(day,columns,include_labels=False):
    data=group_matrix(day,include_labels=include_labels)
    frame_parts=[]
    for chunk in pd.read_csv(ROOT / "2026ml/train.csv",usecols=["pt_d"]+CATEGORIES,dtype="int32",chunksize=200000):
        frame_parts.append(chunk.loc[chunk.pt_d==20191100+day,CATEGORIES])
    raw=pd.concat(frame_parts)
    np.testing.assert_array_equal(raw.index.to_numpy(),data["rows"])
    group_values=data["x"][:,columns]
    row_values=group_values[data["groups"]]
    base_logit=logit(data["base"].clip(1e-6,1-1e-6))
    np.testing.assert_array_equal(raw.to_numpy().astype("float32").astype("int64"),raw.to_numpy().astype("int64"))
    values=np.column_stack([raw.to_numpy(),row_values,base_logit]).astype("float32")
    del row_values,raw
    names=CATEGORIES+["group__"+data["names"][int(i)] for i in columns]+["reference_logit"]
    return {"x":values,"base":data["base"],"keys":data["keys"],"groups":data["groups"],
            "rows":data["rows"],"y":data.get("y"),"names":names}


def dependencies():
    return [Path(__file__),HERE / "context_rank_boost.py",HERE / "feed_behavior_userday.py",
            HERE / "personal_context_residual.py",HERE / "extended_history_residual.py",
            HERE / "rich_group_residual.py",HERE / "rich_userday_activity.py",HERE / "fit_stacker.py",
            HERE / "verify_stacker.py",SELECTOR,HERE / "runs/feed_behavior_userday_v1/news_schema.json",
            HERE / "runs/weighted_v1_day16/context.npz",HERE / "runs/weighted_v1_day16/context_manifest.json"]+[
                p for p in SOURCES.values() if p is not None]+[
                REFERENCE / f"day{d}_{s}.npy" for d in [15,16] for s in ["indices","prediction"]]


def main():
    if OUT.exists():
        raise FileExistsError("Preserving row/context residual run")
    OUT.mkdir(parents=True)
    columns=select_columns()
    protocol={"params":PARAMS,"leaves":LEAVES,"powers":POWERS,"columns":columns.tolist(),
              "max_rounds":500,"patience":60,"fit_date":20191115,"evaluation_date":20191116,
              "inputs_sha256":{str(p):digest(p) for p in dependencies()},
              "raw_datasets_sha256":{p.name:digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
              "selection":"Prior15-only group ranking model selects100 context columns; user3fold15 chooses row residual rounds/power. Three models locked before16 labels.",
              "selection_caveat":"Context column selection is not nested within15 folds; no unbiased15 score claim.",
              "evaluation16_labels_fit":False,"development16_model_selection":True,"sources":list(SOURCES)}
    write(OUT / "protocol.json",protocol)
    data=matrix(15,columns,include_labels=True)
    userfold=((data["keys"][:,1].astype("uint64")*2654435761)%3).astype("int32")
    folds=userfold[data["groups"]]
    offset=logit(data["base"].clip(1e-6,1-1e-6))
    records,locks=[],[]
    print(json.dumps({"event":"row_context_ready","shape":list(data["x"].shape)}),flush=True)
    for leaves in LEAVES:
        rounds,oof=[],np.empty(len(offset))
        for fold in range(3):
            tr,va=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
            train=lgb.Dataset(data["x"][tr],label=data["y"][tr],init_score=offset[tr],categorical_feature=list(range(len(CATEGORIES))))
            valid=lgb.Dataset(data["x"][va],label=data["y"][va],init_score=offset[va],reference=train)
            model=lgb.train(dict(PARAMS,num_leaves=leaves),train,num_boost_round=500,valid_sets=[valid],
                            callbacks=[lgb.early_stopping(60,verbose=False)])
            rounds.append(model.best_iteration)
            oof[va]=model.predict(data["x"][va],raw_score=True)
        scores=[{"power":p,"auc":rank_sum_auc(data["y"],offset+p*np.clip(oof,-2,2))} for p in POWERS]
        best=max(scores,key=lambda r:r["auc"])
        r={"name":f"leaves{leaves}","leaves":leaves,"rounds":rounds,"refit_rounds":max(1,int(np.median(rounds))),
           "power":best["power"],"oof15_auc":best["auc"],"scores":scores}
        records.append(r)
        np.save(OUT / (r["name"]+"_oof.npy"),oof)
        write(OUT / "cv.json",records)
        print(json.dumps(r),flush=True)
    for r in records:
        train=lgb.Dataset(data["x"],label=data["y"],init_score=offset,categorical_feature=list(range(len(CATEGORIES))))
        model=lgb.train(dict(PARAMS,num_leaves=r["leaves"]),train,num_boost_round=r["refit_rounds"])
        path=OUT / (r["name"]+".txt")
        path.write_text(model.model_to_string(),encoding="utf8")
        locks.append(dict(r,model_sha256=digest(path)))
    write(OUT / "models_locked.json",{"models":locks,"protocol_sha256":digest(OUT / "protocol.json"),
          "x_sha256":array_hash(data["x"]),"y_sha256":array_hash(data["y"]),"rows_sha256":array_hash(data["rows"])})
    del data,offset
    gc.collect()
    evaluate(False)


def evaluate(verifying):
    protocol=json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    lock=json.loads((OUT / "models_locked.json").read_text(encoding="utf8"))
    if digest(OUT / "protocol.json")!=lock["protocol_sha256"]:
        raise AssertionError("Row context protocol changed")
    for p,h in protocol["inputs_sha256"].items():
        if digest(p)!=h:
            raise AssertionError("Row context dependency changed")
    for name,h in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name)!=h:
            raise AssertionError("Original data changed")
    columns=np.asarray(protocol["columns"],dtype="int32")
    if verifying:
        data=matrix(15,columns,include_labels=True)
        for field in ["x","y","rows"]:
            if array_hash(data[field])!=lock[field+"_sha256"]:
                raise AssertionError("Row context training reconstruction differs")
        del data
        gc.collect()
    data=matrix(16,columns,include_labels=True)
    if len(data["y"])!=421983 or int(data["y"].sum())!=3724:
        raise AssertionError("Full16 cohort changed")
    records,predictions=[],[]
    for source,path in SOURCES.items():
        base=data["base"] if path is None else np.load(path)
        offset=logit(base.clip(1e-6,1-1e-6))
        for r in lock["models"]:
            path=OUT / (r["name"]+".txt")
            if digest(path)!=r["model_sha256"]:
                raise AssertionError("Row model changed")
            model=lgb.Booster(model_str=path.read_text(encoding="utf8"))
            correction=np.clip(model.predict(data["x"],raw_score=True),-2,2)
            p=expit(offset+r["power"]*correction)
            records.append({"name":r["name"],"source":source,"power":r["power"],"oof15_auc":r["oof15_auc"],"auc":rank_sum_auc(data["y"],p)})
            predictions.append(p)
    best=int(np.argmax([r["auc"] for r in records]));p=predictions[best]
    if verifying:
        error=float(np.max(np.abs(p-np.load(OUT / "prediction.npy"))))
        if error>1e-12:
            raise AssertionError("Row context independent replay differs")
        np.testing.assert_array_equal(data["rows"],np.load(OUT / "valid_indices.npy"))
    else:
        error=0.
        np.save(OUT / "prediction.npy",p);np.save(OUT / "valid_indices.npy",data["rows"])
    result={"status":"passed" if verifying else "evaluated","auc":records[best]["auc"],"best":records[best],
            "records":records,"rows":len(p),"clicks":3724,"maximum_replay_error":error,
            "original_datasets_unchanged":True,"evaluation16_labels_fit":False,"development16_model_selection":True,"target_met":records[best]["auc"]>=.82}
    write(OUT / ("verification.json" if verifying else "result.json"),result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--verify",action="store_true")
    evaluate(True) if parser.parse_args().verify else main()
