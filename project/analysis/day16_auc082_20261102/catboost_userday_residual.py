"""Categorical user-day residual trees, trained with day15 outcomes only."""
import os
for _key in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:
    os.environ[_key]="2"
import argparse
import gc
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from catboost import CatBoost,Pool
from scipy.special import expit
from extended_history_residual import (ROOT,HERE,REFERENCE,digest,write,matrix,target_offset,
                                       correct,rank_sum_auc,array_hash)

OUT=HERE / "runs/catboost_userday_residual_v1"
CATEGORIES=["age","sex","residence","city","city_rank","series_dev","series_group",
            "device_dev","device_name","net_type"]
DEPTHS=[4,6]
POWERS=[0.,.25,.5,.75,1.]
PARAMS={"learning_rate":.04,"l2_leaf_reg":10.,"random_strength":.5,"rsm":.8,
        "bootstrap_type":"Bernoulli","subsample":.8,"border_count":32,
        "one_hot_max_size":10,"max_ctr_complexity":1,"thread_count":2,
        "random_seed":42,"task_type":"CPU","allow_writing_files":False,"verbose":False}


def save_model(model,path):
    with tempfile.TemporaryDirectory(prefix="ctr_cb_") as folder:
        temporary=Path(folder) / "model.cbm"
        model.save_model(str(temporary))
        Path(path).write_bytes(temporary.read_bytes())


def load_model(path):
    model=CatBoost()
    with tempfile.TemporaryDirectory(prefix="ctr_cb_") as folder:
        temporary=Path(folder) / "model.cbm"
        temporary.write_bytes(Path(path).read_bytes())
        model.load_model(str(temporary))
    return model


def features(day,source="reference",evaluation_labels=False):
    data=matrix(day,"reference" if source=="group_meta" else source,evaluation_labels)
    parts=[]
    for chunk in pd.read_csv(ROOT / "2026ml/train.csv",usecols=["pt_d","user_id"]+CATEGORIES,dtype="int32",chunksize=200000):
        parts.append(chunk.loc[chunk.pt_d.to_numpy()==20191100+day])
    raw=pd.concat(parts).drop_duplicates(["pt_d","user_id"],keep="first").set_index(["pt_d","user_id"])
    index=pd.MultiIndex.from_arrays([data["keys"][:,0],data["keys"][:,1]])
    categories=raw.reindex(index)[CATEGORIES]
    if categories.isna().any().any():
        raise AssertionError("Missing original user-day category snapshot")
    frame=pd.DataFrame(data["x"],columns=[f"x{i}" for i in range(data["x"].shape[1])])
    for field in CATEGORIES:
        frame[field]=categories[field].to_numpy(dtype="int64")
    if source=="group_meta":
        from rich_group_residual import aggregate
        data["base"]=np.load(HERE / "runs/group_meta_calibration_v1/prediction.npy")
        data["mass"],data["any"],block=aggregate(data["base"],data["groups"],len(data["keys"]))
        # Locate existing baseline features before the enhanced history columns.
        position=data["names"].index("base_log_size")
        data["x"][:,position:position+4]=block
        frame.iloc[:,position:position+4]=block
    data["frame"]=frame
    return data


def predict(model,data,objective,rows=None):
    rows=np.arange(len(data["keys"])) if rows is None else rows
    _,offset=target_offset(data,objective)
    raw=model.predict(Pool(data["frame"].iloc[rows],cat_features=CATEGORIES),prediction_type="RawFormulaVal")
    z=offset[rows]+np.clip(raw,-3,3)
    return expit(z) if objective=="binary" else np.exp(z)


def dependencies():
    return [Path(__file__),HERE / "extended_history_residual.py",HERE / "rich_group_residual.py",
            HERE / "rich_userday_activity.py",HERE / "fit_stacker.py",HERE / "verify_stacker.py",
            HERE / "runs/weighted_v1_day16/context.npz",HERE / "runs/weighted_v1_day16/context_manifest.json",
            HERE / "runs/weighted_calibration_transfer_v1/best_prediction.npy",
            HERE / "runs/pairwise_group_boost_v1/prediction.npy",HERE / "runs/group_meta_calibration_v1/prediction.npy",
            HERE / "runs/group_meta_calibration_v1/verification.json"]+[
                REFERENCE / f"day{d}_{suffix}.npy" for d in [15,16] for suffix in ["indices","prediction"]]


def main():
    if OUT.exists():
        raise FileExistsError("Preserving categorical residual models")
    OUT.mkdir(parents=True)
    protocol={"params":PARAMS,"depths":DEPTHS,"powers":POWERS,"categorical_fields":CATEGORIES,
              "objectives":["binary","poisson"],"max_iterations":800,"early_stopping":80,
              "inputs_sha256":{str(p):digest(p) for p in dependencies()},
              "raw_datasets_sha256":{p.name:digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
              "fit_date":20191115,"evaluation_date":20191116,"evaluation16_labels_fit":False,
              "raw_category_rule":"First supplied raw snapshot per user/date, no presumed within-day chronology",
              "selection":"Three user folds on15 choose median iterations and row-AUC power. All4 models saved before16 comparisons.",
              "development16_model_selection":True}
    write(OUT / "protocol.json",protocol)
    data=features(15,evaluation_labels=True)
    folds=((data["keys"][:,1].astype("uint64")*2654435761)%3).astype("int32")
    records,locks=[],[]
    for objective in protocol["objectives"]:
        target,offset=target_offset(data,objective)
        loss="Logloss" if objective=="binary" else "Poisson"
        for depth in DEPTHS:
            oof,iterations=np.empty(len(target)),[]
            for fold in range(3):
                tr,va=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
                train=Pool(data["frame"].iloc[tr],label=target[tr],baseline=offset[tr],cat_features=CATEGORIES)
                valid=Pool(data["frame"].iloc[va],label=target[va],baseline=offset[va],cat_features=CATEGORIES)
                model=CatBoost(dict(PARAMS,loss_function=loss,eval_metric=loss,depth=depth,iterations=800))
                model.fit(train,eval_set=valid,early_stopping_rounds=80,use_best_model=True)
                oof[va]=predict(model,data,objective,va)
                iterations.append(model.get_best_iteration()+1)
            scores=[{"power":p,"auc":rank_sum_auc(data["y"],correct(data["base"],oof,data["groups"],objective,p))} for p in POWERS]
            best=max(scores,key=lambda r:r["auc"])
            record={"name":f"{objective}_d{depth}","objective":objective,"depth":depth,"iterations":iterations,
                    "refit_iterations":max(1,int(np.median(iterations))),"power":best["power"],"oof15_auc":best["auc"],"scores":scores}
            records.append(record)
            np.save(OUT / (record["name"]+"_oof.npy"),oof)
            write(OUT / "cv.json",records)
            print(json.dumps(record),flush=True)
    for record in records:
        target,offset=target_offset(data,record["objective"])
        loss="Logloss" if record["objective"]=="binary" else "Poisson"
        model=CatBoost(dict(PARAMS,loss_function=loss,depth=record["depth"],iterations=record["refit_iterations"]))
        model.fit(Pool(data["frame"],label=target,baseline=offset,cat_features=CATEGORIES))
        path=OUT / (record["name"]+".cbm")
        save_model(model,path)
        locks.append(dict(record,model_sha256=digest(path)))
    write(OUT / "models_locked.json",{"models":locks,"protocol_sha256":digest(OUT / "protocol.json"),
          "training_x_sha256":array_hash(data["frame"].to_numpy()),"training_counts_sha256":array_hash(data["count"]),
          "training_keys_sha256":array_hash(data["keys"])})
    del data
    gc.collect()
    evaluate(False)


def evaluate(verifying):
    protocol=json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    lock=json.loads((OUT / "models_locked.json").read_text(encoding="utf8"))
    if digest(OUT / "protocol.json")!=lock["protocol_sha256"]:
        raise AssertionError("Protocol changed")
    for path,expected in protocol["inputs_sha256"].items():
        if digest(path)!=expected:
            raise AssertionError("Bound source/input changed")
    for name,expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name)!=expected:
            raise AssertionError("Original data changed")
    if verifying:
        train=features(15)
        if array_hash(train["frame"].to_numpy())!=lock["training_x_sha256"] or array_hash(train["count"])!=lock["training_counts_sha256"] or array_hash(train["keys"])!=lock["training_keys_sha256"]:
            raise AssertionError("Training reconstruction differs")
        del train
    records,predictions=[],[]
    for source in ["reference","weighted_transfer","pairwise","group_meta"]:
        data=features(16,source,evaluation_labels=True)
        if len(data["y"])!=421983 or int(data["y"].sum())!=3724 or np.any(data["count"]!=-1):
            raise AssertionError("Held-out cohort changed")
        for record in lock["models"]:
            path=OUT / (record["name"]+".cbm")
            if digest(path)!=record["model_sha256"]:
                raise AssertionError("Model changed")
            model=load_model(path)
            group_prediction=predict(model,data,record["objective"])
            p=correct(data["base"],group_prediction,data["groups"],record["objective"],record["power"])
            records.append({"name":record["name"],"source":source,"power":record["power"],"oof15_auc":record["oof15_auc"],"auc":rank_sum_auc(data["y"],p)})
            predictions.append(p)
    best=int(np.argmax([r["auc"] for r in records]))
    p=predictions[best]
    if verifying:
        error=float(np.max(np.abs(p-np.load(OUT / "prediction.npy"))))
        if error>1e-12:
            raise AssertionError("Reloaded prediction differs")
        np.testing.assert_array_equal(data["rows"],np.load(OUT / "valid_indices.npy"))
    else:
        error=0.
        np.save(OUT / "prediction.npy",p)
        np.save(OUT / "valid_indices.npy",data["rows"])
    result={"status":"passed" if verifying else "evaluated","auc":records[best]["auc"],"best":records[best],
            "records":records,"rows":len(p),"clicks":3724,"maximum_replay_error":error,
            "original_datasets_unchanged":True,"evaluation16_labels_fit":False,"development16_model_selection":True,
            "target_met":records[best]["auc"]>=.82}
    write(OUT / ("verification.json" if verifying else "result.json"),result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--verify",action="store_true")
    evaluate(True) if parser.parse_args().verify else main()
