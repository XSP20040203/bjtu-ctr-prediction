"""Use changes from a user's earlier label-free daily context to correct scores."""
import os
for _key in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:
    os.environ[_key]="2"
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from extended_history_residual import matrix as history_matrix
from rich_group_residual import (ROOT,HERE,REFERENCE,digest,write,target_offset,
                                 prediction,lgb,array_hash,rank_sum_auc,aggregate)
from rich_userday_activity import load_day,correct

OUT=HERE / "runs/personal_context_residual_v1"
LEAVES=[3,7,15]
POWERS=[0.,.25,.5,.75,1.]
PARAMS={"learning_rate":.025,"min_data_in_leaf":150,"lambda_l2":20.,
        "feature_fraction":.65,"max_bin":31,"num_threads":2,"seed":42,
        "verbosity":-1,"deterministic":True,"force_col_wise":True,
        "boost_from_average":False}
SOURCES={"reference":None,
         "weighted_transfer":HERE / "runs/weighted_calibration_transfer_v1/best_prediction.npy",
         "pairwise":HERE / "runs/pairwise_group_boost_v1/prediction.npy",
         "extended_history":HERE / "runs/extended_history_residual_v1/prediction.npy",
         "catboost":HERE / "runs/catboost_userday_residual_v1/prediction.npy",
         "hurdle":HERE / "runs/hurdle_userday_v1/prediction.npy"}


def personal_changes(keys,values):
    """Compute today's deviations, querying each user before inserting today."""
    if len(keys)!=len(values) or values.ndim!=2:
        raise ValueError("Unaligned daily values")
    result=np.zeros((len(keys),2*values.shape[1]+2),dtype="float32")
    prior={}
    for day in np.unique(keys[:,0]):
        ids=np.flatnonzero(keys[:,0]==day)
        users=keys[ids,1]
        if len(np.unique(users))!=len(users):
            raise ValueError("Repeated user/day key")
        for i in ids:
            entry=prior.get(int(keys[i,1]))
            if entry is not None:
                previous,total,count=entry
                width=values.shape[1]
                result[i,:width]=values[i]-previous
                result[i,width:2*width]=values[i]-total/count
                result[i,-2:]=[1.,np.log1p(count)]
        for i in ids:
            user=int(keys[i,1])
            entry=prior.get(user)
            total=values[i].astype("float64") if entry is None else entry[1]+values[i]
            count=1 if entry is None else entry[2]+1
            prior[user]=(values[i].copy(),total,count)
    return result


def matrix(day,source="reference",evaluation_labels=False):
    data=history_matrix(day,evaluation_labels=evaluation_labels)
    context=load_day(16,False,evaluation_labels=False)
    changes=personal_changes(context["keys"],context["x"])
    chosen=context["keys"][:,0]==20191100+day
    np.testing.assert_array_equal(data["keys"],context["keys"][chosen])
    data["x"]=np.column_stack([data["x"],changes[chosen]]).astype("float32")
    data["names"] += ["previous_delta__"+s for s in context["names"]]
    data["names"] += ["personal_mean_delta__"+s for s in context["names"]]
    data["names"] += ["has_personal_context","log_personal_context_days"]
    source_path=SOURCES[source]
    if source_path is not None:
        if day!=16:
            raise ValueError("Alternate sources are only fixed16 transfer comparisons")
        data["base"]=np.load(source_path)
        data["mass"],data["any"],block=aggregate(data["base"],data["groups"],len(data["keys"]))
        pos=data["names"].index("base_log_size")
        data["x"][:,pos:pos+4]=block
    return data


def dependencies():
    return [Path(__file__),HERE / "extended_history_residual.py",HERE / "rich_group_residual.py",
            HERE / "rich_userday_activity.py",HERE / "fit_stacker.py",HERE / "verify_stacker.py",
            HERE / "runs/weighted_v1_day16/context.npz",HERE / "runs/weighted_v1_day16/context_manifest.json",
            HERE / "runs/extended_history_residual_v1/verification.json"]+[
                p for p in SOURCES.values() if p is not None]+[
                REFERENCE / f"day{d}_{s}.npy" for d in [15,16] for s in ["indices","prediction"]]


def main():
    if OUT.exists():
        raise FileExistsError("Preserving personal-context experiment")
    OUT.mkdir(parents=True)
    protocol={"params":PARAMS,"leaves":LEAVES,"powers":POWERS,"objectives":["binary","poisson"],
              "max_rounds":500,"patience":60,"fit_date":20191115,"evaluation_date":20191116,
              "inputs_sha256":{str(p):digest(p) for p in dependencies()},
              "raw_datasets_sha256":{p.name:digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
              "context_rule":"Current minus previous observation and all earlier user days; query before insertion; predictor-only context.",
              "selection":"User3fold15 selects rounds and power; six models locked before16 labels.",
              "evaluation16_labels_fit":False,"development16_model_selection":True,"sources":list(SOURCES)}
    write(OUT / "protocol.json",protocol)
    data=matrix(15,evaluation_labels=True)
    print(json.dumps({"event":"personal_context_ready","shape":list(data["x"].shape)}),flush=True)
    if np.any(data["count"]<0):
        raise AssertionError("Missing15 training count")
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
            record={"name":f"{objective}_{leaves}","objective":objective,"leaves":leaves,"rounds":rounds,
                    "refit_rounds":max(1,int(np.median(rounds))),"power":best["power"],"oof15_auc":best["auc"],"scores":scores}
            records.append(record)
            np.save(OUT / (record["name"]+"_oof.npy"),oof)
            write(OUT / "cv.json",records)
            print(json.dumps(record),flush=True)
    for r in records:
        target,offset=target_offset(data,r["objective"])
        model=lgb.train(dict(PARAMS,objective=r["objective"],num_leaves=r["leaves"]),
                        lgb.Dataset(data["x"],label=target,init_score=offset),num_boost_round=r["refit_rounds"])
        path=OUT / (r["name"]+".txt")
        path.write_text(model.model_to_string(),encoding="utf8")
        locks.append(dict(r,model_sha256=digest(path)))
    write(OUT / "models_locked.json",{"models":locks,"protocol_sha256":digest(OUT / "protocol.json"),
          "training_x_sha256":array_hash(data["x"]),"training_counts_sha256":array_hash(data["count"]),
          "training_keys_sha256":array_hash(data["keys"])})
    del data
    gc.collect()
    evaluate(False)


def evaluate(verifying):
    protocol=json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    lock=json.loads((OUT / "models_locked.json").read_text(encoding="utf8"))
    if digest(OUT / "protocol.json")!=lock["protocol_sha256"]:
        raise AssertionError("Personal context protocol changed")
    for path,h in protocol["inputs_sha256"].items():
        if digest(path)!=h:
            raise AssertionError("Bound input changed")
    for name,h in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name)!=h:
            raise AssertionError("Original data changed")
    if verifying:
        data=matrix(15)
        for field,key in [("x","training_x_sha256"),("count","training_counts_sha256"),("keys","training_keys_sha256")]:
            if array_hash(data[field])!=lock[key]:
                raise AssertionError("Personal context training reconstruction differs")
        del data
        gc.collect()
    records,predictions=[],[]
    data=matrix(16,evaluation_labels=True)
    for source in SOURCES:
        print(json.dumps({"event":"personal_context_replay_source","source":source,"verify":verifying}),flush=True)
        source_path=SOURCES[source]
        if source_path is not None:
            data["base"]=np.load(source_path)
            data["mass"],data["any"],block=aggregate(data["base"],data["groups"],len(data["keys"]))
            pos=data["names"].index("base_log_size")
            data["x"][:,pos:pos+4]=block
        if len(data["y"])!=421983 or int(data["y"].sum())!=3724 or np.any(data["count"]!=-1):
            raise AssertionError("Held16 boundary failed")
        for r in lock["models"]:
            path=OUT / (r["name"]+".txt")
            if digest(path)!=r["model_sha256"]:
                raise AssertionError("Model changed")
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
            raise AssertionError("Personal context replay differs")
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
