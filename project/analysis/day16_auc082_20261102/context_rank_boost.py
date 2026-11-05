"""Rank advertisement rows using personal-context and raw news group features."""
import os
for _key in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:
    os.environ[_key]="2"
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from personal_context_residual import matrix as personal_matrix
from feed_behavior_userday import prepare as feed_prepare,dependency_paths as feed_dependencies
from pairwise_group_boost import make_objective,row_auc_metric,PARAMS as BASE_PARAMS
from rich_group_residual import ROOT,HERE,REFERENCE,digest,write,lgb,array_hash,rank_sum_auc,aggregate

OUT=HERE / "runs/context_rank_boost_v1"
LEAVES=[7,15]
TEMPERATURES=[.5,1.]
POWERS=[0.,.25,.5,.75,1.]
PARAMS=dict(BASE_PARAMS,feature_fraction=.65,min_data_in_leaf=150,lambda_l2=5.)
SOURCES={"reference":None,
         "weighted_transfer":HERE / "runs/weighted_calibration_transfer_v1/best_prediction.npy",
         "pairwise":HERE / "runs/pairwise_group_boost_v1/prediction.npy",
         "catboost":HERE / "runs/catboost_userday_residual_v1/prediction.npy",
         "personal_context":HERE / "runs/personal_context_residual_v1/prediction.npy",
         "feed_behavior":HERE / "runs/feed_behavior_userday_v1/prediction.npy",
         "neural_userday":HERE / "runs/neural_userday_v1/prediction.npy"}


def matrix(day,include_labels=False):
    data=personal_matrix(day,evaluation_labels=include_labels)
    schema=json.loads((HERE / "runs/feed_behavior_userday_v1/news_schema.json").read_text(encoding="utf8"))
    feed,_=feed_prepare(schema)
    selected=feed["keys"][:,0]==20191100+day
    np.testing.assert_array_equal(data["keys"],feed["keys"][selected])
    columns=[i for i,name in enumerate(feed["names"]) if name.startswith("news_")]
    data["x"]=np.column_stack([data["x"],feed["x"][selected][:,columns]]).astype("float32")
    data["names"] += [feed["names"][i] for i in columns]
    if data["x"].shape[1]!=len(data["names"]):
        raise AssertionError("Feature/name width mismatch")
    return data


def dependencies():
    return list(dict.fromkeys([Path(__file__),HERE / "pairwise_group_boost.py",
            HERE / "personal_context_residual.py",HERE / "feed_behavior_userday.py",
            HERE / "runs/feed_behavior_userday_v1/news_schema.json"]+feed_dependencies()+[
                p for p in SOURCES.values() if p is not None]))


def main():
    if OUT.exists():
        raise FileExistsError("Preserving context rank experiment")
    OUT.mkdir(parents=True)
    protocol={"params":PARAMS,"leaves":LEAVES,"temperatures":TEMPERATURES,"powers":POWERS,
              "max_rounds":400,"patience":60,"fit_date":20191115,"evaluation_date":20191116,
              "inputs_sha256":{str(p):digest(p) for p in dependencies()},
              "raw_datasets_sha256":{p.name:digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
              "selection":"User3fold15 chooses median rounds and power; all four rank models locked before16 labels.",
              "pairs":"64 uniform negative advertisement rows per positive, training user folds only; same-group pairs removed; ridge0.02",
              "features":"Previous/all-past personal context deltas plus supplied same-day news event summaries and their personal deltas",
              "evaluation16_labels_fit":False,"development16_model_selection":True,"sources":list(SOURCES)}
    write(OUT / "protocol.json",protocol)
    data=matrix(15,include_labels=True)
    folds=((data["keys"][:,1].astype("uint64")*2654435761)%3).astype("int32")
    print(json.dumps({"event":"rank_context_ready","shape":list(data["x"].shape)}),flush=True)
    records,locks=[],[]
    for temperature in TEMPERATURES:
        for leaves in LEAVES:
            oof,rounds=np.empty(len(data["keys"])),[]
            for fold in range(3):
                tr,va=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
                objective=make_objective(data,tr,temperature,4242+fold)
                train=lgb.Dataset(data["x"][tr],label=np.zeros(len(tr)))
                valid=lgb.Dataset(data["x"][va],label=np.zeros(len(va)),reference=train)
                model=lgb.train(dict(PARAMS,num_leaves=leaves,objective=objective),train,
                                num_boost_round=400,valid_sets=[valid],feval=row_auc_metric(data,va),
                                callbacks=[lgb.early_stopping(60,verbose=False)])
                oof[va]=model.predict(data["x"][va],raw_score=True)
                rounds.append(model.best_iteration)
            margin=logit(data["base"].clip(1e-6,1-1e-6))
            scores=[{"power":p,"auc":rank_sum_auc(data["y"],margin+p*np.clip(oof[data["groups"]],-3,3))} for p in POWERS]
            best=max(scores,key=lambda r:r["auc"])
            r={"name":f"t{temperature:g}_l{leaves}","temperature":temperature,"leaves":leaves,
               "rounds":rounds,"refit_rounds":max(1,int(np.median(rounds))),"power":best["power"],
               "oof15_auc":best["auc"],"scores":scores}
            records.append(r)
            np.save(OUT / (r["name"]+"_oof.npy"),oof)
            write(OUT / "cv.json",records)
            print(json.dumps(r),flush=True)
    for r in records:
        objective=make_objective(data,np.arange(len(data["keys"])),r["temperature"],4242)
        model=lgb.train(dict(PARAMS,num_leaves=r["leaves"],objective=objective),
                        lgb.Dataset(data["x"],label=np.zeros(len(data["keys"]))),num_boost_round=r["refit_rounds"])
        path=OUT / (r["name"]+".txt")
        path.write_text(model.model_to_string(),encoding="utf8")
        locks.append(dict(r,model_sha256=digest(path),pair_a_sha256=array_hash(objective.a),
                          pair_b_sha256=array_hash(objective.b),pair_margin_sha256=array_hash(objective.margin)))
    write(OUT / "models_locked.json",{"models":locks,"protocol_sha256":digest(OUT / "protocol.json"),
          "training_x_sha256":array_hash(data["x"]),"training_y_sha256":array_hash(data["y"]),
          "training_keys_sha256":array_hash(data["keys"])})
    del data
    gc.collect()
    evaluate(False)


def evaluate(verifying):
    protocol=json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    lock=json.loads((OUT / "models_locked.json").read_text(encoding="utf8"))
    if digest(OUT / "protocol.json")!=lock["protocol_sha256"]:
        raise AssertionError("Rank protocol changed")
    for p,h in protocol["inputs_sha256"].items():
        if digest(p)!=h:
            raise AssertionError("Rank input dependency changed")
    for name,h in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name)!=h:
            raise AssertionError("Original data changed")
    if verifying:
        data=matrix(15,include_labels=True)
        for field,key in [("x","training_x_sha256"),("y","training_y_sha256"),("keys","training_keys_sha256")]:
            if array_hash(data[field])!=lock[key]:
                raise AssertionError("Rank training reconstruction differs")
        for r in lock["models"]:
            objective=make_objective(data,np.arange(len(data["keys"])),r["temperature"],4242)
            for attr in ["a","b","margin"]:
                if array_hash(getattr(objective,attr))!=r[f"pair_{attr}_sha256"]:
                    raise AssertionError("Rank pair sample changed")
        del data
        gc.collect()
    data=matrix(16,include_labels=True)
    if len(data["y"])!=421983 or int(data["y"].sum())!=3724 or np.any(data["count"]!=-1):
        raise AssertionError("Rank full16 boundary failed")
    records,predictions=[],[]
    for source,path in SOURCES.items():
        if path is not None:
            data["base"]=np.load(path)
            data["mass"],data["any"],block=aggregate(data["base"],data["groups"],len(data["keys"]))
            start=data["names"].index("base_log_size")
            data["x"][:,start:start+4]=block
        margin=logit(data["base"].clip(1e-6,1-1e-6))
        for r in lock["models"]:
            path=OUT / (r["name"]+".txt")
            if digest(path)!=r["model_sha256"]:
                raise AssertionError("Rank model changed")
            model=lgb.Booster(model_str=path.read_text(encoding="utf8"))
            correction=np.clip(model.predict(data["x"],raw_score=True),-3,3)
            p=expit(margin+r["power"]*correction[data["groups"]])
            records.append({"name":r["name"],"source":source,"power":r["power"],"oof15_auc":r["oof15_auc"],"auc":rank_sum_auc(data["y"],p)})
            predictions.append(p)
    best=int(np.argmax([r["auc"] for r in records]))
    p=predictions[best]
    if verifying:
        error=float(np.max(np.abs(p-np.load(OUT / "prediction.npy"))))
        if error>1e-12:
            raise AssertionError("Context rank replay differs")
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
