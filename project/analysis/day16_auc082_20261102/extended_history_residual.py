"""Richer strictly past user dynamics for user-day residual calibration."""
import os
for _key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[_key] = "2"
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from rich_group_residual import ROOT,HERE,REFERENCE,digest,write,prepare,target_offset,prediction,lgb,array_hash,rank_sum_auc
from rich_userday_activity import load_day,correct

OUT=HERE / "runs/extended_history_residual_v1"
PARAMS={"learning_rate":.03,"min_data_in_leaf":100,"lambda_l2":10.,"feature_fraction":.8,
        "max_bin":31,"num_threads":2,"seed":42,"verbosity":-1,"deterministic":True,
        "force_col_wise":True,"boost_from_average":False}
LEAVES=[3,7,15]
POWERS=[0.,.25,.5,.75,1.]
WINDOWS=[1,2,3,5,7,30]
HALFLIVES=[1.,2.,4.,8.]


def enhanced_history(keys,sizes,counts,cutoff):
    if np.any(counts[keys[:,0]>=cutoff] != -1) or np.any(counts[keys[:,0]<cutoff]<0):
        raise ValueError("Held counts must be masked and past counts observed")
    names=[f"window{w}_{s}" for w in WINDOWS for s in ["log_n","log_k","days","active_fraction","ctr","current_vs_prior_n"]]
    names += [f"decay{h:g}_{s}" for h in HALFLIVES for s in ["log_n","log_k","days","ctr","active_fraction"]]
    names += ["previous_log_n","previous_log_k","previous_ctr","previous_active","previous_age",
              "previous2_log_n","previous2_log_k","previous2_ctr","maximum_previous_log_k","previous_k_vs_long_mean"]
    result=np.empty((len(keys),len(names)),dtype="float32")
    histories={}
    for day in np.unique(keys[:,0]):
        ordinal=pd.Timestamp(str(int(day))).toordinal()
        ids=np.flatnonzero(keys[:,0]==day)
        for i in ids:
            prior=histories.get(int(keys[i,1]),[])
            values=[]
            for window in WINDOWS:
                records=[r for r in prior if r[0]>=ordinal-window]
                n=sum(r[1] for r in records)
                k=sum(r[2] for r in records)
                active=sum(r[2]>0 for r in records)
                values.extend([np.log1p(n),np.log1p(k),len(records),(active+.1)/(len(records)+.5),
                               (k+.5)/(n+50),np.log1p(sizes[i])-np.log1p(n/max(1,len(records)))])
            for h in HALFLIVES:
                weights=[2**(-(ordinal-r[0]-1)/h) for r in prior]
                n=sum(w*r[1] for w,r in zip(weights,prior))
                k=sum(w*r[2] for w,r in zip(weights,prior))
                d=sum(weights)
                active=sum(w*(r[2]>0) for w,r in zip(weights,prior))
                values.extend([np.log1p(n),np.log1p(k),d,(k+.5)/(n+50),(active+.1)/(d+.5)])
            previous=prior[-1] if prior else (ordinal-30,0,0)
            prev2=prior[-2] if len(prior)>1 else (ordinal-30,0,0)
            values.extend([np.log1p(previous[1]),np.log1p(previous[2]),(previous[2]+.5)/(previous[1]+50),
                           float(previous[2]>0),min(30,ordinal-previous[0]),np.log1p(prev2[1]),np.log1p(prev2[2]),
                           (prev2[2]+.5)/(prev2[1]+50),np.log1p(max([r[2] for r in prior],default=0)),
                           np.log1p(previous[2])-np.log1p(sum(r[2] for r in prior)/max(1,len(prior)))])
            result[i]=values
        if day<cutoff:
            for i in ids:
                histories.setdefault(int(keys[i,1]),[]).append((ordinal,float(sizes[i]),float(counts[i])))
    return result,names


def matrix(day,source="reference",evaluation_labels=False):
    data=prepare(day,"reference" if source=="pairwise" else source,include_labels=evaluation_labels)
    all_days=load_day(16,False,evaluation_labels=False)
    history,names=enhanced_history(all_days["keys"],all_days["sizes"],all_days["counts"],20191116)
    chosen=all_days["keys"][:,0]==20191100+day
    np.testing.assert_array_equal(data["keys"],all_days["keys"][chosen])
    data["x"]=np.column_stack([data["x"],history[chosen]]).astype("float32")
    data["names"]+=names
    if source=="pairwise":
        from rich_group_residual import aggregate
        data["base"]=np.load(HERE / "runs/pairwise_group_boost_v1/prediction.npy")
        data["mass"],data["any"],block=aggregate(data["base"],data["groups"],len(data["keys"]))
        # Existing base features precede the new historical block.
        data["x"][:,-len(names)-4:-len(names)]=block
    return data


def dependencies():
    return [Path(__file__),HERE / "rich_group_residual.py",HERE / "rich_userday_activity.py",
            HERE / "fit_stacker.py",HERE / "verify_stacker.py",
            HERE / "runs/weighted_v1_day16/context.npz",HERE / "runs/weighted_v1_day16/context_manifest.json",
            HERE / "runs/weighted_calibration_transfer_v1/best_prediction.npy",
            HERE / "runs/pairwise_group_boost_v1/prediction.npy",HERE / "runs/pairwise_group_boost_v1/verification.json"]+[
                REFERENCE / f"day{d}_{suffix}.npy" for d in [15,16] for suffix in ["indices","prediction"]]


def main():
    if OUT.exists():
        raise FileExistsError("Preserving extended history run")
    OUT.mkdir(parents=True)
    protocol={"params":PARAMS,"leaves":LEAVES,"powers":POWERS,"windows":WINDOWS,"halflives":HALFLIVES,
              "objectives":["binary","poisson"],"max_rounds":400,"patience":50,
              "inputs_sha256":{str(p):digest(p) for p in dependencies()},
              "raw_datasets_sha256":{p.name:digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
              "fit_date":20191115,"evaluation_date":20191116,"evaluation16_labels_fit":False,
              "selection":"User3fold day15 chooses median rounds and correction power; all6 models locked before16",
              "development16_model_selection":True}
    write(OUT / "protocol.json",protocol)
    data=matrix(15,evaluation_labels=True)
    folds=((data["keys"][:,1].astype("uint64")*2654435761)%3).astype("int32")
    records,locks=[],[]
    for objective in protocol["objectives"]:
        target,offset=target_offset(data,objective)
        params=dict(PARAMS,objective=objective,metric="binary_logloss" if objective=="binary" else "poisson")
        for leaves in LEAVES:
            oof,rounds=np.empty(len(target)),[]
            for fold in range(3):
                tr,va=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
                train=lgb.Dataset(data["x"][tr],label=target[tr],init_score=offset[tr])
                valid=lgb.Dataset(data["x"][va],label=target[va],init_score=offset[va],reference=train)
                model=lgb.train(dict(params,num_leaves=leaves),train,num_boost_round=400,valid_sets=[valid],
                                callbacks=[lgb.early_stopping(50,verbose=False)])
                oof[va]=prediction(model,data,objective,va)
                rounds.append(model.best_iteration)
            scores=[{"power":p,"auc":rank_sum_auc(data["y"],correct(data["base"],oof,data["groups"],objective,p))} for p in POWERS]
            best=max(scores,key=lambda r:r["auc"])
            record={"name":f"{objective}_{leaves}","objective":objective,"leaves":leaves,"rounds":rounds,
                    "refit_rounds":max(1,int(np.median(rounds))),"power":best["power"],"oof15_auc":best["auc"],"scores":scores}
            records.append(record)
            np.save(OUT / (record["name"]+"_oof.npy"),oof)
            write(OUT / "cv.json",records)
            print(json.dumps(record),flush=True)
    for record in records:
        target,offset=target_offset(data,record["objective"])
        params=dict(PARAMS,objective=record["objective"],num_leaves=record["leaves"])
        model=lgb.train(params,lgb.Dataset(data["x"],label=target,init_score=offset),num_boost_round=record["refit_rounds"])
        path=OUT / (record["name"]+".txt")
        path.write_text(model.model_to_string(),encoding="utf8")
        locks.append(dict(record,model_sha256=digest(path)))
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
        raise AssertionError("Protocol changed")
    for path,expected in protocol["inputs_sha256"].items():
        if digest(path)!=expected:
            raise AssertionError("Input/source changed")
    for name,expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name)!=expected:
            raise AssertionError("Original data changed")
    if verifying:
        train=matrix(15)
        for field,key in [("x","training_x_sha256"),("count","training_counts_sha256"),("keys","training_keys_sha256")]:
            if array_hash(train[field])!=lock[key]:
                raise AssertionError("Training reconstruction differs")
        del train
    records,predictions=[],[]
    for source in ["reference","weighted_transfer","pairwise"]:
        data=matrix(16,source,evaluation_labels=True)
        if len(data["y"])!=421983 or int(data["y"].sum())!=3724 or np.any(data["count"]!=-1):
            raise AssertionError("Held cohort changed")
        for record in lock["models"]:
            path=OUT / (record["name"]+".txt")
            if digest(path)!=record["model_sha256"]:
                raise AssertionError("Model changed")
            model=lgb.Booster(model_str=path.read_text(encoding="utf8"))
            group_prediction=prediction(model,data,record["objective"])
            p=correct(data["base"],group_prediction,data["groups"],record["objective"],record["power"])
            records.append({"name":record["name"],"source":source,"power":record["power"],"oof15_auc":record["oof15_auc"],"auc":rank_sum_auc(data["y"],p)})
            predictions.append(p)
    best=int(np.argmax([r["auc"] for r in records]))
    p=predictions[best]
    if verifying:
        error=float(np.max(np.abs(p-np.load(OUT / "prediction.npy"))))
        if error>1e-12:
            raise AssertionError("Prediction replay differs")
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
