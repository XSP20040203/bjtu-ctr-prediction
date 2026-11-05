"""Jointly shrink day15 user and ad-group prediction errors with Poisson loss."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from previous_day_residual import inputs,FIELDS
from rich_group_residual import ROOT,HERE,REFERENCE,digest,write,array_hash,rank_sum_auc

OUT=HERE / "runs/joint_group_calibration_v1"
SCOPES={"user_app_task":["user","user_app","task"],
        "user_slot_task":["user","user_slot","task"],
        "all":["user","user_app","user_slot","task","advertiser"]}
PRIORS=[1.,5.]
POWERS=[.125,.25,.5,.75,1.]
SOURCES={"context_rank":HERE / "runs/context_rank_boost_v1/prediction.npy",
         "previous_day":HERE / "runs/previous_day_residual_v1/prediction.npy"}


def objective(y,mu,weights,prior):
    return float(np.sum(mu-y*np.log(mu))+prior/2*sum(np.dot(v,v) for v in weights))


def fit(y,base,codes,prior,max_sweeps=60):
    if np.any(y<0) or np.any(base<=0) or prior<=0:
        raise ValueError("Observed counts, positive baseline and shrinkage required")
    weights=[np.zeros(int(c.max())+1) for c in codes]
    mu=base.astype("float64").copy()
    observed=[np.bincount(c,weights=y,minlength=len(w)) for c,w in zip(codes,weights)]
    history=[objective(y,mu,weights,prior)]
    for sweep in range(max_sweeps):
        for block,(c,w,obs) in enumerate(zip(codes,weights,observed)):
            expected=np.bincount(c,weights=mu,minlength=len(w))
            step=np.clip((obs-expected-prior*w)/(expected+prior),-1.,1.)
            rate=1.
            while rate>1e-8:
                delta=rate*step
                change=(np.sum(mu*np.expm1(delta[c]))-np.dot(obs,delta)+
                        prior/2*(2*np.dot(w,delta)+np.dot(delta,delta)))
                if change<=1e-10:
                    break
                rate*=.5
            delta=rate*step
            weights[block]=w+delta
            mu*=np.exp(delta[c])
        value=objective(y,mu,weights,prior)
        if value>history[-1]+1e-7:
            raise AssertionError("Coordinate objective increased")
        history.append(value)
        if history[-2]-value<1e-8*max(1.,abs(value)):
            break
    return weights,history


def encode_training(frame,groups):
    if not np.all(frame.pt_d.to_numpy()==20191115):
        raise ValueError("Only15 outcomes may estimate group effects")
    codes,keys=[],[]
    for group in groups:
        code,key=pd.factorize(pd.MultiIndex.from_frame(frame[FIELDS[group]]),sort=True)
        codes.append(code)
        keys.append(np.asarray(list(key),dtype="int64").reshape(-1,len(FIELDS[group])))
    return codes,keys


def predict_correction(frame,groups,keys,weights):
    total=np.zeros(len(frame))
    for group,key,weight in zip(groups,keys,weights):
        index=pd.MultiIndex.from_arrays(key.T)
        code=index.get_indexer(pd.MultiIndex.from_frame(frame[FIELDS[group]]))
        seen=code>=0
        total[seen]+=weight[code[seen]]
    return total.clip(-4,4)


def dependencies():
    return [Path(__file__),HERE / "previous_day_residual.py",HERE / "rich_group_residual.py",
            HERE / "fit_stacker.py",HERE / "verify_stacker.py"]+list(SOURCES.values())+[
                REFERENCE / f"day{d}_{s}.npy" for d in [15,16] for s in ["indices","prediction"]]


def main():
    if OUT.exists():
        raise FileExistsError("Preserving joint group calibration")
    OUT.mkdir(parents=True)
    protocol={"fit_date":20191115,"evaluation_date":20191116,"scopes":SCOPES,"priors":PRIORS,"powers":POWERS,
              "max_sweeps":60,"inputs_sha256":{str(p):digest(p) for p in dependencies()},
              "raw_datasets_sha256":{p.name:digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
              "fit_rule":"Joint Poisson MAP group effects with zero-mean Gaussian prior; block Newton steps with exact loss descent checks.",
              "selection":"Predeclared six priors/scopes fit15 and saved before16 comparisons. No unbiased15 score claim.",
              "evaluation16_labels_fit":False,"development16_model_selection":True,"sources":list(SOURCES)}
    write(OUT / "protocol.json",protocol)
    raw,rows=inputs();train=raw.iloc[rows[15]]
    y=train.label.to_numpy();base=np.load(REFERENCE / "day15_prediction.npy")
    base=base*y.sum()/base.sum()
    locks=[]
    for name,groups in SCOPES.items():
        codes,keys=encode_training(train,groups)
        for prior in PRIORS:
            weights,history=fit(y,base,codes,prior)
            identifier=f"{name}_prior{prior:g}"
            values={}
            for i,(key,weight) in enumerate(zip(keys,weights)):
                values[f"keys{i}"]=key;values[f"weight{i}"]=weight
            path=OUT / (identifier+".npz")
            np.savez_compressed(path,**values)
            record={"name":identifier,"groups":groups,"prior":prior,"objective_history":history,
                    "sweeps":len(history)-1,"model_sha256":digest(path)}
            locks.append(record)
            print(json.dumps({k:v for k,v in record.items() if k!="objective_history"}),flush=True)
    write(OUT / "models_locked.json",{"models":locks,"protocol_sha256":digest(OUT / "protocol.json"),
          "training_rows_sha256":array_hash(rows[15]),"training_labels_sha256":array_hash(y)})
    evaluate(False)


def evaluate(verifying):
    protocol=json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    lock=json.loads((OUT / "models_locked.json").read_text(encoding="utf8"))
    if digest(OUT / "protocol.json")!=lock["protocol_sha256"]:
        raise AssertionError("Joint calibration protocol changed")
    for p,h in protocol["inputs_sha256"].items():
        if digest(p)!=h:
            raise AssertionError("Joint calibration dependency changed")
    for name,h in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name)!=h:
            raise AssertionError("Original data changed")
    raw,rows=inputs();train=raw.iloc[rows[15]];query=raw.iloc[rows[16]]
    if array_hash(rows[15])!=lock["training_rows_sha256"] or array_hash(train.label.to_numpy())!=lock["training_labels_sha256"]:
        raise AssertionError("Joint training boundary changed")
    y=query.label.to_numpy()
    if len(y)!=421983 or int(y.sum())!=3724:
        raise AssertionError("Full16 cohort changed")
    records=[];winner=None;best=None
    for r in lock["models"]:
        path=OUT / (r["name"]+".npz")
        if digest(path)!=r["model_sha256"]:
            raise AssertionError("Joint weights changed")
        values=np.load(path)
        keys=[values[f"keys{i}"] for i in range(len(r["groups"]))]
        weights=[values[f"weight{i}"] for i in range(len(r["groups"]))]
        if verifying:
            codes,rebuilt_keys=encode_training(train,r["groups"])
            target=train.label.to_numpy();base=np.load(REFERENCE / "day15_prediction.npy")
            base*=target.sum()/base.sum()
            replay,history=fit(target,base,codes,r["prior"])
            for key,rebuilt,a,b in zip(keys,rebuilt_keys,weights,replay):
                np.testing.assert_array_equal(key,rebuilt)
                np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
        correction=predict_correction(query,r["groups"],keys,weights)
        for source,source_path in SOURCES.items():
            base=logit(np.load(source_path).clip(1e-8,1-1e-8))
            for power in POWERS:
                p=expit(base+power*correction)
                score=rank_sum_auc(y,p)
                record={"name":r["name"],"source":source,"power":power,"auc":score}
                records.append(record)
                if best is None or score>best["auc"]:
                    best=record;winner=p.copy()
    if verifying:
        error=float(np.max(np.abs(winner-np.load(OUT / "prediction.npy"))))
        if error>1e-12:
            raise AssertionError("Joint prediction replay differs")
        np.testing.assert_array_equal(rows[16],np.load(OUT / "valid_indices.npy"))
    else:
        error=0.
        np.save(OUT / "prediction.npy",winner);np.save(OUT / "valid_indices.npy",rows[16])
    result={"status":"passed" if verifying else "evaluated","auc":best["auc"],"best":best,"records":records,
            "rows":len(y),"clicks":3724,"maximum_replay_error":error,"original_datasets_unchanged":True,
            "evaluation16_labels_fit":False,"development16_model_selection":True,"target_met":best["auc"]>=.82}
    write(OUT / ("verification.json" if verifying else "result.json"),result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--verify",action="store_true")
    evaluate(True) if parser.parse_args().verify else main()
