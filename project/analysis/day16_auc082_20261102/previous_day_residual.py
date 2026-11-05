"""Empirical Bayes transfer of day15 observed-versus-expected group errors."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from rich_group_residual import ROOT,HERE,REFERENCE,digest,write,array_hash,rank_sum_auc

OUT=HERE / "runs/previous_day_residual_v1"
FIELDS={"user":["user_id"],"task":["task_id"],"user_app":["user_id","spread_app_id"],
        "user_slot":["user_id","slot_id"],"advertiser":["adv_prim_id"]}
ALPHAS=[1.,3.,10.]
POWERS=[.125,.25,.5]
SOURCES={"context_rank":HERE / "runs/context_rank_boost_v1/prediction.npy",
         "neural_userday":HERE / "runs/neural_userday_v1/prediction.npy",
         "personal_context":HERE / "runs/personal_context_residual_v1/prediction.npy"}


def posterior_error(count,mass,alpha):
    if alpha<=0 or np.any(count<0) or np.any(mass<0):
        raise ValueError("Observed nonnegative counts, expectation and positive prior required")
    return np.log((count+alpha)/(mass+alpha)).clip(-3,3)


def inputs():
    raw=pd.read_csv(ROOT / "2026ml/train.csv",usecols=["pt_d","label","user_id","task_id","spread_app_id","slot_id","adv_prim_id"],dtype="int32")
    rows={d:np.flatnonzero(raw.pt_d.to_numpy()==20191100+d) for d in [15,16]}
    for d in rows:
        np.testing.assert_array_equal(rows[d],np.load(REFERENCE / f"day{d}_indices.npy"))
    return raw,rows


def make_statistics(train,base):
    if not np.all(train.pt_d.to_numpy()==20191115):
        raise ValueError("Only15 labels may estimate residuals")
    normalized=base*train.label.sum()/base.sum()
    result={}
    for name,fields in FIELDS.items():
        index=pd.MultiIndex.from_frame(train[fields])
        codes,keys=pd.factorize(index,sort=True)
        result[name+"_keys"]=np.asarray(list(keys),dtype="int64").reshape(-1,len(fields))
        result[name+"_count"]=np.bincount(codes,weights=train.label.to_numpy(),minlength=len(keys))
        result[name+"_mass"]=np.bincount(codes,weights=normalized,minlength=len(keys))
    return result


def apply_model(frame,statistics,name,alpha):
    fields=FIELDS[name]
    keys=statistics[name+"_keys"]
    index=pd.MultiIndex.from_arrays(keys.T)
    position=index.get_indexer(pd.MultiIndex.from_frame(frame[fields]))
    value=np.zeros(len(frame))
    seen=position>=0
    correction=posterior_error(statistics[name+"_count"],statistics[name+"_mass"],alpha)
    value[seen]=correction[position[seen]]
    return value


def dependencies():
    return [Path(__file__),HERE / "rich_group_residual.py",HERE / "fit_stacker.py",HERE / "verify_stacker.py"]+list(SOURCES.values())+[
        REFERENCE / f"day{d}_{s}.npy" for d in [15,16] for s in ["indices","prediction"]]


def main():
    if OUT.exists():
        raise FileExistsError("Preserving previous-day error model")
    OUT.mkdir(parents=True)
    protocol={"fit_date":20191115,"evaluation_date":20191116,"fields":FIELDS,"alphas":ALPHAS,"powers":POWERS,
              "sources":list(SOURCES),"inputs_sha256":{str(p):digest(p) for p in dependencies()},
              "raw_datasets_sha256":{p.name:digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
              "recipe":"Day15 counts / calibrated expected counts, Gamma-Poisson shrinkage with fixed alpha. Apply same group residual to16 base logit.",
              "evaluation16_labels_fit":False,"development16_model_selection":True,
              "selection":"All fixed group/prior/power candidates declared before16 evaluation;16 reused for selection; no unbiased15 AUC claimed."}
    write(OUT / "protocol.json",protocol)
    raw,rows=inputs()
    train=raw.iloc[rows[15]]
    statistics=make_statistics(train,np.load(REFERENCE / "day15_prediction.npy"))
    np.savez_compressed(OUT / "statistics.npz",**statistics)
    write(OUT / "model_locked.json",{"protocol_sha256":digest(OUT / "protocol.json"),
          "model_sha256":digest(OUT / "statistics.npz"),"training_labels_sha256":array_hash(train.label.to_numpy()),
          "training_rows_sha256":array_hash(rows[15]),"fit_rows":len(train),"fit_positive":int(train.label.sum())})
    evaluate(False)


def evaluate(verifying):
    protocol=json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    lock=json.loads((OUT / "model_locked.json").read_text(encoding="utf8"))
    if digest(OUT / "protocol.json")!=lock["protocol_sha256"] or digest(OUT / "statistics.npz")!=lock["model_sha256"]:
        raise AssertionError("Previous-day model changed")
    for p,h in protocol["inputs_sha256"].items():
        if digest(p)!=h:
            raise AssertionError("Previous-day dependency changed")
    for name,h in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name)!=h:
            raise AssertionError("Original data changed")
    raw,rows=inputs()
    stats=dict(np.load(OUT / "statistics.npz"))
    if verifying:
        rebuilt=make_statistics(raw.iloc[rows[15]],np.load(REFERENCE / "day15_prediction.npy"))
        for key in stats:
            np.testing.assert_array_equal(stats[key],rebuilt[key])
        if array_hash(raw.label.to_numpy()[rows[15]])!=lock["training_labels_sha256"] or array_hash(rows[15])!=lock["training_rows_sha256"]:
            raise AssertionError("Training boundary changed")
    frame=raw.iloc[rows[16]]
    y=frame.label.to_numpy()
    if len(y)!=421983 or int(y.sum())!=3724:
        raise AssertionError("Full16 cohort changed")
    records=[]
    best_score=-1.
    winner=None
    for source,path in SOURCES.items():
        base=logit(np.load(path).clip(1e-8,1-1e-8))
        for group in FIELDS:
            for alpha in ALPHAS:
                correction=apply_model(frame,stats,group,alpha)
                for power in POWERS:
                    p=expit(base+power*correction)
                    score=rank_sum_auc(y,p)
                    r={"source":source,"group":group,"alpha":alpha,"power":power,"auc":score}
                    records.append(r)
                    if score>best_score:
                        best_score=score;winner=p.copy();best=r
    if verifying:
        error=float(np.max(np.abs(winner-np.load(OUT / "prediction.npy"))))
        if error>1e-12:
            raise AssertionError("Previous-day replay differs")
        np.testing.assert_array_equal(rows[16],np.load(OUT / "valid_indices.npy"))
    else:
        error=0.
        np.save(OUT / "prediction.npy",winner);np.save(OUT / "valid_indices.npy",rows[16])
    result={"status":"passed" if verifying else "evaluated","auc":best_score,"best":best,"records":records,
            "rows":len(y),"clicks":3724,"maximum_replay_error":error,"original_datasets_unchanged":True,
            "evaluation16_labels_fit":False,"development16_model_selection":True,"target_met":best_score>=.82}
    write(OUT / ("verification.json" if verifying else "result.json"),result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--verify",action="store_true")
    evaluate(True) if parser.parse_args().verify else main()
