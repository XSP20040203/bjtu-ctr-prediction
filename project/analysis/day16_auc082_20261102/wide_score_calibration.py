"""Expand calibration to aligned row predictors and independent any-click heads."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from fit_stacker import ROOT, HERE, REFERENCE, fit, digest, write
from group_score_calibration import FIELDS, group_features
from verify_stacker import rank_sum_auc

OUT = HERE/"runs/wide_score_calibration_v1"
RIDGES = [.001, .01, .1]
KINDS = ["rows", "auxiliary", "joint"]


def auxiliary_correction(reference, auxiliary, users):
    if len(reference) != len(auxiliary) or len(reference) != len(users):
        raise ValueError("Auxiliary alignment")
    codes,_ = pd.factorize(users,sort=True)
    p = np.asarray(reference).clip(1e-6,1-1e-6)
    q = np.asarray(auxiliary).clip(1e-6,1-1e-6)
    mass = np.bincount(codes,weights=-np.log1p(-p))
    return np.log(-np.log1p(-q))-np.log(mass[codes])


def select_sources(inventory):
    selected = []
    for row in inventory["records"]:
        if not row["alignment_verified"]:
            continue
        path = row["path15"]
        if "any_prediction" in path and ("/base/" in path or "/context/" in path):
            continue  # These heads were never trained with an auxiliary objective.
        selected.append(dict(row,kind="auxiliary" if "any_prediction" in path else "rows"))
    return selected


def build(raw,day,sources):
    rows = np.flatnonzero(raw.pt_d.to_numpy()==20191100+day)
    np.testing.assert_array_equal(rows,np.load(REFERENCE/f"day{day}_indices.npy"))
    p0 = np.load(REFERENCE/f"day{day}_prediction.npy")
    z0 = logit(p0.clip(1e-6,1-1e-6))
    context,context_names = group_features(raw.iloc[rows][FIELDS],p0)
    columns,names,kinds = [context],list(context_names),["context"]*context.shape[1]
    users = raw.user_id.to_numpy()[rows]
    for source in sources:
        path = ROOT/source[f"path{day}"]
        if digest(path) != source["sha256"][0 if day==15 else 1]:
            raise AssertionError("Prediction source changed")
        index_path = source["indices"][0 if day==15 else 1]
        np.testing.assert_array_equal(rows,np.load(index_path))
        p = np.load(path)
        column = (auxiliary_correction(p0,p,users) if source["kind"]=="auxiliary"
                  else logit(p.clip(1e-6,1-1e-6))-z0)
        columns.append(column[:,None])
        names.append(source["path15"])
        kinds.append(source["kind"])
    return np.column_stack(columns),z0,rows,p0,users,names,kinds


def design(features,base,center,scale):
    return np.column_stack([np.ones(len(base)),base,np.clip((features-center)/scale,-8,8)])


def main():
    if OUT.exists():
        raise FileExistsError("Preserving expanded score calibration")
    inventory_path = HERE/"paired_prediction_inventory.json"
    sources = select_sources(json.loads(inventory_path.read_text(encoding="utf8")))
    if not sources:
        raise AssertionError("No aligned source predictions")
    OUT.mkdir(parents=True)
    protocol = {"fit_date":20191115,"evaluation_date":20191116,"ridges":RIDGES,"kinds":KINDS,
                "sources":sources,"inventory_sha256":digest(inventory_path),
                "source_sha256":{str(p):digest(p) for p in [Path(__file__),HERE/"fit_stacker.py",HERE/"group_score_calibration.py"]},
                "reference_sha256":{str(REFERENCE/f"day{d}_prediction.npy"):digest(REFERENCE/f"day{d}_prediction.npy") for d in [15,16]},
                "raw_datasets_sha256":{p.name:digest(p) for p in (ROOT/"2026ml").glob("*.csv")},
                "selection":"3 user folds on15 only; maximize pooled OOF AUC across scopes and ridges",
                "auxiliary_feature":"log(-log(1-q_any)) - log(sum_user(-log(1-p_reference)))",
                "evaluation16_labels_fit":False,"development_day_reused":True}
    write(OUT/"protocol.json",protocol)
    raw = pd.read_csv(ROOT/"2026ml/train.csv",usecols=FIELDS+["pt_d"],dtype="int32")
    values,base,rows,p0,users,names,kinds = build(raw,15,sources)
    labels = pd.read_csv(ROOT/"2026ml/train.csv",usecols=["label","pt_d"],dtype="int32")
    y = labels.label.to_numpy()[rows].copy()
    del labels
    folds = ((users.astype("uint64")*2654435761)%3).astype(int)
    records = []
    for scope in KINDS:
        active = [i for i,k in enumerate(kinds) if k=="context" or scope=="joint" or k==scope]
        for ridge in RIDGES:
            oof = np.empty(len(y))
            for fold in range(3):
                tr,va = folds!=fold,folds==fold
                center,scale = values[tr].mean(0),values[tr].std(0).clip(.05)
                x = design(values,base,center,scale)
                beta,info = fit(x[tr],y[tr],ridge,active)
                if not info["success"]:
                    raise AssertionError("CV solver did not converge")
                oof[va] = expit(x[va]@beta)
            record = {"scope":scope,"ridge":ridge,"oof_auc":rank_sum_auc(y,oof)}
            records.append(record)
            write(OUT/"cv.json",records)
            print(json.dumps(record),flush=True)
    selected = max(records,key=lambda r:r["oof_auc"])
    active = [i for i,k in enumerate(kinds) if k=="context" or selected["scope"]=="joint" or k==selected["scope"]]
    center,scale = values.mean(0),values.std(0).clip(.05)
    beta,info = fit(design(values,base,center,scale),y,selected["ridge"],active)
    if not info["success"]:
        raise AssertionError("Final solver did not converge")
    np.savez(OUT/"model.npz",center=center,scale=scale,beta=beta)
    write(OUT/"fit15.json",{"selected":selected,"records":records,"baseline_auc":rank_sum_auc(y,p0),
          "feature_names":names,"feature_kinds":kinds,"fit":info,"source_count":len(sources)})
    write(OUT/"weights_locked.json",{"model_sha256":digest(OUT/"model.npz"),
          "protocol_sha256":digest(OUT/"protocol.json"),"evaluation16_labels_fit":False})
    values16,base16,rows16,_,_,_,_ = build(raw,16,sources)
    p = expit(design(values16,base16,center,scale)@beta)
    np.save(OUT/"prediction.npy",p)
    np.save(OUT/"valid_indices.npy",rows16)
    y16 = pd.read_csv(ROOT/"2026ml/train.csv",usecols=["label"],dtype="int32").label.to_numpy()[rows16]
    result = {"auc":rank_sum_auc(y16,p),"rows":len(y16),"clicks":int(y16.sum()),"target_met":rank_sum_auc(y16,p)>=.82}
    write(OUT/"result.json",result)
    print(json.dumps(result,indent=2),flush=True)


def verify():
    protocol = json.loads((OUT/"protocol.json").read_text(encoding="utf8"))
    lock = json.loads((OUT/"weights_locked.json").read_text(encoding="utf8"))
    for key in ["source_sha256","reference_sha256"]:
        for path,expected in protocol[key].items():
            if digest(path)!=expected:
                raise AssertionError("Frozen source changed")
    for name,expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT/"2026ml"/name)!=expected:
            raise AssertionError("Original CSV changed")
    if digest(OUT/"model.npz")!=lock["model_sha256"] or digest(OUT/"protocol.json")!=lock["protocol_sha256"]:
        raise AssertionError("Frozen model/protocol changed")
    raw = pd.read_csv(ROOT/"2026ml/train.csv",usecols=FIELDS+["pt_d"],dtype="int32")
    values,base,rows,_,_,_,_ = build(raw,16,protocol["sources"])
    np.testing.assert_array_equal(rows,np.load(OUT/"valid_indices.npy"))
    model = np.load(OUT/"model.npz")
    p = expit(design(values,base,model["center"],model["scale"])@model["beta"])
    error = float(np.max(np.abs(p-np.load(OUT/"prediction.npy"))))
    if error>1e-12:
        raise AssertionError("Reloaded model differs")
    y = pd.read_csv(ROOT/"2026ml/train.csv",usecols=["label"],dtype="int32").label.to_numpy()[rows]
    if len(y)!=421983 or int(y.sum())!=3724:
        raise AssertionError("Full16 cohort changed")
    result = {"status":"passed","auc":rank_sum_auc(y,p),"rows":len(y),"clicks":int(y.sum()),
              "maximum_reload_error":error,"source_inputs_hashed_and_aligned":True,
              "underlying_models_replayed_in_this_verification":False,
              "original_datasets_unchanged":True,"evaluation16_labels_fit":False,"target_met":rank_sum_auc(y,p)>=.82}
    write(OUT/"verification.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify",action="store_true")
    verify() if parser.parse_args().verify else main()
