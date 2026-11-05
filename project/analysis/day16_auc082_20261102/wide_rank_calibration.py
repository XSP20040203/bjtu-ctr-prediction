"""Pairwise ranking on aligned model scores, fitted only with day15 outcomes."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from wide_score_calibration import ROOT,HERE,FIELDS,build,digest,write,rank_sum_auc
from rank_calibration import fit as rank_fit,predict as rank_predict,CONFIGS,PAIR_COUNT

OUT = HERE/"runs/wide_rank_calibration_v1"
SOURCE = HERE/"runs/wide_score_calibration_v1"
SCOPES = ["auxiliary","joint"]


def indices(kinds,scope):
    return np.array([i for i,k in enumerate(kinds) if k=="context" or k==scope or scope=="joint"],dtype="int64")


def main():
    if OUT.exists(): raise FileExistsError("Preserving wide ranking calibration")
    source = json.loads((SOURCE/"protocol.json").read_text(encoding="utf8"))
    OUT.mkdir(parents=True)
    protocol = dict(source,selection="3-fold user CV on15 chooses temperature/ridge per scope; both locked scope models then evaluated on reused16",
                    scopes=SCOPES,rank_configs=CONFIGS,pair_count=PAIR_COUNT,
                    implementation_sha256=digest(__file__),rank_implementation_sha256=digest(HERE/"rank_calibration.py"))
    write(OUT/"protocol.json",protocol)
    raw = pd.read_csv(ROOT/"2026ml/train.csv",usecols=FIELDS+["pt_d"],dtype="int32")
    x,base,rows,p0,users,names,kinds = build(raw,15,source["sources"])
    y = pd.read_csv(ROOT/"2026ml/train.csv",usecols=["label"],dtype="int32").label.to_numpy()[rows]
    folds = ((users.astype("uint64")*2654435761)%3).astype(int)
    records,stored,chosen = [],{},[]
    for scope in SCOPES:
        active = indices(kinds,scope)
        values = x[:,active]
        scoped = []
        for temperature,ridge in CONFIGS:
            oof = np.empty(len(y))
            for fold in range(3):
                tr,va = folds!=fold,folds==fold
                model,info = rank_fit(values[tr],base[tr],y[tr],temperature,ridge,6160+fold)
                oof[va] = rank_predict(values[va],base[va],model)
            record = {"scope":scope,"temperature":temperature,"ridge":ridge,"oof_auc":rank_sum_auc(y,oof)}
            records.append(record)
            scoped.append(record)
            write(OUT/"cv.json",records)
            print(json.dumps(record),flush=True)
        selected = max(scoped,key=lambda r:r["oof_auc"])
        model,info = rank_fit(values,base,y,selected["temperature"],selected["ridge"],6160)
        for name,value in model.items(): stored[f"{scope}_{name}"]=value
        stored[f"{scope}_indices"] = active
        chosen.append(dict(selected,fit=info))
    np.savez(OUT/"models.npz",**stored)
    write(OUT/"weights_locked.json",{"model_sha256":digest(OUT/"models.npz"),"protocol_sha256":digest(OUT/"protocol.json"),
          "selected_per_scope":chosen,"primary_scope":max(chosen,key=lambda r:r["oof_auc"])["scope"],
          "baseline15_auc":rank_sum_auc(y,p0),"feature_names":names,"evaluation16_coefficients_fit":False})
    del x,values,y
    x,base,rows,_,_,_,_ = build(raw,16,source["sources"])
    predictions = {}
    for scope in SCOPES:
        model = {name:stored[f"{scope}_{name}"] for name in ["center","scale","beta"]}
        p = rank_predict(x[:,stored[f"{scope}_indices"]],base,model)
        predictions[scope]=p
        np.save(OUT/f"{scope}_prediction.npy",p)
    np.save(OUT/"valid_indices.npy",rows)
    y = pd.read_csv(ROOT/"2026ml/train.csv",usecols=["label"],dtype="int32").label.to_numpy()[rows]
    results = [{"scope":scope,"auc":rank_sum_auc(y,p)} for scope,p in predictions.items()]
    best = max(results,key=lambda r:r["auc"])
    np.save(OUT/"prediction.npy",predictions[best["scope"]])
    result = {"auc":best["auc"],"best_scope":best["scope"],"results":results,"rows":len(y),"clicks":int(y.sum()),
              "target_met":best["auc"]>=.82,"development16_used_for_scope_comparison":True}
    write(OUT/"result.json",result)
    print(json.dumps(result,indent=2),flush=True)


def verify():
    protocol = json.loads((OUT/"protocol.json").read_text(encoding="utf8"))
    lock = json.loads((OUT/"weights_locked.json").read_text(encoding="utf8"))
    if digest(__file__)!=protocol["implementation_sha256"] or digest(HERE/"rank_calibration.py")!=protocol["rank_implementation_sha256"]:
        raise AssertionError("Ranking implementation changed")
    for group in ["source_sha256","reference_sha256"]:
        for path,expected in protocol[group].items():
            if digest(path)!=expected: raise AssertionError("Source changed")
    for name,expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT/"2026ml"/name)!=expected: raise AssertionError("Original data changed")
    if digest(OUT/"models.npz")!=lock["model_sha256"] or digest(OUT/"protocol.json")!=lock["protocol_sha256"]:
        raise AssertionError("Frozen models/protocol changed")
    raw = pd.read_csv(ROOT/"2026ml/train.csv",usecols=FIELDS+["pt_d"],dtype="int32")
    x,base,rows,_,_,names,_ = build(raw,16,protocol["sources"])
    np.testing.assert_array_equal(rows,np.load(OUT/"valid_indices.npy"))
    if names!=lock["feature_names"]: raise AssertionError("Feature schema changed")
    stored = np.load(OUT/"models.npz")
    y = pd.read_csv(ROOT/"2026ml/train.csv",usecols=["label"],dtype="int32").label.to_numpy()[rows]
    if len(y)!=421983 or int(y.sum())!=3724: raise AssertionError("Complete16 cohort changed")
    records,predictions = [],{}
    for scope in SCOPES:
        model = {name:stored[f"{scope}_{name}"] for name in ["center","scale","beta"]}
        p = rank_predict(x[:,stored[f"{scope}_indices"]],base,model)
        error = float(np.max(np.abs(p-np.load(OUT/f"{scope}_prediction.npy"))))
        if error>1e-12: raise AssertionError("Model replay differs")
        records.append({"scope":scope,"auc":rank_sum_auc(y,p),"maximum_replay_error":error})
        predictions[scope]=p
    best = max(records,key=lambda r:r["auc"])
    if not np.allclose(predictions[best["scope"]],np.load(OUT/"prediction.npy"),atol=1e-12,rtol=0):
        raise AssertionError("Best prediction differs")
    report = {"status":"passed","auc":best["auc"],"results":records,"rows":len(y),"clicks":int(y.sum()),
              "original_datasets_unchanged":True,"evaluation16_coefficients_fit":False,
              "development16_used_for_scope_comparison":True,"target_met":best["auc"]>=.82}
    write(OUT/"verification.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify",action="store_true")
    verify() if parser.parse_args().verify else main()
