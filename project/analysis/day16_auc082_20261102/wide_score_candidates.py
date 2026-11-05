"""Fit every predeclared15-day scope/ridge, lock them, then compare on dev16."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from wide_score_calibration import (ROOT,HERE,REFERENCE,OUT as SOURCE,FIELDS,RIDGES,KINDS,
                                     build,design,fit,digest,write,rank_sum_auc)

OUT = HERE/"runs/wide_score_candidates_v1"


def main():
    if OUT.exists():
        raise FileExistsError("Preserving candidates")
    source = json.loads((SOURCE/"protocol.json").read_text(encoding="utf8"))
    OUT.mkdir(parents=True)
    protocol = dict(source,source_run=str(SOURCE),source_run_protocol_sha256=digest(SOURCE/"protocol.json"),
                    selection="All nine originally declared scopes/ridges fit15 and locked before comparing16; exploratory reused-development selection",
                    implementation_sha256=digest(__file__),evaluation16_coefficients_fit=False)
    write(OUT/"protocol.json",protocol)
    raw = pd.read_csv(ROOT/"2026ml/train.csv",usecols=FIELDS+["pt_d"],dtype="int32")
    values,base,rows,_,_,names,kinds = build(raw,15,source["sources"])
    y = pd.read_csv(ROOT/"2026ml/train.csv",usecols=["label"],dtype="int32").label.to_numpy()[rows]
    center,scale = values.mean(0),values.std(0).clip(.05)
    x = design(values,base,center,scale)
    betas,records = [],[]
    for scope in KINDS:
        active = [i for i,k in enumerate(kinds) if k=="context" or scope=="joint" or k==scope]
        for ridge in RIDGES:
            beta,info = fit(x,y,ridge,active)
            if not info["success"]:
                raise AssertionError("Solver failed")
            betas.append(beta)
            records.append({"scope":scope,"ridge":ridge,"fit":info})
    np.savez(OUT/"models.npz",center=center,scale=scale,betas=np.column_stack(betas))
    write(OUT/"weights_locked.json",{"model_sha256":digest(OUT/"models.npz"),
          "protocol_sha256":digest(OUT/"protocol.json"),"records":records,"feature_names":names,
          "evaluation16_coefficients_fit":False})
    del values,x,y
    values,base,rows,_,_,_,_ = build(raw,16,source["sources"])
    ps = expit(design(values,base,center,scale)@np.column_stack(betas))
    np.save(OUT/"all_predictions.npy",ps)
    np.save(OUT/"valid_indices.npy",rows)
    y = pd.read_csv(ROOT/"2026ml/train.csv",usecols=["label"],dtype="int32").label.to_numpy()[rows]
    scored = [dict(record,auc=rank_sum_auc(y,ps[:,i])) for i,record in enumerate(records)]
    best = int(np.argmax([r["auc"] for r in scored]))
    np.save(OUT/"prediction.npy",ps[:,best])
    result = {"auc":scored[best]["auc"],"best_column":best,"records":scored,"rows":len(y),"clicks":int(y.sum()),
              "target_met":scored[best]["auc"]>=.82,"development16_used_for_model_selection":True}
    write(OUT/"result.json",result)
    print(json.dumps(result,indent=2),flush=True)


def verify():
    protocol = json.loads((OUT/"protocol.json").read_text(encoding="utf8"))
    lock = json.loads((OUT/"weights_locked.json").read_text(encoding="utf8"))
    if digest(__file__)!=protocol["implementation_sha256"]:
        raise AssertionError("Implementation changed")
    for key in ["source_sha256","reference_sha256"]:
        for path,expected in protocol[key].items():
            if digest(path)!=expected:
                raise AssertionError("Frozen source changed")
    for name,expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT/"2026ml"/name)!=expected:
            raise AssertionError("Raw data changed")
    if digest(OUT/"models.npz")!=lock["model_sha256"] or digest(OUT/"protocol.json")!=lock["protocol_sha256"]:
        raise AssertionError("Model/protocol changed")
    raw = pd.read_csv(ROOT/"2026ml/train.csv",usecols=FIELDS+["pt_d"],dtype="int32")
    values,base,rows,_,_,_,_ = build(raw,16,protocol["sources"])
    np.testing.assert_array_equal(rows,np.load(OUT/"valid_indices.npy"))
    model = np.load(OUT/"models.npz")
    ps = expit(design(values,base,model["center"],model["scale"])@model["betas"])
    error = float(np.max(np.abs(ps-np.load(OUT/"all_predictions.npy"))))
    if error>1e-12:
        raise AssertionError("Replayed candidates differ")
    y = pd.read_csv(ROOT/"2026ml/train.csv",usecols=["label"],dtype="int32").label.to_numpy()[rows]
    if len(y)!=421983 or int(y.sum())!=3724:
        raise AssertionError("Full16 cohort changed")
    scores = [rank_sum_auc(y,ps[:,i]) for i in range(ps.shape[1])]
    best = int(np.argmax(scores))
    if not np.array_equal(ps[:,best],np.load(OUT/"prediction.npy")):
        raise AssertionError("Best saved candidate differs")
    report = {"status":"passed","auc":scores[best],"best_column":best,"all_auc":scores,
              "maximum_reload_error":error,"rows":len(y),"clicks":int(y.sum()),"original_datasets_unchanged":True,
              "evaluation16_coefficients_fit":False,"development16_used_for_model_selection":True,"target_met":scores[best]>=.82}
    write(OUT/"verification.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify",action="store_true")
    verify() if parser.parse_args().verify else main()
