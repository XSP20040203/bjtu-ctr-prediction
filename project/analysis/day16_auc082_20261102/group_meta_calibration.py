"""Fit small15-only ensembles of cross-fitted group correction models."""
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
from rich_group_residual import ROOT,HERE,REFERENCE,prepare,prediction,digest,write,rank_sum_auc,lgb
from rich_userday_activity import correct
from subgroup_residual import subgroup_matrix,predict as subgroup_predict
from extended_history_residual import matrix as extended_matrix
from fit_stacker import fit as logistic_fit
from rank_calibration import fit as rank_fit,predict as rank_predict

OUT=HERE / "runs/group_meta_calibration_v1"
FAMILIES={"group":"rich_group_residual_v1","pair":"pairwise_group_boost_v1",
          "subgroup":"subgroup_residual_v1","extended":"extended_history_residual_v1"}
RIDGES=[.001,.01,.1]
RANK_RIDGES=[.03,.3,3.]


def choose_sources():
    chosen=[]
    for kind,folder in FAMILIES.items():
        directory=HERE / "runs" / folder
        verified=json.loads((directory / "verification.json").read_text(encoding="utf8"))
        if verified["status"]!="passed":
            raise AssertionError("Source requires independent verification")
        lock=json.loads((directory / "models_locked.json").read_text(encoding="utf8"))
        buckets={}
        for record in lock["models"]:
            key=(record["temperature"],) if kind=="pair" else (record.get("field","all"),record["objective"])
            if key not in buckets or record["oof15_auc"]>buckets[key]["oof15_auc"]:
                buckets[key]=record
        for record in buckets.values():
            suffix="_oof_groups.npy" if kind=="group" else "_oof.npy"
            chosen.append({"kind":kind,"directory":str(directory),"record":record,
                           "oof_path":str(directory / (record["name"]+suffix)),
                           "model_path":str(directory / (record["name"]+".txt"))})
    return chosen


def design(day,sources,base_source="reference"):
    common=prepare(day,base_source,include_labels=False)
    rows,base,users=common["rows"],common["base"],common["keys"][:,1][common["groups"]]
    margin=logit(base.clip(1e-6,1-1e-6))
    columns,names=[],[]
    extended=None
    subgroup_cache={}
    for source in sources:
        kind,record=source["kind"],source["record"]
        if kind=="subgroup":
            field=record["field"]
            if field not in subgroup_cache:
                protocol=json.loads((Path(source["directory"]) / "protocol.json").read_text(encoding="utf8"))
                subgroup_cache[field]=subgroup_matrix(day,field,protocol["selected_context_columns"],base_source)
            data=subgroup_cache[field]
        elif kind=="extended" and day==16:
            if extended is None:
                extended=extended_matrix(day,base_source)
            data=extended
        else:
            data=common
        np.testing.assert_array_equal(rows,data["rows"])
        if day==15:
            group_prediction=np.load(source["oof_path"])
        else:
            model=lgb.Booster(model_str=Path(source["model_path"]).read_text(encoding="utf8"))
            group_prediction=(model.predict(data["x"],raw_score=True) if kind=="pair" else
                              subgroup_predict(model,data,record["objective"]) if kind=="subgroup" else
                              prediction(model,data,record["objective"]))
        if kind=="pair":
            p=expit(margin+record["power"]*np.clip(group_prediction[data["groups"]],-3,3))
        else:
            p=correct(base,group_prediction,data["groups"],record["objective"],record["power"])
        if len(p)!=len(rows) or not np.isfinite(p).all():
            raise AssertionError("Invalid correction source")
        columns.append(logit(p.clip(1e-6,1-1e-6))-margin)
        names.append(kind+"/"+record["name"])
    return np.column_stack(columns),margin,rows,users,names


def linear_design(x,base,center,scale):
    return np.column_stack([np.ones(len(base)),base,np.clip((x-center)/scale,-8,8)])


def main():
    if OUT.exists():
        raise FileExistsError("Preserving group meta calibration")
    sources=choose_sources()
    paths=[Path(__file__),HERE / "rich_group_residual.py",HERE / "rich_userday_activity.py",HERE / "subgroup_residual.py",
           HERE / "extended_history_residual.py",HERE / "fit_stacker.py",HERE / "rank_calibration.py",HERE / "verify_stacker.py"]
    for source in sources:
        paths += [Path(source["oof_path"]),Path(source["model_path"]),Path(source["directory"]) / "protocol.json",
                  Path(source["directory"]) / "models_locked.json",Path(source["directory"]) / "verification.json"]
        original=json.loads((Path(source["directory"]) / "protocol.json").read_text(encoding="utf8"))
        paths += [Path(p) for p in original["inputs_sha256"]]
    paths=list(dict.fromkeys(paths))
    OUT.mkdir(parents=True)
    protocol={"sources":sources,"logistic_ridges":RIDGES,"rank_ridges":RANK_RIDGES,
              "inputs_sha256":{str(p):digest(p) for p in paths},
              "raw_datasets_sha256":{p.name:digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
              "fit_date":20191115,"evaluation_date":20191116,"evaluation16_labels_fit":False,
              "selection":"One source per15-only family/objective bucket; all6 meta fits on15 then locked before16",
              "nested_meta_cv":False,"note":"Base OOF predictions share folds. No nested-validation or unbiased15 performance claim.",
              "development16_model_selection":True}
    write(OUT / "protocol.json",protocol)
    x,base,rows,_,names=design(15,sources)
    y=pd.read_csv(ROOT / "2026ml/train.csv",usecols=["label"],dtype="int8").label.to_numpy()[rows]
    stored,models={},[]
    center,scale=x.mean(0),x.std(0).clip(.05)
    z=linear_design(x,base,center,scale)
    for ridge in RIDGES:
        beta,info=logistic_fit(z,y,ridge,list(range(x.shape[1])))
        if not info["success"]:
            raise AssertionError("Logistic meta fit did not converge")
        name=f"logistic_{ridge:g}"
        model={"center":center,"scale":scale,"beta":beta}
        stored.update({name+"_"+k:v for k,v in model.items()})
        models.append({"name":name,"kind":"logistic","ridge":ridge,"info":info})
        print(json.dumps(models[-1]),flush=True)
    for ridge in RANK_RIDGES:
        model,info=rank_fit(x,base,y,1.,ridge,9182)
        name=f"rank_{ridge:g}"
        stored.update({name+"_"+k:v for k,v in model.items()})
        models.append({"name":name,"kind":"rank","ridge":ridge,"info":info})
        print(json.dumps(models[-1]),flush=True)
    np.savez(OUT / "models.npz",**stored)
    write(OUT / "weights_locked.json",{"models":models,"model_sha256":digest(OUT / "models.npz"),
          "protocol_sha256":digest(OUT / "protocol.json"),"features":names,"training_rows":len(y),"training_clicks":int(y.sum())})
    del x,z,y
    gc.collect()
    evaluate(False)


def evaluate(verifying):
    protocol=json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    lock=json.loads((OUT / "weights_locked.json").read_text(encoding="utf8"))
    if digest(OUT / "protocol.json")!=lock["protocol_sha256"] or digest(OUT / "models.npz")!=lock["model_sha256"]:
        raise AssertionError("Locked ensemble changed")
    for path,expected in protocol["inputs_sha256"].items():
        if digest(path)!=expected:
            raise AssertionError("Source changed")
    for name,expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name)!=expected:
            raise AssertionError("Original data changed")
    stored=np.load(OUT / "models.npz")
    results,predictions=[],[]
    for base_source in ["reference","weighted_transfer"]:
        x,base,rows,_,names=design(16,protocol["sources"],base_source)
        if names!=lock["features"]:
            raise AssertionError("Feature order changed")
        y=pd.read_csv(ROOT / "2026ml/train.csv",usecols=["label"],dtype="int8").label.to_numpy()[rows]
        if len(y)!=421983 or int(y.sum())!=3724:
            raise AssertionError("Complete16 cohort changed")
        for record in lock["models"]:
            model={k:stored[record["name"]+"_"+k] for k in ["center","scale","beta"]}
            p=(rank_predict(x,base,model) if record["kind"]=="rank" else
               expit(linear_design(x,base,model["center"],model["scale"])@model["beta"]))
            results.append({"name":record["name"],"source":base_source,"auc":rank_sum_auc(y,p)})
            predictions.append(p)
    best=int(np.argmax([r["auc"] for r in results]))
    p=predictions[best]
    if verifying:
        error=float(np.max(np.abs(p-np.load(OUT / "prediction.npy"))))
        if error>1e-12:
            raise AssertionError("Independent ensemble replay differs")
        np.testing.assert_array_equal(rows,np.load(OUT / "valid_indices.npy"))
    else:
        error=0.
        np.save(OUT / "prediction.npy",p)
        np.save(OUT / "valid_indices.npy",rows)
    result={"status":"passed" if verifying else "evaluated","auc":results[best]["auc"],"best":results[best],
            "results":results,"rows":len(rows),"clicks":3724,"maximum_replay_error":error,
            "original_datasets_unchanged":True,"evaluation16_labels_fit":False,"nested_meta_cv":False,
            "development16_model_selection":True,"target_met":results[best]["auc"]>=.82}
    write(OUT / ("verification.json" if verifying else "result.json"),result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--verify",action="store_true")
    evaluate(True) if parser.parse_args().verify else main()
