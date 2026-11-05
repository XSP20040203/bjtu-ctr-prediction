"""Independent temporal-stat probes and full checkpoint replay for daily interests."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"auc085"))
sys.path.insert(0,str(HERE.parent/"model_comparison"))
from target_statistics import SPECS
from train_models import load_arrays
from daily_model import DailyInterestDCN
from train_daily import digest
from interest_data import request_keys


def read(path):return json.loads(path.read_text(encoding="utf-8"))


def audit_features(folder,cache):
    m=read(folder/"manifest.json");base=read(cache/"manifest.json")
    fields=sorted({field for spec,_ in SPECS for field in spec}|{"pt_d","label"})
    raw=pd.read_csv(HERE.parents[1]/"2026ml/train.csv",usecols=fields,dtype="int32")
    train=np.load(cache/"train_indices.npy");valid=np.load(cache/"valid_indices.npy")
    numeric=np.load(folder/"numeric.npy",mmap_mode="r");mapping=np.load(folder/"request_index.npy",mmap_mode="r")
    keys=np.load(folder/"keys.npy",mmap_mode="r");weights=np.load(folder/"topic_weights.npy",mmap_mode="r")
    context=np.load(folder/"interest_context.npy",mmap_mode="r")
    eligible=np.r_[train,valid];wanted=request_keys(raw.user_id.to_numpy()[eligible],raw.pt_d.to_numpy()[eligible])
    np.testing.assert_array_equal(keys[mapping[eligible]],wanted)
    sampled=np.unique(np.r_[train[np.linspace(0,len(train)-1,10,dtype=int)],valid[np.linspace(0,len(valid)-1,10,dtype=int)]])
    maximum=0.;checks=0
    for row in sampled:
        history=raw.loc[raw.pt_d<int(raw.iloc[row].pt_d)]
        prior=(history.label.sum()+1000*.0084)/(len(history)+1000)
        for index,(fields,alpha) in enumerate(SPECS):
            matches=np.ones(len(history),dtype=bool)
            for field in fields:matches &= history[field].to_numpy()==raw.iloc[row][field]
            count=int(matches.sum());clicks=int(history.loc[matches,"label"].sum())
            probability=np.clip((clicks+alpha*prior)/(count+alpha),1e-6,1-1e-6)
            values=[np.log1p(count),np.log(probability/(1-probability))-np.log(prior/(1-prior))]
            for offset,value in enumerate(values):
                name=base["numeric_fields"][39+index*2+offset];scale=m["numeric_scaler"][name]
                expected=(np.float32(value)-scale["mean"])/scale["std"]
                error=abs(float(numeric[row,39+index*2+offset])-expected)
                maximum=max(maximum,error);checks+=1
                if error>2e-5:raise AssertionError(f"Statistic mismatch {row} {name} {error}")
    sums=np.asarray(weights).sum(-1);mass=np.load(folder/"topic_mass.npy",mmap_mode="r")
    np.testing.assert_allclose(sums[mass>0],1,atol=2e-5)
    assert np.isfinite(numeric).all() and np.isfinite(context).all()
    if m["validation_day"]==20191115:
        future=raw.pt_d.to_numpy()>20191115
        assert np.all(mapping[future]==0) and np.all(numeric[future]==0)
    return {"day":m["validation_day"],"stat_probes":checks,"maximum_stat_error":maximum,
            "all_request_keys_match":True,"normalized_topic_channels":True,"train_rows":len(train),"valid_rows":len(valid)}


def main(args):
    torch.set_num_threads(4);device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audits=[];records=[];seen=set()
    from sklearn.metrics import roc_auc_score,log_loss
    for path in sorted((HERE/"results").glob("day*/*/seed*/metrics.json")):
        record=read(path);cache=Path(record["cache"]);features=Path(record["features"])
        provenance=read(path.parent/"provenance.json")
        for name,value in provenance["source_sha256"].items():
            if digest(Path(name))!=value:raise AssertionError("Source or input changed "+name)
        if features not in seen:
            audits.append(audit_features(features,cache));seen.add(features)
        arrays=load_arrays(cache);valid=arrays["valid_indices"]
        numeric=np.load(features/"numeric.npy",mmap_mode="r");requests=np.load(features/"request_index.npy",mmap_mode="r")
        weights=np.load(features/"topic_weights.npy",mmap_mode="r");context=np.load(features/"interest_context.npy",mmap_mode="r")
        checkpoint=torch.load(path.parent/"model.pt",map_location="cpu",weights_only=False);m=checkpoint["manifest"]
        model=DailyInterestDCN(m["categorical_fields"],m["sequence_fields"],m["group_for_field"],m["vocabulary_sizes"],**checkpoint["model_options"]).to(device)
        model.load_state_dict(checkpoint["state_dict"],strict=True);model.eval();output=[]
        with torch.no_grad():
            for start in range(0,len(valid),1024):
                rows=valid[start:start+1024];req=requests[rows]
                inputs=(torch.as_tensor(arrays["categories"][rows].astype(np.int64),device=device),
                        torch.as_tensor(arrays["sequences"][rows].astype(np.int64),device=device),
                        torch.as_tensor(np.array(numeric[rows]),device=device),torch.as_tensor(np.array(weights[req]),device=device),
                        torch.as_tensor(np.array(context[req]),device=device))
                output.append(torch.sigmoid(model(*inputs)).cpu().numpy())
        prediction=np.concatenate(output);saved=np.load(path.parent/"validation_prediction.npy")
        error=float(np.max(np.abs(prediction-saved)))
        if error>2e-7:raise AssertionError("Full replay mismatch")
        auc=float(roc_auc_score(arrays["label"][valid],prediction));ll=float(log_loss(arrays["label"][valid],prediction,labels=[0,1]))
        if abs(auc-record["auc"])>1e-10 or abs(ll-record["logloss"])>1e-10:raise AssertionError("Metric mismatch")
        if digest(path.parent/"model.pt")!=record["model_sha256"]:raise AssertionError("Checkpoint changed")
        result={"day":record["day"],"arm":record["arm"],"seed":record["seed"],"auc":auc,"logloss":ll,"rows":len(valid),"maximum_replay_error":error}
        records.append(result);print(json.dumps(result),flush=True);del model
    result={"status":"passed","runs":records,"feature_audits":audits,"final_labels_available":False}
    (HERE/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("VERIFIED_DAILY_RUNS "+str(len(records)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();main(parser.parse_args())
