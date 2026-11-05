"""Compare ordinary and sharpness-aware continuation on identical pre16 rows."""
import argparse
import gc
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from train_weighted import (ROOT,HERE,SETTINGS,configure,prepare_inputs,make_model,batch,predict,
                            combined_scores,digest,write,array_hash,clip_separately)
from recent_finetune import checked_source,reference_inputs,recipe_prediction,SOURCE,WEIGHT
from label_boundary_audit import signed_labels
from day_model import auxiliary_loss
from sam_step import sam_step

OUT = HERE/"runs/sam_continuation_v1_day16"
RHOS = [0.,.01,.05]


def save_checkpoint(model,directory,endpoint,record,p,protocol):
    torch.save({"state_dict":{n:v.detach().cpu().clone() for n,v in model.state_dict().items()},
                "record":record,"protocol_sha256":digest(protocol)},directory/f"{endpoint}.pt")
    np.save(directory/f"{endpoint}_raw_prediction.npy",p)


def main():
    if OUT.exists(): raise FileExistsError("Preserve the SAM comparison")
    source_path = checked_source()
    source_protocol = json.loads((SOURCE/"protocol.json").read_text(encoding="utf8"))
    OUT.mkdir(parents=True)
    paths = [Path(__file__),HERE/"sam_step.py",HERE/"train_weighted.py",HERE/"recent_finetune.py",HERE/"label_boundary_audit.py"]
    protocol = {"day":20191116,"rhos":RHOS,"learning_rate":.0001,"epochs":1,
                "batch_size":1024,"positive_weight":WEIGHT,"auxiliary_weight":.02,
                "seed":5021,"source_checkpoint":str(source_path),"source_checkpoint_sha256":digest(source_path),
                "reference":source_protocol["reference"],"raw_datasets_sha256":source_protocol["raw_datasets_sha256"],
                "source_sha256":{str(p):digest(p) for p in paths},
                "method":"Two passes with the same dropout masks; normalized total-loss gradient ascent; exact parameter restoration; AdamW descent",
                "selection":"Best fixed recipe at initial and four quarter-epoch checks; final retained independently",
                "evaluation16_labels_fit":False,"development_day_reused":True,
                "paper":"Foret et al. Sharpness-Aware Minimization, ICLR2021; arXiv2010.01412",
                "reference_implementation":"https://github.com/google-research/sam"}
    for name,expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT/"2026ml"/name)!=expected: raise AssertionError("Raw dataset changed")
    write(OUT/"protocol.json",protocol)
    configure(5021)
    data,contexts = prepare_inputs(20191116,OUT)
    data["label"] = signed_labels(data["label"],data["train"],data["valid"])
    raw_dates = pd.read_csv(ROOT/"2026ml/train.csv",usecols=["pt_d"],dtype="int32").pt_d.to_numpy()[data["global_rows"]]
    if np.any(raw_dates[data["train"]]>=20191116) or not np.all(data["label"][data["valid"]]==-1):
        raise AssertionError("Training boundary failed")
    reference,rows,v7,old_core = reference_inputs(protocol)
    np.testing.assert_array_equal(rows,data["global_rows"][data["valid"]])
    source = torch.load(source_path,map_location="cpu",weights_only=False)
    order = np.random.default_rng(5022).permutation(data["train"])
    write(OUT/"training_contract.json",{"maximum_fit_date":int(raw_dates[order].max()),"rows":len(order),
          "global_order_sha256":array_hash(data["global_rows"][order]),"held_label_dtype":str(data["label"].dtype),
          "held_label_values":np.unique(data["label"][data["valid"]]).tolist()})
    steps = math.ceil(len(order)/1024)
    checks = {math.ceil(steps*f) for f in [.25,.5,.75,1.]}
    normalizer = 1+(WEIGHT-1)*float(data["label"][data["train"]].mean())
    summaries = []
    for rho in RHOS:
        directory = OUT/f"rho_{rho:g}"
        directory.mkdir()
        configure(5021)
        model = make_model(data,contexts,WEIGHT)
        model.load_state_dict(source["state_dict"],strict=True)
        if any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in model.modules()):
            raise AssertionError("Stateful BatchNorm needs an explicit two-pass policy")
        initial = predict(model,data,data["valid"],WEIGHT)
        np.testing.assert_allclose(initial,np.load(SOURCE/"positive_weight_5/ema_best_prediction.npy"),rtol=0,atol=2e-7)
        metrics = combined_scores(data["evaluation_labels"],initial,reference,v7,old_core)
        best_recipe = max(metrics,key=lambda k:metrics[k]["auc"])
        best = metrics[best_recipe]["auc"]
        record = {"fraction":0.,"recipe":best_recipe,"metrics":metrics}
        save_checkpoint(model,directory,"best",record,initial,OUT/"protocol.json")
        history = [record]
        optimizer = torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
        model.train()
        totals = np.zeros(3)
        for step,start in enumerate(range(0,len(order),1024),1):
            selected = order[start:start+1024]
            if np.any(raw_dates[selected]>=20191116): raise AssertionError("Held day reached optimizer")
            target = torch.as_tensor(data["label"][selected],dtype=torch.float32,device="cuda")
            if bool(torch.any((target<0)|(target>1))): raise AssertionError("Unknown optimization label")
            inputs = batch(data,selected)
            group = data["day_rows"][selected]
            aux_target = torch.as_tensor(data["any_targets"][group],dtype=torch.float32,device="cuda")
            sizes = torch.as_tensor(data["day_sizes"][group],dtype=torch.float32,device="cuda")
            def loss():
                logits,aux = model(*inputs)
                ordinary = nn.functional.binary_cross_entropy_with_logits(logits,target,pos_weight=target.new_tensor(WEIGHT))/normalizer
                return ordinary+.02*auxiliary_loss(aux,aux_target,sizes)
            info = sam_step(model,optimizer,loss,rho,lambda params:clip_separately(model,params,5.))
            totals += [info["original_loss"],info["perturbed_loss"],info["ascent_gradient_norm"]]
            if step not in checks: continue
            p = predict(model,data,data["valid"],WEIGHT)
            metrics = combined_scores(data["evaluation_labels"],p,reference,v7,old_core)
            recipe = max(metrics,key=lambda k:metrics[k]["auc"])
            record = {"fraction":step/steps,"recipe":recipe,"metrics":metrics,"training_means":(totals/step).tolist()}
            history.append(record)
            if metrics[recipe]["auc"]>best:
                best = metrics[recipe]["auc"]
                save_checkpoint(model,directory,"best",record,p,OUT/"protocol.json")
            if step==steps: save_checkpoint(model,directory,"final",record,p,OUT/"protocol.json")
            write(directory/"history.json",history)
            print(json.dumps({"rho":rho,"fraction":record["fraction"],"best_recipe":recipe,
                              "auc":metrics[recipe]["auc"],"best_including_initial":best}),flush=True)
            model.train()
        summaries.append({"rho":rho,"best_auc":best,"checkpoint_sha256":{
            endpoint:digest(directory/f"{endpoint}.pt") for endpoint in ["best","final"]}})
        del model,optimizer
        gc.collect()
        torch.cuda.empty_cache()
    write(OUT/"completion.json",{"status":"trained","arms":summaries,"independent_verification_pending":True})


def verify():
    from verify_stacker import rank_sum_auc
    protocol = json.loads((OUT/"protocol.json").read_text(encoding="utf8"))
    completed = json.loads((OUT/"completion.json").read_text(encoding="utf8"))
    for path,expected in protocol["source_sha256"].items():
        if digest(path)!=expected: raise AssertionError("Training source changed")
    for name,expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT/"2026ml"/name)!=expected: raise AssertionError("Original data changed")
    if digest(protocol["source_checkpoint"])!=protocol["source_checkpoint_sha256"]:
        raise AssertionError("Source weights changed")
    configure(5021)
    directory = OUT/"independent_rebuild"
    directory.mkdir()
    data,contexts = prepare_inputs(20191116,directory)
    data["label"] = signed_labels(data["label"],data["train"],data["valid"])
    raw = pd.read_csv(ROOT/"2026ml/train.csv",usecols=["pt_d","label"],dtype="int32")
    contract = json.loads((OUT/"training_contract.json").read_text(encoding="utf8"))
    dates = raw.pt_d.to_numpy()[data["global_rows"]]
    order = np.random.default_rng(5022).permutation(data["train"])
    if (np.any(dates[order]>=20191116)
            or contract["global_order_sha256"]!=array_hash(data["global_rows"][order])
            or not np.all(data["label"][data["valid"]]==-1)):
        raise AssertionError("Independent training boundary audit failed")
    rows = np.flatnonzero(raw.pt_d.to_numpy()==20191116)
    np.testing.assert_array_equal(rows,np.load(OUT/"valid_indices.npy"))
    y = raw.label.to_numpy()[rows]
    if len(y)!=421983 or int(y.sum())!=3724: raise AssertionError("Full16 cohort changed")
    reference,refrows,v7,old_core = reference_inputs(protocol)
    np.testing.assert_array_equal(rows,refrows)
    records,candidates = [],[]
    for arm in completed["arms"]:
        target = OUT/f"rho_{arm['rho']:g}"
        for endpoint in ["best","final"]:
            if digest(target/f"{endpoint}.pt")!=arm["checkpoint_sha256"][endpoint]:
                raise AssertionError("SAM checkpoint changed")
            checkpoint = torch.load(target/f"{endpoint}.pt",map_location="cpu",weights_only=False)
            if checkpoint["protocol_sha256"]!=digest(OUT/"protocol.json"): raise AssertionError("Protocol mismatch")
            model = make_model(data,contexts,WEIGHT)
            model.load_state_dict(checkpoint["state_dict"],strict=True)
            p = predict(model,data,data["valid"],WEIGHT)
            error = float(np.max(np.abs(p-np.load(target/f"{endpoint}_raw_prediction.npy"))))
            if error>2e-7: raise AssertionError("SAM replay mismatch")
            recipe = checkpoint["record"]["recipe"]
            combined = recipe_prediction(recipe,p,reference,v7,old_core)
            score = rank_sum_auc(y,combined)
            expected = checkpoint["record"]["metrics"][recipe]["auc"]
            if abs(score-expected)>2e-8: raise AssertionError("Independent AUC mismatch")
            records.append({"rho":arm["rho"],"endpoint":endpoint,"auc":score,"recipe":recipe,
                            "fraction":checkpoint["record"]["fraction"],"replay_error":error})
            candidates.append(combined)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    best = int(np.argmax([r["auc"] for r in records]))
    np.save(OUT/"prediction.npy",candidates[best])
    result = {"status":"passed","auc":records[best]["auc"],"records":records,"rows":len(y),"clicks":int(y.sum()),
              "original_datasets_unchanged":True,"training_boundary_independently_rebuilt":True,
              "evaluation16_labels_fit":False,"target_met":records[best]["auc"]>=.82}
    write(OUT/"verification.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify",action="store_true")
    verify() if parser.parse_args().verify else main()
