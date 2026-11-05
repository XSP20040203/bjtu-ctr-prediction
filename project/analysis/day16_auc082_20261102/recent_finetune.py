"""Continue a verified weighted model on the most recent observed ad days."""
import argparse
import gc
import json
import math
from pathlib import Path

from train_weighted import (HERE, ROOT, ARCH, SETTINGS, prepare_inputs, make_model, batch,
                            predict, combined_scores, configure, digest, write, array_hash,
                            clip_separately, load_reference)
import numpy as np
import pandas as pd
import torch
from torch import nn
from day_model import auxiliary_loss
from label_boundary_audit import signed_labels, audit as audit_boundary

OUT = HERE/"runs/recent_finetune_v1_day16"
SOURCE = HERE/"runs/weighted_v1_day16"
WEIGHT = 5.
LEARNING_RATES = [.0001, .00005]


def select_recent_rows(dates, train_rows, days, cutoff=20191116):
    train_rows = np.asarray(train_rows)
    if days < 1 or not len(train_rows) or np.any(dates[train_rows] >= cutoff):
        raise ValueError("Only earlier observed training days are allowed")
    observed = np.unique(dates[train_rows])[-days:]
    return train_rows[np.isin(dates[train_rows], observed)]


def checked_source():
    verification = json.loads((SOURCE/"verification.json").read_text(encoding="utf8"))
    if verification["status"] != "passed":
        raise AssertionError("Weighted source must pass independent verification")
    audit_path = SOURCE/"label_boundary_audit.json"
    if not audit_path.is_file():
        audit_boundary("weighted_v1")
    audit = json.loads(audit_path.read_text(encoding="utf8"))
    if audit["status"] != "passed" or not audit["all_training_indices_strictly_before16"]:
        raise AssertionError("Source label boundary audit failed")
    summary = json.loads((SOURCE/"positive_weight_5/summary.json").read_text(encoding="utf8"))
    path = SOURCE/"positive_weight_5/ema_best.pt"
    if digest(path) != summary["checkpoint_sha256"]["ema_best.pt"]:
        raise AssertionError("Source checkpoint changed")
    return path


def reference_inputs(protocol):
    reference,rows,_ = load_reference(20191116,protocol["reference"])
    old_core = np.load(ARCH/"results/gated166_day16/validation_prediction.npy")
    v7 = np.load(ARCH/"fixed_gated20_allocated10_core30_day16.npy").astype("float64")
    return reference,rows,v7,old_core


def main():
    if OUT.exists():
        raise FileExistsError("Preserving recent fine tuning")
    source_path = checked_source()
    if not (HERE/"runs/duplicate_mean_v1_day16/label_boundary_audit.json").is_file():
        audit_boundary("duplicate_mean_v1")
    OUT.mkdir(parents=True)
    source_protocol = json.loads((SOURCE/"protocol.json").read_text(encoding="utf8"))
    protocol = {"day":20191116,"source_checkpoint":str(source_path),"source_sha256":digest(source_path),
                "reference":source_protocol["reference"],"source_verification_sha256":digest(SOURCE/"verification.json"),
                "source_files_sha256":{str(p):digest(p) for p in [Path(__file__),HERE/"train_weighted.py",HERE/"label_boundary_audit.py"]},
                "raw_datasets_sha256":source_protocol["raw_datasets_sha256"],
                "recent_days":[1,2],"learning_rates":LEARNING_RATES,"positive_weight":WEIGHT,
                "optimizer":"Fresh AdamW state; weight_decay0.0001; clip5; existing auxiliary0.02",
                "selection":"Maximum of the prespecified score recipes at quarter-epoch checks on reused development16",
                "evaluation16_labels_fit":False,"development_day_reused":True}
    for name,expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT/"2026ml"/name) != expected:
            raise AssertionError("Raw dataset changed")
    write(OUT/"protocol.json",protocol)
    configure(42)
    data,contexts = prepare_inputs(20191116,OUT)
    data["label"] = signed_labels(data["label"],data["train"],data["valid"])
    if not np.all(data["label"][data["valid"]] == -1):
        raise AssertionError("Signed held-out mask failed")
    dates = pd.read_csv(ROOT/"2026ml/train.csv",usecols=["pt_d"],dtype="int32").pt_d.to_numpy()[data["global_rows"]]
    reference,rows,v7,old_core = reference_inputs(protocol)
    np.testing.assert_array_equal(rows,data["global_rows"][data["valid"]])
    checkpoint = torch.load(source_path,map_location="cpu",weights_only=False)
    reports = []
    for days in protocol["recent_days"]:
        directory = OUT/f"recent_{days}"
        directory.mkdir()
        configure(4216)
        model = make_model(data,contexts,WEIGHT)
        model.load_state_dict(checkpoint["state_dict"],strict=True)
        initial = predict(model,data,data["valid"],WEIGHT)
        np.testing.assert_allclose(initial,np.load(SOURCE/"positive_weight_5/ema_best_prediction.npy"),rtol=0,atol=2e-7)
        selected = select_recent_rows(dates,data["train"],days)
        prior = float(data["label"][selected].mean())
        normalizer = 1+(WEIGHT-1)*prior
        write(directory/"training_contract.json",{"days":np.unique(dates[selected]).tolist(),"rows":len(selected),
              "clicks":int(data["label"][selected].sum()),"global_indices_sha256":array_hash(data["global_rows"][selected]),
              "maximum_date":int(dates[selected].max()),"evaluation_outcomes_masked":True})
        optimizer = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATES[0],weight_decay=.0001)
        best,history = -1.,[]
        steps = math.ceil(len(selected)/SETTINGS["batch_size"])
        checks = {math.ceil(steps*f) for f in [.25,.5,.75,1.]}
        for epoch,lr in enumerate(LEARNING_RATES):
            for group in optimizer.param_groups:
                group["lr"] = lr
            order = np.random.default_rng(4216+epoch).permutation(selected)
            model.train()
            for step,start in enumerate(range(0,len(order),SETTINGS["batch_size"]),1):
                ids = order[start:start+SETTINGS["batch_size"]]
                labels = data["label"][ids]
                if np.any(dates[ids] >= 20191116) or not np.isin(labels,[0,1]).all():
                    raise AssertionError("Invalid training labels/date")
                optimizer.zero_grad(set_to_none=True)
                logits,aux = model(*batch(data,ids))
                target = torch.as_tensor(labels,dtype=torch.float32,device="cuda")
                loss = nn.functional.binary_cross_entropy_with_logits(logits,target,pos_weight=target.new_tensor(WEIGHT))/normalizer
                groups = data["day_rows"][ids]
                aux_target = torch.as_tensor(data["any_targets"][groups],dtype=torch.float32,device="cuda")
                sizes = torch.as_tensor(data["day_sizes"][groups],dtype=torch.float32,device="cuda")
                loss = loss+.02*auxiliary_loss(aux,aux_target,sizes)
                if not bool(torch.isfinite(loss)):
                    raise AssertionError("Nonfinite loss")
                loss.backward()
                clip_separately(model,model.parameters(),5.)
                optimizer.step()
                if step not in checks:
                    continue
                p = predict(model,data,data["valid"],WEIGHT)
                metrics = combined_scores(data["evaluation_labels"],p,reference,v7,old_core)
                recipe = max(metrics,key=lambda key:metrics[key]["auc"])
                row = {"epoch":epoch+step/steps,"recipe":recipe,"metrics":metrics}
                history.append(row)
                if metrics[recipe]["auc"] > best:
                    best = metrics[recipe]["auc"]
                    torch.save({"state_dict":{n:v.detach().cpu().clone() for n,v in model.state_dict().items()},
                                "recipe":recipe,"record":row,"protocol_sha256":digest(OUT/"protocol.json")},directory/"best.pt")
                    np.save(directory/"raw_prediction.npy",p)
                write(directory/"history.json",history)
                print(json.dumps({"recent_days":days,"epoch":row["epoch"],"recipe":recipe,
                                  "auc":metrics[recipe]["auc"]}),flush=True)
                model.train()
        reports.append({"recent_days":days,"best_auc":best,"checkpoint_sha256":digest(directory/"best.pt")})
        del model,optimizer
        gc.collect()
        torch.cuda.empty_cache()
    write(OUT/"completion.json",{"status":"trained","arms":reports,"independent_verification_pending":True})


def recipe_prediction(recipe,p,reference,v7,old_core):
    if recipe == "single":
        return p.astype("float64")
    if recipe == "core_replacement_030":
        return v7+.3*(p-old_core)
    if recipe.startswith("reference_blend_"):
        alpha = float(recipe.rsplit("_",1)[1])
        return (1-alpha)*reference+alpha*p
    raise ValueError("Unrecognized score recipe")


def verify():
    from verify_stacker import rank_sum_auc
    protocol = json.loads((OUT/"protocol.json").read_text(encoding="utf8"))
    completion = json.loads((OUT/"completion.json").read_text(encoding="utf8"))
    checked_source()
    if digest(SOURCE/"verification.json") != protocol["source_verification_sha256"]:
        raise AssertionError("Source verification changed")
    if digest(protocol["source_checkpoint"]) != protocol["source_sha256"]:
        raise AssertionError("Source changed")
    for path,expected in protocol["source_files_sha256"].items():
        if digest(path) != expected:
            raise AssertionError("Training code changed")
    for name,expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT/"2026ml"/name) != expected:
            raise AssertionError("Raw data changed")
    configure(42)
    directory = OUT/"independent_rebuild"
    directory.mkdir()
    data,contexts = prepare_inputs(20191116,directory)
    data["label"] = signed_labels(data["label"],data["train"],data["valid"])
    raw = pd.read_csv(ROOT/"2026ml/train.csv",usecols=["pt_d","label"],dtype="int32")
    rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191116)
    np.testing.assert_array_equal(rows,np.load(OUT/"valid_indices.npy"))
    y = raw.label.to_numpy()[rows]
    if len(y) != 421983 or int(y.sum()) != 3724:
        raise AssertionError("Complete16 cohort changed")
    dates = raw.pt_d.to_numpy()[data["global_rows"]]
    reference,ref_rows,v7,old_core = reference_inputs(protocol)
    np.testing.assert_array_equal(rows,ref_rows)
    records,predictions = [],[]
    for arm in completion["arms"]:
        target = OUT/f"recent_{arm['recent_days']}"
        selected = select_recent_rows(dates,data["train"],arm["recent_days"])
        contract = json.loads((target/"training_contract.json").read_text(encoding="utf8"))
        if (contract["global_indices_sha256"] != array_hash(data["global_rows"][selected])
                or contract["maximum_date"] >= 20191116
                or contract["clicks"] != int(data["label"][selected].sum())):
            raise AssertionError("Recent training boundary changed")
        if digest(target/"best.pt") != arm["checkpoint_sha256"]:
            raise AssertionError("Checkpoint changed")
        ckpt = torch.load(target/"best.pt",map_location="cpu",weights_only=False)
        if ckpt["protocol_sha256"] != digest(OUT/"protocol.json"):
            raise AssertionError("Wrong protocol")
        model = make_model(data,contexts,WEIGHT)
        model.load_state_dict(ckpt["state_dict"],strict=True)
        p = predict(model,data,data["valid"],WEIGHT)
        error = float(np.max(np.abs(p-np.load(target/"raw_prediction.npy"))))
        if error > 2e-7:
            raise AssertionError("Replayed model differs")
        combined = recipe_prediction(ckpt["recipe"],p,reference,v7,old_core)
        score = rank_sum_auc(y,combined)
        if abs(score-arm["best_auc"]) > 2e-8:
            raise AssertionError("Independent score differs")
        records.append(dict(arm,auc=score,recipe=ckpt["recipe"],replay_error=error))
        predictions.append(combined)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    best = int(np.argmax([r["auc"] for r in records]))
    np.save(OUT/"prediction.npy",predictions[best])
    result = {"status":"passed","auc":records[best]["auc"],"records":records,"rows":len(y),"clicks":int(y.sum()),
              "original_datasets_unchanged":True,"evaluation16_labels_fit":False,"target_met":records[best]["auc"]>=.82}
    write(OUT/"verification.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify",action="store_true")
    verify() if parser.parse_args().verify else main()
