"""Train matched chronological daily-interest arms and replay full validation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model_comparison"))
sys.path.insert(0, str(HERE.parent / "user_personalization"))
from train_models import load_arrays
from evaluation import metrics, full_diagnostics
from daily_model import DailyInterestDCN


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def main(args):
    output = args.output_root/f"day{str(args.day)[-2:]}"/args.arm/f"seed{args.seed}"
    if output.exists():
        raise FileExistsError(f"Preserving existing training: {output}")
    protocol = read(HERE/"protocol.json")
    manifest = read(args.cache/"manifest.json")
    feature_manifest = read(args.features/"manifest.json")
    if feature_manifest.get("validation_day", args.day) != args.day:
        raise ValueError("Feature day mismatch")
    if len(manifest["numeric_fields"]) != 75:
        raise ValueError("Expected the existing 75-input numeric contract")
    arrays = load_arrays(args.cache)
    numeric = np.load(args.features/"numeric.npy", mmap_mode="r")
    requests = np.load(args.features/"request_index.npy", mmap_mode="r")
    topics = np.load(args.features/"topic_weights.npy", mmap_mode="r")
    context = np.load(args.features/"interest_context.npy", mmap_mode="r")
    train, valid = arrays["train_indices"], arrays["valid_indices"]
    if (arrays["date"][train].max() >= args.day or not np.all(arrays["date"][valid] == args.day)
            or np.intersect1d(train,valid).size or np.any(requests[train] == 0) or np.any(requests[valid] == 0)):
        raise AssertionError("Chronological advertisement split or request mapping invalid")
    options = {**protocol["backbone"], **protocol["interest"], **protocol["arms"][args.arm],
               "numeric_dim": numeric.shape[1], "n_topics": topics.shape[-1], "context_dim": context.shape[-1]}
    torch.set_num_threads(6)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DailyInterestDCN(manifest["categorical_fields"],manifest["sequence_fields"],manifest["group_for_field"],
                            manifest["vocabulary_sizes"],initial_ctr=float(arrays["label"][train].mean()),**options).to(device)
    initial = hashlib.sha256()
    for name,value in model.state_dict().items():
        if name.startswith(("encoder.","cross.","deep.","output.")):
            initial.update(name.encode());initial.update(value.cpu().numpy().tobytes())
    optimizer = torch.optim.AdamW(model.parameters(),lr=protocol["learning_rate"],weight_decay=protocol["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=1,min_lr=1e-5)
    output.mkdir(parents=True)
    sources = [Path(__file__), HERE/"daily_model.py",HERE/"interest_data.py",HERE/"prepare_features.py",HERE/"protocol.json",
               args.cache/"manifest.json",args.cache/"train_indices.npy",args.cache/"valid_indices.npy",args.features/"manifest.json",
               args.cache/"categories.npy",args.cache/"sequences.npy",args.cache/"label.npy",args.cache/"date.npy",
               args.features/"numeric.npy",args.features/"request_index.npy",args.features/"topic_weights.npy",args.features/"interest_context.npy"]
    write(output/"provenance.json",{"source_sha256":{str(p.resolve()):digest(p) for p in sources},
          "initial_backbone_sha256":initial.hexdigest(),"cache":str(args.cache.resolve()),"features":str(args.features.resolve()),
          "protocol":protocol,"advertisement_validation_labels_in_features":False,
          "same_day_source_feedback_available":True})

    def batch(rows):
        request = requests[rows]
        return (torch.as_tensor(arrays["categories"][rows].astype(np.int64),device=device),
                torch.as_tensor(arrays["sequences"][rows].astype(np.int64),device=device),
                torch.as_tensor(np.array(numeric[rows]),device=device),
                torch.as_tensor(np.array(topics[request],dtype=np.float32),device=device),
                torch.as_tensor(np.array(context[request],dtype=np.float32),device=device))

    def predict(rows):
        model.eval()
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(model(*batch(rows[start:start+1024]))).cpu().numpy()
                                   for start in range(0,len(rows),1024)])

    y = np.asarray(arrays["label"][valid])
    user_column = manifest["categorical_fields"].index("user_id")
    user_count = manifest["vocabulary_sizes"][manifest["group_for_field"]["user_id"]]
    user_clicks = np.bincount(arrays["categories"][train,user_column],weights=arrays["label"][train],minlength=user_count)
    valid_users = np.asarray(arrays["categories"][valid,user_column])
    task_column = manifest["categorical_fields"].index("task_id")
    new_task = ~np.isin(arrays["categories"][valid,task_column],np.unique(arrays["categories"][train,task_column]))
    size=protocol["batch_size"];batches=math.ceil(len(train)/size)
    checkpoints={math.ceil(batches*x) for x in [.25,.5,.75,1.]}
    started=time.perf_counter();history=[];best_auc=-1.;bad=0;best=None;stop=False
    print(f"START_DAILY day={args.day} arm={args.arm} device={device} rows={len(train)} topics={topics.shape[-1]}",flush=True)
    for epoch in range(1,protocol["max_epochs"]+1):
        order=np.random.default_rng(args.seed+epoch).permutation(train);model.train();running=0.;seen=0
        for step,start in enumerate(range(0,len(order),size),1):
            rows=order[start:start+size]
            target=torch.as_tensor(arrays["label"][rows].astype(np.float32),device=device)
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.binary_cross_entropy_with_logits(model(*batch(rows)),target)
            if not torch.isfinite(loss).item():raise RuntimeError("Nonfinite training loss")
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            running+=float(loss.detach())*len(rows);seen+=len(rows)
            if step not in checkpoints:continue
            prediction=predict(valid);record=metrics(y,prediction)
            entry={"fractional_epoch":epoch-1+step/batches,"online_train_logloss":running/seen,
                   "learning_rate":optimizer.param_groups[0]["lr"],"elapsed_seconds":time.perf_counter()-started,**record}
            history.append(entry);pd.DataFrame(history).to_csv(output/"training_history.csv",index=False)
            write(output/"progress.json",{"status":"training",**entry})
            print(json.dumps({"day":args.day,"arm":args.arm,**entry}),flush=True)
            if record["auc"]>best_auc+1e-5:
                best_auc,bad=record["auc"],0
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best={"day":args.day,"arm":args.arm,"seed":args.seed,"model_class":"DailyInterestDCN",
                      "model_options":options,"best_fractional_epoch":entry["fractional_epoch"],
                      "cache":str(args.cache.resolve()),"features":str(args.features.resolve()),
                      "initial_backbone_sha256":initial.hexdigest(),"parameter_count":sum(p.numel() for p in model.parameters()),**record}
                torch.save({"state_dict":state,"manifest":manifest,"feature_manifest":feature_manifest,"model_options":options,
                            "day":args.day,"arm":args.arm,"seed":args.seed,"best_fractional_epoch":entry["fractional_epoch"]},output/"model.pt")
                np.save(output/"validation_prediction.npy",prediction);write(output/"best_so_far.json",best)
            else:bad+=1
            model.train()
            if bad>=protocol["patience_checkpoints"]:stop=True;break
        scheduler.step(history[-1]["auc"])
        if stop:break
    checkpoint=torch.load(output/"model.pt",map_location="cpu",weights_only=False)
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    replay=predict(valid);saved=np.load(output/"validation_prediction.npy")
    error=float(np.max(np.abs(replay-saved)))
    if error>2e-7:raise AssertionError(f"Checkpoint mismatch: {error}")
    best.update(status="complete_and_verified",maximum_reload_error=error,training_seconds=time.perf_counter()-started,
                checkpoints_completed=len(history),last_fractional_epoch=history[-1]["fractional_epoch"],
                model_sha256=digest(output/"model.pt"),validation_strategy="whole future date",
                same_day_source_available=True,advertisement_statistics="strict per-row past dates",
                diagnostics=full_diagnostics(y,replay,valid_users,user_clicks,new_task),final_labels_used=False)
    write(output/"metrics.json",best);write(output/"progress.json",{"status":"complete_and_verified","auc":best["auc"]})
    print("COMPLETE_DAILY "+json.dumps(best),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--day",type=int,required=True,choices=[20191115,20191116])
    parser.add_argument("--arm",choices=["control","mean","din","today_din"],required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--features",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--output-root",type=Path,default=HERE/"results")
    main(parser.parse_args())
