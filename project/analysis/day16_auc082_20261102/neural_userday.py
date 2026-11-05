"""Direct multi-task daily response networks on news and ad input summaries."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _key in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:
    os.environ[_key]="2"
import argparse
import copy
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from feed_behavior_userday import prepare,dependency_paths,SOURCES
from rich_group_residual import ROOT,HERE,REFERENCE,digest,write,rank_sum_auc
from rich_userday_activity import array_hash,correct

OUT=HERE / "runs/neural_userday_v1"
SOURCES={**SOURCES,"feed_behavior":HERE / "runs/feed_behavior_userday_v1/prediction.npy"}
CONFIG={"seeds":[42,2026],"user_embedding_dims":[0,4],"widths":[96,32],"dropout":.2,
        "batch_size":256,"learning_rate":.001,"weight_decay":.01,"max_epochs":80,
        "patience":10,"count_loss_weight":.25,"powers":[0.,.125,.25,.5,.75,1.]}


def preprocess(x,keys,cutoff):
    training=keys[:,0]<cutoff
    center=x[training].mean(0,dtype="float64").astype("float32")
    scale=x[training].std(0,dtype="float64").clip(.05).astype("float32")
    users=np.unique(keys[training,1])
    return {"center":center,"scale":scale,"users":users}


def transform(x,keys,normalization):
    values=np.clip((x-normalization["center"])/normalization["scale"],-8,8).astype("float32")
    users=normalization["users"]
    position=np.searchsorted(users,keys[:,1])
    seen=(position<len(users)) & (users[position.clip(max=max(0,len(users)-1))]==keys[:,1])
    indices=np.where(seen,position+1,0).astype("int64")
    return values,indices


class DailyNet(nn.Module):
    def __init__(self,width,n_users,embedding_dim):
        super().__init__()
        self.embedding=nn.Embedding(n_users+1,embedding_dim,padding_idx=0) if embedding_dim else None
        self.network=nn.Sequential(nn.Linear(width+embedding_dim,96),nn.SiLU(),nn.Dropout(.2),
                                   nn.Linear(96,32),nn.SiLU(),nn.Dropout(.2),nn.Linear(32,2))

    def forward(self,x,user):
        if self.embedding is not None:
            x=torch.cat([x,self.embedding(user)],dim=1)
        return self.network(x)


def predictions(model,x,users,rows):
    model.eval()
    output=[]
    with torch.no_grad():
        for start in range(0,len(rows),1024):
            ids=rows[start:start+1024]
            z=model(x[ids],users[ids])
            output.append(torch.stack([torch.sigmoid(z[:,0]),torch.exp(z[:,1].clamp(-9,5))],dim=1).cpu().numpy())
    return np.concatenate(output).astype("float64")


def loss_value(z,target):
    return (nn.functional.binary_cross_entropy_with_logits(z[:,0],(target>0).float())+
            CONFIG["count_loss_weight"]*nn.functional.poisson_nll_loss(z[:,1].clamp(-9,5),target,log_input=True))


def fit(x,users,target,training,validation,seed,embedding_dim,n_users,epochs=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model=DailyNet(x.shape[1],n_users,embedding_dim).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG["learning_rate"],weight_decay=CONFIG["weight_decay"])
    rng=np.random.default_rng(seed)
    best_loss=np.inf
    best_state=None
    best_epoch=0
    for epoch in range(1,(epochs or CONFIG["max_epochs"])+1):
        model.train()
        order=rng.permutation(training)
        for start in range(0,len(order),CONFIG["batch_size"]):
            ids=order[start:start+CONFIG["batch_size"]]
            if torch.any(target[ids]<0):
                raise AssertionError("Masked advertisement targets entered optimization")
            optimizer.zero_grad(set_to_none=True)
            loss=loss_value(model(x[ids],users[ids]),target[ids])
            if not torch.isfinite(loss):
                raise AssertionError("Nonfinite daily response loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
        if epochs is not None:
            continue
        model.eval()
        with torch.no_grad():
            current=float(loss_value(model(x[validation],users[validation]),target[validation]))
        if current<best_loss-1e-6:
            best_loss=current
            best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch%10==0:
            print(json.dumps({"event":"neural_validation","seed":seed,"embedding_dim":embedding_dim,
                              "epoch":epoch,"best_epoch":best_epoch,"loss":current}),flush=True)
        if epoch-best_epoch>=CONFIG["patience"]:
            break
    if epochs is None:
        model.load_state_dict(best_state)
    return model,(epochs or best_epoch)


def row_mapping(data,day):
    raw=pd.read_csv(ROOT / "2026ml/train.csv",usecols=["pt_d","user_id","label"],dtype="int32")
    rows=np.flatnonzero(raw.pt_d.to_numpy()==20191100+day)
    np.testing.assert_array_equal(rows,np.load(REFERENCE / f"day{day}_indices.npy"))
    mask=data["keys"][:,0]==20191100+day
    keys=data["keys"][mask]
    index=pd.MultiIndex.from_arrays([keys[:,0],keys[:,1]])
    groups=index.get_indexer(pd.MultiIndex.from_frame(raw.iloc[rows][["pt_d","user_id"]]))
    if np.any(groups<0):
        raise AssertionError("User-day mapping failed")
    return rows,groups,raw.label.to_numpy()[rows]


def select_powers(group_prediction,y,base,groups):
    result=[]
    for index,objective in enumerate(["binary","poisson"]):
        scores=[{"power":p,"auc":rank_sum_auc(y,correct(base,group_prediction[:,index],groups,objective,p))} for p in CONFIG["powers"]]
        best=max(scores,key=lambda r:r["auc"])
        result.append({"objective":objective,"index":index,"power":best["power"],"validation15_auc":best["auc"]})
    return result


def dependencies():
    return list(dict.fromkeys([Path(__file__),HERE / "feed_behavior_userday.py"]+
                dependency_paths()+[p for p in SOURCES.values() if p is not None]))


def main():
    if OUT.exists():
        raise FileExistsError("Preserving direct daily neural models")
    OUT.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("Daily neural experiment requires the existing CUDA environment")
    protocol={"config":CONFIG,"inputs_sha256":{str(p):digest(p) for p in dependencies()},
              "raw_datasets_sha256":{p.name:digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
              "selection":"Pre15 train,15 early stopping and powers; normalization/user vocabulary rebuilt before16 refit.",
              "training_target":"User-day any-click BCE plus0.25 Poisson count NLL; equal user-day weights",
              "evaluation16_labels_fit":False,"development16_model_selection":True,
              "same_day_news_feedback":True,"sources":list(SOURCES)}
    write(OUT / "protocol.json",protocol)
    data,schema=prepare()
    write(OUT / "news_schema.json",schema)
    print(json.dumps({"event":"neural_features_ready","shape":list(data["x"].shape)}),flush=True)
    normal15=preprocess(data["x"],data["keys"],20191115)
    normal16=preprocess(data["x"],data["keys"],20191116)
    np.savez(OUT / "normalization16.npz",**normal16)
    x15,u15=transform(data["x"],data["keys"],normal15)
    x16,u16=transform(data["x"],data["keys"],normal16)
    x15=torch.from_numpy(x15).cuda();u15=torch.from_numpy(u15).cuda()
    x16=torch.from_numpy(x16).cuda();u16=torch.from_numpy(u16).cuda()
    target=torch.as_tensor(data["counts"],dtype=torch.float32,device="cuda")
    training=np.flatnonzero(data["keys"][:,0]<20191115)
    valid=np.flatnonzero(data["keys"][:,0]==20191115)
    refit=np.flatnonzero(data["keys"][:,0]<20191116)
    _,groups,y=row_mapping(data,15)
    base=np.load(REFERENCE / "day15_prediction.npy")
    locks,group15=[],[]
    for embedding_dim in CONFIG["user_embedding_dims"]:
        for seed in CONFIG["seeds"]:
            name=f"user{embedding_dim}_seed{seed}"
            model,epochs=fit(x15,u15,target,training,valid,seed,embedding_dim,len(normal15["users"]))
            gp=predictions(model,x15,u15,valid)
            np.save(OUT / (name+"_group15.npy"),gp)
            powers=select_powers(gp,y,base,groups)
            group15.append(gp)
            del model
            final,_=fit(x16,u16,target,refit,None,seed,embedding_dim,len(normal16["users"]),epochs=epochs)
            path=OUT / (name+".pt")
            torch.save({k:v.detach().cpu() for k,v in final.state_dict().items()},path)
            lock={"name":name,"seed":seed,"embedding_dim":embedding_dim,"epochs":epochs,
                  "powers":powers,"model_sha256":digest(path)}
            locks.append(lock)
            write(OUT / "selection15.json",locks)
            print(json.dumps(lock),flush=True)
            del final
    ensemble=select_powers(np.mean(group15,axis=0),y,base,groups)
    write(OUT / "models_locked.json",{"models":locks,"ensemble_powers":ensemble,
          "protocol_sha256":digest(OUT / "protocol.json"),"normalization_sha256":digest(OUT / "normalization16.npz"),
          "news_schema_sha256":digest(OUT / "news_schema.json"),"features_sha256":array_hash(data["x"]),
          "counts_sha256":array_hash(data["counts"]),"keys_sha256":array_hash(data["keys"])})
    del data,x15,x16,u15,u16,target
    gc.collect();torch.cuda.empty_cache()
    evaluate(False)


def evaluate(verifying):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    protocol=json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    lock=json.loads((OUT / "models_locked.json").read_text(encoding="utf8"))
    for filename,key in [("protocol.json","protocol_sha256"),("normalization16.npz","normalization_sha256"),("news_schema.json","news_schema_sha256")]:
        if digest(OUT / filename)!=lock[key]:
            raise AssertionError("Neural model configuration changed")
    for p,h in protocol["inputs_sha256"].items():
        if digest(p)!=h:
            raise AssertionError("Neural dependency changed")
    for name,h in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name)!=h:
            raise AssertionError("Original dataset changed")
    data,_=prepare(json.loads((OUT / "news_schema.json").read_text(encoding="utf8")))
    for field,key in [("x","features_sha256"),("counts","counts_sha256"),("keys","keys_sha256")]:
        if array_hash(data[field])!=lock[key]:
            raise AssertionError("Neural input reconstruction differs")
    normalization=dict(np.load(OUT / "normalization16.npz"))
    rebuilt=preprocess(data["x"],data["keys"],20191116)
    for k in rebuilt:
        np.testing.assert_array_equal(rebuilt[k],normalization[k])
    x,users=transform(data["x"],data["keys"],normalization)
    x=torch.from_numpy(x).cuda();users=torch.from_numpy(users).cuda()
    held=np.flatnonzero(data["keys"][:,0]==20191116)
    if np.any(data["counts"][held]!=-1):
        raise AssertionError("Held daily target is not masked")
    rows,groups,y=row_mapping(data,16)
    if len(rows)!=421983 or int(y.sum())!=3724:
        raise AssertionError("Full16 cohort changed")
    group_predictions=[]
    specifications=[]
    for r in lock["models"]:
        path=OUT / (r["name"]+".pt")
        if digest(path)!=r["model_sha256"]:
            raise AssertionError("Neural weights changed")
        model=DailyNet(x.shape[1],len(normalization["users"]),r["embedding_dim"]).cuda()
        model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
        group_predictions.append(predictions(model,x,users,held))
        specifications.append((r["name"],r["powers"]))
        del model
    group_predictions.append(np.mean(group_predictions,axis=0))
    specifications.append(("uniform_ensemble",lock["ensemble_powers"]))
    records,predictions_rows=[],[]
    for gp,(name,powers) in zip(group_predictions,specifications):
        for choice in powers:
            for source,source_path in SOURCES.items():
                base=np.load(REFERENCE / "day16_prediction.npy" if source_path is None else source_path)
                p=correct(base,gp[:,choice["index"]],groups,choice["objective"],choice["power"])
                records.append({"name":name,"source":source,**choice,"auc":rank_sum_auc(y,p)})
                predictions_rows.append(p)
    best=int(np.argmax([r["auc"] for r in records]))
    p=predictions_rows[best]
    if verifying:
        error=float(np.max(np.abs(p-np.load(OUT / "prediction.npy"))))
        if error>1e-10:
            raise AssertionError("Neural independent replay differs")
        np.testing.assert_array_equal(rows,np.load(OUT / "valid_indices.npy"))
    else:
        error=0.
        np.save(OUT / "prediction.npy",p);np.save(OUT / "valid_indices.npy",rows)
    result={"status":"passed" if verifying else "evaluated","auc":records[best]["auc"],"best":records[best],
            "records":records,"rows":len(p),"clicks":3724,"maximum_replay_error":error,
            "original_datasets_unchanged":True,"evaluation16_labels_fit":False,
            "same_day_news_feedback":True,"development16_model_selection":True,"target_met":records[best]["auc"]>=.82}
    write(OUT / ("verification.json" if verifying else "result.json"),result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--verify",action="store_true")
    evaluate(True) if parser.parse_args().verify else main()
