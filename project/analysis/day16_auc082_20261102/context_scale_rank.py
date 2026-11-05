"""Learn daily score offset and score spread with a row-pair ranking loss."""
import os
for _key in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"]:
    os.environ[_key]="2"
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from context_rank_boost import matrix,dependencies as rank_dependencies,SOURCES as RANK_SOURCES
from rich_group_residual import ROOT,HERE,REFERENCE,digest,write,lgb,array_hash,rank_sum_auc,aggregate

OUT=HERE / "runs/context_scale_rank_v1"
SOURCES={**RANK_SOURCES,"context_rank":HERE / "runs/context_rank_boost_v1/prediction.npy"}
PARAMS={"learning_rate":.025,"min_data_in_leaf":200,"lambda_l2":5.,"feature_fraction":.7,
        "max_bin":31,"num_threads":2,"seed":42,"verbosity":-1,"deterministic":True,
        "force_col_wise":True,"boost_from_average":False,"metric":"None","max_delta_step":.3}
POWERS=[0.,.25,.5,.75,1.]
RIDGES=[.1,1.]
LEAVES=[7,15]


def score_components(data):
    base=logit(data["base"].clip(1e-6,1-1e-6))
    size=np.bincount(data["groups"],minlength=len(data["keys"]))
    mean=np.bincount(data["groups"],weights=base,minlength=len(size))/size.clip(1)
    return base,np.clip((base-mean[data["groups"]])/2.,-3,3)


def augmented(x):
    return np.vstack([np.column_stack([x,np.zeros(len(x))]),np.column_stack([x,np.ones(len(x))])]).astype("float32")


class ScaleObjective:
    def __init__(self,a,b,za,zb,margin,n,ridge):
        if len(a)==0 or np.any(a==b):
            raise ValueError("Distinct nonempty training user-day pairs required")
        self.a,self.b,self.za,self.zb,self.margin=a,b,za,zb,margin
        self.n,self.ridge=n,ridge

    def penalty(self):
        return np.r_[np.full(self.n,.02),np.full(self.n,self.ridge)]

    def value(self,score):
        delta=self.margin+score[self.a]-score[self.b]+self.za*score[self.n+self.a]-self.zb*score[self.n+self.b]
        return self.n*np.logaddexp(0.,-delta).mean()+.5*np.dot(self.penalty()*score,score)

    def __call__(self,score,dataset):
        delta=self.margin+score[self.a]-score[self.b]+self.za*score[self.n+self.a]-self.zb*score[self.n+self.b]
        prob=expit(-delta)
        h=prob*(1-prob)
        scale=self.n/len(self.a)
        grad=np.r_[np.bincount(self.b,weights=prob,minlength=self.n)-np.bincount(self.a,weights=prob,minlength=self.n),
                   np.bincount(self.b,weights=prob*self.zb,minlength=self.n)-np.bincount(self.a,weights=prob*self.za,minlength=self.n)]*scale
        hess=np.r_[np.bincount(self.a,weights=h,minlength=self.n)+np.bincount(self.b,weights=h,minlength=self.n),
                   np.bincount(self.a,weights=h*self.za**2,minlength=self.n)+np.bincount(self.b,weights=h*self.zb**2,minlength=self.n)]*scale
        return grad+self.penalty()*score,hess+self.penalty()


def objective(data,groups,ridge,seed):
    mapping=np.full(len(data["keys"]),-1,dtype="int32")
    mapping[groups]=np.arange(len(groups))
    eligible=mapping[data["groups"]]>=0
    positives=np.flatnonzero(eligible&(data["y"]==1))
    negatives=np.flatnonzero(eligible&(data["y"]==0))
    if not len(positives) or not len(negatives):
        raise ValueError("Both outcome classes required")
    rng=np.random.default_rng(seed)
    pos=np.repeat(positives,64)
    neg=rng.choice(negatives,len(pos))
    a,b=mapping[data["groups"][pos]],mapping[data["groups"][neg]]
    keep=a!=b
    base,z=score_components(data)
    return ScaleObjective(a[keep],b[keep],z[pos[keep]],z[neg[keep]],(base[pos]-base[neg])[keep],len(groups),ridge)


def row_scores(data,groups,head_scores,power=1.):
    mapping=np.full(len(data["keys"]),-1,dtype="int32")
    mapping[groups]=np.arange(len(groups))
    selected=mapping[data["groups"]]>=0
    positions=mapping[data["groups"][selected]]
    base,z=score_components(data)
    offset=np.clip(head_scores[:len(groups)],-3,3)
    slope=np.clip(head_scores[len(groups):],-.75,.75)
    return selected,base[selected]+power*(offset[positions]+slope[positions]*z[selected])


def metric(data,groups):
    def evaluate(scores,dataset):
        rows,p=row_scores(data,groups,scores)
        return "row_auc",rank_sum_auc(data["y"][rows],p),True
    return evaluate


def dependencies():
    return list(dict.fromkeys([Path(__file__),HERE / "context_rank_boost.py"]+rank_dependencies()+[
        HERE / "runs/context_rank_boost_v1/prediction.npy",HERE / "runs/context_rank_boost_v1/verification.json"]))


def main():
    if OUT.exists():
        raise FileExistsError("Preserving daily scale ranking experiment")
    OUT.mkdir(parents=True)
    protocol={"params":PARAMS,"ridges":RIDGES,"leaves":LEAVES,"powers":POWERS,
              "max_rounds":400,"patience":60,"slope_clip":[-.75,.75],"offset_clip":[-3,3],
              "inputs_sha256":{str(p):digest(p) for p in dependencies()},
              "raw_datasets_sha256":{p.name:digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
              "selection":"User3fold15 selects rounds and correction power; four models locked before16 comparison.",
              "formula":"logit(base)+power*(offset+clip(slope,-.75,.75)*clip((logit(base)-user_day_mean)/2,-3,3))",
              "evaluation16_labels_fit":False,"development16_model_selection":True,"sources":list(SOURCES)}
    write(OUT / "protocol.json",protocol)
    data=matrix(15,include_labels=True)
    folds=((data["keys"][:,1].astype("uint64")*2654435761)%3).astype("int32")
    records,locks=[],[]
    print(json.dumps({"event":"scale_rank_ready","shape":list(data["x"].shape)}),flush=True)
    for ridge in RIDGES:
        for leaves in LEAVES:
            oof=np.empty(2*len(data["keys"]))
            rounds=[]
            for fold in range(3):
                tr,va=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
                obj=objective(data,tr,ridge,700+fold)
                train=lgb.Dataset(augmented(data["x"][tr]),label=np.zeros(2*len(tr)))
                valid=lgb.Dataset(augmented(data["x"][va]),label=np.zeros(2*len(va)),reference=train)
                model=lgb.train(dict(PARAMS,num_leaves=leaves,objective=obj),train,num_boost_round=400,
                                valid_sets=[valid],feval=metric(data,va),callbacks=[lgb.early_stopping(60,verbose=False)])
                head=model.predict(augmented(data["x"][va]),raw_score=True)
                oof[va]=head[:len(va)]
                oof[len(data["keys"])+va]=head[len(va):]
                rounds.append(model.best_iteration)
            scores=[]
            for power in POWERS:
                _,p=row_scores(data,np.arange(len(data["keys"])),oof,power)
                scores.append({"power":power,"auc":rank_sum_auc(data["y"],p)})
            best=max(scores,key=lambda r:r["auc"])
            r={"name":f"ridge{ridge:g}_l{leaves}","ridge":ridge,"leaves":leaves,"rounds":rounds,
               "refit_rounds":max(1,int(np.median(rounds))),"power":best["power"],"oof15_auc":best["auc"],"scores":scores}
            records.append(r)
            np.save(OUT / (r["name"]+"_oof.npy"),oof)
            write(OUT / "cv.json",records)
            print(json.dumps(r),flush=True)
    for r in records:
        obj=objective(data,np.arange(len(data["keys"])),r["ridge"],700)
        model=lgb.train(dict(PARAMS,num_leaves=r["leaves"],objective=obj),
                        lgb.Dataset(augmented(data["x"]),label=np.zeros(2*len(data["keys"]))),num_boost_round=r["refit_rounds"])
        path=OUT / (r["name"]+".txt")
        path.write_text(model.model_to_string(),encoding="utf8")
        locks.append(dict(r,model_sha256=digest(path),pairs_sha256=array_hash(np.column_stack([obj.a,obj.b,obj.za,obj.zb,obj.margin]))))
    write(OUT / "models_locked.json",{"models":locks,"protocol_sha256":digest(OUT / "protocol.json"),
          "x_sha256":array_hash(data["x"]),"y_sha256":array_hash(data["y"]),"keys_sha256":array_hash(data["keys"])})
    del data
    gc.collect()
    evaluate(False)


def evaluate(verifying):
    protocol=json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    lock=json.loads((OUT / "models_locked.json").read_text(encoding="utf8"))
    if digest(OUT / "protocol.json")!=lock["protocol_sha256"]:
        raise AssertionError("Scale model protocol changed")
    for p,h in protocol["inputs_sha256"].items():
        if digest(p)!=h:
            raise AssertionError("Scale rank dependency changed")
    for name,h in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name)!=h:
            raise AssertionError("Original data changed")
    if verifying:
        data=matrix(15,include_labels=True)
        for key in ["x","y","keys"]:
            if array_hash(data[key])!=lock[key+"_sha256"]:
                raise AssertionError("Scale training reconstruction differs")
        for r in lock["models"]:
            obj=objective(data,np.arange(len(data["keys"])),r["ridge"],700)
            if array_hash(np.column_stack([obj.a,obj.b,obj.za,obj.zb,obj.margin]))!=r["pairs_sha256"]:
                raise AssertionError("Scale pair sample differs")
        del data
        gc.collect()
    data=matrix(16,include_labels=True)
    if len(data["y"])!=421983 or int(data["y"].sum())!=3724 or np.any(data["count"]!=-1):
        raise AssertionError("Full16 scale target boundary failed")
    records,predictions=[],[]
    all_groups=np.arange(len(data["keys"]))
    for source,path in SOURCES.items():
        if path is not None:
            data["base"]=np.load(path)
            data["mass"],data["any"],block=aggregate(data["base"],data["groups"],len(data["keys"]))
            start=data["names"].index("base_log_size")
            data["x"][:,start:start+4]=block
        for r in lock["models"]:
            path=OUT / (r["name"]+".txt")
            if digest(path)!=r["model_sha256"]:
                raise AssertionError("Scale rank model changed")
            model=lgb.Booster(model_str=path.read_text(encoding="utf8"))
            head=model.predict(augmented(data["x"]),raw_score=True)
            _,p=row_scores(data,all_groups,head,r["power"])
            p=expit(p)
            records.append({"name":r["name"],"source":source,"power":r["power"],"oof15_auc":r["oof15_auc"],"auc":rank_sum_auc(data["y"],p)})
            predictions.append(p)
    best=int(np.argmax([r["auc"] for r in records]))
    p=predictions[best]
    if verifying:
        error=float(np.max(np.abs(p-np.load(OUT / "prediction.npy"))))
        if error>1e-12:
            raise AssertionError("Scale rank replay differs")
        np.testing.assert_array_equal(data["rows"],np.load(OUT / "valid_indices.npy"))
    else:
        error=0.
        np.save(OUT / "prediction.npy",p);np.save(OUT / "valid_indices.npy",data["rows"])
    result={"status":"passed" if verifying else "evaluated","auc":records[best]["auc"],"best":records[best],
            "records":records,"rows":len(p),"clicks":3724,"maximum_replay_error":error,
            "original_datasets_unchanged":True,"evaluation16_labels_fit":False,"development16_model_selection":True,
            "target_met":records[best]["auc"]>=.82}
    write(OUT / ("verification.json" if verifying else "result.json"),result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--verify",action="store_true")
    evaluate(True) if parser.parse_args().verify else main()
