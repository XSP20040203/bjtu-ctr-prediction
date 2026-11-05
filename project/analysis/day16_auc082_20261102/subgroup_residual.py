"""Out-of-day residual calibration for user-slot/app/task subgroups."""
import os
for _key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[_key] = "2"
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from rich_group_residual import (ROOT, HERE, REFERENCE, digest, write, prepare, aggregate,
                                 array_hash, rank_sum_auc, lgb)
from rich_userday_activity import correct

OUT = HERE / "runs/subgroup_residual_v1"
GROUP_FIELDS = ["slot_id", "spread_app_id", "task_id"]
READ_FIELDS = ["pt_d", "user_id", "slot_id", "spread_app_id", "task_id", "label"]
PARAMS = {"learning_rate": .03, "min_data_in_leaf": 120, "lambda_l2": 15.,
          "max_bin": 31, "num_threads": 2, "seed": 42, "verbosity": -1,
          "deterministic": True, "force_col_wise": True, "boost_from_average": False,
          "cat_l2": 20., "cat_smooth": 30.}
LEAVES = [7, 15]
POWERS = [0., .25, .5, .75, 1.]


def subgroup_matrix(day, field, selected, source="reference", evaluation_labels=False):
    user = prepare(day, include_labels=False)
    frames = []
    for chunk in pd.read_csv(ROOT / "2026ml/train.csv", usecols=READ_FIELDS, dtype="int32", chunksize=200000):
        frames.append(chunk.loc[chunk.pt_d.to_numpy() == 20191100+day])
    raw = pd.concat(frames)
    np.testing.assert_array_equal(raw.index.to_numpy(), user["rows"])
    # No label is passed to feature aggregation.
    key_frame = raw[["user_id", field]]
    codes, uniques = pd.factorize(pd.MultiIndex.from_frame(key_frame), sort=True)
    group_users = np.array([k[0] for k in uniques], dtype="int64")
    group_ids = np.array([k[1] for k in uniques], dtype="int64")
    user_index = pd.Index(user["keys"][:,1]).get_indexer(group_users)
    if np.any(user_index < 0):
        raise AssertionError("Subgroup user mapping failed")
    if source == "reference":
        base = user["base"]
    elif source == "weighted_transfer":
        base = np.load(HERE / "runs/weighted_calibration_transfer_v1/best_prediction.npy")
    elif source == "userday_residual":
        base = np.load(HERE / "runs/rich_group_residual_v1/prediction.npy")
    else:
        raise ValueError("Unsupported source")
    size = np.bincount(codes, minlength=len(uniques))
    user_size = np.bincount(user["groups"], minlength=len(user["keys"]))
    mass, any_probability, base_features = aggregate(base, codes, len(uniques))
    user_mass, _, _ = aggregate(base, user["groups"], len(user["keys"]))
    app_count = raw.groupby(["user_id", field], sort=True).size().groupby(level=0).size().reindex(group_users).to_numpy()
    extra = np.column_stack([np.log1p(app_count), size/user_size[user_index],
                             mass/user_mass[user_index].clip(1e-8)])
    # A joint unlabeled vocabulary is permitted; integer magnitude carries no meaning.
    all_ids = pd.read_csv(ROOT / "2026ml/train.csv", usecols=[field], dtype="int32")[field]
    vocabulary = pd.Index(np.sort(all_ids.unique()))
    categorical = vocabulary.get_indexer(group_ids)
    x = np.column_stack([user["x"][user_index][:, selected], base_features, extra, categorical]).astype("float32")
    result = {"x":x, "users":group_users, "group_ids":group_ids, "groups":codes,
              "rows":user["rows"], "base":base, "mass":mass, "any":any_probability,
              "sizes":size, "categorical":[x.shape[1]-1]}
    if day == 15:
        result["count"] = np.bincount(codes, weights=raw.label.to_numpy(), minlength=len(uniques))
    else:
        result["count"] = np.full(len(uniques), -1.)
    if evaluation_labels:
        result["y"] = raw.label.to_numpy().copy()
    if x.shape[0] != len(uniques) or not np.isfinite(x).all() or int(size.sum()) != len(base):
        raise AssertionError("Invalid subgroup feature matrix")
    np.testing.assert_array_equal(group_users[codes], raw.user_id.to_numpy())
    np.testing.assert_array_equal(group_ids[codes], raw[field].to_numpy())
    del user, raw, frames, all_ids
    gc.collect()
    return result


def target_offset(data, objective):
    if objective == "binary":
        return (data["count"] > 0).astype("float64"), logit(data["any"].clip(1e-6,1-1e-6))
    return data["count"], np.log(data["mass"].clip(1e-8))


def predict(model, data, objective, groups=None):
    groups = np.arange(len(data["users"])) if groups is None else groups
    _, offset = target_offset(data, objective)
    score = offset[groups]+np.clip(model.predict(data["x"][groups], raw_score=True), -3, 3)
    return expit(score) if objective == "binary" else np.exp(score)


def dependencies():
    return [Path(__file__), HERE / "rich_group_residual.py", HERE / "rich_userday_activity.py",
            HERE / "fit_stacker.py", HERE / "verify_stacker.py",
            HERE / "runs/rich_group_residual_v1/poisson_7.txt",
            HERE / "runs/rich_group_residual_v1/prediction.npy",
            HERE / "runs/rich_group_residual_v1/verification.json",
            HERE / "runs/weighted_v1_day16/context.npz", HERE / "runs/weighted_v1_day16/context_manifest.json",
            HERE / "runs/weighted_calibration_transfer_v1/best_prediction.npy"] + [
                REFERENCE / f"day{day}_{suffix}.npy" for day in [15,16] for suffix in ["indices","prediction"]]


def main():
    if OUT.exists():
        raise FileExistsError("Preserving existing subgroup experiment")
    OUT.mkdir(parents=True)
    selector = lgb.Booster(model_str=(HERE / "runs/rich_group_residual_v1/poisson_7.txt").read_text(encoding="utf8"))
    # Select context features with the existing15-only model. Never use16 targets.
    gain = selector.feature_importance(importance_type="gain")[:-4]
    selected = np.sort(np.argsort(-gain, kind="stable")[:80]).tolist()
    protocol = {"selected_context_columns":selected, "params":PARAMS, "group_fields":GROUP_FIELDS,
                "objectives":["binary", "poisson"], "leaves":LEAVES, "powers":POWERS,
                "max_rounds":350, "patience":50,
                "inputs_sha256":{str(p):digest(p) for p in dependencies()},
                "raw_datasets_sha256":{p.name:digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
                "fit_date":20191115, "evaluation_date":20191116, "evaluation16_labels_fit":False,
                "selection":"Day15 three user folds; median iterations and row AUC power. All12 refits locked before16 evaluation.",
                "development16_model_selection":True}
    write(OUT / "protocol.json", protocol)
    records, locks = [], []
    for field in GROUP_FIELDS:
        data = subgroup_matrix(15, field, selected, evaluation_labels=True)
        folds = ((data["users"].astype("uint64")*2654435761)%3).astype("int32")
        for objective in protocol["objectives"]:
            target, offset = target_offset(data, objective)
            params = dict(PARAMS, objective=objective, metric="binary_logloss" if objective == "binary" else "poisson")
            for leaves in LEAVES:
                oof, rounds = np.empty(len(target)), []
                for fold in range(3):
                    tr, va = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                    train = lgb.Dataset(data["x"][tr], label=target[tr], init_score=offset[tr], categorical_feature=data["categorical"])
                    valid = lgb.Dataset(data["x"][va], label=target[va], init_score=offset[va], categorical_feature=data["categorical"], reference=train)
                    model = lgb.train(dict(params,num_leaves=leaves), train, num_boost_round=350,
                                      valid_sets=[valid], callbacks=[lgb.early_stopping(50,verbose=False)])
                    oof[va] = predict(model,data,objective,va)
                    rounds.append(model.best_iteration)
                scores = [{"power":p,"auc":rank_sum_auc(data["y"],correct(data["base"],oof,data["groups"],objective,p))} for p in POWERS]
                best = max(scores,key=lambda r:r["auc"])
                record = {"name":f"{field}_{objective}_{leaves}","field":field,"objective":objective,"leaves":leaves,
                          "rounds":rounds,"refit_rounds":max(1,int(np.median(rounds))),"power":best["power"],
                          "oof15_auc":best["auc"],"scores":scores}
                records.append(record)
                np.save(OUT / (record["name"]+"_oof.npy"),oof)
                write(OUT / "cv.json",records)
                print(json.dumps(record),flush=True)
                model = lgb.train(dict(params,num_leaves=leaves),lgb.Dataset(data["x"],label=target,init_score=offset,categorical_feature=data["categorical"]),
                                  num_boost_round=record["refit_rounds"])
                path = OUT / (record["name"]+".txt")
                path.write_text(model.model_to_string(),encoding="utf8")
                locks.append(dict(record,model_sha256=digest(path),training_features_sha256=array_hash(data["x"]),
                                  training_counts_sha256=array_hash(data["count"]),training_users_sha256=array_hash(data["users"])))
        del data
        gc.collect()
    write(OUT / "models_locked.json",{"models":locks,"protocol_sha256":digest(OUT / "protocol.json")})
    evaluate(False)


def evaluate(verifying):
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    lock = json.loads((OUT / "models_locked.json").read_text(encoding="utf8"))
    if lock["protocol_sha256"] != digest(OUT / "protocol.json"):
        raise AssertionError("Protocol changed")
    for path,expected in protocol["inputs_sha256"].items():
        if digest(path) != expected:
            raise AssertionError("Frozen source or input changed")
    for name,expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name) != expected:
            raise AssertionError("Original data changed")
    records,predictions = [],[]
    for field in GROUP_FIELDS:
        if verifying:
            train = subgroup_matrix(15,field,protocol["selected_context_columns"])
            for item in [r for r in lock["models"] if r["field"] == field]:
                for data_name,lock_name in [("x","training_features_sha256"),("count","training_counts_sha256"),("users","training_users_sha256")]:
                    if array_hash(train[data_name]) != item[lock_name]:
                        raise AssertionError("Training reconstruction changed")
            del train
        for source in ["reference","weighted_transfer","userday_residual"]:
            data = subgroup_matrix(16,field,protocol["selected_context_columns"],source,evaluation_labels=True)
            if len(data["y"]) != 421983 or int(data["y"].sum()) != 3724 or np.any(data["count"] != -1):
                raise AssertionError("Held-out labels/cohort invalid")
            for item in [r for r in lock["models"] if r["field"] == field]:
                path = OUT / (item["name"]+".txt")
                if digest(path) != item["model_sha256"]:
                    raise AssertionError("Model changed")
                model = lgb.Booster(model_str=path.read_text(encoding="utf8"))
                group_prediction = predict(model,data,item["objective"])
                p = correct(data["base"],group_prediction,data["groups"],item["objective"],item["power"])
                records.append({"name":item["name"],"source":source,"power":item["power"],"oof15_auc":item["oof15_auc"],"auc":rank_sum_auc(data["y"],p)})
                predictions.append(p)
    best = int(np.argmax([r["auc"] for r in records]))
    p = predictions[best]
    if verifying:
        error = float(np.max(np.abs(p-np.load(OUT / "prediction.npy"))))
        if error > 1e-12:
            raise AssertionError("Independent replay differs")
        np.testing.assert_array_equal(data["rows"],np.load(OUT / "valid_indices.npy"))
    else:
        error=0.
        np.save(OUT / "prediction.npy",p)
        np.save(OUT / "valid_indices.npy",data["rows"])
    result = {"status":"passed" if verifying else "evaluated","auc":records[best]["auc"],"best":records[best],
              "records":records,"rows":len(p),"clicks":3724,"maximum_replay_error":error,
              "original_datasets_unchanged":True,"evaluation16_labels_fit":False,"development16_model_selection":True,
              "target_met":records[best]["auc"] >= .82}
    write(OUT / ("verification.json" if verifying else "result.json"),result)
    print(json.dumps({k:v for k,v in result.items() if k != "records"},indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--verify",action="store_true")
    parser.add_argument("--check-inputs",action="store_true")
    args=parser.parse_args()
    if args.check_inputs:
        for field in GROUP_FIELDS:
            data=subgroup_matrix(15,field,list(range(80)),evaluation_labels=True)
            print(json.dumps({"field":field,"groups":len(data["users"]),"shape":data["x"].shape,
                              "rows":len(data["y"]),"group_clicks":int(data["count"].sum()),
                              "row_clicks":int(data["y"].sum())}),flush=True)
    else:
        evaluate(True) if args.verify else main()
