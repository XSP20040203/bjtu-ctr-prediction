"""Out-of-day residual trees with raw categories and history summaries."""
import argparse
import json
from pathlib import Path
import sys

from rank_calibration import ROOT, HERE, FIELDS, features, SOURCES, digest, write, rank_sum_auc
import numpy as np
import pandas as pd
from scipy.special import expit
sys.path.insert(0, str(ROOT/"analysis/auc085_next/deps"))
import lightgbm as lgb

OUT = HERE/"runs/feature_residual_tree_v1"
CATEGORIES = ["age", "sex", "residence", "city", "city_rank", "series_dev", "series_group",
              "device_dev", "device_name", "device_size", "net_type", "creat_type_cd",
              "adv_prim_id", "inter_type_cd", "slot_id", "spread_app_id", "hispace_app_tags",
              "app_second_class", "app_score", "u_refreshTimes", "u_feedLifeCycle"]
HISTORIES = [f"ad_{action}_list_v00{i}" for action in ["click", "close"] for i in [1,2,3]]
HISTORIES += ["u_newsCatInterestsST"]
READ_FIELDS = list(dict.fromkeys(FIELDS+CATEGORIES+HISTORIES+["pt_d"]))
PARAMS = {"objective":"binary", "metric":"auc", "learning_rate":.03,
          "min_data_in_leaf":1000, "lambda_l2":30., "max_bin":63,
          "cat_l2":30., "cat_smooth":50., "verbosity":-1, "num_threads":2,
          "seed":42, "deterministic":True, "force_col_wise":True, "boost_from_average":False}
LEAVES = [7, 15, 31]


def load_rows():
    parts = []
    for block in pd.read_csv(ROOT/"2026ml/train.csv", usecols=READ_FIELDS,
                             dtype={name:str for name in HISTORIES}, chunksize=120000):
        parts.append(block.loc[block.pt_d.isin([20191115,20191116])])
    return pd.concat(parts)


def scalar_and_history(raw):
    if "label" in raw.columns:
        raise ValueError("Outcomes are forbidden in feature construction")
    blocks = [raw[CATEGORIES].to_numpy(dtype="float64")]
    names = list(CATEGORIES)
    for column in HISTORIES:
        cache = {s:tuple(int(t) for t in str(s).split("^") if t and t != "-1")
                 for s in raw[column].unique()}
        tokens = [cache[s] for s in raw[column]]
        blocks.append(np.array([len(t) for t in tokens], dtype="float64")[:,None])
        names.append(column+"_length")
        if column.startswith("ad_"):
            target = ["task_id", "adv_prim_id", "spread_app_id"][int(column[-1])-1]
            match = np.array([int(value) in t for value,t in zip(raw[target],tokens)],dtype="float64")
            blocks.append(match[:,None])
            names.append(column+"_candidate_match")
    return np.column_stack(blocks), names


def matrix(raw, day, active):
    # Preserve the original full-CSV index for reference alignment.
    day_raw = raw.loc[raw.pt_d == 20191100+day]
    from group_score_calibration import REFERENCE, group_features
    rows = day_raw.index.to_numpy()
    np.testing.assert_array_equal(rows, np.load(REFERENCE/f"day{day}_indices.npy"))
    reference = np.load(REFERENCE/f"day{day}_prediction.npy")
    from scipy.special import logit
    base = logit(reference.clip(1e-6, 1-1e-6))
    group, group_names = group_features(day_raw[FIELDS], reference)
    extra, extra_names = scalar_and_history(day_raw)
    blocks, names = [base[:,None],group,extra], ["reference_logit"]+group_names+extra_names
    for name, pattern in active.items():
        p = np.load(ROOT/pattern.format(day=day))
        if p.shape != base.shape or not np.isfinite(p).all():
            raise AssertionError("Prediction source alignment")
        blocks.append((logit(p.clip(1e-6, 1-1e-6))-base)[:,None])
        names.append(name+"_difference")
    return np.column_stack(blocks), base, rows, names, reference, day_raw.user_id.to_numpy()


def main():
    if OUT.exists():
        raise FileExistsError("Preserving feature residual tree")
    OUT.mkdir(parents=True)
    active = {name:p for name,p in SOURCES.items() if all((ROOT/p.format(day=d)).is_file() for d in [15,16])}
    from group_score_calibration import REFERENCE
    inputs = [REFERENCE/f"day{d}_prediction.npy" for d in [15,16]]
    inputs += [ROOT/p.format(day=d) for p in active.values() for d in [15,16]]
    protocol = {"fit_date":20191115,"evaluation_date":20191116,"params":PARAMS,"leaves":LEAVES,
                "sources":active,"max_rounds":400,"patience":50,"categories":CATEGORIES,
                "source_sha256":{str(p):digest(p) for p in [Path(__file__),HERE/"rank_calibration.py",HERE/"group_score_calibration.py"]},
                "input_sha256":{str(p):digest(p) for p in inputs},
                "raw_datasets_sha256":{p.name:digest(p) for p in (ROOT/"2026ml").glob("*.csv")},
                "selection":"Three user folds on15 only; median best iteration for refit",
                "evaluation16_labels_fit":False,"development_day_reused":True}
    write(OUT/"protocol.json",protocol)
    raw = load_rows()
    x,base,rows,names,reference,users = matrix(raw,15,active)
    labels = pd.read_csv(ROOT/"2026ml/train.csv",usecols=["pt_d","label"],dtype="int32")
    y = labels.label.to_numpy()[rows].copy()
    del labels
    categorical = [names.index(n) for n in CATEGORIES]
    folds = ((users.astype("uint64")*2654435761)%3).astype(int)
    records = []
    for leaves in LEAVES:
        oof,rounds = np.empty(len(y)),[]
        for fold in range(3):
            tr,va = folds != fold,folds == fold
            training = lgb.Dataset(x[tr],label=y[tr],init_score=base[tr],categorical_feature=categorical,feature_name=names)
            validation = lgb.Dataset(x[va],label=y[va],init_score=base[va],reference=training)
            model = lgb.train(dict(PARAMS,num_leaves=leaves),training,num_boost_round=400,valid_sets=[validation],
                              callbacks=[lgb.early_stopping(50,verbose=False)])
            rounds.append(model.best_iteration)
            oof[va] = expit(base[va]+np.clip(model.predict(x[va],raw_score=True),-2,2))
        record = {"leaves":leaves,"rounds":rounds,"oof_auc":rank_sum_auc(y,oof)}
        records.append(record)
        np.save(OUT/f"leaves{leaves}_oof15.npy",oof)
        write(OUT/"cv.json",records)
        print(json.dumps(record),flush=True)
    chosen = max(records,key=lambda r:r["oof_auc"])
    rounds = max(1,int(np.median(chosen["rounds"])))
    dataset = lgb.Dataset(x,label=y,init_score=base,categorical_feature=categorical,feature_name=names)
    model = lgb.train(dict(PARAMS,num_leaves=chosen["leaves"]),dataset,num_boost_round=rounds)
    (OUT/"model.txt").write_text(model.model_to_string(),encoding="utf8")
    write(OUT/"weights_locked.json",{"model_sha256":digest(OUT/"model.txt"),"protocol_sha256":digest(OUT/"protocol.json"),
          "selected":chosen,"refit_rounds":rounds,"baseline15_auc":rank_sum_auc(y,reference),"feature_names":names,
          "evaluation16_labels_fit":False})
    x16,base16,rows16,_,_,_ = matrix(raw,16,active)
    p = expit(base16+np.clip(model.predict(x16,raw_score=True),-2,2))
    np.save(OUT/"prediction.npy",p)
    np.save(OUT/"valid_indices.npy",rows16)
    y16 = pd.read_csv(ROOT/"2026ml/train.csv",usecols=["label"],dtype="int32").label.to_numpy()[rows16]
    result = {"auc":rank_sum_auc(y16,p),"rows":len(y16),"clicks":int(y16.sum()),"target_met":rank_sum_auc(y16,p)>=.82}
    write(OUT/"result.json",result)
    print(json.dumps(result,indent=2),flush=True)


def verify():
    protocol = json.loads((OUT/"protocol.json").read_text(encoding="utf8"))
    lock = json.loads((OUT/"weights_locked.json").read_text(encoding="utf8"))
    for field in ["source_sha256","input_sha256"]:
        for path,expected in protocol[field].items():
            if digest(path) != expected:
                raise AssertionError("Frozen source or input changed")
    for name,expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT/"2026ml"/name) != expected:
            raise AssertionError("Original CSV changed")
    if digest(OUT/"model.txt") != lock["model_sha256"] or digest(OUT/"protocol.json") != lock["protocol_sha256"]:
        raise AssertionError("Frozen model or protocol changed")
    x,base,rows,names,_,_ = matrix(load_rows(),16,protocol["sources"])
    if names != lock["feature_names"]:
        raise AssertionError("Feature schema changed")
    np.testing.assert_array_equal(rows,np.load(OUT/"valid_indices.npy"))
    model = lgb.Booster(model_str=(OUT/"model.txt").read_text(encoding="utf8"))
    p = expit(base+np.clip(model.predict(x,raw_score=True),-2,2))
    error = float(np.max(np.abs(p-np.load(OUT/"prediction.npy"))))
    if error > 1e-12:
        raise AssertionError("Model replay differs")
    y = pd.read_csv(ROOT/"2026ml/train.csv",usecols=["label"],dtype="int32").label.to_numpy()[rows]
    if len(y) != 421983 or int(y.sum()) != 3724:
        raise AssertionError("Full16 cohort changed")
    result = {"status":"passed","auc":rank_sum_auc(y,p),"rows":len(y),"clicks":int(y.sum()),
              "maximum_reload_error":error,"original_datasets_unchanged":True,
              "evaluation16_labels_fit":False,"target_met":rank_sum_auc(y,p)>=.82}
    write(OUT/"verification.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify",action="store_true")
    verify() if parser.parse_args().verify else main()
