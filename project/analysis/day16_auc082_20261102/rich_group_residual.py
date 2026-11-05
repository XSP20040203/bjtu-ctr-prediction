"""Correct held-day group prediction errors using prior-day true outcomes only.

Fits six small residual trees on day15 user groups, using a label-free context
schema fitted before16. Day15 group cross-validation selects rounds/power.
"""
import os
for _key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[_key] = "2"
import argparse
import gc
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from fit_stacker import ROOT, HERE, REFERENCE, digest, write
from verify_stacker import rank_sum_auc
from rich_userday_activity import load_day, array_hash, correct
sys.path.insert(0, str(ROOT / "analysis/auc085_next/deps"))
import lightgbm as lgb

OUT = HERE / "runs/rich_group_residual_v1"
PARAMS = {"learning_rate": .03, "min_data_in_leaf": 100, "lambda_l2": 10.,
          "feature_fraction": .8, "max_bin": 31, "num_threads": 2,
          "seed": 42, "verbosity": -1, "deterministic": True,
          "force_col_wise": True, "boost_from_average": False}
LEAVES = [3, 7, 15]
POWERS = [0., .25, .5, .75, 1.]


def aggregate(base, groups, n_groups):
    mass = np.bincount(groups, weights=base, minlength=n_groups)
    hazard = np.bincount(groups, weights=-np.log1p(-base.clip(1e-8, 1-1e-8)), minlength=n_groups)
    maximum = np.zeros(n_groups)
    np.maximum.at(maximum, groups, base)
    size = np.bincount(groups, minlength=n_groups)
    features = np.column_stack([np.log1p(size), np.log(mass.clip(1e-8)),
                                logit((mass/size.clip(1)).clip(1e-6, 1-1e-6)),
                                logit(maximum.clip(1e-6, 1-1e-6))]).astype("float32")
    return mass, -np.expm1(-hazard), features


def prepare(day, source="reference", include_labels=False):
    context = load_day(16, True, evaluation_labels=False)
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["pt_d", "user_id"])
    rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
    np.testing.assert_array_equal(rows, np.load(REFERENCE / f"day{day}_indices.npy"))
    group_keys = context["keys"][context["keys"][:, 0] == 20191100+day]
    key_index = pd.MultiIndex.from_arrays([group_keys[:, 0], group_keys[:, 1]])
    groups = key_index.get_indexer(pd.MultiIndex.from_frame(raw.iloc[rows][["pt_d", "user_id"]]))
    if np.any(groups < 0):
        raise AssertionError("Group/row mismatch")
    if source == "reference":
        base = np.load(REFERENCE / f"day{day}_prediction.npy").astype("float64")
    elif source == "weighted_transfer" and day == 16:
        base = np.load(HERE / "runs/weighted_calibration_transfer_v1/best_prediction.npy")
    else:
        raise ValueError("Invalid base source")
    mask = context["keys"][:, 0] == 20191100+day
    mass, any_probability, base_features = aggregate(base, groups, len(group_keys))
    x = np.column_stack([context["x"][mask], base_features]).astype("float32")
    result = {"x": x, "keys": group_keys, "groups": groups, "rows": rows,
              "base": base, "mass": mass, "any": any_probability,
              "count": context["counts"][mask],
              "names": context["names"]+["base_log_size", "base_log_mass", "base_mean_logit", "base_max_logit"]}
    if include_labels:
        result["y"] = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["label"], dtype="int8").label.to_numpy()[rows]
    del context, raw
    gc.collect()
    return result


def target_offset(data, objective):
    if objective == "binary":
        return (data["count"] > 0).astype("float64"), logit(data["any"].clip(1e-6, 1-1e-6))
    return data["count"].copy(), np.log(data["mass"].clip(1e-8))


def prediction(model, data, objective, rows=None):
    rows = np.arange(len(data["keys"])) if rows is None else rows
    _, offset = target_offset(data, objective)
    residual = np.clip(model.predict(data["x"][rows], raw_score=True), -3., 3.)
    return expit(offset[rows]+residual) if objective == "binary" else np.exp(offset[rows]+residual)


def paths():
    return [Path(__file__), HERE / "rich_userday_activity.py", HERE / "fit_stacker.py", HERE / "verify_stacker.py",
            HERE / "runs/weighted_v1_day16/context.npz", HERE / "runs/weighted_v1_day16/context_manifest.json",
            HERE / "runs/weighted_calibration_transfer_v1/best_prediction.npy",
            HERE / "runs/weighted_calibration_transfer_v1/verification.json"] + [
                REFERENCE / f"day{day}_{suffix}.npy" for day in [15,16] for suffix in ["indices", "prediction"]]


def main():
    if OUT.exists():
        raise FileExistsError("Preserving existing residual experiment")
    OUT.mkdir(parents=True)
    protocol = {"fit_date": 20191115, "evaluation_date": 20191116, "params": PARAMS,
                "objectives": ["binary", "poisson"], "leaves": LEAVES, "powers": POWERS,
                "max_rounds": 400, "patience": 50,
                "inputs_sha256": {str(p): digest(p) for p in paths()},
                "raw_datasets_sha256": {p.name: digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
                "selection": "Day15 user-group3fold chooses median rounds and row-AUC correction power; all6 refits locked before day16 comparison",
                "schema": "Use common outcome-free pre16 scaler/schema for15 and16; histories queried before each day is inserted",
                "evaluation16_labels_fit": False, "development16_model_selection": True}
    write(OUT / "protocol.json", protocol)
    data = prepare(15, include_labels=True)
    if np.any(data["count"] < 0):
        raise AssertionError("Training target unavailable")
    folds = ((data["keys"][:, 1].astype("uint64")*2654435761)%3).astype("int32")
    records, locks = [], []
    for objective in protocol["objectives"]:
        target, offset = target_offset(data, objective)
        params = dict(PARAMS, objective=objective, metric="binary_logloss" if objective == "binary" else "poisson")
        for leaves in LEAVES:
            oof, rounds = np.empty(len(target)), []
            for fold in range(3):
                tr, va = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                train = lgb.Dataset(data["x"][tr], label=target[tr], init_score=offset[tr])
                valid = lgb.Dataset(data["x"][va], label=target[va], init_score=offset[va], reference=train)
                model = lgb.train(dict(params, num_leaves=leaves), train, num_boost_round=400,
                                  valid_sets=[valid], callbacks=[lgb.early_stopping(50, verbose=False)])
                oof[va] = prediction(model, data, objective, va)
                rounds.append(model.best_iteration)
            scores = [{"power": power, "auc": rank_sum_auc(data["y"], correct(data["base"], oof, data["groups"], objective, power))}
                      for power in POWERS]
            choice = max(scores, key=lambda r: r["auc"])
            record = {"name": f"{objective}_{leaves}", "objective": objective, "leaves": leaves,
                      "cv_rounds": rounds, "refit_rounds": max(1, int(np.median(rounds))),
                      "power": choice["power"], "oof15_auc": choice["auc"], "scores": scores}
            records.append(record)
            np.save(OUT / (record["name"]+"_oof_groups.npy"), oof)
            write(OUT / "cv.json", records)
            print(json.dumps(record), flush=True)
    write(OUT / "selection15_locked.json", {"best": max(records, key=lambda r:r["oof15_auc"]),
                                             "cv_sha256": digest(OUT / "cv.json")})
    for record in records:
        objective = record["objective"]
        target, offset = target_offset(data, objective)
        params = dict(PARAMS, objective=objective, metric="binary_logloss" if objective == "binary" else "poisson", num_leaves=record["leaves"])
        model = lgb.train(params, lgb.Dataset(data["x"], label=target, init_score=offset), num_boost_round=record["refit_rounds"])
        path = OUT / (record["name"]+".txt")
        path.write_text(model.model_to_string(), encoding="utf8")
        locks.append(dict(record, model_sha256=digest(path)))
    write(OUT / "models_locked.json", {"models": locks, "protocol_sha256": digest(OUT / "protocol.json"),
                                         "training_features_sha256": array_hash(data["x"]),
                                         "training_group_keys_sha256": array_hash(data["keys"]),
                                         "training_target_count_sha256": array_hash(data["count"]),
                                         "evaluation16_labels_fit": False})
    del data
    gc.collect()
    evaluate(False)


def evaluate(verifying):
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    locks = json.loads((OUT / "models_locked.json").read_text(encoding="utf8"))
    if digest(OUT / "protocol.json") != locks["protocol_sha256"]:
        raise AssertionError("Protocol changed")
    for path, expected in protocol["inputs_sha256"].items():
        if digest(path) != expected:
            raise AssertionError("Bound source/input changed")
    for name, expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name) != expected:
            raise AssertionError("Original dataset changed")
    if verifying:
        train = prepare(15)
        for name, key in [("x", "training_features_sha256"), ("keys", "training_group_keys_sha256"), ("count", "training_target_count_sha256")]:
            if array_hash(train[name]) != locks[key]:
                raise AssertionError("Training reconstruction differs")
        del train
    all_records, predictions = [], []
    for source in ["reference", "weighted_transfer"]:
        data = prepare(16, source, include_labels=True)
        if len(data["rows"]) != 421983 or int(data["y"].sum()) != 3724 or np.any(data["count"] != -1):
            raise AssertionError("Held target boundary changed")
        for record in locks["models"]:
            path = OUT / (record["name"]+".txt")
            if digest(path) != record["model_sha256"]:
                raise AssertionError("Model changed")
            model = lgb.Booster(model_str=path.read_text(encoding="utf8"))
            group_prediction = prediction(model, data, record["objective"])
            p = correct(data["base"], group_prediction, data["groups"], record["objective"], record["power"])
            all_records.append({k:record[k] for k in ["name", "power", "oof15_auc"]} | {"source":source, "auc":rank_sum_auc(data["y"], p)})
            predictions.append(p)
    best = int(np.argmax([r["auc"] for r in all_records]))
    p = predictions[best]
    if verifying:
        error = float(np.max(np.abs(p-np.load(OUT / "prediction.npy"))))
        if error > 1e-12:
            raise AssertionError("Independent replay differs")
        np.testing.assert_array_equal(data["rows"], np.load(OUT / "valid_indices.npy"))
    else:
        error = 0.
        np.save(OUT / "prediction.npy", p)
        np.save(OUT / "valid_indices.npy", data["rows"])
    result = {"status": "passed" if verifying else "evaluated", "auc":all_records[best]["auc"],
              "best":all_records[best], "records":all_records, "rows":len(p), "clicks":3724,
              "maximum_replay_error":error, "original_datasets_unchanged":True,
              "evaluation16_labels_fit":False, "development16_model_selection":True,
              "target_met": all_records[best]["auc"] >= .82}
    write(OUT / ("verification.json" if verifying else "result.json"), result)
    print(json.dumps({k:v for k,v in result.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    evaluate(True) if parser.parse_args().verify else main()
