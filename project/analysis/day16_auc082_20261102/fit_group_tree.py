"""User-group CV on15 chooses a small nonlinear residual model;16 is evaluation."""
import os
for _key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[_key] = "2"
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from group_score_calibration import HERE, ROOT, REFERENCE, FIELDS, read_day, digest, write, rank_sum_auc
sys.path.insert(0, str(ROOT / "analysis/auc085_next/deps"))
import lightgbm as lgb

LEAVES = [3, 7, 15]
PARAMS = {"objective": "binary", "metric": "auc", "learning_rate": .03,
          "min_data_in_leaf": 1000, "lambda_l2": 10., "max_bin": 63,
          "feature_fraction": 1., "verbosity": -1, "num_threads": 2,
          "seed": 42, "deterministic": True, "force_col_wise": True,
          "boost_from_average": False}


def matrix(data):
    base = logit(data["reference"].clip(1e-6, 1-1e-6))
    return np.column_stack([base, data["features"]]), base


def metric(y, p):
    return {"auc": rank_sum_auc(y, p),
            "logloss": float(np.mean(np.logaddexp(0., logit(p.clip(1e-12, 1-1e-12))) - y*logit(p.clip(1e-12, 1-1e-12))))}


def main():
    out = HERE / "runs/group_tree_v2"
    if out.exists():
        raise FileExistsError("Preserving tree experiment")
    out.mkdir(parents=True)
    protocol = {"fit_date": 20191115, "evaluation_date": 20191116,
                "source_sha256": {str(p): digest(p) for p in [Path(__file__), HERE / "group_score_calibration.py"]},
                "params": PARAMS, "leaves": LEAVES, "maximum_rounds": 500, "patience": 60,
                "selection": "Highest OOF15 user-group AUC; final round count is median CV best iteration",
                "residual_prediction": "sigmoid(reference_logit + tree_raw_score)",
                "residual_clip": [-2., 2.], "library_version": lgb.__version__,
                "train_csv_sha256": digest(ROOT / "2026ml/train.csv"),
                "evaluation16_labels_fit": False, "development_day_reused": True,
                "reference_sha256": {f"day{d}_prediction.npy": digest(REFERENCE / f"day{d}_prediction.npy") for d in [15,16]}}
    write(out / "protocol.json", protocol)
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=FIELDS+["pt_d", "label"], dtype="int32")
    development = read_day(raw, 15)
    x, base = matrix(development)
    y = raw.label.to_numpy()[development["rows"]]
    folds = ((development["users"].astype("uint64")*2654435761) % 3).astype(int)
    records = []
    for leaves in LEAVES:
        predictions = np.empty(len(y))
        rounds = []
        for fold in range(3):
            tr, va = folds != fold, folds == fold
            dataset = lgb.Dataset(x[tr], label=y[tr], init_score=base[tr])
            validation = lgb.Dataset(x[va], label=y[va], init_score=base[va], reference=dataset)
            params = dict(PARAMS, num_leaves=leaves)
            model = lgb.train(params, dataset, num_boost_round=500, valid_sets=[validation],
                              callbacks=[lgb.early_stopping(60, verbose=False)])
            rounds.append(model.best_iteration)
            residual = model.predict(x[va], raw_score=True)
            predictions[va] = expit(base[va]+np.clip(residual, -2, 2))
            (out / f"leaves{leaves}_fold{fold}.txt").write_text(model.model_to_string(), encoding="utf8")
        record = {"leaves": leaves, "rounds": rounds, "oof15": metric(y, predictions)}
        records.append(record)
        np.save(out / f"leaves{leaves}_oof15.npy", predictions)
        print(json.dumps(record), flush=True)
    selected = max(records, key=lambda record: record["oof15"]["auc"])
    rounds = max(1, int(np.median(selected["rounds"])))
    dataset = lgb.Dataset(x, label=y, init_score=base)
    model = lgb.train(dict(PARAMS, num_leaves=selected["leaves"]), dataset, num_boost_round=rounds)
    (out / "model.txt").write_text(model.model_to_string(), encoding="utf8")
    write(out / "selection_and_weights_locked.json", {"selection": selected, "final_rounds": rounds,
           "all_cv_results": records, "protocol_sha256": digest(out / "protocol.json"),
           "model_sha256": digest(out / "model.txt"), "evaluation16_labels_fit": False})
    evaluation = read_day(raw, 16)
    x16, base16 = matrix(evaluation)
    y16 = raw.label.to_numpy()[evaluation["rows"]]
    residual = model.predict(x16, raw_score=True)
    p = expit(base16+np.clip(residual, -2, 2))
    np.save(out / "prediction.npy", p)
    np.save(out / "valid_indices.npy", evaluation["rows"])
    result = {"baseline": metric(y16, evaluation["reference"]), "candidate": metric(y16, p),
              "target_met": rank_sum_auc(y16,p) >= .82}
    write(out / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def verify():
    out = HERE / "runs/group_tree_v2"
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf8"))
    lock = json.loads((out / "selection_and_weights_locked.json").read_text(encoding="utf8"))
    if digest(out / "model.txt") != lock["model_sha256"] or digest(out / "protocol.json") != lock["protocol_sha256"]:
        raise AssertionError("Weights/protocol changed")
    for path, expected in protocol["source_sha256"].items():
        if digest(path) != expected:
            raise AssertionError("Source changed")
    if digest(ROOT / "2026ml/train.csv") != protocol["train_csv_sha256"]:
        raise AssertionError("Original CSV changed")
    for name, expected in protocol["reference_sha256"].items():
        if digest(REFERENCE / name) != expected:
            raise AssertionError("Reference changed")
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=FIELDS+["pt_d", "label"], dtype="int32")
    data = read_day(raw,16)
    x, base = matrix(data)
    model = lgb.Booster(model_str=(out / "model.txt").read_text(encoding="utf8"))
    p = expit(base+np.clip(model.predict(x,raw_score=True),-2,2))
    error = float(np.max(np.abs(p-np.load(out / "prediction.npy"))))
    if error > 1e-12:
        raise AssertionError("Reload prediction mismatch")
    np.testing.assert_array_equal(data["rows"],np.load(out / "valid_indices.npy"))
    y = raw.label.to_numpy()[data["rows"]]
    result = {"status": "passed", "rows": len(y), "clicks": int(y.sum()),
              "auc": rank_sum_auc(y,p), "maximum_reload_error": error,
              "target_met": rank_sum_auc(y,p)>=.82, "original_train_csv_unchanged": True,
              "evaluation16_labels_fit": False}
    write(out / "verification.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify",action="store_true")
    args = parser.parse_args()
    verify() if args.verify else main()
