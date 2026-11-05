"""Calibrate out-of-day scores using label-free exposure-group context."""
import os
for _key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[_key] = "2"
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from fit_stacker import HERE, ROOT, REFERENCE, digest, fit, metrics, write
from verify_stacker import rank_sum_auc

GROUPS = [("user", ["user_id"]), ("task", ["task_id"]),
          ("user_slot", ["user_id", "slot_id"]),
          ("user_app", ["user_id", "spread_app_id"])]
FIELDS = ["user_id", "task_id", "slot_id", "spread_app_id"]
RIDGES = [.001, .01, .1]


def group_features(raw, probability):
    if set(raw.columns) != set(FIELDS):
        raise ValueError("Group-score context accepts only declared predictors")
    if probability.shape != (len(raw),) or not np.isfinite(probability).all():
        raise ValueError("Probability alignment")
    result, names = [], []
    for group_name, columns in GROUPS:
        codes, keys = pd.factorize(pd.MultiIndex.from_frame(raw[columns]), sort=True)
        count = np.bincount(codes, minlength=len(keys))
        mass = np.bincount(codes, weights=probability, minlength=len(keys))
        maximum = np.zeros(len(keys))
        np.maximum.at(maximum, codes, probability)
        values = pd.DataFrame({"code": codes, "p": probability})
        ranks = values.groupby("code", sort=False).p.rank(method="average", pct=True).to_numpy()
        block = [np.log1p(count[codes]), np.log(mass[codes].clip(1e-8)),
                 logit((mass/count)[codes].clip(1e-6, 1-1e-6)),
                 logit(maximum[codes].clip(1e-6, 1-1e-6)), ranks]
        result.extend(block)
        names.extend([f"{group_name}_{suffix}" for suffix in ["log_count", "log_mass", "mean_logit", "max_logit", "rank"]])
    return np.column_stack(result), names


def transform(features, reference, center, scale):
    return np.column_stack([np.ones(len(reference)), logit(reference.clip(1e-6, 1-1e-6)),
                            np.clip((features-center)/scale, -8, 8)])


def read_day(raw, day):
    rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
    np.testing.assert_array_equal(rows, np.load(REFERENCE / f"day{day}_indices.npy"))
    reference = np.load(REFERENCE / f"day{day}_prediction.npy")
    features, names = group_features(raw.iloc[rows][FIELDS], reference)
    return {"rows": rows, "reference": reference, "features": features,
            "names": names, "users": raw.user_id.to_numpy()[rows]}


def run():
    out = HERE / "runs/group_score_calibration_v1"
    if out.exists():
        raise FileExistsError("Preserve group score calibration")
    out.mkdir(parents=True)
    protocol = {"fit_date": 20191115, "evaluation_date": 20191116, "ridges": RIDGES,
                "groups": GROUPS, "input_fields": FIELDS,
                "source_sha256": digest(__file__),
                "train_csv_sha256": digest(ROOT / "2026ml/train.csv"),
                "reference_sha256": {f"day{d}_prediction.npy": digest(REFERENCE / f"day{d}_prediction.npy") for d in [15, 16]},
                "selection": "Three folds by user on15; mean held-user AUC",
                "evaluation16_outcomes_fit": False,
                "same_day_unlabelled_exposure_group_scores_used": True,
                "development_day_reused": True}
    write(out / "protocol.json", protocol)
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=FIELDS+["pt_d", "label"], dtype="int32")
    development = read_day(raw, 15)
    features = development["features"]
    y = raw.label.to_numpy()[development["rows"]]
    folds = ((development["users"].astype("uint64") * 2654435761) % 3).astype(int)
    records = []
    for ridge in RIDGES:
        p = np.empty(len(y))
        for fold in range(3):
            train, valid = folds != fold, folds == fold
            center = features[train].mean(0)
            scale = features[train].std(0).clip(.05)
            x = transform(features, development["reference"], center, scale)
            beta, info = fit(x[train], y[train], ridge, list(range(features.shape[1])))
            if not info["success"]:
                raise AssertionError("Optimizer did not converge")
            p[valid] = expit(x[valid] @ beta)
        records.append({"ridge": ridge, **metrics(y, p)})
    selected = max(records, key=lambda row: row["auc"])["ridge"]
    center = features.mean(0)
    scale = features.std(0).clip(.05)
    x = transform(features, development["reference"], center, scale)
    beta, info = fit(x, y, selected, list(range(features.shape[1])))
    np.savez(out / "model.npz", center=center, scale=scale, beta=beta)
    write(out / "fit15.json", {"cv": records, "selected_ridge": selected, "fit": info,
                              "features": development["names"], "beta": beta.tolist()})
    write(out / "weights_locked.json", {"model_sha256": digest(out / "model.npz"),
                                        "protocol_sha256": digest(out / "protocol.json"),
                                        "evaluation16_labels_fit": False})
    evaluation = read_day(raw, 16)
    p = expit(transform(evaluation["features"], evaluation["reference"], center, scale) @ beta)
    np.save(out / "prediction.npy", p)
    np.save(out / "valid_indices.npy", evaluation["rows"])
    y16 = raw.label.to_numpy()[evaluation["rows"]]
    result = {"status": "fitted_and_evaluated", "baseline": metrics(y16, evaluation["reference"]),
              "candidate": metrics(y16, p), "rows": len(y16), "clicks": int(y16.sum()),
              "target_met": metrics(y16,p)["auc"] >= .82}
    write(out / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def verify():
    out = HERE / "runs/group_score_calibration_v1"
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf8"))
    lock = json.loads((out / "weights_locked.json").read_text(encoding="utf8"))
    if digest(__file__) != protocol["source_sha256"] or digest(ROOT / "2026ml/train.csv") != protocol["train_csv_sha256"]:
        raise AssertionError("Source or original CSV changed")
    if digest(out / "model.npz") != lock["model_sha256"] or digest(out / "protocol.json") != lock["protocol_sha256"]:
        raise AssertionError("Locked model or protocol changed")
    for name, expected in protocol["reference_sha256"].items():
        if digest(REFERENCE / name) != expected:
            raise AssertionError("Reference changed")
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=FIELDS+["pt_d", "label"], dtype="int32")
    evaluation = read_day(raw, 16)
    np.testing.assert_array_equal(evaluation["rows"], np.load(out / "valid_indices.npy"))
    model = np.load(out / "model.npz")
    p = expit(transform(evaluation["features"], evaluation["reference"], model["center"], model["scale"]) @ model["beta"])
    error = float(np.max(np.abs(p-np.load(out / "prediction.npy"))))
    if error > 1e-12:
        raise AssertionError("Prediction replay mismatch")
    y = raw.label.to_numpy()[evaluation["rows"]]
    result = {"status": "passed", "auc": rank_sum_auc(y, p), "maximum_replay_error": error,
              "rows": len(y), "clicks": int(y.sum()), "target_met": rank_sum_auc(y, p) >= .82,
              "raw_csv_unchanged": True, "evaluation16_outcomes_fit": False}
    write(out / "verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    verify() if args.verify else run()
