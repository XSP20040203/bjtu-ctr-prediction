"""Fit a convex pairwise ranking correction using only day15 outcomes."""
import os
for _key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[_key] = "2"
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from fit_stacker import SOURCES, digest, write
from group_score_calibration import HERE, ROOT, REFERENCE, FIELDS, read_day
from verify_stacker import rank_sum_auc

OUT = HERE / "runs/rank_calibration_v1"
CONFIGS = [(t, r) for t in [.5, 1.] for r in [.03, .3, 3.]]
PAIR_COUNT = 131072


def sample_pairs(y, seed, count=PAIR_COUNT):
    y = np.asarray(y)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Only observed binary training labels are permitted")
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Pairwise fitting requires both classes")
    rng = np.random.default_rng(seed)
    return rng.choice(positive, count), rng.choice(negative, count)


def objective(beta, difference, base_difference, temperature, ridge):
    z = (base_difference + difference @ beta) / temperature
    value = np.logaddexp(0., -z).mean() + .5*ridge*np.dot(beta, beta)
    gradient = difference.T @ (-expit(-z)) / (len(z)*temperature) + ridge*beta
    return float(value), gradient


def fit(x, base, y, temperature, ridge, seed):
    center = x.mean(0)
    scale = x.std(0).clip(.05)
    design = np.clip((x-center)/scale, -8, 8)
    positive, negative = sample_pairs(y, seed)
    difference = design[positive] - design[negative]
    base_difference = base[positive] - base[negative]
    result = minimize(objective, np.zeros(x.shape[1]),
                      args=(difference, base_difference, temperature, ridge),
                      jac=True, method="L-BFGS-B", bounds=[(-.75, .75)]*x.shape[1],
                      options={"maxiter": 250, "ftol": 1e-12, "gtol": 1e-7, "maxls": 40})
    if not result.success or not np.isfinite(result.x).all():
        raise AssertionError(str(result.message))
    model = dict(center=center, scale=scale, beta=result.x)
    info = {"iterations": int(result.nit), "objective": float(result.fun),
            "pair_positive_rows": int(len(np.unique(positive))),
            "pair_negative_rows": int(len(np.unique(negative))),
            "pair_count": len(positive), "success": bool(result.success)}
    return model, info


def predict(x, base, model):
    return expit(base + np.clip((x-model["center"])/model["scale"], -8, 8) @ model["beta"])


def features(raw, day, active):
    data = read_day(raw, day)
    base = logit(data["reference"].clip(1e-6, 1-1e-6))
    blocks, names = [data["features"]], list(data["names"])
    for name, pattern in active.items():
        path = ROOT / pattern.format(day=day)
        p = np.load(path)
        if p.shape != base.shape or not np.isfinite(p).all():
            raise AssertionError("Bad paired source: " + name)
        rows_path = path.parent / "valid_indices.npy"
        if rows_path.is_file():
            np.testing.assert_array_equal(data["rows"], np.load(rows_path))
        blocks.append((logit(p.clip(1e-6, 1-1e-6))-base)[:, None])
        names.append(name + "_logit_difference")
    data.update(x=np.column_stack(blocks), base=base, feature_names=names)
    return data


def load_predictors():
    return pd.read_csv(ROOT / "2026ml/train.csv", usecols=FIELDS+["pt_d"], dtype="int32")


def main():
    if OUT.exists():
        raise FileExistsError("Preserving ranking calibration run")
    OUT.mkdir(parents=True)
    active = {name: p for name, p in SOURCES.items()
              if all((ROOT / p.format(day=d)).is_file() for d in [15, 16])}
    input_paths = [REFERENCE / f"day{d}_prediction.npy" for d in [15, 16]]
    input_paths += [ROOT / p.format(day=d) for p in active.values() for d in [15, 16]]
    protocol = {"fit_date": 20191115, "evaluation_date": 20191116,
                "configs": CONFIGS, "pair_count": PAIR_COUNT, "sources": active,
                "selection": "Three user folds on15 only; largest pooled OOF AUC",
                "base_logit_coefficient": 1., "correction_bounds": [-.75, .75],
                "source_sha256": {str(p): digest(p) for p in [Path(__file__), HERE/"group_score_calibration.py", HERE/"fit_stacker.py"]},
                "prediction_sha256": {str(p): digest(p) for p in input_paths},
                "raw_datasets_sha256": {p.name: digest(p) for p in (ROOT/"2026ml").glob("*.csv")},
                "evaluation16_labels_fit": False, "development_day_reused": True}
    write(OUT/"protocol.json", protocol)
    raw = load_predictors()
    data = features(raw, 15, active)
    labels = pd.read_csv(ROOT/"2026ml/train.csv", usecols=["label", "pt_d"], dtype="int32")
    y = labels.loc[labels.pt_d == 20191115, "label"].to_numpy(copy=True)
    del labels
    if len(y) != len(data["rows"]):
        raise AssertionError("Training row alignment")
    folds = ((data["users"].astype("uint64")*2654435761) % 3).astype(int)
    records = []
    for temperature, ridge in CONFIGS:
        oof = np.empty(len(y))
        for fold in range(3):
            tr, va = folds != fold, folds == fold
            model, info = fit(data["x"][tr], data["base"][tr], y[tr], temperature, ridge, 4215+fold)
            oof[va] = predict(data["x"][va], data["base"][va], model)
        record = {"temperature": temperature, "ridge": ridge, "oof_auc": rank_sum_auc(y, oof)}
        records.append(record)
        print(json.dumps(record), flush=True)
        write(OUT/"cv.json", records)
    selected = max(records, key=lambda r: r["oof_auc"])
    model, info = fit(data["x"], data["base"], y, selected["temperature"], selected["ridge"], 4215)
    np.savez(OUT/"model.npz", **model)
    write(OUT/"fit15.json", {"selected": selected, "records": records, "fit": info,
          "baseline_oof_auc": rank_sum_auc(y, data["reference"]),
          "feature_names": data["feature_names"], "training_rows": len(y), "training_clicks": int(y.sum())})
    write(OUT/"weights_locked.json", {"model_sha256": digest(OUT/"model.npz"),
          "protocol_sha256": digest(OUT/"protocol.json"), "evaluation16_labels_fit": False})
    data16 = features(raw, 16, active)
    p = predict(data16["x"], data16["base"], model)
    np.save(OUT/"prediction.npy", p)
    np.save(OUT/"valid_indices.npy", data16["rows"])
    labels = pd.read_csv(ROOT/"2026ml/train.csv", usecols=["label"], dtype="int32").label.to_numpy()
    y16 = labels[data16["rows"]]
    result = {"auc": rank_sum_auc(y16, p), "baseline_auc": rank_sum_auc(y16, data16["reference"]),
              "rows": len(y16), "clicks": int(y16.sum()), "target_met": rank_sum_auc(y16, p) >= .82}
    write(OUT/"result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def verify():
    protocol = json.loads((OUT/"protocol.json").read_text(encoding="utf8"))
    lock = json.loads((OUT/"weights_locked.json").read_text(encoding="utf8"))
    for group in ["source_sha256", "prediction_sha256"]:
        for path, expected in protocol[group].items():
            if digest(path) != expected:
                raise AssertionError("Frozen source/input changed: " + path)
    for name, expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT/"2026ml"/name) != expected:
            raise AssertionError("Original CSV changed")
    if digest(OUT/"model.npz") != lock["model_sha256"] or digest(OUT/"protocol.json") != lock["protocol_sha256"]:
        raise AssertionError("Model/protocol changed")
    data = features(load_predictors(), 16, protocol["sources"])
    np.testing.assert_array_equal(data["rows"], np.load(OUT/"valid_indices.npy"))
    model = np.load(OUT/"model.npz")
    p = predict(data["x"], data["base"], model)
    error = float(np.max(np.abs(p-np.load(OUT/"prediction.npy"))))
    if error > 1e-12:
        raise AssertionError("Replayed predictions differ")
    labels = pd.read_csv(ROOT/"2026ml/train.csv", usecols=["label"], dtype="int32").label.to_numpy()[data["rows"]]
    if len(labels) != 421983 or int(labels.sum()) != 3724:
        raise AssertionError("Complete16 cohort changed")
    result = {"status": "passed", "auc": rank_sum_auc(labels, p), "maximum_reload_error": error,
              "rows": len(labels), "clicks": int(labels.sum()), "evaluation16_labels_fit": False,
              "original_datasets_unchanged": True, "target_met": rank_sum_auc(labels, p) >= .82}
    write(OUT/"verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    verify() if parser.parse_args().verify else main()
