"""Replay a fixed day15 calibrator on independently verified continuation models."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from group_score_calibration import HERE, ROOT, REFERENCE, FIELDS, group_features, transform
from fit_stacker import digest, write
from verify_stacker import rank_sum_auc

FAMILIES = {"recent": "recent_finetune_v1_day16", "sam": "sam_continuation_v1_day16"}


def compute(source):
    predictors = pd.read_csv(ROOT / "2026ml/train.csv", usecols=FIELDS+["pt_d"], dtype="int32")
    rows = np.flatnonzero(predictors.pt_d.to_numpy() == 20191116)
    np.testing.assert_array_equal(rows, np.load(source / "valid_indices.npy"))
    np.testing.assert_array_equal(rows, np.load(REFERENCE / "day16_indices.npy"))
    original = np.load(REFERENCE / "day16_prediction.npy")
    new = np.load(source / "prediction.npy")
    model = np.load(HERE / "runs/group_score_calibration_v1/model.npz")
    frame = predictors.iloc[rows][FIELDS]
    old_features, _ = group_features(frame, original)
    new_features, _ = group_features(frame, new)
    old_design = transform(old_features, original, model["center"], model["scale"])
    fixed = old_design.copy()
    fixed[:, 1] = logit(new.clip(1e-6, 1-1e-6))
    fresh = transform(new_features, new, model["center"], model["scale"])
    zold = old_design @ model["beta"]
    return rows, {"source": new, "calibrated_reference": expit(zold),
                  "fixed_context": expit(fixed @ model["beta"]),
                  "recomputed_context": expit(fresh @ model["beta"]),
                  "carry_logit_correction": expit(logit(new.clip(1e-6, 1-1e-6))+zold-logit(original.clip(1e-6, 1-1e-6)))}


def main(family, verifying):
    source = HERE / "runs" / FAMILIES[family]
    out = HERE / "runs" / f"{family}_calibration_transfer_v1"
    calibration = HERE / "runs/group_score_calibration_v1"
    if not verifying:
        if out.exists():
            raise FileExistsError("Preserving existing calibration transfer")
        source_verification = json.loads((source / "verification.json").read_text(encoding="utf8"))
        cal_verification = json.loads((calibration / "verification.json").read_text(encoding="utf8"))
        if source_verification["status"] != "passed" or cal_verification["status"] != "passed":
            raise AssertionError("Independent source/calibrator verification required")
        cal_lock = json.loads((calibration / "weights_locked.json").read_text(encoding="utf8"))
        if digest(calibration / "model.npz") != cal_lock["model_sha256"]:
            raise AssertionError("Frozen calibrator changed")
        source_protocol = json.loads((source / "protocol.json").read_text(encoding="utf8"))
        inputs = [Path(__file__), HERE / "group_score_calibration.py", HERE / "fit_stacker.py", HERE / "verify_stacker.py",
                  source / "verification.json", source / "prediction.npy", source / "valid_indices.npy",
                  source / "protocol.json", calibration / "model.npz", calibration / "weights_locked.json",
                  calibration / "verification.json", REFERENCE / "day16_indices.npy", REFERENCE / "day16_prediction.npy"]
        protocol = {"source": str(source), "source_auc": source_verification["auc"],
                    "inputs_sha256": {str(p):digest(p) for p in inputs},
                    "raw_datasets_sha256": source_protocol["raw_datasets_sha256"],
                    "new_model_fitting_performed": False, "calibrator_fit_date": 20191115,
                    "evaluation16_labels_fit": False, "development16_model_selection": True}
        out.mkdir(parents=True)
        write(out / "protocol.json", protocol)
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf8"))
    for path, expected in protocol["inputs_sha256"].items():
        if digest(path) != expected:
            raise AssertionError("Source/calibrator changed: "+path)
    for name, expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name) != expected:
            raise AssertionError("Original data changed")
    rows, predictions = compute(source)
    if not verifying:
        for name, p in predictions.items():
            np.save(out / (name+"_prediction.npy"), p)
        np.save(out / "valid_indices.npy", rows)
        write(out / "predictions_locked.json", {"protocol_sha256": digest(out / "protocol.json"),
                                                 "prediction_sha256": {name:digest(out / (name+"_prediction.npy")) for name in predictions}})
    lock = json.loads((out / "predictions_locked.json").read_text(encoding="utf8"))
    if digest(out / "protocol.json") != lock["protocol_sha256"]:
        raise AssertionError("Protocol changed")
    y = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["label"], dtype="int8").label.to_numpy()[rows]
    if len(rows) != 421983 or int(y.sum()) != 3724:
        raise AssertionError("Complete16 cohort changed")
    records = []
    for name, p in predictions.items():
        path = out / (name+"_prediction.npy")
        if digest(path) != lock["prediction_sha256"][name]:
            raise AssertionError("Locked prediction changed")
        error = float(np.max(np.abs(p-np.load(path))))
        if error > 1e-12:
            raise AssertionError("Formula replay differs")
        records.append({"rule":name, "auc":rank_sum_auc(y, p), "maximum_replay_error":error})
    original_score = next(r["auc"] for r in records if r["rule"] == "source")
    if abs(original_score-protocol["source_auc"]) > 1e-12:
        raise AssertionError("Independent source AUC mismatch")
    best = max(records, key=lambda r:r["auc"])
    if verifying:
        np.testing.assert_array_equal(np.load(out / "best_prediction.npy"), predictions[best["rule"]])
        np.testing.assert_array_equal(np.load(out / "valid_indices.npy"), rows)
    else:
        np.save(out / "best_prediction.npy", predictions[best["rule"]])
    result = {"status":"passed" if verifying else "computed", "auc":best["auc"], "best_rule":best["rule"],
              "records":records, "rows":len(rows), "clicks":int(y.sum()), "original_datasets_unchanged":True,
              "new_model_fitting_performed":False, "evaluation16_labels_fit":False,
              "development16_model_selection":True, "target_met":best["auc"] >= .82}
    write(out / ("verification.json" if verifying else "result.json"), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    main(args.family, args.verify)
