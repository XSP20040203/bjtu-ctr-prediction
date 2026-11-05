"""Apply a frozen15-day group calibrator to a newly verified16-day predictor."""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from group_score_calibration import HERE, ROOT, REFERENCE, FIELDS, group_features, transform
from fit_stacker import digest, write
from verify_stacker import rank_sum_auc


def main(family):
    source = HERE / "runs" / f"{family}_day16"
    verification_path = source / "verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf8"))
    if verification["status"] != "passed":
        raise AssertionError("Source predictions need independent verification first")
    calibration = HERE / "runs/group_score_calibration_v1"
    calibration_verification = json.loads((calibration / "verification.json").read_text(encoding="utf8"))
    if calibration_verification["status"] != "passed":
        raise AssertionError("Calibrator needs independent verification")
    suffix = "weighted" if family == "weighted_v1" else "duplicate"
    out = HERE / "runs" / f"{suffix}_calibration_transfer_v1"
    if out.exists():
        raise FileExistsError("Preserving calibration transfer")
    out.mkdir(parents=True)
    source_prediction = source / "best_verified_prediction.npy"
    model_path = calibration / "model.npz"
    if digest(source_prediction) != verification["best_prediction_sha256"]:
        raise AssertionError("Verified source predictions changed")
    calibration_lock = json.loads((calibration / "weights_locked.json").read_text(encoding="utf8"))
    if digest(model_path) != calibration_lock["model_sha256"]:
        raise AssertionError("Frozen calibration model changed")
    source_protocol = json.loads((source / "protocol.json").read_text(encoding="utf8"))
    raw_hashes = {name:digest(ROOT / "2026ml" / name)
                  for name in source_protocol["raw_datasets_sha256"]}
    if raw_hashes != source_protocol["raw_datasets_sha256"]:
        raise AssertionError("An original dataset changed")
    protocol = {"new_parameters_fit":False,"calibrator_fit_date":20191115,"evaluation_date":20191116,
                "source_verification":str(verification_path),"source_verification_sha256":digest(verification_path),
                "source_prediction_sha256":digest(source_prediction),"calibrator_sha256":digest(model_path),
                "source_sha256":digest(__file__),
                "rules":["fixed_context","recomputed_context","carry_logit_correction"],
                "development_day_reused":True,"evaluation16_labels_fit":False,
                "original_datasets_sha256":raw_hashes}
    write(out / "protocol.json",protocol)
    predictors = pd.read_csv(ROOT / "2026ml/train.csv",usecols=FIELDS+["pt_d"],dtype="int32")
    rows = np.flatnonzero(predictors.pt_d.to_numpy()==20191116)
    np.testing.assert_array_equal(rows,np.load(source / "valid_indices.npy"))
    np.testing.assert_array_equal(rows,np.load(REFERENCE / "day16_indices.npy"))
    frame = predictors.iloc[rows][FIELDS]
    original = np.load(REFERENCE / "day16_prediction.npy")
    new = np.load(source_prediction)
    calibrator = np.load(model_path)
    old_features,_ = group_features(frame,original)
    new_features,_ = group_features(frame,new)
    old_design = transform(old_features,original,calibrator["center"],calibrator["scale"])
    fixed = old_design.copy()
    fixed[:,1] = logit(new.clip(1e-6,1-1e-6))
    fresh = transform(new_features,new,calibrator["center"],calibrator["scale"])
    zold = old_design @ calibrator["beta"]
    outputs = {"source":new,"calibrated_reference":expit(zold),
               "fixed_context":expit(fixed @ calibrator["beta"]),
               "recomputed_context":expit(fresh @ calibrator["beta"]),
               "carry_logit_correction":expit(logit(new.clip(1e-6,1-1e-6))+zold-logit(original.clip(1e-6,1-1e-6)))}
    # Store every fixed-rule prediction before reading any evaluation outcomes.
    for name,p in outputs.items():
        np.save(out / f"{name}_prediction.npy",p)
    write(out / "predictions_locked.json",{name:digest(out / f"{name}_prediction.npy") for name in outputs})
    labels = pd.read_csv(ROOT / "2026ml/train.csv",usecols=["pt_d","label"],dtype="int32")
    y = labels.label.to_numpy()[rows]
    records = []
    for name,p in outputs.items():
        saved = np.load(out / f"{name}_prediction.npy")
        if not np.array_equal(saved,p):
            raise AssertionError("Saved prediction changed")
        records.append({"rule":name,"auc":rank_sum_auc(y,saved)})
    best = max(records,key=lambda v:v["auc"])
    np.save(out / "best_prediction.npy",outputs[best["rule"]])
    np.save(out / "valid_indices.npy",rows)
    report = {"status":"passed","auc":best["auc"],"best_rule":best["rule"],"records":records,
              "rows":len(y),"clicks":int(y.sum()),"target_met":best["auc"]>=.82,
              "source_independently_verified":True,"new_model_fitting_performed":False,
              "calibrator_not_refit_on_new16_features":True,"development_day_reused":True,
              "original_datasets_unchanged":True,"train_csv_sha256":digest(ROOT / "2026ml/train.csv")}
    write(out / "verification.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family",choices=["weighted_v1","duplicate_mean_v1"],required=True)
    main(parser.parse_args().family)
