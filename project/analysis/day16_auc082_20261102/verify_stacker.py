"""Audit stored15-only score models and16 scores with a rank-sum AUC."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata

from fit_stacker import HERE, ROOT, REFERENCE, SOURCES, digest, write


def rank_sum_auc(y, p):
    npos = int(np.sum(y == 1))
    nneg = int(np.sum(y == 0))
    if not npos or not nneg:
        raise ValueError("Both classes required")
    ranks = rankdata(p, method="average")
    return float((ranks[y == 1].sum() - npos*(npos+1)/2)/(npos*nneg))


def main():
    out = HERE / "runs/score_stacker_v1"
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf8"))
    lock = json.loads((out / "weights_locked.json").read_text(encoding="utf8"))
    if digest(out / "model.npz") != lock["model_sha256"] or digest(out / "protocol.json") != lock["protocol_sha256"]:
        raise AssertionError("Locked weights/protocol changed")
    if digest(HERE / "fit_stacker.py") != protocol["source_sha256"]:
        raise AssertionError("Training source changed")
    if digest(ROOT / "2026ml/train.csv") != protocol["train_csv_sha256"]:
        raise AssertionError("Original train data changed")
    for path, expected in protocol["input_sha256"].items():
        if digest(path) != expected:
            raise AssertionError("Input predictions changed")
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["pt_d", "label", "user_id"], dtype="int32")
    rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191116)
    np.testing.assert_array_equal(rows, np.load(out / "valid_indices.npy"))
    y = raw.label.to_numpy()[rows]
    reference = np.load(REFERENCE / "day16_prediction.npy")
    z = logit(np.clip(reference, 1e-6, 1-1e-6))
    matrix = np.column_stack([np.ones(len(y)), z] + [
        logit(np.clip(np.load(ROOT / SOURCES[name].format(day=16)), 1e-6, 1-1e-6))-z
        for name in protocol["features"]])
    saved = np.load(out / "model.npz")
    probabilities = {"reference": reference, "calibration_only": expit(matrix @ saved["base_beta"]),
                     "regularized_stack": expit(matrix @ saved["beta"])}
    results = {}
    previous = json.loads((out / "verification.json").read_text(encoding="utf8"))
    by_name = {entry["name"]: entry for entry in previous["results16"]}
    for name, p in probabilities.items():
        expected = np.load(out / f"{name}_prediction.npy")
        error = float(np.max(np.abs(expected-p)))
        if error > 1e-12:
            raise AssertionError("Independent prediction reconstruction mismatch")
        auc = rank_sum_auc(y, p)
        if abs(auc - by_name[name]["auc"]) > 1e-12:
            raise AssertionError("Rank-sum AUC differs")
        results[name] = {"auc": auc, "prediction_error": error}
    report = {"status": "passed", "evaluation_date": 20191116, "rows": len(y),
              "clicks": int(y.sum()), "results": results,
              "auc_method": "Average-rank Mann-Whitney sum, independently of sklearn ROC API",
              "fit_labels_date": 20191115, "weights_locked_sha256": digest(out / "weights_locked.json"),
              "target_met": any(v["auc"] >= .82 for v in results.values()),
              "development_date_reused": True}
    write(out / "independent_verification.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
