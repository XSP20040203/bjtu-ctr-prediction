"""Describe earlier-exposure support; held labels never fit a predictor here."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis/strong_interest_control_20261031"))
from features import write, emit
from day_context import past_exposure_features
import numpy as np
import pandas as pd


def run():
    day = 20191115
    raw = pd.read_csv(ROOT / "2026ml/train.csv",
                      usecols=["pt_d", "user_id", "task_id", "label"], dtype="int32")
    raw = raw.loc[raw.pt_d <= day].reset_index(drop=True)
    predictors = raw[["pt_d", "user_id", "task_id"]]
    history = past_exposure_features(predictors)
    held = raw.pt_d.to_numpy() == day
    counts = np.rint(np.expm1(history[held, 0])).astype("int32")
    labels = raw.label.to_numpy()[held]
    prediction = np.load(ROOT / "analysis/learned_audience_20261101/reference/day15_prediction.npy")
    rows = []
    for name, mask in [("0", counts == 0), ("1to2", (counts >= 1) & (counts <= 2)),
                        ("3to10", (counts >= 3) & (counts <= 10)),
                        ("11to30", (counts >= 11) & (counts <= 30)), ("31plus", counts >= 31)]:
        if not mask.any():
            continue
        rows.append({"past_pair_exposures": name, "rows": int(mask.sum()),
                     "clicks": int(labels[mask].sum()), "ctr": float(labels[mask].mean()),
                     "mean_reference_prediction": float(prediction[mask].mean()),
                     "mean_residual": float((labels[mask] - prediction[mask]).mean())})
    result = {"day": day, "status": "descriptive_diagnostic_only", "window_calendar_days": 3,
              "labels_used_for_feature_construction": False, "held_labels_fit_a_model": False,
              "goal_metric": "Original advertisement-row AUC remains unchanged", "bins": rows}
    write(HERE / "history_diagnostic_day15.json", result)
    emit("past_exposure_diagnostic", **result)


if __name__ == "__main__":
    run()
