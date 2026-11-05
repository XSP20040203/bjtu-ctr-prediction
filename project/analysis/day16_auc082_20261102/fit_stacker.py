"""Fit regularized score corrections on real day15 outcomes, then evaluate16."""
import os
for _key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[_key] = "2"
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, log_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {
    "v6": "analysis/nextday_goal_20261029/duplicate_multiplicity/day{day}_prediction.npy",
    "gated500": "analysis/architecture_goal_20261029/results/gated_oof_day{day}/validation_prediction.npy",
    "allocated": "analysis/architecture_goal_20261029/results/set_allocated_day{day}/validation_prediction.npy",
    "gated166": "analysis/architecture_goal_20261029/results/gated166_day{day}/validation_prediction.npy",
    "protected42": "analysis/userday_aux_20261101/runs/protected_day_aux_v1_day{day}_seed42/both/best_prediction.npy",
    "protected2026": "analysis/userday_aux_20261101/runs/protected_confirmation_v1_day{day}_seed2026/both/best_prediction.npy",
}
REFERENCE = ROOT / "analysis/userday_aux_20261101/reference_day_state_protected_seed42_v1"
RIDGES = [0.0001, 0.001, 0.01]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def fit(x, y, ridge, active):
    target = np.zeros(x.shape[1])
    target[1] = 1.
    bounds = [(-5., 5.), (.1, 3.)] + [(-.75, .75) if i in active else (0., 0.) for i in range(x.shape[1]-2)]
    penalties = np.r_[1e-5, ridge*.1, np.full(x.shape[1]-2, ridge)]
    def objective(beta):
        z = x @ beta
        delta = beta-target
        value = np.mean(np.logaddexp(0., z) - y*z) + .5*np.dot(penalties, delta*delta)
        gradient = x.T @ (expit(z)-y)/len(y) + penalties*delta
        return value, gradient
    result = minimize(objective, target, jac=True, bounds=bounds, method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-9, "maxls": 50})
    if not np.isfinite(result.x).all():
        raise AssertionError("Nonfinite fitted stacker")
    return result.x, {"success": bool(result.success), "message": str(result.message),
                      "iterations": int(result.nit), "objective": float(result.fun),
                      "max_gradient": float(np.abs(result.jac).max())}


def metrics(y, p):
    return {"auc": float(roc_auc_score(y, p)), "logloss": float(log_loss(y, p, labels=[0, 1]))}


def main():
    out = HERE / "runs/score_stacker_v1"
    if out.exists():
        raise FileExistsError("Preserving completed stacker")
    out.mkdir(parents=True)
    started = time.perf_counter()
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["pt_d", "user_id", "label"], dtype="int32")
    # Establish source existence before locking the experiment; old optional seed
    # results may have different tags, so only exact, complete pairs qualify.
    active_sources = {name: pattern for name, pattern in SOURCES.items()
                      if all((ROOT / pattern.format(day=d)).is_file() for d in [15, 16])}
    if len(active_sources) < 4:
        raise AssertionError("Too few complete, paired model predictions")
    protocol = {"fit_date": 20191115, "evaluation_date": 20191116, "ridges": RIDGES,
                "regularization_selection": "3-fold user-group cross-validation on15 only; largest mean AUC",
                "features": list(active_sources), "reference": str(REFERENCE),
                "source_sha256": digest(__file__), "train_csv_sha256": digest(ROOT / "2026ml/train.csv"),
                "input_sha256": {str(ROOT / pattern.format(day=d)): digest(ROOT / pattern.format(day=d))
                                  for pattern in active_sources.values() for d in [15, 16]},
                "training16_labels_used": False, "evaluation_date_reused": True,
                "all_underlying_models_fit_before_their_prediction_date": True,
                "final_prediction_read": False}
    write(out / "protocol.json", protocol)
    data = {}
    for day in [15, 16]:
        rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191100+day)
        np.testing.assert_array_equal(rows, np.load(REFERENCE / f"day{day}_indices.npy"))
        reference = np.load(REFERENCE / f"day{day}_prediction.npy")
        z0 = logit(np.clip(reference, 1e-6, 1-1e-6))
        pieces = [np.ones(len(rows)), z0]
        for name, pattern in active_sources.items():
            path = ROOT / pattern.format(day=day)
            indices_path = path.parent / "valid_indices.npy"
            if indices_path.is_file():
                np.testing.assert_array_equal(rows, np.load(indices_path))
            p = np.load(path)
            if p.shape != reference.shape or not np.isfinite(p).all():
                raise AssertionError("Bad prediction source: " + name)
            pieces.append(logit(np.clip(p, 1e-6, 1-1e-6))-z0)
        data[day] = (np.column_stack(pieces), rows, reference)
    x15, rows15, p15 = data[15]
    y15 = raw.label.to_numpy()[rows15]
    folds = ((raw.user_id.to_numpy()[rows15].astype("uint64") * 2654435761) % 3).astype(int)
    records = []
    for ridge in RIDGES:
        oof = np.empty(len(y15))
        for fold in range(3):
            train, val = folds != fold, folds == fold
            beta, result = fit(x15[train], y15[train], ridge, list(range(len(active_sources))))
            oof[val] = expit(x15[val] @ beta)
        records.append({"ridge": ridge, **metrics(y15, oof)})
    selected = max(records, key=lambda row: row["auc"])["ridge"]
    beta, result = fit(x15, y15, selected, list(range(len(active_sources))))
    base_beta, base_result = fit(x15, y15, selected, [])
    np.savez(out / "model.npz", beta=beta, base_beta=base_beta)
    write(out / "fit15.json", {"cv": records, "selected_ridge": selected, "fit": result,
                                "base_fit": base_result, "coefficients": beta.tolist(),
                                "columns": ["intercept", "reference_logit"] + list(active_sources)})
    write(out / "weights_locked.json", {"model_sha256": digest(out / "model.npz"),
                                        "protocol_sha256": digest(out / "protocol.json"),
                                        "evaluation16_labels_used": False})
    # Day16 responses are first used here, after parameters and selection lock.
    x16, rows16, p16 = data[16]
    y16 = raw.label.to_numpy()[rows16]
    output = {"reference": p16, "calibration_only": expit(x16 @ base_beta),
              "regularized_stack": expit(x16 @ beta)}
    results = []
    for name, p in output.items():
        np.save(out / f"{name}_prediction.npy", p)
        results.append({"name": name, **metrics(y16, p)})
    reloaded = np.load(out / "model.npz")
    error = float(np.max(np.abs(expit(x16 @ reloaded["beta"])-output["regularized_stack"])))
    np.save(out / "valid_indices.npy", rows16)
    best = max(results, key=lambda row: row["auc"])
    verification = {"status": "passed", "results16": results, "best": best,
                    "reload_error": error, "rows": len(y16), "clicks": int(y16.sum()),
                    "fit_date": 20191115, "evaluation_date": 20191116,
                    "evaluation16_labels_fit": False, "target_met": best["auc"] >= .82,
                    "original_train_csv_unchanged": digest(ROOT / "2026ml/train.csv") == protocol["train_csv_sha256"],
                    "seconds": time.perf_counter()-started}
    write(out / "verification.json", verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
