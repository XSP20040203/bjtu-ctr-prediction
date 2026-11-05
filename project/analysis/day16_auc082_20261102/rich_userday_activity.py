"""Uniform user-day trees on label-free context and strictly past ad outcomes.

All candidates and correction powers are specified before day16 scoring. Day15
selects rounds and powers. Day16 is reused development data, not a blind test.
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
from sklearn.metrics import roc_auc_score, log_loss
from fit_stacker import HERE, ROOT, REFERENCE, digest, write
from verify_stacker import rank_sum_auc
sys.path.insert(0, str(ROOT / "analysis/auc085_next/deps"))
import lightgbm as lgb

OUT = HERE / "runs/rich_userday_activity_v1"
CONTEXTS = {
    15: ROOT / "analysis/userday_aux_20261101/runs/protected_day_aux_v1_day15_seed42",
    16: HERE / "runs/weighted_v1_day16",
}
POWERS = [0., .125, .25, .5, .75, 1.]
LEAVES = [7, 15, 31]
PARAMS = {"learning_rate": .03, "min_data_in_leaf": 100, "lambda_l2": 10.,
          "feature_fraction": .8, "max_bin": 63, "num_threads": 2,
          "seed": 42, "verbosity": -1, "deterministic": True,
          "force_col_wise": True}


def array_hash(value):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def past_outcomes(keys, sizes, counts, cutoff):
    """Today is queried before insertion. Held outcomes must be -1."""
    if not (len(keys) == len(sizes) == len(counts)):
        raise ValueError("Unaligned user-day arrays")
    if np.any(counts[keys[:, 0] >= cutoff] != -1):
        raise ValueError("Held-day outcomes must be masked")
    if np.any(counts[keys[:, 0] < cutoff] < 0):
        raise ValueError("Earlier outcome counts must be observed")
    names = [f"history_{window}_{kind}" for window in ["all", "3d", "7d"]
             for kind in ["log_exposures", "log_clicks", "log_days", "smoothed_ctr", "active_day_fraction"]]
    names += ["days_since_observed", "days_since_click"]
    result = np.zeros((len(keys), len(names)), dtype="float32")
    history = {}
    for day in np.unique(keys[:, 0]):
        rows = np.flatnonzero(keys[:, 0] == day)
        ordinal = pd.Timestamp(str(int(day))).toordinal()
        for row in rows:
            records = history.get(int(keys[row, 1]), [])
            values = []
            for window in [None, 3, 7]:
                prior = records if window is None else [r for r in records if r[0] >= ordinal-window]
                exposures = sum(r[1] for r in prior)
                clicks = sum(r[2] for r in prior)
                active = sum(r[2] > 0 for r in prior)
                values += [np.log1p(exposures), np.log1p(clicks), np.log1p(len(prior)),
                           (clicks+.5)/(exposures+50), (active+.2)/(len(prior)+1)]
            positive = [r[0] for r in records if r[2] > 0]
            values += [min(30, ordinal-records[-1][0]) if records else 30,
                       min(30, ordinal-positive[-1]) if positive else 30]
            result[row] = values
        # No held outcomes are ever inserted, including when several held dates exist.
        if day < cutoff:
            for row in rows:
                history.setdefault(int(keys[row, 1]), []).append((ordinal, int(sizes[row]), float(counts[row])))
    return result, names


def load_day(day, with_history, evaluation_labels=True):
    cutoff = 20191100 + day
    directory = CONTEXTS[day]
    context = np.load(directory / "context.npz")
    metadata = json.loads((directory / "context_manifest.json").read_text(encoding="utf8"))
    keys, sizes = context["keys"], context["sizes"]
    values = context["values"]
    if metadata["ad_labels_in_context"] is not False:
        raise AssertionError("Context label contract failed")
    if array_hash(values) != metadata["context_values_sha256"]:
        raise AssertionError("Context content changed")
    if any("target_stat__" in name for name in metadata["features"]):
        raise AssertionError("Outcome-derived context found")
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["pt_d", "user_id", "label"], dtype="int32")
    earlier = raw.pt_d.to_numpy() < cutoff
    valid_rows = np.flatnonzero(raw.pt_d.to_numpy() == cutoff)
    np.testing.assert_array_equal(valid_rows, np.load(REFERENCE / f"day{day}_indices.npy"))
    key_index = pd.MultiIndex.from_arrays([keys[:, 0], keys[:, 1]])
    positions = key_index.get_indexer(pd.MultiIndex.from_frame(raw[["pt_d", "user_id"]]))
    if np.any(positions[earlier] < 0) or np.any(positions[valid_rows] < 0):
        raise AssertionError("Raw records missing context keys")
    np.testing.assert_array_equal(np.bincount(positions[earlier | (raw.pt_d.to_numpy() == cutoff)], minlength=len(keys)), sizes)
    counts = np.full(len(keys), -1., dtype="float64")
    training = keys[:, 0] < cutoff
    totals = np.bincount(positions[earlier], weights=raw.label.to_numpy()[earlier], minlength=len(keys))
    counts[training] = totals[training]
    if "any_targets" in context.files:
        np.testing.assert_array_equal(context["any_targets"][training], (counts[training] > 0).astype("int8"))
        if not np.all(context["any_targets"][~training] == -1):
            raise AssertionError("Held context targets not masked")
    names = list(metadata["features"])
    if with_history:
        past, past_names = past_outcomes(keys, sizes, counts, cutoff)
        values = np.column_stack([values, past]).astype("float32")
        names += past_names
    reference = np.load(REFERENCE / f"day{day}_prediction.npy")
    row_groups = positions[valid_rows]
    result = {"keys": keys, "x": values, "names": names, "sizes": sizes,
              "counts": counts, "training": training, "valid": keys[:, 0] == cutoff,
              "rows": valid_rows, "row_groups": row_groups,
              "reference": reference}
    if evaluation_labels:
        result["y"] = raw.label.to_numpy()[valid_rows].copy()
    # A masked training target array is the only target array retained for fitting.
    del raw, totals, positions
    return result


def correct(base, group_prediction, row_groups, objective, power):
    if power == 0:
        return base.astype("float64").copy()
    base = np.clip(np.asarray(base, dtype="float64"), 1e-8, 1-1e-8)
    if objective == "binary":
        hazard = -np.log1p(-base)
        target = -np.log1p(-np.clip(group_prediction, 1e-6, 1-1e-6))
    elif objective == "poisson":
        hazard = base
        target = np.maximum(group_prediction, 1e-6)
    else:
        raise ValueError("Unknown user-day objective")
    mass = np.bincount(row_groups, weights=hazard, minlength=len(group_prediction))
    ratio = np.clip(target / mass.clip(1e-8), .05, 20.) ** power
    if objective == "binary":
        return -np.expm1(-hazard*ratio[row_groups])
    odds = base/(1-base)*ratio[row_groups]
    return odds/(1+odds)


def full_predictions(model, data):
    result = np.zeros(len(data["keys"]), dtype="float64")
    result[data["valid"]] = model.predict(data["x"][data["valid"]])
    return result


def fit_candidate(data, objective, leaves, rounds=None):
    target = data["counts"].copy()
    if objective == "binary":
        target[data["training"]] = target[data["training"]] > 0
    train = lgb.Dataset(data["x"][data["training"]], label=target[data["training"]],
                        feature_name=[f"f{i}" for i in range(len(data["names"]))])
    params = dict(PARAMS, objective=objective, metric="binary_logloss" if objective == "binary" else "poisson", num_leaves=leaves)
    if rounds is not None:
        return lgb.train(params, train, num_boost_round=rounds)
    validation_target = np.bincount(data["row_groups"], weights=data["y"], minlength=len(target))[data["valid"]]
    if objective == "binary":
        validation_target = validation_target > 0
    validation = lgb.Dataset(data["x"][data["valid"]], label=validation_target, reference=train)
    return lgb.train(params, train, num_boost_round=700, valid_sets=[validation],
                     callbacks=[lgb.early_stopping(70, verbose=False)])


def input_paths():
    paths = [Path(__file__), HERE / "fit_stacker.py", HERE / "verify_stacker.py",
             ROOT / "analysis/userday_aux_20261101/day_context.py"]
    for day, directory in CONTEXTS.items():
        paths += [directory / "context.npz", directory / "context_manifest.json",
                  REFERENCE / f"day{day}_indices.npy", REFERENCE / f"day{day}_prediction.npy"]
    paths += [HERE / "runs/weighted_calibration_transfer_v1/best_prediction.npy",
              HERE / "runs/weighted_calibration_transfer_v1/verification.json"]
    return paths


def main():
    if OUT.exists():
        raise FileExistsError("Preserving existing user-day experiment")
    OUT.mkdir(parents=True)
    raw_hashes = {p.name: digest(p) for p in (ROOT / "2026ml").glob("*.csv")}
    protocol = {"inputs_sha256": {str(p): digest(p) for p in input_paths()}, "raw_datasets_sha256": raw_hashes,
                "training_cutoffs": [20191115, 20191116], "objectives": ["binary", "poisson"],
                "scopes": ["context", "context_history"], "leaves": LEAVES, "powers": POWERS,
                "params": PARAMS, "max_rounds": 700, "patience": 70,
                "group_weighting": "Each user-day has unit weight, regardless of impression count",
                "selection": "Day15 group loss selects rounds; day15 row AUC selects correction power. All12 models locked before16 outcomes compared.",
                "evaluation16_labels_fit": False, "development16_model_selection": True,
                "prior_work_difference": "Old119-feature Poisson tree replaced by591/598-feature context, binary any-click objective, and optional strictly past outcomes"}
    write(OUT / "protocol.json", protocol)
    selected, locks = [], []
    for history in [False, True]:
        scope = "context_history" if history else "context"
        data = load_day(15, history)
        for objective in protocol["objectives"]:
            for leaves in LEAVES:
                name = f"{scope}_{objective}_{leaves}"
                model = fit_candidate(data, objective, leaves)
                group_prediction = full_predictions(model, data)
                records = [{"power": power, "auc": rank_sum_auc(data["y"], correct(data["reference"], group_prediction, data["row_groups"], objective, power))}
                           for power in POWERS]
                best = max(records, key=lambda r: r["auc"])
                record = {"name": name, "scope": scope, "history": history, "objective": objective,
                          "leaves": leaves, "rounds": model.best_iteration, "power": best["power"],
                          "selected15_auc": best["auc"], "day15_scores": records}
                selected.append(record)
                write(OUT / "selection15.json", selected)
                print(json.dumps(record), flush=True)
        del data
        gc.collect()
    write(OUT / "selection15_locked.json", {"selected": max(selected, key=lambda r: r["selected15_auc"]),
                                             "selection_sha256": digest(OUT / "selection15.json")})
    # Fit all day16 candidate models without giving their fitter held outcomes.
    for history in [False, True]:
        data = load_day(16, history, evaluation_labels=False)
        for record in [r for r in selected if r["history"] == history]:
            model = fit_candidate(data, record["objective"], record["leaves"], record["rounds"])
            path = OUT / (record["name"] + ".txt")
            path.write_text(model.model_to_string(), encoding="utf8")
            locks.append(dict(record, model_sha256=digest(path), features_sha256=array_hash(data["x"]),
                              training_indices_sha256=array_hash(data["keys"][data["training"]]),
                              training_targets_sha256=array_hash(data["counts"][data["training"]])))
        del data
        gc.collect()
    write(OUT / "models_locked.json", {"models": locks, "protocol_sha256": digest(OUT / "protocol.json"),
                                         "evaluation16_outcomes_fit": False})
    evaluate(False)


def evaluate(verifying):
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    lock = json.loads((OUT / "models_locked.json").read_text(encoding="utf8"))
    if digest(OUT / "protocol.json") != lock["protocol_sha256"]:
        raise AssertionError("Protocol changed")
    for path, expected in protocol["inputs_sha256"].items():
        if digest(path) != expected:
            raise AssertionError("Source/input changed: " + path)
    for name, expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name) != expected:
            raise AssertionError("Original data changed")
    rows, all_records, all_predictions = None, [], []
    for history in [False, True]:
        data = load_day(16, history)
        rows = data["rows"]
        if len(rows) != 421983 or int(data["y"].sum()) != 3724:
            raise AssertionError("Complete real16 cohort changed")
        for record in [r for r in lock["models"] if r["history"] == history]:
            path = OUT / (record["name"] + ".txt")
            if digest(path) != record["model_sha256"] or array_hash(data["x"]) != record["features_sha256"]:
                raise AssertionError("Model/features changed")
            if array_hash(data["keys"][data["training"]]) != record["training_indices_sha256"]:
                raise AssertionError("Training boundary changed")
            if array_hash(data["counts"][data["training"]]) != record["training_targets_sha256"]:
                raise AssertionError("Training outcomes changed")
            model = lgb.Booster(model_str=path.read_text(encoding="utf8"))
            group_prediction = full_predictions(model, data)
            sources = {"reference": data["reference"], "weighted_transfer": np.load(HERE / "runs/weighted_calibration_transfer_v1/best_prediction.npy")}
            for source, base in sources.items():
                prediction = correct(base, group_prediction, data["row_groups"], record["objective"], record["power"])
                item = {k: record[k] for k in ["name", "scope", "objective", "leaves", "rounds", "power", "selected15_auc"]}
                item.update(source=source, auc=rank_sum_auc(data["y"], prediction))
                all_records.append(item)
                all_predictions.append(prediction)
        del data
        gc.collect()
    best = int(np.argmax([r["auc"] for r in all_records]))
    chosen = all_predictions[best]
    if verifying:
        error = float(np.max(np.abs(chosen-np.load(OUT / "prediction.npy"))))
        if error > 1e-12:
            raise AssertionError("Independent replay differs")
        np.testing.assert_array_equal(rows, np.load(OUT / "valid_indices.npy"))
    else:
        np.save(OUT / "prediction.npy", chosen)
        np.save(OUT / "valid_indices.npy", rows)
        error = 0.
    result = {"status": "passed" if verifying else "fitted_and_evaluated", "auc": all_records[best]["auc"],
              "best": all_records[best], "records": all_records, "rows": len(rows), "clicks": 3724,
              "maximum_replay_error": error, "evaluation16_labels_fit": False,
              "development16_model_selection": True, "original_datasets_unchanged": True,
              "target_met": all_records[best]["auc"] >= .82}
    write(OUT / ("verification.json" if verifying else "result.json"), result)
    print(json.dumps({k:v for k,v in result.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    evaluate(True) if parser.parse_args().verify else main()
