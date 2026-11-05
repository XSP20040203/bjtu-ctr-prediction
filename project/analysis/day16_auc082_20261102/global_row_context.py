"""Fit cross-day row trees with detailed daily news context and past ad rates."""
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
from rich_group_residual import ROOT, HERE, REFERENCE, digest, write, lgb, array_hash, rank_sum_auc
from feed_behavior_userday import prepare as daily_inputs
from row_context_residual import CATEGORIES
sys.path.insert(0, str(ROOT/"analysis/auc085_next"))
from prepare_multiscale_stats import build_multiscale_statistics, feature_names

OUT = HERE/"runs/global_row_context_v1"
SELECTOR = HERE/"runs/feed_behavior_userday_v1/binary_15.txt"
SCHEMA = HERE/"runs/feed_behavior_userday_v1/news_schema.json"
BASE = HERE/"runs/previous_day_residual_v1"
CAT = ["user_id"]+CATEGORIES
HISTORY = [f"ad_{action}_list_v00{i}" for action in ["click", "close"] for i in [1, 2, 3]]
TARGET = ["task_id", "adv_prim_id", "spread_app_id"]*2
CHECKPOINTS = [250, 500, 900]
LEAVES = [31, 63]
PARAMS = {"objective": "binary", "learning_rate": .025, "min_data_in_leaf": 500,
          "lambda_l2": 30., "cat_l2": 40., "cat_smooth": 60., "max_bin": 63,
          "max_cat_threshold": 32, "feature_fraction": .85, "num_threads": 2,
          "seed": 4163, "verbosity": -1, "deterministic": True, "force_col_wise": True,
          "feature_pre_filter": False}


def selected_training(labels, dates):
    train = np.flatnonzero(dates < 20191116)
    if not np.isin(labels[train], [0, 1]).all():
        raise ValueError("Only observed earlier outcomes are trainable")
    positive, negative = train[labels[train] == 1], train[labels[train] == 0]
    selected_negative = np.random.default_rng(4163).choice(negative, len(negative)//3, replace=False)
    chosen = np.sort(np.r_[positive, selected_negative])
    weights = np.where(labels[chosen] == 1, 1., len(negative)/len(selected_negative))
    return chosen, weights.astype("float32")


def histories(frame):
    if "label" in frame.columns:
        raise ValueError("Advertisement outcomes are forbidden in history input")
    result = np.empty((len(frame), 12), dtype="float32")
    for i, (column, target) in enumerate(zip(HISTORY, TARGET)):
        lookup = {value: frozenset(int(s) for s in str(value).split("^") if s and s != "-1")
                  for value in frame[column].unique()}
        tokens = [lookup[value] for value in frame[column]]
        result[:, 2*i] = np.log1p([len(value) for value in tokens])
        result[:, 2*i+1] = [int(value) in token for value, token in zip(frame[target], tokens)]
    return result


def dependencies():
    return [Path(__file__), HERE/"row_context_residual.py", HERE/"feed_behavior_userday.py",
            HERE/"personal_context_residual.py", HERE/"extended_history_residual.py",
            HERE/"rich_userday_activity.py", HERE/"rich_group_residual.py", HERE/"fit_stacker.py",
            HERE/"verify_stacker.py", ROOT/"analysis/auc085_next/prepare_multiscale_stats.py",
            SELECTOR, SCHEMA, BASE/"prediction.npy", BASE/"verification.json",
            HERE/"runs/weighted_v1_day16/context.npz",
            HERE/"runs/weighted_v1_day16/context_manifest.json"]


def build(directory):
    directory.mkdir(parents=True)
    raw = pd.read_csv(ROOT/"2026ml/train.csv", usecols=CAT+["pt_d", "label"], dtype="int32")
    labels, dates = raw.label.to_numpy(copy=True), raw.pt_d.to_numpy()
    train, weights = selected_training(labels, dates)
    held = np.flatnonzero(dates == 20191116)
    rows = np.sort(np.r_[train, held])
    train_positions = np.flatnonzero(dates[rows] < 20191116)
    valid_positions = np.flatnonzero(dates[rows] == 20191116)
    np.testing.assert_array_equal(rows[train_positions], train)
    labels[dates >= 20191116] = -1
    raw["label"] = labels
    np.testing.assert_array_equal(held, np.load(REFERENCE/"day16_indices.npy"))
    selector = lgb.Booster(model_str=SELECTOR.read_text(encoding="utf8"))
    columns = np.argsort(-selector.feature_importance("gain"), kind="stable")[:96]
    stats = np.lib.format.open_memmap(directory/"past_rates.npy", mode="w+", dtype="float32",
                                     shape=(len(raw), len(feature_names())), fortran_order=True)
    def progress(index, total, fields, size):
        print(json.dumps({"event": "global_row_past_rates", "group": index, "of": total}), flush=True)
    stats, names = build_multiscale_statistics(raw, 20191116, output=stats, progress=progress)
    schema = json.loads(SCHEMA.read_text(encoding="utf8"))
    daily, _ = daily_inputs(schema)
    if np.any(daily["counts"][daily["keys"][:, 0] >= 20191116] != -1):
        raise AssertionError("Held daily outcomes must remain masked")
    groups = pd.MultiIndex.from_arrays([daily["keys"][:, 0], daily["keys"][:, 1]])
    positions = groups.get_indexer(pd.MultiIndex.from_frame(raw.iloc[rows][["pt_d", "user_id"]]))
    if np.any(positions < 0):
        raise AssertionError("Missing row context")
    names = CAT+names+["daily__"+daily["names"][int(i)] for i in columns]
    names += [column+suffix for column in HISTORY for suffix in ["_log_unique", "_match"]]
    x = np.lib.format.open_memmap(directory/"features.npy", mode="w+", dtype="float32",
                                 shape=(len(rows), len(names)))
    for start in range(0, len(rows), 20000):
        stop = min(start+20000, len(rows))
        current = rows[start:stop]
        categorical = raw.iloc[current][CAT].to_numpy()
        np.testing.assert_array_equal(categorical.astype("float32").astype("int64"), categorical)
        x[start:stop, :len(CAT)] = categorical
        x[start:stop, len(CAT):len(CAT)+len(feature_names())] = stats[current]
        x[start:stop, len(CAT)+len(feature_names()):-12] = daily["x"][positions[start:stop]][:, columns]
    x.flush()
    del stats, daily
    gc.collect()
    # Only the dedicated temporary generated matrix is removed; raw inputs stay untouched.
    (directory/"past_rates.npy").unlink()
    global_to_local = np.full(len(raw), -1, dtype="int32")
    global_to_local[rows] = np.arange(len(rows))
    offset = 0
    for chunk in pd.read_csv(ROOT/"2026ml/train.csv", usecols=HISTORY+list(set(TARGET)),
                             dtype={name: str for name in HISTORY}, chunksize=100000):
        local = global_to_local[offset:offset+len(chunk)]
        keep = local >= 0
        if np.any(keep):
            x[local[keep], -12:] = histories(chunk.loc[keep])
        offset += len(chunk)
    x.flush()
    if offset != len(raw):
        raise AssertionError("History chunk alignment changed")
    for start in range(0, len(rows), 20000):
        if not np.isfinite(x[start:start+20000]).all():
            raise AssertionError("Nonfinite global row input")
    contract = {"rows_sha256": array_hash(rows), "train_rows_sha256": array_hash(train),
                "training_labels_sha256": array_hash(labels[train]), "features_sha256": digest(directory/"features.npy"),
                "weights_sha256": array_hash(weights), "feature_names": names,
                "selector_columns": columns.tolist(), "fit_rows": len(train),
                "fit_positive_rows": int(labels[train].sum()), "max_fit_date": int(dates[train].max()),
                "negative_inverse_probability_weight": float(weights.max()), "held_label_values": [-1]}
    return x, rows, train_positions, valid_positions, labels[train], weights, contract


def recipes(p, base):
    yield "single", p
    for weight in [.05, .1, .25, .5]:
        yield f"probability_{weight:g}", (1.-weight)*base+weight*p
        a, b = logit(base.clip(1e-6, 1.-1e-6)), logit(p.clip(1e-6, 1.-1e-6))
        b = (b-b.mean())*(a.std()/max(b.std(), 1e-6))+a.mean()
        yield f"logit_{weight:g}", expit((1.-weight)*a+weight*b)


def main():
    if OUT.exists():
        raise FileExistsError("Preserving global row context experiment")
    expected_raw = json.loads((HERE/"runs/weighted_v1_day16/protocol.json").read_text(encoding="utf8"))["raw_datasets_sha256"]
    for name, expected in expected_raw.items():
        if digest(ROOT/"2026ml"/name) != expected:
            raise AssertionError("Original dataset changed before fitting")
    OUT.mkdir(parents=True)
    protocol = {"inputs_sha256": {str(p): digest(p) for p in dependencies()},
                "raw_datasets_sha256": expected_raw,
                "params": PARAMS, "leaves": LEAVES, "checkpoints": CHECKPOINTS,
                "selection": "Fixed tree counts; all models locked before reading16 outcomes. Reused16 selects fixed blend recipes.",
                "sampling": "All earlier positives and one third of earlier negatives, without replacement; inverse inclusion weighting",
                "context_columns": "96 highest-gain columns of a verified pre16 daily classifier; no independent15 validation claim",
                "evaluation16_labels_fit": False, "development_day_reused": True}
    write(OUT/"protocol.json", protocol)
    x, rows, train, valid, y, weights, contract = build(OUT/"input")
    write(OUT/"input_contract.json", contract)
    np.save(OUT/"valid_indices.npy", rows[valid])
    print(json.dumps({"event": "global_row_input_ready", "shape": list(x.shape),
                      "fit_rows": len(train), "clicks": int(y.sum())}), flush=True)
    dataset = lgb.Dataset(np.asarray(x[train]), label=y, weight=weights,
                          categorical_feature=list(range(len(CAT))), free_raw_data=True,
                          params=PARAMS)
    dataset.construct()
    locks = []
    for leaves in LEAVES:
        def checkpoint(environment):
            count = environment.iteration+1
            if count in CHECKPOINTS:
                name = f"leaves{leaves}_trees{count}"
                path = OUT/(name+".txt")
                path.write_text(environment.model.model_to_string(), encoding="utf8")
                locks.append({"name": name, "leaves": leaves, "trees": count, "sha256": digest(path)})
                print(json.dumps({"event": "global_row_checkpoint", "name": name}), flush=True)
        lgb.train(dict(PARAMS, num_leaves=leaves), dataset, num_boost_round=max(CHECKPOINTS), callbacks=[checkpoint])
    write(OUT/"models_locked.json", {"models": locks, "protocol_sha256": digest(OUT/"protocol.json")})
    evaluate(x, rows[valid], valid, False)


def evaluate(x, rows, valid, verifying):
    lock = json.loads((OUT/"models_locked.json").read_text(encoding="utf8"))
    base = np.load(BASE/"prediction.npy")
    raw = pd.read_csv(ROOT/"2026ml/train.csv", usecols=["label", "pt_d"], dtype="int32")
    np.testing.assert_array_equal(rows, np.flatnonzero(raw.pt_d.to_numpy() == 20191116))
    np.testing.assert_array_equal(rows, np.load(OUT/"valid_indices.npy"))
    y = raw.label.to_numpy()[rows]
    if len(y) != 421983 or int(y.sum()) != 3724:
        raise AssertionError("Complete16 cohort changed")
    records, predictions = [], []
    for model in lock["models"]:
        path = OUT/(model["name"]+".txt")
        if digest(path) != model["sha256"]:
            raise AssertionError("Global row model changed")
        predictor = lgb.Booster(model_str=path.read_text(encoding="utf8"))
        p = predictor.predict(x[valid])
        for recipe, values in recipes(p, base):
            records.append({"name": model["name"], "recipe": recipe, "auc": rank_sum_auc(y, values)})
            predictions.append(values)
    best = int(np.argmax([r["auc"] for r in records]))
    p = predictions[best]
    error = 0.
    if verifying:
        error = float(np.max(np.abs(p-np.load(OUT/"prediction.npy"))))
        if error > 1e-12:
            raise AssertionError("Global row model replay differs")
    else:
        np.save(OUT/"prediction.npy", p)
    result = {"status": "passed" if verifying else "evaluated", "auc": records[best]["auc"],
              "best": records[best], "records": records, "rows": len(y), "clicks": int(y.sum()),
              "maximum_replay_error": error, "original_datasets_unchanged": True,
              "evaluation16_labels_fit": False, "development_day_reused": True,
              "target_met": records[best]["auc"] >= .82}
    write(OUT/("verification.json" if verifying else "result.json"), result)
    print(json.dumps({k:v for k,v in result.items() if k != "records"}, indent=2), flush=True)


def verify():
    protocol = json.loads((OUT/"protocol.json").read_text(encoding="utf8"))
    lock = json.loads((OUT/"models_locked.json").read_text(encoding="utf8"))
    if digest(OUT/"protocol.json") != lock["protocol_sha256"]:
        raise AssertionError("Global row protocol changed")
    for p, h in protocol["inputs_sha256"].items():
        if digest(p) != h:
            raise AssertionError("Global row dependency changed")
    for name, h in protocol["raw_datasets_sha256"].items():
        if digest(ROOT/"2026ml"/name) != h:
            raise AssertionError("Original dataset changed")
    x, rows, train, valid, y, weights, contract = build(OUT/"independent_input")
    expected = json.loads((OUT/"input_contract.json").read_text(encoding="utf8"))
    if contract != expected:
        raise AssertionError("Independent global row reconstruction differs")
    evaluate(x, rows[valid], valid, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    verify() if parser.parse_args().verify else main()
