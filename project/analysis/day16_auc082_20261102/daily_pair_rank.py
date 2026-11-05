"""Learn daily click propensity by ranking historical positive/negative rows."""
import os
for _key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[_key] = "2"
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from feed_behavior_userday import prepare, dependency_paths
from pairwise_group_boost import PairObjective, PARAMS as RANK_PARAMS, lgb
from fit_stacker import HERE, ROOT, REFERENCE, digest, write
from verify_stacker import rank_sum_auc
from rich_userday_activity import array_hash

OUT = HERE/"runs/daily_pair_rank_v1"
SCHEMA = HERE/"runs/feed_behavior_userday_v1/news_schema.json"
BASE = HERE/"runs/compose_verified_v1"
POWERS = [0., .125, .25, .5, .75, 1.]
PARAMS = dict(RANK_PARAMS, lambda_l2=10., feature_fraction=.75, min_data_in_leaf=100)


def pairs(keys, sizes, counts, selected, temperature, seed=4462):
    if np.any(keys[selected, 0] >= 20191116) or np.any(counts[selected] < 0):
        raise ValueError("Historical observed groups required")
    negative = sizes-counts
    if np.any(negative[selected] < 0):
        raise ValueError("Click counts exceed exposure counts")
    index = np.full(len(keys), -1, dtype="int32")
    index[selected] = np.arange(len(selected))
    rng = np.random.default_rng(seed)
    aa, bb = [], []
    for day in np.unique(keys[selected, 0]):
        day_rows = selected[keys[selected, 0] == day]
        pos = day_rows[counts[day_rows] > 0]
        neg = day_rows[negative[day_rows] > 0]
        if not len(pos) or not len(neg):
            continue
        total = int(counts[pos].sum())*48
        a = rng.choice(pos, total, p=counts[pos]/counts[pos].sum())
        b = rng.choice(neg, total, p=negative[neg]/negative[neg].sum())
        keep = a != b
        aa.append(index[a[keep]])
        bb.append(index[b[keep]])
    a, b = np.concatenate(aa), np.concatenate(bb)
    return PairObjective(a, b, np.zeros(len(a)), len(selected), temperature, ridge=.02)


def correct(base, group_scores, groups, sizes, power):
    if power == 0:
        return base.astype("float64").copy()
    mean = np.bincount(groups, weights=base, minlength=len(group_scores))/sizes
    delta = group_scores-logit(mean.clip(1e-6, 1.-1e-6))
    return expit(logit(base.clip(1e-6, 1.-1e-6))+power*delta[groups])


def row_mapping(data, day):
    raw = pd.read_csv(ROOT/"2026ml/train.csv", usecols=["pt_d", "user_id", "label"], dtype="int32")
    rows = np.flatnonzero(raw.pt_d.to_numpy() == day)
    chosen = data["keys"][:, 0] == day
    index = pd.MultiIndex.from_arrays([data["keys"][chosen, 0], data["keys"][chosen, 1]])
    groups = index.get_indexer(pd.MultiIndex.from_frame(raw.iloc[rows][["pt_d", "user_id"]]))
    if np.any(groups < 0):
        raise AssertionError("Row group absent")
    return chosen, rows, groups, raw.label.to_numpy()[rows]


def deps():
    return list(dict.fromkeys([Path(__file__), HERE/"pairwise_group_boost.py", SCHEMA,
        BASE/"verification.json", BASE/"prediction.npy"]+dependency_paths()))


def main():
    if OUT.exists():
        raise FileExistsError("Preserving direct daily ranking")
    expected = json.loads((HERE/"runs/weighted_v1_day16/protocol.json").read_text(encoding="utf8"))["raw_datasets_sha256"]
    for name, value in expected.items():
        if digest(ROOT/"2026ml"/name) != value:
            raise AssertionError("Original dataset changed")
    OUT.mkdir(parents=True)
    protocol = {"params": PARAMS, "leaves": [7, 15], "temperatures": [.5, 1.], "powers": POWERS,
                "max_rounds": 700, "patience": 70, "inputs_sha256": {str(p): digest(p) for p in deps()},
                "raw_datasets_sha256": expected,
                "pairs": "Within each historical date, sample positive groups by click count and negative groups by nonclick exposure count; remove constant same-group pairs",
                "fit_rule": "Pre15 selects rounds/power; pre16 refit; four models locked before16 scoring",
                "evaluation16_labels_fit": False, "development16_model_selection": True}
    write(OUT/"protocol.json", protocol)
    data, _ = prepare(json.loads(SCHEMA.read_text(encoding="utf8")))
    chosen, rows, groups, y = row_mapping(data, 20191115)
    selected = np.flatnonzero(data["keys"][:, 0] < 20191115)
    refit = np.flatnonzero(data["keys"][:, 0] < 20191116)
    base = np.load(REFERENCE/"day15_prediction.npy")
    sizes = data["sizes"][chosen]
    np.testing.assert_array_equal(rows, np.load(REFERENCE/"day15_indices.npy"))
    records = []
    for temperature in protocol["temperatures"]:
        for leaves in protocol["leaves"]:
            objective = pairs(data["keys"], data["sizes"], data["counts"], selected, temperature)
            def metric(score, dataset):
                scores = [rank_sum_auc(y, correct(base, score, groups, sizes, power)) for power in POWERS[1:]]
                return "row_auc", max(scores), True
            model = lgb.train(dict(PARAMS, num_leaves=leaves, objective=objective),
                              lgb.Dataset(data["x"][selected], label=np.zeros(len(selected))),
                              num_boost_round=700,
                              valid_sets=[lgb.Dataset(data["x"][chosen], label=np.zeros(chosen.sum()))],
                              feval=metric, callbacks=[lgb.early_stopping(70, verbose=False)])
            score = model.predict(data["x"][chosen], raw_score=True)
            power_scores = [{"power": power, "auc": rank_sum_auc(y, correct(base, score, groups, sizes, power))} for power in POWERS]
            best = max(power_scores, key=lambda row: row["auc"])
            name = f"t{temperature:g}_l{leaves}"
            objective = pairs(data["keys"], data["sizes"], data["counts"], refit, temperature)
            final = lgb.train(dict(PARAMS, num_leaves=leaves, objective=objective),
                              lgb.Dataset(data["x"][refit], label=np.zeros(len(refit))),
                              num_boost_round=max(model.best_iteration, 1))
            path = OUT/(name+".txt")
            path.write_text(final.model_to_string(), encoding="utf8")
            record = {"name": name, "temperature": temperature, "leaves": leaves,
                      "rounds": model.best_iteration, "power": best["power"], "validation15_auc": best["auc"],
                      "model_sha256": digest(path), "a_sha256": array_hash(objective.a), "b_sha256": array_hash(objective.b)}
            records.append(record)
            write(OUT/"selection15.json", records)
            print(json.dumps(record), flush=True)
    write(OUT/"models_locked.json", {"models": records, "protocol_sha256": digest(OUT/"protocol.json"),
          "features_sha256": array_hash(data["x"]), "counts_sha256": array_hash(data["counts"]),
          "keys_sha256": array_hash(data["keys"])})
    del data
    gc.collect()
    evaluate(False)


def evaluate(verifying):
    protocol = json.loads((OUT/"protocol.json").read_text(encoding="utf8"))
    lock = json.loads((OUT/"models_locked.json").read_text(encoding="utf8"))
    if digest(OUT/"protocol.json") != lock["protocol_sha256"]:
        raise AssertionError("Daily rank protocol changed")
    for p, h in protocol["inputs_sha256"].items():
        if digest(p) != h:
            raise AssertionError("Daily rank source changed")
    for name, h in protocol["raw_datasets_sha256"].items():
        if digest(ROOT/"2026ml"/name) != h:
            raise AssertionError("Original data changed")
    data, _ = prepare(json.loads(SCHEMA.read_text(encoding="utf8")))
    for field, key in [("x", "features_sha256"), ("counts", "counts_sha256"), ("keys", "keys_sha256")]:
        if array_hash(data[field]) != lock[key]:
            raise AssertionError("Daily ranking reconstruction differs")
    chosen, rows, groups, y = row_mapping(data, 20191116)
    if len(y) != 421983 or int(y.sum()) != 3724 or np.any(data["counts"][chosen] != -1):
        raise AssertionError("Complete16 and label mask contract failed")
    refit = np.flatnonzero(data["keys"][:, 0] < 20191116)
    base = np.load(BASE/"prediction.npy")
    records, predictions = [], []
    for r in lock["models"]:
        obj = pairs(data["keys"], data["sizes"], data["counts"], refit, r["temperature"])
        if array_hash(obj.a) != r["a_sha256"] or array_hash(obj.b) != r["b_sha256"]:
            raise AssertionError("Daily rank sample replay differs")
        path = OUT/(r["name"]+".txt")
        if digest(path) != r["model_sha256"]:
            raise AssertionError("Daily rank model changed")
        score = lgb.Booster(model_str=path.read_text(encoding="utf8")).predict(data["x"][chosen], raw_score=True)
        p = correct(base, score, groups, data["sizes"][chosen], r["power"])
        records.append({"name": r["name"], "power": r["power"], "auc": rank_sum_auc(y, p)})
        predictions.append(p)
    best = int(np.argmax([r["auc"] for r in records]))
    p = predictions[best]
    error = 0.
    if verifying:
        error = float(np.max(np.abs(p-np.load(OUT/"prediction.npy"))))
        if error > 1e-12:
            raise AssertionError("Daily rank saved prediction differs")
        np.testing.assert_array_equal(rows, np.load(OUT/"valid_indices.npy"))
    else:
        np.save(OUT/"prediction.npy", p)
        np.save(OUT/"valid_indices.npy", rows)
    result = {"status": "passed" if verifying else "evaluated", "auc": records[best]["auc"],
              "records": records, "rows": len(y), "clicks": int(y.sum()), "maximum_replay_error": error,
              "original_datasets_unchanged": True, "evaluation16_labels_fit": False,
              "development16_model_selection": True, "target_met": records[best]["auc"] >= .82}
    write(OUT/("verification.json" if verifying else "result.json"), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    evaluate(True) if parser.parse_args().verify else main()
