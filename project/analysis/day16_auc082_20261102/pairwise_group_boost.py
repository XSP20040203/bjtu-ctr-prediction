"""Learn user-day logit corrections with a sampled global row-ranking objective."""
import os
for _key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[_key] = "2"
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from rich_group_residual import (ROOT, HERE, REFERENCE, digest, write, prepare, aggregate,
                                 array_hash, rank_sum_auc, lgb)

OUT = HERE / "runs/pairwise_group_boost_v1"
TEMPERATURES = [.5, 1.]
LEAVES = [3, 7, 15]
PARAMS = {"learning_rate": .03, "min_data_in_leaf": 100, "lambda_l2": 2.,
          "feature_fraction": .8, "max_bin": 31, "num_threads": 2,
          "seed": 42, "verbosity": -1, "deterministic": True,
          "force_col_wise": True, "boost_from_average": False, "metric": "None",
          "max_delta_step": .5}
NEGATIVES_PER_POSITIVE = 64
RIDGE = .02


class PairObjective:
    def __init__(self, a, b, margin, n, temperature, ridge=RIDGE):
        self.a, self.b, self.margin = a, b, margin
        self.n, self.temperature, self.ridge = n, temperature, ridge
        if len(a) == 0 or len(a) != len(b) or len(a) != len(margin) or np.any(a == b):
            raise ValueError("Nonempty cross-group pairs required")

    def value(self, score):
        difference = (self.margin+score[self.a]-score[self.b])/self.temperature
        return self.n*np.logaddexp(0., -difference).mean()+.5*self.ridge*np.dot(score, score)

    def __call__(self, score, dataset):
        difference = (self.margin+score[self.a]-score[self.b])/self.temperature
        probability = expit(-difference)
        first = probability/self.temperature
        second = probability*(1-probability)/(self.temperature**2)
        scale = self.n/len(self.a)
        grad = (np.bincount(self.b, weights=first, minlength=self.n)-np.bincount(self.a, weights=first, minlength=self.n))*scale
        hess = (np.bincount(self.a, weights=second, minlength=self.n)+np.bincount(self.b, weights=second, minlength=self.n))*scale
        return grad+self.ridge*score, hess+self.ridge


def make_objective(data, group_rows, temperature, seed):
    group_map = np.full(len(data["keys"]), -1, dtype="int32")
    group_map[group_rows] = np.arange(len(group_rows))
    eligible = group_map[data["groups"]] >= 0
    positives = np.flatnonzero(eligible & (data["y"] == 1))
    negatives = np.flatnonzero(eligible & (data["y"] == 0))
    if not len(positives) or not len(negatives):
        raise ValueError("Both classes required")
    rng = np.random.default_rng(seed)
    pos = np.repeat(positives, NEGATIVES_PER_POSITIVE)
    neg = rng.choice(negatives, size=len(pos), replace=True)
    a, b = group_map[data["groups"][pos]], group_map[data["groups"][neg]]
    keep = a != b
    base_margin = logit(data["base"].clip(1e-6, 1-1e-6))
    margin = base_margin[pos]-base_margin[neg]
    return PairObjective(a[keep], b[keep], margin[keep], len(group_rows), temperature)


def row_auc_metric(data, group_rows):
    group_map = np.full(len(data["keys"]), -1, dtype="int32")
    group_map[group_rows] = np.arange(len(group_rows))
    row_mask = group_map[data["groups"]] >= 0
    group_positions = group_map[data["groups"][row_mask]]
    margin = logit(data["base"][row_mask].clip(1e-6, 1-1e-6))
    y = data["y"][row_mask]
    def metric(score, dataset):
        return "row_auc", rank_sum_auc(y, margin+score[group_positions]), True
    return metric


def matrix(day, source="reference", include_labels=False):
    data = prepare(day, "reference" if source == "rich_activity" else source, include_labels)
    if source == "rich_activity":
        data["base"] = np.load(HERE / "runs/rich_userday_activity_v1/prediction.npy")
        mass, any_probability, features = aggregate(data["base"], data["groups"], len(data["keys"]))
        data["mass"], data["any"], data["x"][:, -4:] = mass, any_probability, features
    return data


def main():
    if OUT.exists():
        raise FileExistsError("Preserving pairwise group experiment")
    OUT.mkdir(parents=True)
    source_paths = [Path(__file__), HERE / "rich_group_residual.py", HERE / "rich_userday_activity.py",
                    HERE / "fit_stacker.py", HERE / "verify_stacker.py",
                    HERE / "runs/weighted_v1_day16/context.npz", HERE / "runs/weighted_v1_day16/context_manifest.json",
                    HERE / "runs/weighted_calibration_transfer_v1/best_prediction.npy",
                    HERE / "runs/rich_userday_activity_v1/prediction.npy",
                    HERE / "runs/rich_userday_activity_v1/verification.json"] + [
                        REFERENCE / f"day{day}_{suffix}.npy" for day in [15,16] for suffix in ["indices", "prediction"]]
    protocol = {"params":PARAMS, "temperatures":TEMPERATURES, "leaves":LEAVES, "ridge":RIDGE,
                "negatives_per_positive":NEGATIVES_PER_POSITIVE, "max_rounds":400, "patience":60,
                "inputs_sha256":{str(p):digest(p) for p in source_paths},
                "raw_datasets_sha256":{p.name:digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
                "fit_date":20191115, "evaluation_date":20191116, "evaluation16_labels_fit":False,
                "pair_sampling":"Fixed uniform negative advertisement rows; only training user groups; self-group pairs removed",
                "selection":"User-group3fold15 rowAUC selects rounds/power; six models locked before16 comparison",
                "powers":[0., .25, .5, 1.], "sources":["reference", "weighted_transfer", "rich_activity"],
                "development16_model_selection":True}
    write(OUT / "protocol.json", protocol)
    data = matrix(15, include_labels=True)
    folds = ((data["keys"][:,1].astype("uint64")*2654435761)%3).astype("int32")
    records, locks = [], []
    for temperature in TEMPERATURES:
        for leaves in LEAVES:
            oof, rounds = np.empty(len(data["keys"])), []
            for fold in range(3):
                tr, va = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                objective = make_objective(data, tr, temperature, 4242+fold)
                params = dict(PARAMS, num_leaves=leaves, objective=objective)
                train = lgb.Dataset(data["x"][tr], label=np.zeros(len(tr)))
                validation = lgb.Dataset(data["x"][va], label=np.zeros(len(va)), reference=train)
                model = lgb.train(params, train, num_boost_round=400, valid_sets=[validation],
                                  feval=row_auc_metric(data, va), callbacks=[lgb.early_stopping(60, verbose=False)])
                oof[va] = model.predict(data["x"][va], raw_score=True)
                rounds.append(model.best_iteration)
            base_margin = logit(data["base"].clip(1e-6, 1-1e-6))
            scores = [{"power":p, "auc":rank_sum_auc(data["y"], base_margin+p*np.clip(oof[data["groups"]],-3,3))}
                      for p in protocol["powers"]]
            best = max(scores, key=lambda r:r["auc"])
            record = {"name":f"t{temperature:g}_l{leaves}", "temperature":temperature, "leaves":leaves,
                      "rounds":rounds, "refit_rounds":max(1,int(np.median(rounds))), "scores":scores,
                      "power":best["power"], "oof15_auc":best["auc"]}
            records.append(record)
            np.save(OUT / (record["name"]+"_oof.npy"), oof)
            write(OUT / "cv.json", records)
            print(json.dumps(record), flush=True)
    for record in records:
        all_groups = np.arange(len(data["keys"]))
        objective = make_objective(data, all_groups, record["temperature"], 4242)
        model = lgb.train(dict(PARAMS, num_leaves=record["leaves"], objective=objective),
                          lgb.Dataset(data["x"], label=np.zeros(len(all_groups))), num_boost_round=record["refit_rounds"])
        path = OUT / (record["name"]+".txt")
        path.write_text(model.model_to_string(), encoding="utf8")
        locks.append(dict(record, model_sha256=digest(path), pair_a_sha256=array_hash(objective.a),
                          pair_b_sha256=array_hash(objective.b), pair_margin_sha256=array_hash(objective.margin)))
    write(OUT / "models_locked.json", {"models":locks, "protocol_sha256":digest(OUT / "protocol.json"),
              "training_features_sha256":array_hash(data["x"]), "training_y_sha256":array_hash(data["y"]),
              "training_keys_sha256":array_hash(data["keys"])})
    del data
    gc.collect()
    evaluate(False)


def evaluate(verifying):
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    lock = json.loads((OUT / "models_locked.json").read_text(encoding="utf8"))
    if lock["protocol_sha256"] != digest(OUT / "protocol.json"):
        raise AssertionError("Protocol changed")
    for path, expected in protocol["inputs_sha256"].items():
        if digest(path) != expected:
            raise AssertionError("Bound source/input changed")
    for name, expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name) != expected:
            raise AssertionError("Original data changed")
    if verifying:
        train = matrix(15, include_labels=True)
        for field in ["x", "y", "keys"]:
            key = {"x":"training_features_sha256", "y":"training_y_sha256", "keys":"training_keys_sha256"}[field]
            if array_hash(train[field]) != lock[key]:
                raise AssertionError("Training reconstruction differs")
        for record in lock["models"]:
            objective = make_objective(train, np.arange(len(train["keys"])), record["temperature"], 4242)
            for attr in ["a", "b", "margin"]:
                if array_hash(getattr(objective, attr)) != record[f"pair_{attr}_sha256"]:
                    raise AssertionError("Pair sampling changed")
        del train
    records, predictions = [], []
    for source in protocol["sources"]:
        data = matrix(16, source, include_labels=True)
        if len(data["y"]) != 421983 or int(data["y"].sum()) != 3724 or np.any(data["count"] != -1):
            raise AssertionError("Held data boundary differs")
        base_margin = logit(data["base"].clip(1e-6, 1-1e-6))
        for record in lock["models"]:
            path = OUT / (record["name"]+".txt")
            if digest(path) != record["model_sha256"]:
                raise AssertionError("Model changed")
            model = lgb.Booster(model_str=path.read_text(encoding="utf8"))
            residual = np.clip(model.predict(data["x"], raw_score=True), -3, 3)
            p = expit(base_margin+record["power"]*residual[data["groups"]])
            records.append({"name":record["name"], "source":source, "power":record["power"],
                            "oof15_auc":record["oof15_auc"], "auc":rank_sum_auc(data["y"],p)})
            predictions.append(p)
    best = int(np.argmax([r["auc"] for r in records]))
    p = predictions[best]
    if verifying:
        error = float(np.max(np.abs(p-np.load(OUT / "prediction.npy"))))
        if error > 1e-12:
            raise AssertionError("Prediction replay differs")
        np.testing.assert_array_equal(data["rows"], np.load(OUT / "valid_indices.npy"))
    else:
        error = 0.
        np.save(OUT / "prediction.npy", p)
        np.save(OUT / "valid_indices.npy", data["rows"])
    result = {"status":"passed" if verifying else "evaluated", "auc":records[best]["auc"], "best":records[best],
              "records":records, "rows":len(p), "clicks":3724, "maximum_replay_error":error,
              "original_datasets_unchanged":True, "evaluation16_labels_fit":False,
              "development16_model_selection":True, "target_met":records[best]["auc"] >= .82}
    write(OUT / ("verification.json" if verifying else "result.json"), result)
    print(json.dumps({k:v for k,v in result.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    evaluate(True) if parser.parse_args().verify else main()
