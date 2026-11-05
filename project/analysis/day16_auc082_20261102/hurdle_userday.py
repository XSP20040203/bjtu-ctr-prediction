"""Forecast user-day click mass as P(any click) times conditional click count.

The occurrence model is selected from frozen day15 out-of-fold predictions.
Severity models are selected using a genuine pre15 -> day15 time split and
refitted only on positive user-days strictly before16. Day16 remains reused
development data, and is never used as a fitting label.
"""
import os
for _key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[_key] = "2"
import argparse
import gc
import json
from pathlib import Path
import numpy as np
from extended_history_residual import enhanced_history, matrix
from rich_group_residual import (ROOT, HERE, REFERENCE, digest, write, aggregate,
                                 prediction, lgb, rank_sum_auc)
from rich_userday_activity import load_day, array_hash, correct

OUT = HERE / "runs/hurdle_userday_v1"
OCCURRENCE = HERE / "runs/extended_history_residual_v1"
LEAVES = [3, 7, 15]
ALPHAS = [0., .5, 1.]
POWERS = [0., .25, .5, .75, 1.]
PARAMS = {"objective": "poisson", "metric": "poisson", "learning_rate": .03,
          "min_data_in_leaf": 80, "lambda_l2": 20., "feature_fraction": .8,
          "max_bin": 31, "num_threads": 2, "seed": 42, "verbosity": -1,
          "deterministic": True, "force_col_wise": True}
SOURCES = {"reference": None,
           "weighted_transfer": HERE / "runs/weighted_calibration_transfer_v1/best_prediction.npy",
           "pairwise": HERE / "runs/pairwise_group_boost_v1/prediction.npy",
           "extended_history": OCCURRENCE / "prediction.npy",
           "catboost": HERE / "runs/catboost_userday_residual_v1/prediction.npy"}


def severity_rows(keys, counts, cutoff):
    """Only observed positive days may teach the conditional count model."""
    return np.flatnonzero((keys[:, 0] < cutoff) & (counts > 0))


def severity_data(history):
    data = load_day(16, history, evaluation_labels=False)
    if history:
        more, names = enhanced_history(data["keys"], data["sizes"], data["counts"], 20191116)
        data["x"] = np.column_stack([data["x"], more]).astype("float32")
        data["names"] += names
    data["x"] = np.column_stack([data["x"], np.log1p(data["sizes"])]).astype("float32")
    return data


def expected_clicks(any_probability, excess, base_mass, base_any, alpha):
    """A shifted Poisson severity ensures E(K|K>0)>=1."""
    conditional = 1. + np.clip(np.asarray(excess, dtype="float64"), 0., 100.)
    implied = np.maximum(np.asarray(base_mass) / np.maximum(base_any, 1e-8), 1.)
    severity = np.exp((1-alpha)*np.log(implied) + alpha*np.log(conditional))
    return np.clip(any_probability, 1e-8, 1-1e-8) * severity


def dependencies():
    paths = [Path(__file__), HERE / "extended_history_residual.py",
             HERE / "rich_group_residual.py", HERE / "rich_userday_activity.py",
             HERE / "fit_stacker.py", HERE / "verify_stacker.py",
             HERE / "runs/weighted_v1_day16/context.npz",
             HERE / "runs/weighted_v1_day16/context_manifest.json",
             OCCURRENCE / "cv.json", OCCURRENCE / "models_locked.json",
             OCCURRENCE / "verification.json"]
    for leaves in LEAVES:
        paths += [OCCURRENCE / f"binary_{leaves}.txt", OCCURRENCE / f"binary_{leaves}_oof.npy"]
    paths += [p for p in SOURCES.values() if p is not None]
    paths += [REFERENCE / f"day{d}_{suffix}.npy" for d in [15,16] for suffix in ["indices", "prediction"]]
    return paths


def main():
    if OUT.exists():
        raise FileExistsError("Preserving hurdle experiment")
    OUT.mkdir(parents=True)
    cv = json.loads((OCCURRENCE / "cv.json").read_text(encoding="utf8"))
    occurrence = max((r for r in cv if r["objective"] == "binary"), key=lambda r:r["oof15_auc"])
    protocol = {"inputs_sha256": {str(p):digest(p) for p in dependencies()},
                "raw_datasets_sha256": {p.name:digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
                "params": PARAMS, "leaves": LEAVES, "alphas": ALPHAS, "powers": POWERS,
                "occurrence": occurrence, "max_rounds": 700, "patience": 70,
                "fit_cutoff": 20191116, "severity_selection_cutoff": 20191115,
                "selection": "Pre15 positive days select severity rounds on15; day15 OOF occurrence selects alpha/power; refit severity on all pre16 positive days.",
                "conditional_target": "click_count - 1 given click_count > 0",
                "evaluation16_labels_fit": False, "development16_model_selection": True,
                "sources": list(SOURCES)}
    write(OUT / "protocol.json", protocol)
    day15 = matrix(15, evaluation_labels=True)
    any15 = np.load(OCCURRENCE / (occurrence["name"] + "_oof.npy"))
    records, locks = [], []
    for history in [False, True]:
        data = severity_data(history)
        print(json.dumps({"event":"severity_features_ready","history":history,"shape":list(data["x"].shape)}),flush=True)
        train = severity_rows(data["keys"], data["counts"], 20191115)
        valid = np.flatnonzero((data["keys"][:,0] == 20191115) & (data["counts"] > 0))
        refit = severity_rows(data["keys"], data["counts"], 20191116)
        day15_rows = np.flatnonzero(data["keys"][:,0] == 20191115)
        np.testing.assert_array_equal(data["keys"][day15_rows], day15["keys"])
        if not len(valid) or np.any(data["counts"][data["keys"][:,0] >= 20191116] != -1):
            raise AssertionError("Severity split boundary failed")
        for leaves in LEAVES:
            params = dict(PARAMS, num_leaves=leaves)
            training = lgb.Dataset(data["x"][train], label=data["counts"][train]-1)
            validation = lgb.Dataset(data["x"][valid], label=data["counts"][valid]-1, reference=training)
            model = lgb.train(params, training, num_boost_round=700, valid_sets=[validation],
                              callbacks=[lgb.early_stopping(70, verbose=False)])
            excess = model.predict(data["x"][day15_rows])
            scores = []
            for alpha in ALPHAS:
                mass = expected_clicks(any15, excess, day15["mass"], day15["any"], alpha)
                for power in POWERS:
                    p = correct(day15["base"], mass, day15["groups"], "poisson", power)
                    scores.append({"alpha":alpha, "power":power, "auc":rank_sum_auc(day15["y"],p)})
            best = max(scores,key=lambda r:r["auc"])
            name = f"history{int(history)}_leaves{leaves}"
            record = {"name":name, "history":history, "leaves":leaves, "rounds":model.best_iteration,
                      "selection15":best, "scores":scores,
                      "training_positive_days":len(train), "validation_positive_days":len(valid),
                      "refit_positive_days":len(refit)}
            np.save(OUT / (name+"_excess15.npy"), excess)
            records.append(record)
            write(OUT / "selection15.json",records)
            final_model = lgb.train(params, lgb.Dataset(data["x"][refit], label=data["counts"][refit]-1),
                                    num_boost_round=max(1,model.best_iteration))
            path = OUT / (name+".txt")
            path.write_text(final_model.model_to_string(),encoding="utf8")
            locks.append(dict(record,model_sha256=digest(path),
                              features_sha256=array_hash(data["x"]),
                              targets_sha256=array_hash(data["counts"][refit]-1),
                              training_keys_sha256=array_hash(data["keys"][refit])))
            print(json.dumps({k:v for k,v in record.items() if k != "scores"}),flush=True)
        del data
        gc.collect()
    write(OUT / "models_locked.json",{"models":locks,"protocol_sha256":digest(OUT / "protocol.json")})
    del day15
    gc.collect()
    evaluate(False)


def evaluate(verifying):
    protocol=json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    lock=json.loads((OUT / "models_locked.json").read_text(encoding="utf8"))
    if digest(OUT / "protocol.json") != lock["protocol_sha256"]:
        raise AssertionError("Hurdle protocol changed")
    for p,h in protocol["inputs_sha256"].items():
        if digest(p)!=h:
            raise AssertionError("Hurdle dependency changed: "+p)
    for name,h in protocol["raw_datasets_sha256"].items():
        if digest(ROOT / "2026ml" / name)!=h:
            raise AssertionError("Original input changed")
    excess16={}
    for history in [False,True]:
        data=severity_data(history)
        refit=severity_rows(data["keys"],data["counts"],20191116)
        held=np.flatnonzero(data["keys"][:,0]==20191116)
        for r in [r for r in lock["models"] if r["history"]==history]:
            path=OUT / (r["name"]+".txt")
            if digest(path)!=r["model_sha256"]:
                raise AssertionError("Severity model changed")
            if verifying:
                for actual,expected in [(data["x"],r["features_sha256"]),
                        (data["counts"][refit]-1,r["targets_sha256"]),
                        (data["keys"][refit],r["training_keys_sha256"])]:
                    if array_hash(actual)!=expected:
                        raise AssertionError("Severity training replay differs")
            model=lgb.Booster(model_str=path.read_text(encoding="utf8"))
            excess16[r["name"]]=model.predict(data["x"][held])
        del data
        gc.collect()
    occurrence=lgb.Booster(model_str=(OCCURRENCE / (protocol["occurrence"]["name"]+".txt")).read_text(encoding="utf8"))
    records,predictions=[],[]
    data=matrix(16,evaluation_labels=True)
    for source,source_path in SOURCES.items():
        print(json.dumps({"event":"hurdle_replay_source","source":source,"verify":verifying}),flush=True)
        if source_path is not None:
            data["base"]=np.load(source_path)
            data["mass"],data["any"],block=aggregate(data["base"],data["groups"],len(data["keys"]))
            pos=data["names"].index("base_log_size")
            data["x"][:,pos:pos+4]=block
        if len(data["y"])!=421983 or int(data["y"].sum())!=3724 or np.any(data["count"]!=-1):
            raise AssertionError("Full16 boundary failed")
        any16=prediction(occurrence,data,"binary")
        for r in lock["models"]:
            choice=r["selection15"]
            mass=expected_clicks(any16,excess16[r["name"]],data["mass"],data["any"],choice["alpha"])
            p=correct(data["base"],mass,data["groups"],"poisson",choice["power"])
            records.append({"name":r["name"],"source":source,**choice,
                            "selection15_auc":choice["auc"],"auc":rank_sum_auc(data["y"],p)})
            predictions.append(p)
    winner=int(np.argmax([r["auc"] for r in records]))
    p=predictions[winner]
    if verifying:
        error=float(np.max(np.abs(p-np.load(OUT / "prediction.npy"))))
        if error>1e-12:
            raise AssertionError("Hurdle prediction replay differs")
        np.testing.assert_array_equal(data["rows"],np.load(OUT / "valid_indices.npy"))
    else:
        error=0.
        np.save(OUT / "prediction.npy",p)
        np.save(OUT / "valid_indices.npy",data["rows"])
    result={"status":"passed" if verifying else "evaluated","auc":records[winner]["auc"],
            "best":records[winner],"records":records,"rows":len(p),"clicks":3724,
            "maximum_replay_error":error,"original_datasets_unchanged":True,
            "evaluation16_labels_fit":False,"development16_model_selection":True,
            "target_met":records[winner]["auc"]>=.82}
    write(OUT / ("verification.json" if verifying else "result.json"),result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--verify",action="store_true")
    evaluate(True) if parser.parse_args().verify else main()
