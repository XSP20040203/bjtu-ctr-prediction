"""Rescore predictions, reload weights, and independently audit target statistics."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "model_comparison"
sys.path.insert(0, str(BENCH))
from train_models import load_arrays, metrics, write_json
from bounded_id_models import BoundedIDDCN
from mainstream_models import DCNV2Model


def audit_target_features(cache):
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    arrays = load_arrays(cache)
    train, valid = arrays["train_indices"], arrays["valid_indices"]
    folds = np.load(cache / "target_stat_oof_fold.npy")
    hashes = np.load(HERE / "raw_predictor_hash.npy", mmap_mode="r")
    order = np.argsort(hashes[train])
    equal = hashes[train][order][1:] == hashes[train][order][:-1]
    assert np.all(folds[train][order][1:][equal] == folds[train][order][:-1][equal])
    assert np.all(folds[train] >= 0) and np.all(folds[valid] == -1)
    labels = arrays["label"]
    prior = float(np.clip(labels[train].mean(), 1e-6, 1 - 1e-6))
    probes = np.r_[valid[[0, len(valid) // 2, -1]], train[[0, len(train) // 2, -1]]]
    for definition in manifest["target_stat_definitions"]:
        fields, alpha = definition["fields"], definition["alpha"]
        keys = np.zeros(len(labels), dtype=np.int64)
        for name in fields:
            column = manifest["categorical_fields"].index(name)
            size = manifest["vocabulary_sizes"][manifest["group_for_field"][name]]
            keys = keys * size + arrays["categories"][:, column]
        for row in probes:
            fit = train if folds[row] < 0 else train[folds[train] != folds[row]]
            matching = fit[keys[fit] == keys[row]]
            probability_prior = float(np.clip(labels[fit].mean(), 1e-6, 1 - 1e-6))
            probability = np.float32((labels[matching].sum() + alpha * probability_prior) / (len(matching) + alpha))
            raw = [float(np.log1p(np.array([len(matching)], dtype=np.float32))[0]),
                   float((np.log(np.array([probability], dtype=np.float32) / (1 - np.array([probability], dtype=np.float32))) - np.log(prior / (1 - prior)))[0])]
            for suffix, value in zip(["log_count", "relative_logit"], raw):
                name = "target_stat__" + "__".join(fields) + "__" + suffix
                scale = manifest["numeric_scaler"][name]
                expected = (value - scale["mean"]) / scale["std"]
                actual = arrays["numeric"][row, manifest["numeric_fields"].index(name)]
                np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-4)
    return {"cache": str(cache), "feature_groups": len(manifest["target_stat_definitions"]), "sampled_rows_per_group": len(probes),
            "exact_duplicate_groups_kept_in_one_oof_fold": True, "target_features_recomputed_from_excluded_fold_labels": True}


def main():
    torch.set_num_threads(4)
    records = []
    for path in sorted((HERE / "results").glob("*/*/seed*/metrics.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        cache = Path(record["cache_dir"])
        arrays = load_arrays(cache)
        train, valid = arrays["train_indices"], arrays["valid_indices"]
        assert not np.intersect1d(train, valid).size
        if record["validation_day"] is not None:
            assert arrays["date"][train].max() < record["validation_day"]
            assert np.all(arrays["date"][valid] == record["validation_day"])
        prediction = np.load(path.parent / "validation_prediction.npy")
        recalculated = metrics(arrays["label"][valid], prediction)
        for name, value in recalculated.items():
            assert abs(value - record[name]) < 1e-10
        checkpoint = torch.load(path.parent / "model.pt", map_location="cpu", weights_only=False)
        manifest = checkpoint["manifest"]
        assert manifest == json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        builder = {"DCNV2Model": DCNV2Model, "BoundedIDDCN": BoundedIDDCN}[checkpoint["model_class"]]
        model = builder(manifest["categorical_fields"], manifest["sequence_fields"], manifest["group_for_field"], manifest["vocabulary_sizes"], **checkpoint["model_options"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        chosen = np.linspace(0, len(valid) - 1, 128, dtype=int)
        rows = valid[chosen]
        with torch.no_grad():
            restored = torch.sigmoid(model(torch.from_numpy(arrays["categories"][rows].astype(np.int64)),
                                           torch.from_numpy(arrays["sequences"][rows].astype(np.int64)),
                                           torch.from_numpy(np.array(arrays["numeric"][rows])))).numpy()
        np.testing.assert_allclose(restored, prediction[chosen], rtol=3e-5, atol=2e-7)
        record.update(experiment=path.parts[-4], results_dir=str(path.parent), max_reload_error=float(np.max(np.abs(restored - prediction[chosen]))))
        records.append(record)
    assert len(records) >= 6
    audits = [audit_target_features(HERE / name) for name in ["cache_rich_stats_time16", "cache_rich_stats_random20"]]
    random_cache = HERE / "cache_random20_seed42"
    hashes = np.load(HERE / "raw_predictor_hash.npy", mmap_mode="r")
    train, valid = np.load(random_cache / "train_indices.npy"), np.load(random_cache / "valid_indices.npy")
    overlap = float(np.isin(hashes[valid], np.unique(hashes[train])).mean())
    write_json(HERE / "phase1_verified_results.json", records)
    write_json(HERE / "phase1_verification.json", {"status": "passed", "runs_verified": len(records),
               "predictions_rescored": True, "checkpoints_reloaded": True, "target_feature_audits": audits,
               "random_validation_duplicate_predictor_fraction": overlap,
               "max_reload_probability_error": max(r["max_reload_error"] for r in records),
               "final_test_scoring": "deferred_by_user", "final_test_auc_verified": False,
               "goal_auc_085_achieved": False})
    print(pd.DataFrame([{key: record[key] for key in ["experiment", "validation_strategy", "model", "auc", "logloss", "best_epoch"]} for record in records]).to_string(index=False))
    print("All saved models and sampled cross-fitted target features independently verified.")


if __name__ == "__main__":
    main()
