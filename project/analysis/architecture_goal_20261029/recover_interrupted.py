"""Verify the one interrupted day15 dropout run, without resuming training.

Run with E:/Users/env/Machine_Learning/python.exe -B. The only runtime output
is a new recovery_verification.json; existing run artifacts and caches stay
read-only. This script deliberately supports no other result directory.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

# Set before importing either NumPy or PyTorch; do not initialize CUDA.
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUPPORTED = Path("E:/AI/ml/大作业上传/analysis/architecture_goal_20261029/results/gated_dropout20_day15").resolve()
CHUNK = 2048
TOLERANCE = 2e-6
BASELINE_AUC = 0.7905480723412239
WEIGHTS = (.05, .1, .2, .3)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def scores(y, prediction):
    require(prediction.shape == y.shape, "Prediction shape differs from validation labels")
    require(bool(np.isfinite(prediction).all()), "Nonfinite prediction")
    require(bool(((prediction >= 0) & (prediction <= 1)).all()), "Prediction outside [0, 1]")
    return {"auc": float(roc_auc_score(y, prediction)),
            "logloss": float(log_loss(y, prediction)),
            "average_precision": float(average_precision_score(y, prediction)),
            "mean_probability": float(prediction.mean())}


def main():
    started = time.perf_counter()
    out = (HERE / "results/gated_dropout20_day15").resolve()
    require(out == SUPPORTED, "Only the original interrupted result directory is supported")
    report_path = out / "recovery_verification.json"
    if report_path.exists():
        raise FileExistsError("Preserving existing recovery report: " + str(report_path))
    require(not (out / "metrics.json").exists(), "Run already has normal completion metrics")
    protected_names = ("protocol.json", "history.json", "model.pt", "model_snapshot.py",
                       "trainer_snapshot.py", "valid_indices.npy", "validation_prediction.npy")
    original_hashes = {name: sha256(out / name) for name in protected_names}
    protocol, history = read(out / "protocol.json"), read(out / "history.json")
    require(protocol["day"] == 20191115 and protocol["model"] == "GatedCrossCTR",
            "Unexpected interrupted run protocol")
    require(protocol["options"] == {"numeric_dim": 500, "group_dropout_rates":
            {"user_id": .2, "ad_task": .2, "adv_id": .2}}, "Unexpected model options")
    require(original_hashes["model_snapshot.py"] == protocol["model_source_sha256"],
            "Saved model snapshot SHA256 differs from protocol")
    require(original_hashes["trainer_snapshot.py"] == protocol["trainer_sha256"],
            "Saved trainer snapshot SHA256 differs from protocol")
    require(bool(history) and round(history[-1]["fractional_epoch"], 2) == 1.25,
            "History does not end at the expected interrupted quarter epoch")
    require(history[-1]["fractional_epoch"] < protocol["epochs"],
            "History reaches the planned training duration")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    saved = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    require(saved["model"] == protocol["model"] and saved["options"] == protocol["options"],
            "Checkpoint model/options differ from protocol")
    require(saved["record"] in history, "Checkpoint record absent from preserved history")
    require(saved["record"] == max(history, key=lambda item: item["auc"]),
            "Checkpoint is not the best recorded validation checkpoint")

    # The manifest identifies the original tensor cache and records the already
    # fitted 500-column scaler/schema. The existing numeric memmap is in local
    # [train_indices, valid_indices] order, unlike global categories/sequences.
    input_cache = Path(protocol["input_cache"]).resolve()
    require(input_cache == (HERE / "cache/day15_oof").resolve(), "Unexpected input cache")
    input_meta = read(input_cache / "manifest.json")
    tensor_cache = Path(input_meta["base_numeric"]).resolve().parent
    require(tensor_cache == (ROOT / "analysis/user_personalization/cache_day15_stats").resolve(),
            "Unexpected source tensor cache")
    meta = read(tensor_cache / "manifest.json")
    require(saved["manifest"] == meta, "Checkpoint manifest differs from source cache manifest")
    require(input_meta["day"] == protocol["day"] and input_meta["statistics"] == "oof",
            "Input manifest split/statistics differ")
    require(input_meta["numeric_fields"] == protocol["numeric_fields"], "Numeric schema differs")
    require(len(input_meta["numeric_fields"]) == 500 and len(set(input_meta["numeric_fields"])) == 500,
            "Expected 500 unique numeric fields")
    require(set(input_meta["scalers"]) == set(input_meta["numeric_fields"]), "Incomplete saved scalers")
    require(input_meta["clip"] == [-8, 8] and input_meta["scaler_fit_scope"] == "Only outer-training rows",
            "Unexpected numeric normalization protocol")
    require(input_meta["advertisement_validation_labels_used"] is False and
            meta["target_stat_validation_labels_used"] is False and
            protocol["validation_labels_used_for_features_or_optimizer"] is False,
            "Validation-label boundary differs")
    arrays = {name: np.load(tensor_cache / (name + ".npy"), mmap_mode="r")
              for name in ("categories", "sequences", "date", "label", "train_indices", "valid_indices")}
    train = np.load(input_cache / "train_indices.npy", mmap_mode="r")
    valid = np.load(input_cache / "valid_indices.npy", mmap_mode="r")
    for name, expected in (("train_indices", train), ("valid_indices", valid)):
        np.testing.assert_array_equal(arrays[name], expected)
    np.testing.assert_array_equal(valid, np.load(out / "valid_indices.npy", mmap_mode="r"))
    np.testing.assert_array_equal(train, np.flatnonzero(arrays["date"] < protocol["day"]))
    np.testing.assert_array_equal(valid, np.flatnonzero(arrays["date"] == protocol["day"]))
    ntrain, nvalid = len(train), len(valid)
    for manifest in (protocol, input_meta, meta):
        require(manifest["train_rows"] == ntrain and manifest["valid_rows"] == nvalid,
                "Manifest row counts differ from cached split")
    numeric = np.load(input_cache / "numeric.npy", mmap_mode="r")
    require(numeric.shape == (ntrain + nvalid, 500) and numeric.dtype == np.float32,
            "Unexpected cached numeric shape/dtype")
    y = np.asarray(arrays["label"][valid])
    require(nvalid == 424780 and int(y.sum()) == 3523, "Incomplete day15 validation")
    train_ctr = float(arrays["label"][train].mean())

    # Snapshot imports use the original shared encoder dependencies. Import the
    # saved snapshot itself, never the potentially changed live model module.
    sys.path.insert(0, str(ROOT / "analysis/model_comparison"))
    spec = importlib.util.spec_from_file_location("gated_dropout20_recovery_snapshot", out / "model_snapshot.py")
    require(spec is not None and spec.loader is not None, "Cannot import saved model snapshot")
    snapshot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(snapshot)
    model = getattr(snapshot, saved["model"])(
        meta["categorical_fields"], meta["sequence_fields"], meta["group_for_field"],
        meta["vocabulary_sizes"], initial_ctr=train_ctr, **saved["options"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    require(all(parameter.device.type == "cpu" for parameter in model.parameters()), "Model left CPU")
    require(sum(parameter.numel() for parameter in model.parameters()) == protocol["parameter_count"],
            "Model parameter count differs from protocol")
    print("RECOVERY_REPLAY", json.dumps({"device": "cpu", "threads": 2,
          "chunk_size": CHUNK, "validation_rows": nvalid, "training_interrupted": True}), flush=True)
    replay = np.empty(nvalid, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, nvalid, CHUNK):
            stop = min(start + CHUNK, nvalid)
            rows = valid[start:stop]
            values = np.array(numeric[ntrain + start:ntrain + stop], copy=True)
            require(bool(np.isfinite(values).all()) and bool((np.abs(values) <= 8).all()),
                    "Invalid cached validation numeric values")
            categories = torch.from_numpy(arrays["categories"][rows].astype(np.int64))
            sequences = torch.from_numpy(arrays["sequences"][rows].astype(np.int64))
            replay[start:stop] = torch.sigmoid(model(categories, sequences, torch.from_numpy(values))).numpy()
    original = np.load(out / "validation_prediction.npy", mmap_mode="r")
    require(original.shape == replay.shape, "Saved prediction shape differs")
    replay_scores, saved_scores = scores(y, replay), scores(y, original)
    difference = np.abs(replay.astype(np.float64) - original.astype(np.float64))
    maximum_error = float(difference.max())
    require(maximum_error <= TOLERANCE, "CPU checkpoint replay exceeds absolute tolerance")
    require(abs(saved_scores["auc"] - saved["record"]["auc"]) <= 1e-12,
            "Saved prediction AUC differs from checkpoint record")

    baseline_dir = ROOT / "analysis/nextday_goal_20261029/duplicate_multiplicity"
    np.testing.assert_array_equal(valid, np.load(baseline_dir / "day15_indices.npy", mmap_mode="r"))
    baseline = np.load(baseline_dir / "day15_prediction.npy", mmap_mode="r")
    baseline_scores = scores(y, baseline)
    require(abs(baseline_scores["auc"] - BASELINE_AUC) <= 1e-12, "Baseline AUC differs")
    blends = [{"weight": weight, **scores(y, (1 - weight) * baseline + weight * replay)} for weight in WEIGHTS]
    for blend in blends:
        blend["auc_delta_vs_baseline"] = blend["auc"] - baseline_scores["auc"]
    best_blend = max(blends, key=lambda item: item["auc"])

    # Completed FinalMLP is summarized from its existing metrics only; no replay.
    finalmlp_path = HERE / "results/finalmlp_oof_day15/metrics.json"
    finalmlp = read(finalmlp_path)
    require(finalmlp["status"] == "complete_and_verified", "FinalMLP is not completed")
    require(finalmlp["day"] == protocol["day"] and finalmlp["valid_rows"] == nvalid and
            finalmlp["validation_clicks"] == int(y.sum()), "FinalMLP validation split differs")
    finalmlp_best = max(finalmlp["fixed_blends"], key=lambda item: item["auc"])
    report = {
        "status": "recovered_checkpoint_verified", "training_interrupted": True,
        "training_resumed": False, "training_completed": False, "day": protocol["day"],
        "model": saved["model"], "last_fractional_epoch": round(history[-1]["fractional_epoch"], 2),
        "last_fractional_epoch_exact": history[-1]["fractional_epoch"],
        "planned_epochs": protocol["epochs"], "best_checkpoint_record": saved["record"],
        "best_fractional_epoch": saved["record"]["fractional_epoch"],
        "best_checkpoint_score": saved["record"]["auc"],
        "train_rows": ntrain, "valid_rows": nvalid, "validation_clicks": int(y.sum()),
        **replay_scores, "saved_validation_metrics": saved_scores,
        "device": "cpu", "threads": 2, "interop_threads": 2, "chunk_size": CHUNK,
        "strict_state_dict_load": True, "complete_day_validation_verified": True,
        "source_verification": {"model_snapshot_sha256": original_hashes["model_snapshot.py"],
            "model_snapshot_matches_protocol": True,
            "trainer_snapshot_sha256": original_hashes["trainer_snapshot.py"],
            "trainer_snapshot_matches_protocol": True,
            "shared_dependency_sha256": {name: sha256(ROOT / "analysis/model_comparison" / name)
                for name in ("id_regularized_models.py", "mainstream_models.py", "benchmark_core.py")},
            "shared_dependencies_note": "Current shared dependencies are hashed for traceability; no saved dependency hash is available."},
        "input_verification": {"input_cache": str(input_cache), "manifest_sha256": sha256(input_cache / "manifest.json"),
            "numeric_source": "Existing read-only float32 memmap, validated against input manifest schema and split; saved scalers are not refitted.",
            "numeric_shape": list(numeric.shape), "numeric_columns": len(input_meta["numeric_fields"]),
            "numeric_row_order": "train_indices followed by valid_indices; validation starts at train_rows",
            "tensor_cache": str(tensor_cache), "checkpoint_manifest_matches_cache": True,
            "complete_chronological_split_verified": True},
        "backend_rounding": {"reference_backend": "Original GPU training validation prediction",
            "replay_backend": "CPU float32; chunk size differs from original batch size 1024",
            "absolute_tolerance": TOLERANCE, "relative_tolerance": 0.0,
            "maximum_absolute_error": maximum_error, "mean_absolute_error": float(difference.mean()),
            "nonidentical_prediction_rows": int(np.count_nonzero(difference)),
            "within_tolerance": True, "auc_delta_vs_saved": replay_scores["auc"] - saved_scores["auc"],
            "interpretation": "Observed differences are within the allowed CPU/GPU float32 replay tolerance."},
        "baseline_auc": baseline_scores["auc"], "baseline_metrics": baseline_scores,
        "baseline_name": "duplicate_multiplicity", "fixed_blends": blends, "best_fixed_blend": best_blend,
        "goal_auc": .85, "goal_met": replay_scores["auc"] > .85,
        "any_blend_goal_met": any(blend["auc"] > .85 for blend in blends), "final_auc": None,
        "scope": "Previously reused development day; recovered checkpoint evaluation does not establish training completion or blind-test improvement.",
        "finalmlp_completed_summary": {"source": str(finalmlp_path), "source_sha256": sha256(finalmlp_path),
            "evaluation": "Existing completed metrics only; no duplicate replay", "metrics": finalmlp,
            "best_fixed_blend": finalmlp_best,
            "best_blend_auc_delta_vs_baseline": finalmlp_best["auc"] - baseline_scores["auc"]},
        "original_artifact_sha256": original_hashes,
        "original_artifacts_unchanged": True, "seconds": time.perf_counter() - started}
    require(original_hashes == {name: sha256(out / name) for name in protected_names},
            "An original run artifact changed during recovery")
    require(not (out / "metrics.json").exists(), "Normal completion metrics appeared during recovery")
    # Exclusive creation makes reruns preserve the first recovery report.
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("RECOVERY_VERIFIED", json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
