"""Independently rescore saved controls and the fixed v7 replacement test."""
from __future__ import annotations

import math
from pathlib import Path
import sys

from features import HERE, ROOT, ARCH, read, write, digest, emit, cache_dir
from experiment import SETTINGS, sources, scores
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

V7_HASH = "c44a9e803b8b4566b9bcef4e1bf85181c2ee00eee71a9f84474b73c619c121b9"


def influences(y, prediction):
    positive = np.sort(prediction[y == 1])
    negative = np.sort(prediction[y == 0])
    vpos = (np.searchsorted(negative, prediction[y == 1], side="left") +
            np.searchsorted(negative, prediction[y == 1], side="right")) / (2 * len(negative))
    vneg = 1 - (np.searchsorted(positive, prediction[y == 0], side="left") +
                np.searchsorted(positive, prediction[y == 0], side="right")) / (2 * len(positive))
    auc = float(vpos.mean())
    if abs(auc - roc_auc_score(y, prediction)) > 1e-12:
        raise AssertionError("AUC influence mismatch")
    influence = np.empty(len(y), dtype="float64")
    influence[y == 1] = (vpos - auc) / len(positive)
    influence[y == 0] = (vneg - auc) / len(negative)
    return auc, influence


def paired_interval(y, first, second, users):
    first_auc, first_influence = influences(y, first)
    second_auc, second_influence = influences(y, second)
    _, codes = np.unique(users, return_inverse=True)
    clusters = np.bincount(codes, weights=first_influence - second_influence)
    variance = len(clusters) / max(len(clusters) - 1, 1) * np.square(clusters - clusters.mean()).sum()
    delta = first_auc - second_auc
    margin = 1.96 * math.sqrt(variance)
    return {"auc_delta": delta, "user_cluster_approximate_95_interval": [delta - margin, delta + margin],
            "logloss_delta": scores(y, first)["logloss"] - scores(y, second)["logloss"]}


def verify_arm(path, out, raw, day):
    metrics = read(path / "metrics.json")
    if metrics["status"] != "complete_and_verified" or metrics["fixed_epochs_completed"] != 1:
        raise AssertionError("Incomplete arm")
    if metrics["source_sha256"] != sources():
        raise AssertionError("Trained source changed")
    if digest(out / "feature_manifest.json") != metrics["feature_manifest_sha256"]:
        raise AssertionError("Feature metadata changed")
    for name, expected in metrics["files_sha256"].items():
        if digest(path / name) != expected:
            raise AssertionError("Experiment output changed: " + name)
    valid = np.flatnonzero(raw.pt_d.to_numpy() == day)
    np.testing.assert_array_equal(np.load(path / "valid_indices.npy"), valid)
    y = raw.label.to_numpy()[valid]
    predictions = {tag: np.load(path / f"{tag}_prediction.npy") for tag in ["final", "best"]}
    for tag, prediction in predictions.items():
        if len(prediction) != len(valid) or np.any((prediction < 0) | (prediction > 1)):
            raise AssertionError("Invalid prediction")
        if tag == "final":
            actual = scores(y, prediction)
            for name in actual:
                if abs(actual[name] - metrics["scores"][name]) > 1e-12:
                    raise AssertionError("Saved fixed scores differ: " + name)
    return metrics, predictions


def evaluate():
    raw = pd.read_csv(ROOT / "2026ml/train.csv",
                      usecols=["pt_d", "user_id", "label", "task_id", "log_id"], dtype="int32")
    rows, differences, replacements, controls, ensemble_differences = [], [], [], [], []
    files = {}
    preprocessing_assets = {
        str(ROOT / "analysis/auc085/daily_source_profiles.pkl"):
            digest(ROOT / "analysis/auc085/daily_source_profiles.pkl")}
    reference_manifest = read(ROOT / "output/best_nextday_arch_v7/manifest.json")
    if digest(ROOT / "output/best_nextday_arch_v7/submission.csv") != V7_HASH:
        raise AssertionError("Original recommended submission changed")
    for day in [20191115, 20191116]:
        out = HERE / "runs" / f"day{day % 100}_seed42"
        for name, field in [("date", "pt_d"), ("label", "label"), ("log_id", "log_id")]:
            np.testing.assert_array_equal(np.load(cache_dir(day) / f"{name}.npy"),
                                          raw[field].to_numpy())
        feature_meta = read(out / "feature_manifest.json")
        for name, expected in feature_meta["input_sha256"].items():
            if digest(ROOT / "2026ml" / name) != expected:
                raise AssertionError("Raw input changed")
        for name, expected in feature_meta["metadata_sha256"].items():
            if digest(name) != expected:
                raise AssertionError("Feature source/metadata changed")
        replay = read(out / "original_replay.json")
        if replay["status"] != "passed" or replay["maximum_probability_error"] > 2e-7:
            raise AssertionError("Original strong replay not passed")
        original_directory = ARCH / f"results/gated166_day{day % 100}"
        if digest(original_directory / "model.pt") != replay["original_checkpoint_sha256"]:
            raise AssertionError("Original checkpoint changed")
        if digest(original_directory / "validation_prediction.npy") != replay["prediction_sha256"]:
            raise AssertionError("Original prediction changed")
        for path in sorted((cache_dir(day) / "target_stat_models").glob("*.npz")):
            preprocessing_assets[str(path)] = digest(path)
        results, probabilities = {}, {}
        for mode in ["control", "attention"]:
            path = out / mode
            results[mode], probabilities[mode] = verify_arm(path, out, raw, day)
            files[str(path / "metrics.json")] = digest(path / "metrics.json")
        for key in ["initial_backbone_sha256", "batch_order_sha256"]:
            if results["control"][key] != results["attention"][key]:
                raise AssertionError("Matched arm initialization/order differs")
        valid = np.flatnonzero(raw.pt_d.to_numpy() == day)
        y, users = raw.label.to_numpy()[valid], raw.user_id.to_numpy()[valid]
        original = np.load(ARCH / f"results/gated166_day{day % 100}/validation_prediction.npy")
        reference_path = ARCH / f"fixed_gated20_allocated10_core30_day{day % 100}.npy"
        reference = np.load(reference_path).astype("float64")
        old_reference_auc = reference_manifest["development"][str(day)]["auc"]
        if abs(roc_auc_score(y, reference) - old_reference_auc) > 1e-12:
            raise AssertionError("Fixed v7 reference differs")
        rows.append({"day": day, "mode": "original_v7", "endpoint": "original_fixed_blend",
                     **scores(y, reference)})
        for tag in ["final", "best"]:
            matched_blends = {}
            for mode in ["control", "attention"]:
                prediction = probabilities[mode][tag]
                rows.append({"day": day, "mode": mode, "endpoint": tag, **scores(y, prediction)})
                # Keep the other original ensemble components exactly fixed.
                replacement = reference + .30 * (prediction.astype("float64") - original)
                if np.any((replacement < -1e-7) | (replacement > 1 + 1e-7)):
                    raise AssertionError("Replacement is not a valid original-weight blend")
                np.save(out / f"v7_replace_{mode}_{tag}.npy", replacement)
                matched_blends[mode] = replacement
                replacements.append({
                    "day": day, "mode": mode, "endpoint": tag, **scores(y, replacement),
                    **paired_interval(y, replacement, reference, users)})
            differences.append({
                "day": day, "endpoint": tag,
                **paired_interval(y, probabilities["attention"][tag],
                                  probabilities["control"][tag], users)})
            ensemble_differences.append({
                "day": day, "endpoint": tag,
                **paired_interval(y, matched_blends["attention"],
                                  matched_blends["control"], users)})
        controls.append({
            "day": day, "original_checkpoint_replay_error": replay["maximum_probability_error"],
            "new_control_best_auc": scores(y, probabilities["control"]["best"])["auc"],
            "old_control_best_auc": scores(y, original)["auc"],
            "new_control_best_max_probability_difference": float(
                np.max(np.abs(probabilities["control"]["best"] - original))),
        })
    fixed = [row for row in differences if row["endpoint"] == "final"]
    eligible = all(
        row["auc_delta"] >= SETTINGS["promotion_min_auc_gain_both_days"] and
        row["logloss_delta"] <= SETTINGS["promotion_max_logloss_worsening_both_days"]
        for row in fixed)
    v7_improved = all(
        row["auc_delta"] > 0 for row in replacements
        if row["mode"] == "attention" and row["endpoint"] == "best")
    decision = {
        "status": "awaiting_seed_confirmation" if eligible and v7_improved else "not_promoted",
        "selected_submission": "output/best_nextday_arch_v7/submission.csv",
        "selected_submission_sha256": V7_HASH,
        "fixed_budget_interest_gain_rule_passed": eligible,
        "v7_replacement_both_day_gain": v7_improved,
        "confirmation_seed": SETTINGS["confirmation_seed"],
        "paired_interest_changes": differences,
        "paired_ensemble_interest_changes": ensemble_differences,
        "fixed_v7_replacement_changes": replacements,
        "original_control_reproduction": controls,
        "final_test_auc": None,
        "limitations": [
            "Two previously reused development dates, not an untouched blind test",
            "Approximate paired AUC interval clusters by user, without shared-ad or model-selection adjustment",
            "Only this controlled interest pooling change was tested; no general attention impossibility claim",
        ],
    }
    pd.DataFrame(rows).to_csv(HERE / "comparison.csv", index=False)
    pd.DataFrame(differences).to_csv(HERE / "paired_differences.csv", index=False)
    pd.DataFrame(replacements).to_csv(HERE / "v7_replacements.csv", index=False)
    pd.DataFrame(ensemble_differences).to_csv(HERE / "paired_v7_differences.csv", index=False)
    write(HERE / "decision.json", decision)
    write(HERE / "verification.json", {
        "status": "passed", "raw_label_rescore": True, "raw_row_order_verified": True,
        "all_cached_training_labels_dates_and_log_ids_match_raw_csv": True,
        "original_strong_checkpoints_replayed": True, "complete_fixed_epochs": 4,
        "matched_backbones_and_batch_orders": True, "v7_submission_unchanged": True,
        "metric_files_sha256": files, "source_sha256": sources(),
        "preprocessing_assets_sha256": preprocessing_assets,
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "pandas": pd.__version__, "torch": torch.__version__,
                    "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
        "evaluator_sha256": digest(Path(__file__)),
    })
    emit("controlled_experiment_decision", **decision)


if __name__ == "__main__":
    evaluate()
