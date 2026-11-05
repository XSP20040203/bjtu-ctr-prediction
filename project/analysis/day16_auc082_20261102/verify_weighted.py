"""Rebuild inputs, independently read raw outcomes and replay selected weights."""
from pathlib import Path
import argparse
import gc
import time

from train_weighted import (ROOT, HERE, ARCH, SETTINGS, read, write, digest, array_hash,
                            configure, prepare_inputs, make_model, predict, load_reference)
import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from sklearn.metrics import roc_auc_score, log_loss


def evaluate(labels, probability):
    if probability.shape != labels.shape or not np.isfinite(probability).all():
        raise AssertionError("Prediction alignment/finiteness failed")
    return {"auc": float(roc_auc_score(labels, probability)),
            "logloss": float(log_loss(labels, probability, labels=[0, 1]))}


def main(day, family="weighted_v1"):
    if family not in ["weighted_v1", "duplicate_mean_v1"]:
        raise ValueError("Unknown experiment family")
    out = HERE / "runs" / f"{family}_day{day % 100}"
    protocol = read(out / "protocol.json")
    read(out / "completion.json")
    for path, expected in protocol["source_sha256"].items():
        if digest(path) != expected:
            raise AssertionError("Training source changed: " + path)
    actual_hashes = {name: digest(ROOT / "2026ml" / name) for name in protocol["raw_datasets_sha256"]}
    if actual_hashes != protocol["raw_datasets_sha256"]:
        raise AssertionError("Original datasets changed")
    started = time.perf_counter()
    verification = out / "independent_rebuild"
    verification.mkdir(exist_ok=True)
    configure(SETTINGS["seed"])
    data, contexts = prepare_inputs(day, verification)
    if family == "duplicate_mean_v1":
        from duplicate_targets import duplicate_targets
        soft, grouping = duplicate_targets(data)
        grouping["target_array_sha256"] = array_hash(soft)
        if grouping != read(out / "duplicate_groups.json"):
            raise AssertionError("Duplicate training target reconstruction differs")
        del soft
        gc.collect()
    fresh_contract = read(verification / "input_contract.json")
    if fresh_contract != read(out / "input_contract.json"):
        raise AssertionError("Rebuilt input contract changed")
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["pt_d", "user_id", "label", "log_id"], dtype="int32")
    indices = np.flatnonzero(raw.pt_d.to_numpy() == day)
    np.testing.assert_array_equal(indices, np.load(out / "valid_indices.npy"))
    y = raw.label.to_numpy()[indices]
    np.testing.assert_array_equal(y, data["evaluation_labels"])
    reference, ref_indices, _ = load_reference(day, protocol["reference"])
    np.testing.assert_array_equal(indices, ref_indices)
    old_core = np.load(ARCH / f"results/gated166_day{day % 100}/validation_prediction.npy")
    v7 = np.load(ARCH / f"fixed_gated20_allocated10_core30_day{day % 100}.npy").astype("float64")
    records = [{"arm": "reference", "endpoint": "frozen", "recipe": "identity", **evaluate(y, reference)}]
    candidates = {("reference", "frozen", "identity"): reference}
    replay_errors = {}
    for weight in protocol["settings"]["positive_weights"]:
        arm = f"positive_weight_{weight:g}"
        directory = out / arm
        for endpoint in ["raw", "ema"]:
            checkpoint = torch.load(directory / f"{endpoint}_best.pt", map_location="cpu", weights_only=False)
            if checkpoint["protocol_sha256"] != digest(out / "protocol.json"):
                raise AssertionError("Checkpoint protocol changed")
            model = make_model(data, contexts, weight)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            replay = predict(model, data, data["valid"], weight)
            saved = np.load(directory / f"{endpoint}_best_prediction.npy")
            error = float(np.max(np.abs(replay - saved)))
            replay_errors[f"{arm}/{endpoint}"] = error
            if error > 2e-7:
                raise AssertionError("Independent model replay mismatch")
            variants = {"single": replay.astype("float64"),
                        "core_replacement_030": v7 + .3 * (replay - old_core)}
            variants.update({f"reference_blend_{alpha:g}": (1-alpha)*reference + alpha*replay
                             for alpha in SETTINGS["blend_weights"]})
            for recipe, p in variants.items():
                record = {"arm": arm, "endpoint": endpoint, "recipe": recipe,
                          "selected_epoch": checkpoint["record"]["epoch"], **evaluate(y, p)}
                expected = checkpoint["record"]["scores"][recipe]
                if abs(record["auc"] - expected["auc"]) > 2e-8:
                    raise AssertionError("Independent AUC differs from training report")
                records.append(record)
                candidates[(arm, endpoint, recipe)] = p
            del model, checkpoint
            gc.collect()
            torch.cuda.empty_cache()
    table = pd.DataFrame(records).sort_values("auc", ascending=False)
    table.to_csv(out / "verified_comparison.csv", index=False)
    best = {key: (None if pd.isna(value) else value)
            for key, value in table.iloc[0].to_dict().items()}
    p = candidates[(best["arm"], best["endpoint"], best["recipe"])]
    np.save(out / "best_verified_prediction.npy", p)
    pd.DataFrame({"row_index": indices, "log_id": raw.log_id.to_numpy()[indices],
                  "label": y, "pctr": p}).to_csv(out / "best_verified_day16.csv", index=False)
    if day == 20191116 and best["auc"] >= .82:
        target_met = True
    else:
        target_met = False
    result = {"status": "passed", "evaluation_date": day, "rows": len(indices), "clicks": int(y.sum()),
              "all_original_dataset_hashes_unchanged": True,
              "all_training_source_hashes_unchanged": True,
              "inputs_rebuilt_from_original_files": True,
              "duplicate_target_reconstruction_verified": family == "duplicate_mean_v1",
              "complete_day_indices_match_raw_csv": True,
              "evaluation_labels_masked_in_training_array": bool(np.all(data["label"][data["valid"]] == -1)),
              "maximum_probability_errors": replay_errors,
              "reference": evaluate(y, reference), "best_candidate": best,
              "real16_auc_target": .82, "target_met": target_met,
              "development_day_reused": True, "blind_test_claim": False,
              "best_prediction_sha256": digest(out / "best_verified_prediction.npy"),
              "best_prediction_csv_sha256": digest(out / "best_verified_day16.csv"),
              "seconds": time.perf_counter() - started}
    write(out / "verification.json", result)
    print(table.head(12).to_string(index=False), flush=True)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, choices=[20191115, 20191116], default=20191116)
    parser.add_argument("--family", choices=["weighted_v1", "duplicate_mean_v1"], default="weighted_v1")
    args = parser.parse_args()
    main(args.day, args.family)
