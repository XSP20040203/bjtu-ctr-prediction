"""Cross-fitted advertisement statistics with duplicate groups kept in one fold."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from numpy.lib.format import open_memmap
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "model_comparison"
sys.path.insert(0, str(BENCH))
from train_models import write_json

SPECS = [(["user_id"], 200), (["task_id"], 100), (["adv_id"], 100), (["adv_prim_id"], 200),
         (["spread_app_id"], 200), (["slot_id"], 500), (["app_second_class"], 500),
         (["user_id", "slot_id"], 100), (["user_id", "adv_prim_id"], 100),
         (["user_id", "spread_app_id"], 100), (["user_id", "app_second_class"], 100),
         (["task_id", "slot_id"], 50), (["adv_prim_id", "slot_id"], 100),
         (["spread_app_id", "slot_id"], 100), (["app_second_class", "slot_id"], 200),
         (["age", "slot_id"], 200), (["sex", "slot_id"], 200), (["device_name", "slot_id"], 100)]


def oof_target_statistics(keys, labels, train_indices, fold_for_row, alpha=100.):
    if alpha <= 0:
        raise ValueError("Smoothing must be positive")
    unique, codes = np.unique(keys, return_inverse=True)
    train = np.asarray(train_indices)
    assert np.all(fold_for_row[train] >= 0)
    counts = np.bincount(codes[train], minlength=len(unique))
    clicks = np.bincount(codes[train], weights=labels[train], minlength=len(unique))
    total_clicks = float(labels[train].sum())
    prior = float(np.clip(total_clicks / len(train), 1e-6, 1 - 1e-6))
    result = np.column_stack([counts[codes], (clicks[codes] + alpha * prior) / (counts[codes] + alpha)]).astype(np.float32)
    for fold in np.unique(fold_for_row[train]):
        held = train[fold_for_row[train] == fold]
        removed_count = np.bincount(codes[held], minlength=len(unique))
        removed_clicks = np.bincount(codes[held], weights=labels[held], minlength=len(unique))
        fit_count, fit_clicks = counts - removed_count, clicks - removed_clicks
        fold_prior = float(np.clip((total_clicks - labels[held].sum()) / (len(train) - len(held)), 1e-6, 1 - 1e-6))
        result[held, 0] = fit_count[codes[held]]
        result[held, 1] = (fit_clicks[codes[held]] + alpha * fold_prior) / (fit_count[codes[held]] + alpha)
    return result, {"keys": unique, "counts": counts, "clicks": clicks, "prior": prior, "alpha": alpha}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving existing cache {args.output_dir}")
    started = time.perf_counter()
    manifest = json.loads((args.base_cache / "manifest.json").read_text(encoding="utf-8"))
    labels = np.load(args.base_cache / "label.npy", mmap_mode="r")
    train = np.load(args.base_cache / "train_indices.npy")
    valid = np.load(args.base_cache / "valid_indices.npy")
    raw_hash = np.load(HERE / "raw_predictor_hash.npy", mmap_mode="r")
    folds = np.full(len(labels), -1, dtype=np.int8)
    print("Assigning five duplicate-group folds for target statistics", flush=True)
    for fold, (_, held) in enumerate(GroupKFold(5).split(np.empty(len(train)), labels[train], groups=raw_hash[train])):
        folds[train[held]] = fold
    assert np.all(folds[train] >= 0) and np.all(folds[valid] == -1)
    # Equal raw predictors must have the same cross-fitting fold.
    order = np.argsort(raw_hash[train])
    same = raw_hash[train][order][1:] == raw_hash[train][order][:-1]
    assert np.all(folds[train][order][1:][same] == folds[train][order][:-1][same])
    args.output_dir.mkdir()
    for name in ["categories", "sequences", "label", "date", "log_id", "baseline", "train_indices", "valid_indices"]:
        shutil.copy2(args.base_cache / (name + ".npy"), args.output_dir / (name + ".npy"))
    shutil.copy2(args.base_cache / "vocabs.json", args.output_dir / "vocabs.json")
    np.save(args.output_dir / "target_stat_oof_fold.npy", folds)
    category = np.load(args.base_cache / "categories.npy", mmap_mode="r")
    numeric = np.load(args.base_cache / "numeric.npy", mmap_mode="r")
    width = numeric.shape[1]
    output = open_memmap(args.output_dir / "numeric.npy", mode="w+", shape=(len(labels), width + len(SPECS) * 2), dtype="float32")
    for start in range(0, len(labels), 100000):
        output[start:start + 100000, :width] = numeric[start:start + 100000]
    names, scaler, definitions = list(manifest["numeric_fields"]), dict(manifest["numeric_scaler"]), []
    model_dir = args.output_dir / "target_stat_models"
    model_dir.mkdir()
    for position, (fields, alpha) in enumerate(SPECS):
        keys = np.zeros(len(labels), dtype=np.int64)
        for field in fields:
            column = manifest["categorical_fields"].index(field)
            size = manifest["vocabulary_sizes"][manifest["group_for_field"][field]]
            keys = keys * size + category[:, column]
        encoded, model = oof_target_statistics(keys, labels, train, folds, alpha)
        group = "__".join(fields)
        count = np.log1p(encoded[:, 0])
        probability = np.clip(encoded[:, 1], 1e-6, 1 - 1e-6)
        relative_logit = np.log(probability / (1 - probability)) - np.log(model["prior"] / (1 - model["prior"]))
        for offset, (suffix, values) in enumerate([("log_count", count), ("relative_logit", relative_logit)]):
            name = "target_stat__" + group + "__" + suffix
            mean, std = float(values[train].mean()), max(float(values[train].std()), 1e-6)
            output[:, width + position * 2 + offset] = (values - mean) / std
            scaler[name] = {"mean": mean, "std": std}
            names.append(name)
        np.savez_compressed(model_dir / (group + ".npz"), **model)
        definitions.append({"fields": fields, "alpha": alpha})
        print(f"TARGET_STAT {position + 1}/{len(SPECS)} {group}", flush=True)
    output.flush()
    assert np.isfinite(output).all()
    manifest.update(numeric_fields=names, numeric_scaler=scaler,
                    feature_variant=manifest.get("feature_variant", "historical") + "+oof_target_stats",
                    target_stat_definitions=definitions, target_stat_crossfit_folds=5,
                    target_stat_fold_unit="64-bit hash of identical raw advertisement predictors including date; all matching training rows stay in the same fold",
                    target_stat_validation_labels_used=False,
                    preparation_seconds=time.perf_counter() - started)
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"output": str(args.output_dir), "numeric_fields": len(names), "seconds": manifest["preparation_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
