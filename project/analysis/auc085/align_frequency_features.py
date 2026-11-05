"""Separate unsupervised training-input frequency from cross-fitted target rates."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from numpy.lib.format import open_memmap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model_comparison"))
from train_models import write_json


def training_input_frequency(keys, train_indices):
    if not len(train_indices):
        raise ValueError("Training inputs are required to fit frequencies")
    observed, counts = np.unique(keys[train_indices], return_counts=True)
    position = np.searchsorted(observed, keys)
    safe = np.minimum(position, len(observed) - 1)
    known = (position < len(observed)) & (observed[safe] == keys)
    result = np.zeros(len(keys), dtype=np.int64)
    result[known] = counts[safe[known]]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving existing cache {args.output_dir}")
    started = time.perf_counter()
    manifest = json.loads((args.base_cache / "manifest.json").read_text(encoding="utf-8"))
    category = np.load(args.base_cache / "categories.npy", mmap_mode="r")
    numeric = np.load(args.base_cache / "numeric.npy", mmap_mode="r")
    train = np.load(args.base_cache / "train_indices.npy")
    args.output_dir.mkdir(parents=True)
    for name in ["categories", "sequences", "label", "date", "log_id", "baseline", "train_indices", "valid_indices", "target_stat_oof_fold"]:
        shutil.copy2(args.base_cache / (name + ".npy"), args.output_dir / (name + ".npy"))
    shutil.copy2(args.base_cache / "vocabs.json", args.output_dir / "vocabs.json")
    shutil.copytree(args.base_cache / "target_stat_models", args.output_dir / "target_stat_models")
    output = open_memmap(args.output_dir / "numeric.npy", mode="w+", shape=numeric.shape, dtype="float32")
    for start in range(0, len(numeric), 100000):
        output[start:start + 100000] = numeric[start:start + 100000]
    changed = []
    for definition in manifest["target_stat_definitions"]:
        fields = definition["fields"]
        keys = np.zeros(len(category), dtype=np.int64)
        for name in fields:
            index = manifest["categorical_fields"].index(name)
            size = manifest["vocabulary_sizes"][manifest["group_for_field"][name]]
            keys = keys * size + category[:, index]
        count = training_input_frequency(keys, train)
        values = np.log1p(count.astype(np.float32))
        name = "target_stat__" + "__".join(fields) + "__log_count"
        index = manifest["numeric_fields"].index(name)
        mean, std = float(values[train].mean()), max(float(values[train].std()), 1e-6)
        output[:, index] = (values - mean) / std
        manifest["numeric_scaler"][name] = {"mean": mean, "std": std}
        changed.append(name)
    output.flush()
    assert np.isfinite(output).all()
    manifest.update(feature_variant=manifest["feature_variant"] + "+aligned_training_frequencies",
                    frequency_alignment={"frequency_counts": "All and only advertisement training input rows; no labels accepted by frequency fitting",
                                         "target_rates": "Existing duplicate-group cross-fitted target rates preserved exactly",
                                         "modified_fields": changed},
                    base_cache=str(args.base_cache.resolve()), preparation_seconds=time.perf_counter() - started)
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"output": str(args.output_dir), "changed_frequency_fields": len(changed),
                      "target_rate_columns_changed": False, "seconds": manifest["preparation_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
