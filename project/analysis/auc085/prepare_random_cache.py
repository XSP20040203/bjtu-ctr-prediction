"""Fit a random-partition vocabulary/scaler without altering historical caches."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "model_comparison"
sys.path.insert(0, str(BENCH))
from benchmark_core import make_vocab, encode_sequence
from train_models import write_json
from split_utils import make_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = HERE / f"cache_random20_seed{args.seed}"
    if output.exists():
        raise FileExistsError(f"Preserving existing cache: {output}")
    source = BENCH / "cache"
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    labels = np.load(source / "label.npy", mmap_mode="r")
    dates = np.load(source / "date.npy", mmap_mode="r")
    train, valid = make_split(labels, dates, "random20", args.seed)
    fit_mask = np.zeros(len(labels), dtype=bool)
    fit_mask[train] = True
    categories, sequences = manifest["categorical_fields"], manifest["sequence_fields"]
    group_for = manifest["group_for_field"]
    groups = {group: set() for group in sorted(manifest["vocabulary_sizes"])}
    data = HERE.parents[1] / "2026ml/train.csv"
    started = time.perf_counter()
    position = 0
    print("Fitting vocabularies using only random training rows", flush=True)
    for chunk in pd.read_csv(data, usecols=categories + sequences, dtype={name: str for name in sequences}, chunksize=60000):
        fit = chunk.loc[fit_mask[position:position + len(chunk)]]
        for name in categories:
            groups[group_for[name]].update(int(value) for value in fit[name].unique())
        for name in sequences:
            for value in fit[name].dropna().unique():
                groups[group_for[name]].update(int(token) for token in value.split("^") if token)
        position += len(chunk)
    vocabs = {name: make_vocab(values) for name, values in groups.items()}
    numeric_vocabs = {name: {int(raw): index for raw, index in vocab.items()} for name, vocab in vocabs.items()}
    output.mkdir()
    write_json(output / "vocabs.json", vocabs)
    for name in ["label", "date", "log_id", "baseline"]:
        shutil.copy2(source / (name + ".npy"), output / (name + ".npy"))
    np.save(output / "train_indices.npy", train)
    np.save(output / "valid_indices.npy", valid)
    encoded_categories = open_memmap(output / "categories.npy", mode="w+", shape=(len(labels), len(categories)), dtype="int32")
    encoded_sequences = open_memmap(output / "sequences.npy", mode="w+", shape=(len(labels), len(sequences), 5), dtype="int32")
    caches = {name: {} for name in sequences}
    position = 0
    print("Encoding random training and validation rows", flush=True)
    for chunk in pd.read_csv(data, usecols=categories + sequences, dtype={name: str for name in sequences}, chunksize=60000):
        end = position + len(chunk)
        for i, name in enumerate(categories):
            encoded_categories[position:end, i] = chunk[name].map(numeric_vocabs[group_for[name]]).fillna(1).to_numpy(dtype=np.int32)
        for i, name in enumerate(sequences):
            values = chunk[name].fillna("")
            for value in values.unique():
                if value not in caches[name]:
                    caches[name][value] = encode_sequence(value, vocabs[group_for[name]], 5)
            encoded_sequences[position:end, i] = np.stack([caches[name][value] for value in values])
        position = end
    original = np.load(source / "numeric.npy", mmap_mode="r")
    numeric = open_memmap(output / "numeric.npy", mode="w+", shape=original.shape, dtype="float32")
    for start in range(0, len(labels), 100000):
        numeric[start:start + 100000] = original[start:start + 100000]
    scaler = {}
    for name, old in manifest["numeric_scaler"].items():
        index = manifest["numeric_fields"].index(name)
        raw = numeric[:, index] * old["std"] + old["mean"]
        mean, std = float(raw[train].mean()), max(float(raw[train].std()), 1e-6)
        numeric[:, index] = (raw - mean) / std
        scaler[name] = {"mean": mean, "std": std}
    for array in [encoded_categories, encoded_sequences, numeric]:
        array.flush()
    assert position == len(labels) and not np.intersect1d(train, valid).size
    manifest.update(version=2, validation_day=None, validation_strategy="random20", split_seed=args.seed,
                    train_rows=len(train), valid_rows=len(valid), train_clicks=int(labels[train].sum()),
                    valid_clicks=int(labels[valid].sum()), numeric_scaler=scaler,
                    vocabulary_sizes={name: len(vocab) + 2 for name, vocab in vocabs.items()},
                    later_rows_excluded_from_training_and_validation=0,
                    preprocessing_fit_note="Vocabulary and scalers fitted only on random training indices; source features remain strictly historical.",
                    preparation_seconds=time.perf_counter() - started)
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "train_rows": len(train), "valid_rows": len(valid),
                      "valid_clicks": manifest["valid_clicks"], "seconds": manifest["preparation_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
