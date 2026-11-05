"""Prepare all training records once, using memory-mapped arrays and past-only source data."""
from __future__ import annotations

import gc
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from benchmark_core import category_key, encode_sequence, make_vocab, source_history_features

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "2026ml"
CACHE = HERE / "cache"
BASE_FIELDS = ["log_id", "user_id", "age", "sex", "residence", "device_name", "device_size", "net_type", "task_id", "adv_id", "creat_type_cd"]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_cache(cache_dir=CACHE, validation_day=20191116):
    started = time.perf_counter()
    CACHE = Path(cache_dir)
    CACHE.mkdir(parents=True, exist_ok=True)
    schema = json.loads((ROOT / "analysis/neural_baseline/feature_schema.json").read_text(encoding="utf-8"))
    categories, sequences = schema["categorical_fields"], schema["sequence_fields"]
    group_for = {name: name for name in categories + sequences}
    for group, names in schema["shared_embedding_groups"].items():
        for name in names:
            group_for[name] = group
    group_for["u_newsCatInterestsST"] = "news_interest"
    groups = {name: set() for name in sorted(set(group_for.values()))}
    usecols = categories + sequences + ["pt_d"]
    row_count = 0
    train_count = 0
    valid_count = 0
    print(f"Pass 1: building vocabulary on days strictly before {validation_day}", flush=True)
    for chunk in pd.read_csv(DATA / "train.csv", usecols=usecols, dtype={name: str for name in sequences}, chunksize=60000):
        row_count += len(chunk)
        fit = chunk.loc[chunk["pt_d"].lt(validation_day)]
        train_count += len(fit)
        valid_count += int(chunk["pt_d"].eq(validation_day).sum())
        for name in categories:
            groups[group_for[name]].update(int(value) for value in fit[name].unique())
        for name in sequences:
            for value in fit[name].dropna().unique():
                groups[group_for[name]].update(int(token) for token in value.split("^") if token)
    vocabs = {name: make_vocab(values) for name, values in groups.items()}
    numeric_vocabs = {name: {int(key): value for key, value in vocabulary.items()} for name, vocabulary in vocabs.items()}
    sizes = {name: len(vocabulary) + 2 for name, vocabulary in vocabs.items()}
    write_json(CACHE / "vocabs.json", vocabs)
    print(f"Rows: {row_count:,}; train: {train_count:,}; validation: {valid_count:,}; later rows excluded: {row_count-train_count-valid_count:,}", flush=True)

    requests = pd.read_csv(DATA / "train.csv", usecols=["user_id", "pt_d"], dtype="int32").drop_duplicates()
    feed = pd.read_csv(DATA / "train_feeds.csv", usecols=["u_userId", "e_et", "label", "pro", "i_cat"], dtype="int32")
    source_values = source_history_features(feed, requests)
    keys = requests["pt_d"].to_numpy(dtype=np.int64) * 10000000 + requests["user_id"].to_numpy(dtype=np.int64)
    source_table = pd.DataFrame(source_values, index=keys)
    del feed, requests, source_values
    gc.collect()

    arrays = {
        "categories": open_memmap(CACHE / "categories.npy", mode="w+", dtype="int32", shape=(row_count, 24)),
        "sequences": open_memmap(CACHE / "sequences.npy", mode="w+", dtype="int32", shape=(row_count, 7, 5)),
        "numeric": open_memmap(CACHE / "numeric.npy", mode="w+", dtype="float32", shape=(row_count, 20)),
        "baseline": open_memmap(CACHE / "baseline.npy", mode="w+", dtype="float32", shape=(row_count, 11)),
        "label": open_memmap(CACHE / "label.npy", mode="w+", dtype="uint8", shape=(row_count,)),
        "date": open_memmap(CACHE / "date.npy", mode="w+", dtype="int32", shape=(row_count,)),
        "log_id": open_memmap(CACHE / "log_id.npy", mode="w+", dtype="int32", shape=(row_count,)),
    }
    sequence_cache = {name: {} for name in sequences}
    token_sets = {name: {} for name in sequences}
    position = 0
    print("Pass 2: encoding all rows and adding past-only source features", flush=True)
    for chunk in pd.read_csv(DATA / "train.csv", dtype={name: str for name in sequences}, chunksize=60000):
        end = position + len(chunk)
        for index, name in enumerate(categories):
            arrays["categories"][position:end, index] = chunk[name].map(numeric_vocabs[group_for[name]]).fillna(1).to_numpy(dtype=np.int32)
        numeric = np.empty((len(chunk), 20), dtype=np.float32)
        for index, name in enumerate(sequences):
            values = chunk[name].fillna("")
            cache = sequence_cache[name]
            memberships = token_sets[name]
            for value in values.unique():
                if value not in cache:
                    cache[value] = encode_sequence(value, vocabs[group_for[name]], 5)
                    memberships[value] = {int(token) for token in value.split("^") if token}
            encoded = np.stack([cache[value] for value in values])
            arrays["sequences"][position:end, index] = encoded
            numeric[:, 6 + index] = np.count_nonzero(encoded, axis=1) / 5.
            if index < 6:
                current = ["task_id", "adv_prim_id", "spread_app_id"][index % 3]
                numeric[:, index] = np.fromiter((int(target) in memberships[value] for value, target in zip(values, chunk[current])), dtype=np.float32, count=len(chunk))
        request_keys = chunk["pt_d"].to_numpy(dtype=np.int64) * 10000000 + chunk["user_id"].to_numpy(dtype=np.int64)
        source = source_table.reindex(request_keys).to_numpy(dtype=np.float32)
        assert np.isfinite(source).all()
        numeric[:, 13:] = source
        numeric[:, 16] /= 100.
        for index in [13, 14, 17, 18, 19]:
            numeric[:, index] = np.log1p(numeric[:, index])
        arrays["numeric"][position:end] = numeric
        arrays["baseline"][position:end] = chunk[BASE_FIELDS].to_numpy(dtype=np.float32)
        for name, column in [("label", "label"), ("date", "pt_d"), ("log_id", "log_id")]:
            arrays[name][position:end] = chunk[column].to_numpy()
        position = end
        if position % 600000 == 0:
            print(f"  Encoded {position:,} rows", flush=True)
    del sequence_cache, token_sets, source_table
    gc.collect()
    train_indices = np.flatnonzero(arrays["date"] < validation_day).astype(np.int32)
    valid_indices = np.flatnonzero(arrays["date"] == validation_day).astype(np.int32)
    np.save(CACHE / "train_indices.npy", train_indices)
    np.save(CACHE / "valid_indices.npy", valid_indices)
    scaler = {}
    for index in [13, 14, 17, 18, 19]:
        fit_values = arrays["numeric"][train_indices, index]
        mean, std = float(fit_values.mean()), max(float(fit_values.std()), 1e-6)
        scaler[schema["numeric_fields"][index]] = {"mean": mean, "std": std}
        arrays["numeric"][:, index] = (arrays["numeric"][:, index] - mean) / std
    for value in arrays.values():
        value.flush()
    assert position == row_count == 2526320
    expected = {20191116: (421983, 3724), 20191115: (424780, 3523)}
    assert len(valid_indices) == expected[validation_day][0]
    assert int(arrays["label"][valid_indices].sum()) == expected[validation_day][1]
    assert int(arrays["date"][train_indices].max()) < validation_day
    assert np.all(arrays["date"][valid_indices] == validation_day)
    assert np.isfinite(arrays["numeric"]).all()
    manifest = {
        "version": 1, "rows": row_count, "train_rows": len(train_indices), "valid_rows": len(valid_indices),
        "later_rows_excluded_from_training_and_validation": row_count - len(train_indices) - len(valid_indices),
        "train_clicks": int(arrays["label"][train_indices].sum()), "valid_clicks": int(arrays["label"][valid_indices].sum()),
        "validation_day": validation_day, "source_rule": "source e_et strictly less than each ad pt_d",
        "categorical_fields": categories, "sequence_fields": sequences, "numeric_fields": schema["numeric_fields"],
        "baseline_fields": BASE_FIELDS, "group_for_field": group_for, "vocabulary_sizes": sizes,
        "numeric_scaler": scaler, "preparation_seconds": time.perf_counter() - started,
        "sources": {name: {"bytes": (DATA/name).stat().st_size, "mtime_ns": (DATA/name).stat().st_mtime_ns} for name in ["train.csv", "train_feeds.csv"]},
    }
    write_json(CACHE / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ["rows", "train_rows", "valid_rows", "train_clicks", "valid_clicks", "preparation_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-day", type=int, choices=[20191115, 20191116], default=20191116)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    args = parser.parse_args()
    prepare_cache(args.cache_dir, args.validation_day)


if __name__ == "__main__":
    main()
