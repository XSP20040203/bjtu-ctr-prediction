"""Append matched source-day profiles with train-only vocabulary and scaling."""
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
from benchmark_core import make_vocab, encode_sequence, category_key
from train_models import write_json
from source_profiles import (DAILY_CATEGORIES, DAILY_SEQUENCES, DAILY_NUMERIC, UNIT100,
                             LOG_STANDARD, load_profiles, interest_overlap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving existing cache: {args.output_dir}")
    started = time.perf_counter()
    manifest = json.loads((args.base_cache / "manifest.json").read_text(encoding="utf-8"))
    vocabs = json.loads((args.base_cache / "vocabs.json").read_text(encoding="utf-8"))
    categories = np.load(args.base_cache / "categories.npy", mmap_mode="r")
    sequences = np.load(args.base_cache / "sequences.npy", mmap_mode="r")
    dates = np.load(args.base_cache / "date.npy", mmap_mode="r")
    train = np.load(args.base_cache / "train_indices.npy")
    old_categories, old_sequences, old_numeric = list(manifest["categorical_fields"]), list(manifest["sequence_fields"]), list(manifest["numeric_fields"])
    group_for = dict(manifest["group_for_field"])
    decoded = np.zeros(manifest["vocabulary_sizes"]["user_id"], dtype=np.int64)
    for raw, index in vocabs["user_id"].items():
        decoded[index] = int(raw)
    users = decoded[categories[:, old_categories.index("user_id")]]
    assert (users > 0).all()
    keys = dates.astype(np.int64) * 10000000 + users
    profile = load_profiles(HERE)
    fit = profile.reindex(np.unique(keys[train]))
    for name in DAILY_CATEGORIES:
        group_for[name] = name
        vocabs[name] = make_vocab(fit[name].fillna("__MISSING__"))
    news = vocabs["news_interest"]
    additions = set()
    for name in DAILY_SEQUENCES:
        group_for[name] = "news_interest"
        for value in fit[name].fillna("").unique():
            additions.update(token for token in str(value).split("^") if token)
    for token in sorted(additions):
        token = category_key(token)
        if token not in news:
            news[token] = len(news) + 2
    args.output_dir.mkdir()
    for name in ["label", "date", "log_id", "baseline", "train_indices", "valid_indices"]:
        shutil.copy2(args.base_cache / (name + ".npy"), args.output_dir / (name + ".npy"))
    cat_out = open_memmap(args.output_dir / "categories.npy", mode="w+", shape=(len(keys), len(old_categories) + len(DAILY_CATEGORIES)), dtype="int32")
    seq_out = open_memmap(args.output_dir / "sequences.npy", mode="w+", shape=(len(keys), len(old_sequences) + len(DAILY_SEQUENCES), 5), dtype="int32")
    numeric_out = open_memmap(args.output_dir / "numeric.npy", mode="w+", shape=(len(keys), len(old_numeric) + len(DAILY_NUMERIC)), dtype="float32")
    base_numeric = np.load(args.base_cache / "numeric.npy", mmap_mode="r")
    token_cache = {name: {} for name in DAILY_SEQUENCES}
    raw_interest = pd.read_csv(HERE.parents[1] / "2026ml/train.csv", usecols=["u_newsCatInterestsST"], dtype=str)["u_newsCatInterestsST"].fillna("")
    overlap_cache = {}
    short_cache = {}
    for start in range(0, len(keys), 60000):
        end = min(start + 60000, len(keys))
        aligned = profile.reindex(keys[start:end])
        missing = aligned["feed_day_impressions"].isna().to_numpy()
        cat_out[start:end, :len(old_categories)] = categories[start:end]
        seq_out[start:end, :len(old_sequences)] = sequences[start:end]
        numeric_out[start:end, :len(old_numeric)] = base_numeric[start:end]
        for i, name in enumerate(DAILY_CATEGORIES):
            values = aligned[name].fillna("__MISSING__").map(category_key)
            cat_out[start:end, len(old_categories) + i] = values.map(vocabs[name]).fillna(1).to_numpy(dtype=np.int32)
        for i, name in enumerate(DAILY_SEQUENCES):
            values = aligned[name].fillna("")
            for value in values.unique():
                if value not in token_cache[name]:
                    token_cache[name][value] = encode_sequence(value, news)
            seq_out[start:end, len(old_sequences) + i] = np.stack([token_cache[name][value] for value in values])
        short_values = raw_interest.iloc[start:end].to_numpy()
        for value in np.unique(short_values):
            if value not in short_cache:
                short_cache[value] = encode_sequence(value, news)
        seq_out[start:end, old_sequences.index("u_newsCatInterestsST")] = np.stack([short_cache[value] for value in short_values])
        added = np.zeros((end - start, len(DAILY_NUMERIC)), dtype=np.float32)
        for i, name in enumerate(DAILY_NUMERIC[:15]):
            values = aligned[name].fillna(.1 if name == "feed_day_ctr_smoothed" else 0).to_numpy(dtype=np.float32)
            if name in UNIT100:
                values = values / 100.
            elif name in LOG_STANDARD:
                values = np.log1p(values)
            added[:, i] = values
        clicked = aligned["_all_feed_clicked_topics"].fillna("").to_numpy()
        exposed = aligned["_all_feed_exposed_topics"].fillna("").to_numpy()
        longterm = aligned["_all_feed_long_interest"].fillna("").to_numpy()
        for i, (key, short, c, e, l) in enumerate(zip(keys[start:end], short_values, clicked, exposed, longterm)):
            cache_key = (int(key), short)
            if cache_key not in overlap_cache:
                overlap_cache[cache_key] = interest_overlap(short, c, e, l)
            added[i, 15:18] = overlap_cache[cache_key]
        added[:, 18] = missing
        numeric_out[start:end, len(old_numeric):] = added
    scaler = dict(manifest["numeric_scaler"])
    for name in LOG_STANDARD:
        index = len(old_numeric) + DAILY_NUMERIC.index(name)
        fit_values = numeric_out[train, index]
        mean, std = float(fit_values.mean()), max(float(fit_values.std()), 1e-6)
        numeric_out[:, index] = (numeric_out[:, index] - mean) / std
        scaler[name] = {"mean": mean, "std": std}
    for array in [cat_out, seq_out, numeric_out]:
        array.flush()
    assert np.isfinite(numeric_out).all()
    manifest.update(categorical_fields=old_categories + DAILY_CATEGORIES, sequence_fields=old_sequences + DAILY_SEQUENCES,
                    numeric_fields=old_numeric + DAILY_NUMERIC, numeric_scaler=scaler, group_for_field=group_for,
                    vocabulary_sizes={name: len(vocab) + 2 for name, vocab in vocabs.items()},
                    feature_variant="rich_offline_same_day_source",
                    source_rule="Original strictly-past features plus separately matched same-day source profiles supplied by the offline task",
                    feature_availability_assumption="A full source-day snapshot is available as auxiliary input; not an online pre-impression benchmark.",
                    source_advertisement_labels_used=False, preprocessing_fit_note="New vocabularies and scalers use only the chosen advertisement training indices; auxiliary source-day labels describe news clicks, not advertisement targets.",
                    preparation_seconds=time.perf_counter() - started)
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "vocabs.json", vocabs)
    print(json.dumps({"output": str(args.output_dir), "categorical_fields": len(manifest["categorical_fields"]),
                      "sequence_fields": len(manifest["sequence_fields"]), "numeric_fields": len(manifest["numeric_fields"]),
                      "seconds": manifest["preparation_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
