"""Prepare a separate cache with strictly historical fixed-window source features."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from benchmark_core import source_history_features
from train_models import write_json

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "2026ml"


def fixed_window_source_features(feeds, requests, window_days=3):
    result = np.empty((len(requests), 7), dtype=np.float32)
    days = requests["pt_d"].to_numpy()
    for day in np.unique(days):
        positions = np.flatnonzero(days == day)
        first = int((datetime.strptime(str(day), "%Y%m%d") - timedelta(days=window_days)).strftime("%Y%m%d"))
        window = feeds.loc[feeds["e_et"].ge(first) & feeds["e_et"].lt(day)]
        values = source_history_features(window, requests.iloc[positions])
        # This is an explicit 'no event in window' sentinel, not an observed gap.
        values[values[:, 0] == 0, 6] = window_days + 1
        result[positions] = values
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, default=HERE / "cache")
    parser.add_argument("--window-days", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving existing directory: {args.output_dir}")
    started = time.perf_counter()
    manifest = json.loads((args.base_cache / "manifest.json").read_text(encoding="utf-8"))
    requests = pd.read_csv(DATA / "train.csv", usecols=["user_id", "pt_d"], dtype="int32").drop_duplicates()
    feeds = pd.read_csv(DATA / "train_feeds.csv", usecols=["u_userId", "e_et", "label", "pro", "i_cat"], dtype="int32")
    source_values = fixed_window_source_features(feeds, requests, args.window_days)
    keys = requests["pt_d"].to_numpy(dtype=np.int64) * 10000000 + requests["user_id"].to_numpy(dtype=np.int64)
    table = pd.DataFrame(source_values, index=keys)
    args.output_dir.mkdir(parents=True)
    for name in ["categories", "sequences", "baseline", "label", "date", "log_id", "train_indices", "valid_indices"]:
        shutil.copy2(args.base_cache / (name + ".npy"), args.output_dir / (name + ".npy"))
    shutil.copy2(args.base_cache / "vocabs.json", args.output_dir / "vocabs.json")
    base = np.load(args.base_cache / "numeric.npy", mmap_mode="r")
    dates = np.load(args.base_cache / "date.npy", mmap_mode="r")
    # Decode users from the exact integer vocabulary rather than the float baseline.
    vocabs = json.loads((args.base_cache / "vocabs.json").read_text(encoding="utf-8"))
    decoded_users = np.zeros(manifest["vocabulary_sizes"]["user_id"], dtype=np.int64)
    for raw, encoded in vocabs["user_id"].items():
        decoded_users[encoded] = int(raw)
    categories = np.load(args.base_cache / "categories.npy", mmap_mode="r")
    user_column = manifest["categorical_fields"].index("user_id")
    assert np.all(categories[:, user_column] >= 2), "This cache builder expects known users in all advertisement rows"
    numeric = open_memmap(args.output_dir / "numeric.npy", mode="w+", shape=base.shape, dtype=np.float32)
    for start in range(0, len(base), 100000):
        end = min(start + 100000, len(base))
        numeric[start:end] = base[start:end]
        request_keys = dates[start:end].astype(np.int64) * 10000000 + decoded_users[categories[start:end, user_column]]
        values = table.reindex(request_keys).to_numpy(dtype=np.float32)
        assert np.isfinite(values).all()
        values[:, 3] /= 100.
        values[:, [0, 1, 4, 5, 6]] = np.log1p(values[:, [0, 1, 4, 5, 6]])
        numeric[start:end, 13:] = values
    train = np.load(args.base_cache / "train_indices.npy")
    scaler = {}
    for index in [13, 14, 17, 18, 19]:
        fit = numeric[train, index]
        mean, std = float(fit.mean()), max(float(fit.std()), 1e-6)
        numeric[:, index] = (numeric[:, index] - mean) / std
        scaler[manifest["numeric_fields"][index]] = {"mean": mean, "std": std}
    numeric.flush()
    assert np.isfinite(numeric).all()
    manifest.update(numeric_scaler=scaler, feature_window_days=args.window_days,
                    base_cache=str(args.base_cache.resolve()),
                    source_rule=f"source day in [advertisement day minus {args.window_days} calendar days, advertisement day)",
                    no_window_history_recency_sentinel=args.window_days + 1,
                    preparation_seconds=time.perf_counter() - started)
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"output": str(args.output_dir), "validation_day": manifest["validation_day"],
                      "window_days": args.window_days, "preparation_seconds": manifest["preparation_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
