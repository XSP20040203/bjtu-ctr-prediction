"""Rebuild the original 166-column GatedCross inputs in RAM."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[name] = "3"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCH = ROOT / "analysis/architecture_goal_20261029"
for path in ["analysis/model_comparison", "analysis/auc085", "analysis/daily_interest_din"]:
    sys.path.insert(0, str(ROOT / path))

import numpy as np
import pandas as pd
from benchmark_core import source_history_features
from predict_final import SavedFeatureTransform
from source_profiles import load_profiles
from target_statistics import oof_target_statistics
from prepare_exposure_features import build_exposure_features
from prepare_snapshot_features import build_features as build_snapshot_features


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def cache_dir(day):
    return ROOT / ("analysis/user_personalization/cache_day15_stats" if day == 20191115
                   else "analysis/auc085/cache_rich_stats_time16")


class FoldTransform(SavedFeatureTransform):
    def __init__(self, day):
        cache = cache_dir(day)
        self.manifest = read(cache / "manifest.json")
        self.manifests = {"signed": self.manifest}
        self.vocabs = read(cache / "vocabs.json")
        self.raw_categories = self.manifest["categorical_fields"][:24]
        self.raw_sequences = self.manifest["sequence_fields"][:7]
        self.read_columns = self.raw_categories + self.raw_sequences + ["pt_d", "log_id"]
        self.raw_vocab = {
            field: {int(raw): code for raw, code in
                    self.vocabs[self.manifest["group_for_field"][field]].items()}
            for field in self.raw_categories
        }
        self.sequence_cache = {field: {} for field in self.manifest["sequence_fields"]}
        self.membership_cache = {field: {} for field in self.raw_sequences[:6]}
        # Chronological caches did not undo a previous random-split scaler.
        self.original_scaler = {
            self.manifest["numeric_fields"][i]: {"mean": 0., "std": 1.}
            for i in [13, 14, 17, 18, 19]
        }
        self.profiles = load_profiles(ROOT / "analysis/auc085")
        self.profiles = self.profiles.loc[self.profiles.index // 10000000 <= day]
        feeds = pd.read_csv(ROOT / "2026ml/train_feeds.csv",
                            usecols=["u_userId", "e_et", "label", "pro", "i_cat"],
                            dtype="int32")
        requests = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["user_id", "pt_d"],
                               dtype="int32").drop_duplicates()
        requests = requests.loc[requests.pt_d <= day]
        keys = (requests.pt_d.to_numpy(dtype="int64") * 10000000 +
                requests.user_id.to_numpy(dtype="int64"))
        self.past_table = pd.DataFrame(source_history_features(feeds, requests), index=keys)
        self.statistics = []
        for definition in self.manifest["target_stat_definitions"]:
            name = "__".join(definition["fields"])
            with np.load(cache / "target_stat_models" / (name + ".npz")) as model:
                self.statistics.append((definition["fields"],
                                        {key: model[key].copy() for key in model.files}))


def standardize_architecture(values, names, architecture):
    means = np.array([architecture["scalers"][name]["mean"] for name in names])
    stds = np.array([architecture["scalers"][name]["std"] for name in names])
    return np.clip((values - means) / stds, -8, 8).astype("float32")


def prepare(day, destination):
    started = time.perf_counter()
    cache = cache_dir(day)
    transform = FoldTransform(day)
    meta = transform.manifest
    architecture = read(ARCH / f"cache/day{day % 100}_oof/manifest.json")
    dates = np.load(cache / "date.npy")
    selected = np.flatnonzero(dates <= day)
    dates = dates[selected]
    n = len(selected)
    categories = np.empty((n, 27), dtype="int32")
    sequences = np.empty((n, 12, 5), dtype="int32")
    numeric = np.empty((n, 166), dtype="float32")
    users = np.empty(n, dtype="int32")
    tasks = np.empty(n, dtype="int32")
    offset = 0
    for chunk in pd.read_csv(ROOT / "2026ml/train.csv", usecols=transform.read_columns,
                             dtype={field: str for field in transform.raw_sequences},
                             chunksize=60000):
        chunk = chunk.loc[chunk.pt_d <= day]
        if not len(chunk):
            continue
        cats, seqs, values = transform.transform(chunk)
        stop = offset + len(chunk)
        categories[offset:stop], sequences[offset:stop] = cats, seqs
        numeric[offset:stop, :75] = values["signed"]
        users[offset:stop] = chunk.user_id.to_numpy()
        tasks[offset:stop] = chunk.task_id.to_numpy()
        offset = stop
        if offset % 600000 < len(chunk) or offset == n:
            emit("encode_original_inputs", day=day, rows=offset, total=n)
    if offset != n:
        raise AssertionError("Input row count differs")
    train, valid = np.flatnonzero(dates < day), np.flatnonzero(dates == day)
    np.testing.assert_array_equal(selected[train], np.load(cache / "train_indices.npy"))
    np.testing.assert_array_equal(selected[valid], np.load(cache / "valid_indices.npy"))
    labels = np.load(cache / "label.npy")[selected]
    folds = np.load(cache / "target_stat_oof_fold.npy")[selected]
    stat_checks = []
    for position, definition in enumerate(meta["target_stat_definitions"]):
        keys = np.zeros(n, dtype="int64")
        for field in definition["fields"]:
            size = meta["vocabulary_sizes"][meta["group_for_field"][field]]
            keys = keys * size + categories[:, meta["categorical_fields"].index(field)]
        # The exact historical duplicate-group folds are retained. Held-day
        # labels are excluded even when computing the training prior.
        encoded, fitted = oof_target_statistics(keys, labels, train, folds, definition["alpha"])
        old = transform.statistics[position][1]
        old_position = np.searchsorted(old["keys"], fitted["keys"])
        if not np.array_equal(old["keys"][old_position], fitted["keys"]):
            raise AssertionError("Target-stat lookup keys differ")
        for name in ["counts", "clicks"]:
            np.testing.assert_array_equal(old[name][old_position], fitted[name])
        if float(old["prior"]) != float(fitted["prior"]):
            raise AssertionError("Training target-stat prior differs")
        count = np.log1p(encoded[:, 0])
        probability = np.clip(encoded[:, 1], 1e-6, 1 - 1e-6)
        logit = (np.log(probability / (1 - probability)) -
                 np.log(fitted["prior"] / (1 - fitted["prior"])))
        for j, raw in enumerate([count, logit]):
            col = 39 + 2 * position + j
            scale = meta["numeric_scaler"][meta["numeric_fields"][col]]
            numeric[:, col] = (raw - scale["mean"]) / scale["std"]
        stat_checks.append({"fields": definition["fields"], "saved_counts_clicks_exact": True})
    names = architecture["numeric_fields"][:166]
    for start in range(0, n, 30000):
        stop = min(start + 30000, n)
        numeric[start:stop, :75] = standardize_architecture(
            numeric[start:stop, :75], names[:75], architecture)
    del transform, cats, seqs, values, encoded, keys, raw, count, probability, logit
    gc.collect()
    fields = ["pt_d", "user_id", "task_id", "adv_id", "adv_prim_id",
              "spread_app_id", "app_second_class", "slot_id"]
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=fields, dtype="int32")
    raw = raw.iloc[selected].reset_index(drop=True)
    exposure, exposure_names = build_exposure_features(raw)
    if exposure_names != names[75:118]:
        raise AssertionError("Exposure schema differs")
    for start in range(0, n, 30000):
        stop = min(start + 30000, n)
        numeric[start:stop, 75:118] = standardize_architecture(
            exposure[start:stop], exposure_names, architecture)
    del exposure, raw
    gc.collect()
    fields = ["user_id", "pt_d", "task_id", "adv_prim_id", "spread_app_id"]
    fields += [f"ad_{action}_list_v00{i}" for action in ["click", "close"] for i in [1, 2, 3]]
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=fields,
                      dtype={field: "category" if "list" in field else "int32" for field in fields})
    raw = raw.iloc[selected].reset_index(drop=True)
    snapshot, snapshot_names = build_snapshot_features(raw)
    if snapshot_names != names[118:166]:
        raise AssertionError("Snapshot schema differs")
    for start in range(0, n, 30000):
        stop = min(start + 30000, n)
        numeric[start:stop, 118:166] = standardize_architecture(
            snapshot[start:stop], snapshot_names, architecture)
    del snapshot, raw
    gc.collect()
    if not np.isfinite(numeric).all():
        raise AssertionError("Nonfinite original input")
    source_files = [Path(__file__), ROOT / "analysis/auc085/predict_final.py",
                    ROOT / "analysis/auc085/source_profiles.py",
                    ROOT / "analysis/auc085/target_statistics.py",
                    ROOT / "analysis/daily_interest_din/prepare_exposure_features.py",
                    ROOT / "analysis/daily_interest_din/prepare_snapshot_features.py",
                    ROOT / "analysis/model_comparison/benchmark_core.py",
                    cache / "manifest.json", cache / "vocabs.json",
                    cache / "target_stat_oof_fold.npy",
                    ARCH / f"cache/day{day % 100}_oof/manifest.json"]
    info = {
        "day": day, "train_rows": len(train), "valid_rows": len(valid),
        "training_clicks": int(labels[train].sum()), "valid_clicks": int(labels[valid].sum()),
        "numeric_fields": names, "categorical_fields": meta["categorical_fields"],
        "sequence_fields": meta["sequence_fields"], "ram_only_feature_matrices": True,
        "availability": architecture["availability"],
        "target_statistics": "Original saved grouped OOF folds; held day never fitted",
        "target_stat_rebuild_checks": stat_checks,
        "final_test_csv_read": False, "final_source_csv_read": False,
        "metadata_sha256": {str(path): digest(path) for path in source_files},
        "input_sha256": {name: digest(ROOT / "2026ml" / name)
                         for name in ["train.csv", "train_feeds.csv"]},
        "seconds": time.perf_counter() - started,
    }
    write(destination / "feature_manifest.json", info)
    emit("original_features_ready", day=day, train_rows=len(train), valid_rows=len(valid),
         seconds=info["seconds"])
    return {"categories": categories, "sequences": sequences, "numeric": numeric,
            "label": labels, "train": train, "valid": valid, "global_rows": selected,
            "user_id": users, "task_id": tasks, "meta": meta, "feature_manifest": info}
