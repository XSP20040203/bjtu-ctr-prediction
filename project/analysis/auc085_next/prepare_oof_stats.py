"""Expanded grouped OOF statistics for offline supervised learning.

Outer train is pt_d < cutoff, validation is the entire cutoff date. Training
features may use labels from ANY other-fold outer-training date. This is NOT
per-row strictly historical. Validation/future labels are never materialized.
Exact raw predictor duplicates (all 33 columns except label/log_id, INCLUDING
pt_d and original list token order) share a deterministic uint64 hash % 5 fold.
Old target_statistics.py already provided 18 grouped OOF groups; this expands
that family to 60 groups, contextual parent smoothing and a new dense base.
"""
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from prepare_multiscale_stats import _factorize_fields, normalize_day

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DAILY = HERE.parent / "daily_interest_din"
FOLDS = 5
PROTOCOL = "whole_date_outer_group_hash5_oof_v1"


@dataclass(frozen=True)
class GroupSpec:
    fields: tuple
    parents: tuple = ()
    alpha: float = 100.0


def group_specs():
    # Retain the old 18 groups' strengths where applicable, without reading any
    # old cached target-stat columns. Parent tuples refer to earlier definitions.
    sys.path.insert(0, str(HERE.parent / "auc085"))
    from target_statistics import SPECS as old_specs
    strengths = {tuple(fields): float(alpha) for fields, alpha in old_specs}
    result = []
    singles = ("user_id", "task_id", "adv_id", "adv_prim_id", "spread_app_id",
               "slot_id", "app_second_class", "creat_type_cd", "inter_type_cd",
               "age", "sex", "device_name", "device_size", "net_type", "city",
               "residence", "city_rank", "series_group", "hispace_app_tags", "app_score")
    for field in singles:
        parent = {"task_id": "adv_id", "adv_id": "adv_prim_id",
                  "adv_prim_id": "spread_app_id", "spread_app_id": "app_second_class"}.get(field)
        # Singles are intentionally global-shrunk; interactions average their
        # own contextual parents and make no assumption of a functional ID map.
        result.append(GroupSpec((field,), (), strengths.get((field,), 200.0)))
    pairs = [("user_id", field) for field in
             ("slot_id", "task_id", "adv_id", "adv_prim_id", "spread_app_id",
              "app_second_class", "creat_type_cd", "inter_type_cd")]
    pairs += [(field, "slot_id") for field in
              ("task_id", "adv_id", "adv_prim_id", "spread_app_id", "app_second_class",
               "age", "sex", "device_name", "device_size", "net_type", "creat_type_cd", "inter_type_cd")]
    pairs += [(field, context) for field in ("creat_type_cd", "inter_type_cd", "age", "sex", "device_name")
              for context in ("spread_app_id", "app_second_class")]
    pairs += [("adv_prim_id", "creat_type_cd"), ("adv_prim_id", "inter_type_cd")]
    for fields in pairs:
        result.append(GroupSpec(fields, tuple((field,) for field in fields), strengths.get(fields, 100.0)))
    triples = (
        (("user_id", "spread_app_id", "slot_id"), (("user_id", "spread_app_id"), ("spread_app_id", "slot_id"))),
        (("user_id", "spread_app_id", "creat_type_cd"), (("user_id", "spread_app_id"), ("creat_type_cd", "spread_app_id"))),
        (("user_id", "app_second_class", "slot_id"), (("user_id", "app_second_class"), ("app_second_class", "slot_id"))),
        (("user_id", "app_second_class", "creat_type_cd"), (("user_id", "app_second_class"), ("creat_type_cd", "app_second_class"))),
        (("user_id", "adv_prim_id", "slot_id"), (("user_id", "adv_prim_id"), ("adv_prim_id", "slot_id"))),
        (("user_id", "creat_type_cd", "slot_id"), (("user_id", "creat_type_cd"), ("creat_type_cd", "slot_id"))),
    )
    result += [GroupSpec(fields, parents, 60.0) for fields, parents in triples]
    result.append(GroupSpec(("task_id", "creat_type_cd"), (("task_id",), ("creat_type_cd",)), 80.0))
    result.append(GroupSpec(("spread_app_id", "creat_type_cd", "slot_id"),
                            (("creat_type_cd", "spread_app_id"), ("spread_app_id", "slot_id")), 80.0))
    return tuple(result)


SPECS = group_specs()
FIELDS = sorted({field for spec in SPECS for field in spec.fields})


def feature_names(specs=SPECS):
    return ["oof60__" + "__".join(spec.fields) + "__" + suffix
            for spec in specs for suffix in ("log_count", "smoothed_rate")] + ["oof60__global_rate"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def base_cache(day):
    return HERE.parent / ("user_personalization/cache_day15_stats" if normalize_day(day) == 20191115
                          else "auc085/cache_rich_stats_time16")


def raw_predictor_hash(frame):
    """Column order and scalar string representation are fixed by CSV ingestion.

    Raw exact duplicates always hash identically; list token order is retained.
    Hash collisions can only merge groups, making exclusion more conservative.
    The old raw hash's serialization was not documented, so it is not reused.
    """
    predictors = frame.drop(columns=["label", "log_id"], errors="ignore")
    return pd.util.hash_pandas_object(predictors.fillna("").astype(str), index=False).to_numpy(dtype=np.uint64)


def fold_assignments(hashes, dates, day):
    folds = np.full(len(dates), -1, dtype=np.int8)
    train = np.flatnonzero(np.asarray(dates) < normalize_day(day))
    folds[train] = (np.asarray(hashes, dtype=np.uint64)[train] % np.uint64(FOLDS)).astype(np.int8)
    return folds


def build_oof_statistics(frame, train_labels, day, predictor_hashes, output=None, specs=SPECS, progress=None):
    """Labels argument contains ONLY outer-training labels, in original row order.

    Aggregate full-training counts and subtract the held fold. All lookup is by
    integer codes, never an index-bearing reindex/merge that could reorder rows.
    Each parent's probability was queried under the SAME row's exclusion fold.
    """
    day = normalize_day(day)
    dates = frame["pt_d"].to_numpy(dtype=np.int64)
    train = np.flatnonzero(dates < day)
    valid = np.flatnonzero(dates == day)
    if len(predictor_hashes) != len(frame) or not len(train) or not len(valid):
        raise ValueError("Aligned hashes and nonempty whole-date train/validation are required")
    labels = np.asarray(train_labels)
    if labels.shape != (len(train),) or not np.isin(labels, (0, 1)).all():
        raise ValueError("Supply binary labels for outer-training rows ONLY")
    folds = fold_assignments(predictor_hashes, dates, day)
    fold_rows = [train[folds[train] == fold] for fold in range(FOLDS)]
    if any(len(rows) == len(train) for rows in fold_rows):
        raise ValueError("Hash folds leave no fit rows; use more distinct predictor groups")
    y = np.zeros(len(frame), dtype=np.int8)
    y[train] = labels
    total_clicks = float(labels.sum())
    priors = [float((total_clicks - y[rows].sum()) / (len(train) - len(rows))) for rows in fold_rows]
    priors.append(total_clicks / len(train))
    query_rows = fold_rows + [valid]
    names = feature_names(specs)
    shape = (len(frame), len(names))
    if output is None:
        result = np.zeros(shape, dtype=np.float32)
    else:
        result = output
        if result.shape != shape or result.dtype != np.float32 or not result.flags.writeable:
            raise ValueError("Expected a writable float32 output of shape " + str(shape))
        for column in range(shape[1]):
            result[:, column] = 0
    for rows, prior in zip(query_rows, priors):
        result[rows, -1] = prior
    parent_columns = {}
    for index, spec in enumerate(specs):
        if spec.alpha <= 0 or any(parent not in parent_columns for parent in spec.parents):
            raise ValueError("Positive smoothing and topologically ordered parent definitions are required")
        codes, size = _factorize_fields(frame, spec.fields)
        counts = np.bincount(codes[train], minlength=size)
        clicks = np.bincount(codes[train], weights=labels, minlength=size)
        for fold, rows in enumerate(query_rows):
            if fold < FOLDS:
                held_count = np.bincount(codes[rows], minlength=size)
                held_clicks = np.bincount(codes[rows], weights=y[rows], minlength=size)
                fit_count, fit_clicks = counts - held_count, clicks - held_clicks
            else:
                fit_count, fit_clicks = counts, clicks
            for start in range(0, len(rows), 65536):
                take = rows[start:start + 65536]
                keys = codes[take]
                count = fit_count[keys]
                if spec.parents:
                    prior = np.zeros(len(take), dtype=np.float64)
                    for parent in spec.parents:
                        prior += result[take, parent_columns[parent]]
                    prior /= len(spec.parents)
                else:
                    prior = priors[fold]
                rate = (fit_clicks[keys] + spec.alpha * prior) / (count + spec.alpha)
                result[take, index * 2] = np.log1p(count)
                result[take, index * 2 + 1] = rate
        parent_columns[spec.fields] = index * 2 + 1
        if isinstance(result, np.memmap):
            result.flush()
        if progress:
            progress(index + 1, len(specs), spec.fields, size)
    for start in range(0, len(frame), 16384):
        if not np.isfinite(result[start:start + 16384]).all():
            raise AssertionError("Nonfinite OOF statistics")
    return result, names, folds


def memory_report():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return {"rss_mib": round(info.rss / 2**20, 1),
                "peak_working_set_mib": round(getattr(info, "peak_wset", info.rss) / 2**20, 1),
                "available_system_mib": round(psutil.virtual_memory().available / 2**20, 1)}
    except ImportError:
        return {"measurement": "psutil unavailable"}


def main(args):
    started = time.perf_counter()
    day = normalize_day(args.day)
    output = (args.output or HERE / f"oof_day{day % 100}.npy").resolve()
    if output.suffix != ".npy":
        raise ValueError("--output must end in .npy")
    meta_path, folds_path = output.with_suffix(".json"), output.with_suffix(".folds.npy")
    if any(path.exists() for path in (output, meta_path, folds_path)):
        raise FileExistsError("Preserving existing OOF cache, metadata or folds")
    cache = base_cache(day)
    cached_dates = np.load(cache / "date.npy", mmap_mode="r")
    n = len(cached_dates)
    raw_path = ROOT / "2026ml/train.csv"
    predictor_fields = [field for field in pd.read_csv(raw_path, nrows=0).columns if field not in ("log_id", "label")]
    if len(predictor_fields) != 33:
        raise ValueError("Expected exactly 33 original predictors")
    # Fixed str dtype avoids chunk-dependent int/float inference splitting exact
    # groups; labels are excluded at the CSV parser, not merely dropped later.
    hashes = np.empty(n, dtype=np.uint64)
    columns = {field: np.empty(n, dtype=np.float64 if field == "app_score" else np.int32)
               for field in FIELDS + ["pt_d"]}
    offset = 0
    for chunk in pd.read_csv(raw_path, usecols=predictor_fields, dtype=str, keep_default_na=False, chunksize=50000):
        end = offset + len(chunk)
        hashes[offset:end] = raw_predictor_hash(chunk)
        for field, values in columns.items():
            values[offset:end] = pd.to_numeric(chunk[field], errors="raise").to_numpy()
        offset = end
    if offset != n or not np.array_equal(columns["pt_d"], cached_dates):
        raise ValueError("Raw CSV rows/dates do not align with the existing cache")
    frame = pd.DataFrame(columns, copy=False)
    train, valid = np.flatnonzero(cached_dates < day), np.flatnonzero(cached_dates == day)
    for name, expected in (("train_indices", train), ("valid_indices", valid)):
        if not np.array_equal(np.load(cache / (name + ".npy"), mmap_mode="r"), expected):
            raise ValueError("Cache split violates the whole-date cutoff")
    # The label mapping is opened, but ONLY train rows are read/materialized.
    train_labels = np.load(cache / "label.npy", mmap_mode="r")[train]
    output.parent.mkdir(parents=True, exist_ok=True)
    result = np.lib.format.open_memmap(output, mode="w+", dtype="float32",
                                      shape=(n, len(feature_names())), fortran_order=True)
    print(json.dumps({"stage": "loaded", "shape": list(result.shape), "groups": len(SPECS),
                      "seconds": round(time.perf_counter() - started, 2), **memory_report()}), flush=True)

    def progress(index, total, fields, size):
        print(json.dumps({"stage": "group", "index": index, "total": total, "fields": fields,
                          "keys": size, "seconds": round(time.perf_counter() - started, 2)}), flush=True)

    result, names, folds = build_oof_statistics(frame, train_labels, day, hashes, result, progress=progress)
    result.flush()
    np.save(folds_path, folds)
    order = np.argsort(hashes[train])
    sorted_hash, sorted_fold = hashes[train][order], folds[train][order]
    same = sorted_hash[1:] == sorted_hash[:-1]
    if not np.all(sorted_fold[1:][same] == sorted_fold[:-1][same]):
        raise AssertionError("A complete raw predictor duplicate group crosses folds")
    meta = {
        "status": "complete", "protocol": PROTOCOL, "shape": list(result.shape), "dtype": "float32",
        "storage_order": "F", "features": names, "validation_day": day,
        "outer_split": "train: pt_d < cutoff; validation: pt_d == cutoff; later rows: zero and unused",
        "train_rows": len(train), "valid_rows": len(valid), "fold_count": FOLDS,
        "fold_rule": "uint64 pandas raw-string predictor hash % 5; label/log_id excluded; pt_d included; original list order retained",
        "raw_predictor_fields": predictor_fields, "raw_predictor_field_count": len(predictor_fields),
        "old_hash_reused": False, "old_hash_note": "Old hash serialization not documented; old 18-group OOF used GroupKFold, not hash%5",
        "fold_rows": np.bincount(folds[train], minlength=FOLDS).tolist(),
        "exact_hash_duplicate_training_rows": int(same.sum()), "duplicate_group_fold_check": "passed",
        "fold_file": str(folds_path), "fold_sha256": sha256(folds_path),
        "advertisement_labels_loaded": True, "advertisement_validation_labels_loaded": False,
        "advertisement_validation_labels_used": False, "advertisement_future_labels_loaded": False,
        "train_oof_allowed_across_train_dates": True, "per_row_strictly_historical": False,
        "rule": "Offline supervised learning: each train row uses all outer-train dates excluding its complete hash fold; validation uses full outer train only. Across-train-date OOF is not future-validation leakage.",
        "shrinkage": "(other_fold_clicks + alpha * parent_probability) / (other_fold_count + alpha); parent probabilities share row fold; multiple parents averaged; root prior is other-fold label mean",
        "validation_shrinkage": "Full outer-train count/clicks, parents and label mean; validation labels unused",
        "future_rows": "pt_d > cutoff are exactly zero", "scaling": "Raw support/log-count and probabilities; downstream scaler fits outer train only",
        "groups": [{"fields": list(spec.fields), "parents": [list(parent) for parent in spec.parents], "alpha": spec.alpha} for spec in SPECS],
        "date_sha256": array_digest(np.asarray(cached_dates)), "train_indices_sha256": array_digest(train),
        "train_label_sha256": array_digest(train_labels), "base_cache": str(cache.resolve()),
        "source_sha256": sha256(Path(__file__)), "factorizer_source_sha256": sha256(HERE / "prepare_multiscale_stats.py"),
        "pandas_version": pd.__version__, "numpy_version": np.__version__, "raw_csv_bytes": raw_path.stat().st_size,
        "elapsed_seconds": round(time.perf_counter() - started, 2), "cache_mib": round(output.stat().st_size / 2**20, 2),
        **memory_report(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OOF_COMPLETE " + json.dumps({key: meta[key] for key in ("validation_day", "shape", "elapsed_seconds", "cache_mib", "fold_rows", "rss_mib", "peak_working_set_mib")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", type=normalize_day, choices=(20191115, 20191116), required=True)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args())
