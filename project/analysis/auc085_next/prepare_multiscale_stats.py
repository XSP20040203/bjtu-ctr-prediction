"""Strictly past advertisement response features at 1-day, 3-day and full-history scales.

Run from any directory, without loading final test data or training a model::

    python analysis/auc085_next/prepare_multiscale_stats.py --day 15 --output stats_day15.npy

Both 15/16 and 20191115/20191116 are accepted by the CLI. Output rows keep the
original train.csv order. Dates after validation_day are deliberately all zero.
The 159 float32 columns occupy about 1.61 GB for 2.53 million rows; the CLI writes
a .npy memory map and processes one grouping at a time rather than constructing
multiple dense feature matrices. Only labels at dates < validation_day are read
into aggregation state. Within those dates, a day's labels are inserted only
after all of that day's features have been queried.
"""
from collections import deque
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WINDOWS = ("1d", "3d", "all")
METRICS = ("log_count", "log_clicks", "relative_logit", "is_new")
INITIAL_PRIOR = 0.0084
GLOBAL_STRENGTH = 1000.0


@dataclass(frozen=True)
class GroupSpec:
    fields: tuple
    parents: tuple
    alpha: float


# These are contextual parent priors, not assumed functional ID mappings. Each
# row looks up its own parent IDs using only outcomes before that row's date.
SPECS = (
    GroupSpec(("app_second_class",), (), 500.0),
    GroupSpec(("spread_app_id",), (("app_second_class",),), 300.0),
    GroupSpec(("adv_prim_id",), (("spread_app_id",),), 300.0),
    GroupSpec(("adv_id",), (("adv_prim_id",),), 100.0),
    GroupSpec(("task_id",), (("adv_id",),), 80.0),
    GroupSpec(("app_second_class", "slot_id"), (("app_second_class",), ("slot_id",)), 100.0),
    GroupSpec(("spread_app_id", "slot_id"), (("spread_app_id",), ("app_second_class", "slot_id")), 100.0),
    GroupSpec(("adv_prim_id", "slot_id"), (("adv_prim_id",), ("spread_app_id", "slot_id")), 100.0),
    GroupSpec(("adv_id", "slot_id"), (("adv_id",), ("adv_prim_id", "slot_id")), 60.0),
    GroupSpec(("task_id", "slot_id"), (("task_id",), ("adv_id", "slot_id")), 40.0),
    GroupSpec(("user_id", "slot_id"), (("user_id",), ("slot_id",)), 100.0),
    GroupSpec(("user_id", "spread_app_id"), (("user_id",), ("spread_app_id",)), 60.0),
)
SUPPORT_PARENTS = ((("user_id",), 100.0), (("slot_id",), 500.0))
FIELDS = sorted({field for spec in SPECS for field in spec.fields})


def normalize_day(value):
    value = int(value)
    return 20191100 + value if value in (15, 16) else value


def feature_names():
    names = []
    for spec in SPECS:
        prefix = "past_multiscale__" + "__".join(spec.fields)
        names.extend(prefix + "__" + window + "__" + metric
                     for window in WINDOWS for metric in METRICS)
        names.append(prefix + "__1d_minus_all_logit")
    names.extend("past_multiscale__global__" + window + "__logit" for window in WINDOWS)
    return names


def _logit(probability):
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(probability) - np.log1p(-probability)


def _factorize_fields(frame, fields):
    """Collision-free integer pair keys, avoiding millions of Python tuples."""
    packed = None
    cardinality = 1
    for field in fields:
        codes, unique = pd.factorize(frame[field], sort=False, use_na_sentinel=False)
        width = max(len(unique), 1)
        if cardinality > np.iinfo(np.int64).max // width:
            raise ValueError("Composite key exceeds int64 capacity")
        packed = codes if packed is None else packed * width + codes
        cardinality *= width
    if len(fields) > 1:
        packed, unique = pd.factorize(packed, sort=False)
        size = len(unique)
    else:
        size = len(unique)
    if size > np.iinfo(np.int32).max:
        raise ValueError("Too many distinct groups")
    return packed.astype(np.int32, copy=False), size


def _prepare_days(frame, validation_day):
    dates = frame["pt_d"].to_numpy(dtype=np.int64)
    distinct = np.unique(dates)
    ordinals = pd.to_datetime(distinct.astype(str), format="%Y%m%d", errors="raise").to_numpy(
        dtype="datetime64[D]").astype(np.int64)
    groups = [(int(ordinal), np.flatnonzero(dates == day))
              for day, ordinal in zip(distinct, ordinals) if day <= validation_day]
    labels = {}
    for day, ordinal in zip(distinct, ordinals):
        if day >= validation_day:
            continue
        positions = np.flatnonzero(dates == day)
        # Deliberately do not inspect or validate labels at/after validation_day.
        values = frame["label"].iloc[positions].to_numpy()
        if not np.isin(values, [0, 1]).all():
            raise ValueError("Historical advertisement labels must be 0/1")
        labels[int(ordinal)] = values.astype(np.int8)
    return groups, labels


def _daily_states(codes, size, groups, labels, multiscale=True):
    """Yield prefix state before inserting current labels; gaps use calendar days."""
    total_count = np.zeros(size, dtype=np.float32)
    total_clicks = np.zeros(size, dtype=np.float32)
    if multiscale:
        recent_count = np.zeros(size, dtype=np.float32)
        recent_clicks = np.zeros(size, dtype=np.float32)
        empty = np.zeros(size, dtype=np.float32)
        recent = deque()
    for ordinal, rows in groups:
        if multiscale:
            while recent and recent[0][0] < ordinal - 3:
                _, old_count, old_clicks = recent.popleft()
                recent_count -= old_count
                recent_clicks -= old_clicks
            last = recent[-1][1:] if recent and recent[-1][0] == ordinal - 1 else (empty, empty)
            states = (last, (recent_count, recent_clicks), (total_count, total_clicks))
        else:
            states = ((total_count, total_clicks),)
        yield rows, states
        # Validation-day labels have no entry, and no later day is queried.
        if ordinal not in labels:
            continue
        daily_codes = codes[rows]
        day_count = np.bincount(daily_codes, minlength=size).astype(np.float32)
        day_clicks = np.bincount(daily_codes, weights=labels[ordinal], minlength=size).astype(np.float32)
        total_count += day_count
        total_clicks += day_clicks
        if multiscale:
            recent_count += day_count
            recent_clicks += day_clicks
            recent.append((ordinal, day_count, day_clicks))


def _global_rates(size, groups, labels, initial_prior, prior_strength, result):
    cumulative = np.zeros(size, dtype=np.float32)
    global_codes = np.zeros(size, dtype=np.int32)
    offset = len(SPECS) * (len(WINDOWS) * len(METRICS) + 1)
    for rows, states in _daily_states(global_codes, 1, groups, labels):
        all_count, all_clicks = states[-1]
        parent = float((all_clicks[0] + prior_strength * initial_prior) /
                       (all_count[0] + prior_strength))
        cumulative[rows] = parent
        for window, (count, clicks) in enumerate(states):
            rate = parent if window == 2 else float((clicks[0] + prior_strength * parent) /
                                                   (count[0] + prior_strength))
            result[rows, offset + window] = _logit(rate)
    return cumulative


def build_multiscale_statistics(frame, validation_day, output=None, initial_prior=INITIAL_PRIOR,
                               prior_strength=GLOBAL_STRENGTH, progress=None):
    """Return (float32 matrix, names); output optionally supplies a writable matrix.

    Keys may include future/current input IDs, but their labels never enter any
    earlier feature. New keys back off exactly to their cumulative parent rate.
    For two parents, the prior is the arithmetic mean of their cumulative
    shrunken probabilities. All three windows share that cumulative parent prior.
    Relative logits use the strictly past cumulative global rate as their origin.
    """
    if not np.isfinite(initial_prior) or not 0 < initial_prior < 1:
        raise ValueError("An interior finite initial prior is required")
    if not np.isfinite(prior_strength) or prior_strength <= 0:
        raise ValueError("A positive finite global prior strength is required")
    missing = set(FIELDS + ["pt_d", "label"]) - set(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    validation_day = normalize_day(validation_day)
    pd.to_datetime(str(validation_day), format="%Y%m%d", errors="raise")
    names = feature_names()
    shape = (len(frame), len(names))
    if output is None:
        result = np.zeros(shape, dtype=np.float32)
    else:
        if output.shape != shape or output.dtype != np.float32 or not output.flags.writeable:
            raise ValueError("Output must be a writable float32 matrix of shape " + str(shape))
        result = output
        if not isinstance(result, np.memmap):
            result.fill(0)
        else:
            # Honor supplied existing maps; contiguous writes avoid repeatedly
            # paging through the whole file when using column-major storage.
            if result.flags.f_contiguous:
                for start in range(0, len(names), 8):
                    result[:, start:start + 8] = 0
                    result.flush()
            else:
                for start in range(0, len(frame), 8192):
                    result[start:start + 8192] = 0
                result.flush()
    groups, labels = _prepare_days(frame, validation_day)
    global_rate = _global_rates(len(frame), groups, labels, initial_prior, prior_strength, result)
    parents = {}
    for fields, alpha in SUPPORT_PARENTS:
        codes, size = _factorize_fields(frame, fields)
        rates = np.zeros(len(frame), dtype=np.float32)
        for rows, states in _daily_states(codes, size, groups, labels, multiscale=False):
            count, clicks = states[0]
            query = codes[rows]
            rates[rows] = (clicks[query] + alpha * global_rate[rows]) / (count[query] + alpha)
        parents[fields] = rates
    for index, spec in enumerate(SPECS):
        codes, size = _factorize_fields(frame, spec.fields)
        cumulative_rate = np.zeros(len(frame), dtype=np.float32)
        offset = index * (len(WINDOWS) * len(METRICS) + 1)
        for rows, states in _daily_states(codes, size, groups, labels):
            query = codes[rows]
            prior = global_rate[rows].copy() if not spec.parents else parents[spec.parents[0]][rows].copy()
            for parent in spec.parents[1:]:
                prior += parents[parent][rows]
            if len(spec.parents) > 1:
                prior /= len(spec.parents)
            origin = _logit(global_rate[rows])
            first_logit = None
            for window, (counts, clicks) in enumerate(states):
                count = counts[query]
                positive = clicks[query]
                probability = (positive + spec.alpha * prior) / (count + spec.alpha)
                relative = _logit(probability) - origin
                column = offset + window * len(METRICS)
                result[rows, column] = np.log1p(count)
                result[rows, column + 1] = np.log1p(positive)
                result[rows, column + 2] = relative
                result[rows, column + 3] = count == 0
                if window == 0:
                    first_logit = relative
                if window == 2:
                    cumulative_rate[rows] = probability
                    result[rows, offset + 12] = first_logit - relative
        parents[spec.fields] = cumulative_rate
        # Flush each completed group so mapped dirty pages remain reclaimable.
        if isinstance(result, np.memmap):
            result.flush()
        if progress:
            progress(index + 1, len(SPECS), spec.fields, size)
    for start in range(0, len(frame), 8192):
        if not np.isfinite(result[start:start + 8192]).all():
            raise AssertionError("Nonfinite multiscale statistics")
    return result, names


def main(args):
    started = time.perf_counter()
    output = args.output.resolve()
    if output.suffix.lower() != ".npy":
        raise ValueError("--output must have a .npy suffix")
    metadata = output.with_suffix(".json")
    if output.exists() or metadata.exists():
        raise FileExistsError("Preserving existing multiscale feature output or metadata")
    frame = pd.read_csv(ROOT / "2026ml" / "train.csv", usecols=FIELDS + ["pt_d", "label"], dtype="int32")
    output.parent.mkdir(parents=True, exist_ok=True)
    shape = (len(frame), len(feature_names()))
    result = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=shape, fortran_order=True)
    print(json.dumps({"stage": "loaded", "shape": list(shape), "validation_day": args.day,
                      "dataframe_mib": round(frame.memory_usage(deep=True).sum() / 2**20, 1)}), flush=True)

    def progress(index, total, fields, size):
        print(json.dumps({"stage": "group", "group": index, "total": total, "fields": fields,
                          "distinct_keys": size, "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)

    result, names = build_multiscale_statistics(frame, args.day, output=result, progress=progress)
    result.flush()
    meta = {
        "shape": list(result.shape),
        "dtype": "float32",
        "storage_order": "F (column major .npy; np.load preserves ordinary row/column semantics)",
        "features": names,
        "advertisement_labels_loaded": True,
        "advertisement_validation_labels_used": False,
        "validation_day": args.day,
        "future_rows": "All features for pt_d > validation_day are zero; original row order is preserved",
        "rule": "Every row uses only labels at strictly earlier calendar dates; validation/current/future labels never update its state",
        "windows": {"1d": "d-1 <= history_date < d", "3d": "d-3 <= history_date < d", "all": "history_date < d"},
        "shrinkage": "(clicks + alpha * parent_rate) / (count + alpha); parent rates use cumulative strictly past labels; multiple parents are averaged",
        "relative_logit_origin": "Cumulative global past rate, smoothed toward initial_prior",
        "new_entity_rule": "is_new = (window historical exposure count == 0)",
        "initial_prior": INITIAL_PRIOR,
        "global_prior_strength": GLOBAL_STRENGTH,
        "groups": [{"fields": list(spec.fields), "parents": [list(parent) for parent in spec.parents],
                    "alpha": spec.alpha} for spec in SPECS],
        "support_only_parents": [{"fields": list(fields), "alpha": alpha} for fields, alpha in SUPPORT_PARENTS],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    metadata.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in meta.items() if key not in ("features", "groups", "support_only_parents")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", type=normalize_day, choices=(20191115, 20191116), required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
