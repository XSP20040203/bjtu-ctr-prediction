"""Outcome-free user-day summaries and separately constructed training targets."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from scipy import sparse

HISTOGRAM_FIELDS = ["spread_app_id", "app_second_class", "slot_id", "creat_type_cd",
                    "inter_type_cd", "u_refreshTimes", "u_feedLifeCycle"]
HISTORY_FIELDS = ["past3_pair_log_count", "past3_pair_seen", "past3_task_seen",
                  "past3_user_log_daily_exposures", "past3_user_seen"]


def hash_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def past_exposure_features(raw):
    """Only inputs in earlier three calendar days, queried before inserting today."""
    if set(raw.columns) != {"pt_d", "user_id", "task_id"}:
        raise ValueError("Past exposure features accept declared predictors only")
    dates = raw.pt_d.to_numpy()
    ordinals = pd.to_datetime(raw.pt_d.astype(str), format="%Y%m%d").to_numpy(
        dtype="datetime64[D]").astype("int64")
    _, pairs = np.unique(raw[["user_id", "task_id"]].to_numpy(), axis=0, return_inverse=True)
    _, users = np.unique(raw.user_id.to_numpy(), return_inverse=True)
    _, tasks = np.unique(raw.task_id.to_numpy(), return_inverse=True)
    pair_count = np.zeros(int(pairs.max()) + 1, dtype="int32")
    user_count = np.zeros(int(users.max()) + 1, dtype="int32")
    task_count = np.zeros(int(tasks.max()) + 1, dtype="int32")
    result = np.empty((len(raw), len(HISTORY_FIELDS)), dtype="float32")
    history = []
    for day in np.unique(dates):
        rows = np.flatnonzero(dates == day)
        ordinal = int(ordinals[rows[0]])
        while history and history[0][0] < ordinal - 3:
            _, expired = history.pop(0)
            for counts, codes in [(pair_count, pairs), (user_count, users), (task_count, tasks)]:
                np.add.at(counts, codes[expired], -1)
        coverage = max(len(history), 1)
        previous_pairs, previous_users = pair_count[pairs[rows]], user_count[users[rows]]
        result[rows] = np.column_stack([
            np.log1p(previous_pairs), previous_pairs > 0, task_count[tasks[rows]] > 0,
            np.log1p(previous_users / coverage), previous_users > 0,
        ])
        for counts, codes in [(pair_count, pairs), (user_count, users), (task_count, tasks)]:
            np.add.at(counts, codes[rows], 1)
        history.append((ordinal, rows))
    return result


def build_context(raw, categories, numeric, numeric_fields, meta, held_day):
    if set(raw.columns) != {"pt_d", "user_id", "task_id"}:
        raise ValueError("User-day context accepts no advertisement outcome")
    if len(raw) != len(categories) or len(raw) != len(numeric) or numeric.shape[1] != len(numeric_fields):
        raise ValueError("Context inputs are not aligned")
    excluded = [i for i, name in enumerate(numeric_fields) if name.startswith("target_stat__")]
    allowed = [i for i in range(len(numeric_fields)) if i not in excluded]
    if not allowed or not excluded:
        raise ValueError("Explicit label-free and target-stat column contracts required")
    keys, groups = np.unique(raw[["pt_d", "user_id"]].to_numpy(dtype="int64"),
                              axis=0, return_inverse=True)
    size = np.bincount(groups, minlength=len(keys)).astype("int32")
    membership = sparse.csr_matrix((1. / size[groups], (groups, np.arange(len(groups)))),
                                    shape=(len(keys), len(raw)))
    mean = np.empty((len(keys), len(allowed)), dtype="float32")
    for start in range(0, len(allowed), 16):
        stop = min(start + 16, len(allowed))
        mean[:, start:stop] = membership @ numeric[:, allowed[start:stop]]
    maximum = np.full_like(mean, -np.inf)
    for start in range(0, len(raw), 20000):
        stop = min(start + 20000, len(raw))
        np.maximum.at(maximum, groups[start:stop], numeric[start:stop, allowed])
    pieces = [mean, maximum, np.log1p(size).astype("float32")[:, None]]
    names = (["mean__" + numeric_fields[i] for i in allowed] +
             ["max__" + numeric_fields[i] for i in allowed] + ["log_exposures"])
    for field in HISTOGRAM_FIELDS:
        column = meta["categorical_fields"].index(field)
        width = meta["vocabulary_sizes"][meta["group_for_field"][field]]
        histogram = sparse.csr_matrix((1. / size[groups], (groups, categories[:, column])),
                                      shape=(len(keys), width)).toarray().astype("float32")
        pieces.append(histogram)
        names += [f"share__{field}__{code}" for code in range(width)]
    history = past_exposure_features(raw)
    history_mean = (membership @ history).astype("float32")
    history_max = np.full((len(keys), len(HISTORY_FIELDS)), -np.inf, dtype="float32")
    np.maximum.at(history_max, groups, history)
    pieces += [history_mean, history_max]
    names += ["mean__" + name for name in HISTORY_FIELDS] + ["max__" + name for name in HISTORY_FIELDS]
    pieces.append((np.log1p(size) - history_mean[:, 3]).astype("float32")[:, None])
    names.append("current_vs_past3_log_exposures")
    values = np.column_stack(pieces).astype("float32")
    training_groups = np.flatnonzero(keys[:, 0] < held_day)
    center = values[training_groups].mean(0, dtype="float64")
    scale = values[training_groups].std(0, dtype="float64").clip(.05)
    values = np.clip((values - center) / scale, -8., 8.).astype("float32")
    if not np.isfinite(values).all():
        raise AssertionError("Nonfinite user-day context")
    return {"keys": keys, "rows": groups.astype("int32"), "sizes": size, "values": values,
            "manifest": {"features": names, "groups": len(keys), "width": len(names),
                         "excluded_target_stat_columns": [numeric_fields[i] for i in excluded],
                         "included_numeric_columns": [numeric_fields[i] for i in allowed],
                         "scaler_fit_scope": "Unique outer-training user-days only",
                         "scaler_mean": center.tolist(), "scaler_std": scale.tolist(),
                         "query_groups_sha256": hash_array(groups.astype("int32")),
                         "context_values_sha256": hash_array(values),
                         "past_window_calendar_days": 3, "ad_labels_in_context": False}}


def training_targets(keys, groups, labels, train_rows, held_day):
    if not len(train_rows) or not np.isin(labels[train_rows], [0, 1]).all():
        raise ValueError("Nonempty binary outer-training labels required")
    if np.any(keys[groups[train_rows], 0] >= held_day):
        raise ValueError("Auxiliary targets may use outer-training dates only")
    counts = np.bincount(groups[train_rows], weights=labels[train_rows], minlength=len(keys))
    targets = np.full(len(keys), -1, dtype="int8")
    training_groups = np.unique(groups[train_rows])
    targets[training_groups] = (counts[training_groups] > 0).astype("int8")
    return targets, float(targets[training_groups].mean())
