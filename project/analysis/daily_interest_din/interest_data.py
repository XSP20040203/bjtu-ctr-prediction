"""Day-prefix ad statistics and unordered role-topic source snapshots."""
import numpy as np
import pandas as pd

STRIDE = 100000000
CONTEXT_NAMES = ["log_exposure_mass", "log_click_mass", "log_progress_mass", "log_dislike_mass",
                 "log_source_rows", "source_missing", "dislike_missing"]


def request_keys(users, dates):
    return np.asarray(users, dtype=np.int64)*STRIDE + np.asarray(dates, dtype=np.int64)


def build_past_statistics(frame, validation_day, specs, initial_prior=.0084, prior_strength=1000., history_days=None):
    if not 0 < initial_prior < 1 or prior_strength <= 0:
        raise ValueError("A finite interior prior and positive prior strength are required")
    frame = frame.reset_index(drop=True)
    dates = frame.pt_d.to_numpy()
    result = np.zeros((len(frame), 2*len(specs)), dtype=np.float32)
    ordinal = pd.to_datetime(frame.pt_d.astype(str), format="%Y%m%d").to_numpy(dtype="datetime64[D]").astype(np.int64)
    for day in sorted(np.unique(dates[dates <= validation_day])):
        positions = np.flatnonzero(dates == day)
        historical = dates < day
        if history_days is not None:
            historical &= ordinal >= ordinal[positions[0]]-history_days
        history = frame.loc[historical]
        if not history.label.isin([0, 1]).all():
            raise ValueError("Historical advertisement labels must be 0/1")
        prior = float((history.label.sum()+prior_strength*initial_prior)/(len(history)+prior_strength))
        if not len(history):continue
        query = frame.iloc[positions]
        for index, (fields, alpha) in enumerate(specs):
            fields = list(fields)
            table = history.groupby(fields, sort=False, observed=True).label.agg(["size", "sum"])
            keys = pd.Index(query[fields[0]]) if len(fields) == 1 else pd.MultiIndex.from_frame(query[fields])
            aligned = table.reindex(keys).fillna(0)
            count, clicks = aligned["size"].to_numpy(), aligned["sum"].to_numpy()
            probability = np.clip((clicks+alpha*prior)/(count+alpha), 1e-6, 1-1e-6)
            result[positions, 2*index] = np.log1p(count)
            result[positions, 2*index+1] = np.log(probability/(1-probability))-np.log(prior/(1-prior))
    if not np.isfinite(result).all():raise AssertionError("Nonfinite historical statistics")
    return result


def build_topic_store(feeds, requests, vocabulary, history_days=3):
    """Use complete source day feedback as supplied input, with no intraday ordering."""
    if history_days < 1:raise ValueError("Positive history window required")
    if not feeds.label.isin([-1, 1]).all():raise ValueError("Expected news labels -1/1")
    width = max(vocabulary.values(), default=1)+1
    source_keys, source_codes = np.unique(request_keys(feeds.u_userId, feeds.e_et), return_inverse=True)
    cube = np.zeros((len(source_keys), 4, width), dtype=np.float32)
    topics = np.fromiter((vocabulary.get(str(int(v)), 1) for v in feeds.i_cat), dtype=np.int64, count=len(feeds))
    click = feeds.label.eq(1).to_numpy(dtype=np.float32)
    progress = np.clip(feeds.pro.to_numpy(dtype=np.float32), 0, 100)/100*click
    for channel, values in enumerate([np.ones(len(feeds), dtype=np.float32), click, progress]):
        np.add.at(cube[:,channel], (source_codes, topics), values)
    if "u_newsCatDislike" in feeds:
        negative = pd.DataFrame({"row": source_codes, "tokens": feeds.u_newsCatDislike.fillna("").astype(str).str.split("^")})
        negative = negative.explode("tokens")
        negative = negative.loc[~negative.tokens.isin(["", "0", "nan", "None", "-1"])]
        codes = negative.tokens.map(vocabulary).fillna(1).to_numpy(dtype=np.int64)
        np.add.at(cube[:,3], (negative.row.to_numpy(dtype=np.int64), codes), 1.)
    source_rows = np.bincount(source_codes, minlength=len(source_keys)).astype(np.float32)
    wanted = np.unique(request_keys(requests.user_id, requests.pt_d))
    wanted = np.r_[np.int64(0), wanted]
    users, dates = wanted[1:]//STRIDE, wanted[1:]%STRIDE
    weights = np.zeros((len(wanted), 2, 4, width), dtype=np.float32)
    totals = np.zeros((len(wanted), 2), dtype=np.float32)
    ordinals = pd.to_datetime(dates.astype(str), format="%Y%m%d").to_numpy(dtype="datetime64[D]")
    for gap in range(history_days+1):
        shifted = pd.to_datetime(ordinals-np.timedelta64(gap,"D")).strftime("%Y%m%d").to_numpy(dtype=np.int64)
        query = request_keys(users, shifted)
        positions = np.searchsorted(source_keys, query)
        safe = np.minimum(positions, max(len(source_keys)-1, 0))
        if not len(source_keys):continue
        found = (positions < len(source_keys)) & (source_keys[safe] == query)
        rows = np.flatnonzero(found)+1
        axis = 1 if gap == 0 else 0
        weights[rows,axis] += cube[safe[found]]
        totals[rows,axis] += source_rows[safe[found]]
    mass = weights.sum(axis=-1)
    weights /= np.maximum(mass[...,None], 1e-12)
    context = np.concatenate([np.log1p(mass), np.log1p(totals)[...,None],
                              (totals == 0)[...,None], (mass[:,:,3] == 0)[...,None]], axis=-1).astype(np.float32)
    context[0] = 0
    return {"keys": wanted, "topic_weights": weights, "topic_mass": mass, "interest_context": context,
            "context_names": CONTEXT_NAMES, "n_topics": width, "context_dim": len(CONTEXT_NAMES)}
