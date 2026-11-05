"""Offline source-day covariates; advertisement labels are never accepted here."""
from pathlib import Path

import numpy as np
import pandas as pd

DAILY_CATEGORIES = ["feed_phone_price", "feed_browser_lifecycle", "feed_browser_mode"]
DAILY_SEQUENCES = ["feed_long_interest", "feed_dislike_topics", "feed_clicked_topics", "feed_click_ca2", "feed_exposed_topics"]
DAILY_NUMERIC = ["feed_day_impressions", "feed_day_clicks", "feed_day_ctr_smoothed", "feed_day_progress_mean",
                 "feed_day_progress_max", "feed_day_finish_count", "feed_day_topic_nunique", "feed_day_source_nunique",
                 "feed_day_refresh_mean", "feed_day_position_mean", "feed_day_click_progress_mean",
                 "feed_day_clicked_topic_nunique", "feed_day_clicked_source_nunique", "feed_day_like_count",
                 "feed_day_long_interest_nunique", "feed_st_click_overlap", "feed_st_exposure_overlap",
                 "feed_st_long_overlap", "feed_day_profile_missing"]
UNIT100 = {"feed_day_progress_mean", "feed_day_progress_max", "feed_day_click_progress_mean"}
LOG_STANDARD = set(DAILY_NUMERIC[:15]) - UNIT100 - {"feed_day_ctr_smoothed"}
SOURCE_COLUMNS = ["u_userId", "e_et", "label", "pro", "cillabel", "i_cat", "i_s_sourceId", "e_rn", "e_po",
                  "u_phonePrice", "u_browserLifeCycle", "u_browserMode", "u_newsCatInterests", "u_newsCatDislike", "u_click_ca2_news"]


def token_counts(frame, column):
    pieces = frame[["_key", column]].copy()
    pieces["_token"] = pieces[column].fillna("").astype(str).str.split("^")
    pieces = pieces[["_key", "_token"]].explode("_token")
    pieces = pieces.loc[pieces["_token"].ne("") & pieces["_token"].notna()]
    return pieces.groupby(["_key", "_token"]).size().rename("count").reset_index()


def top_tokens(counts, limit=5):
    ordered = counts.sort_values(["_key", "count", "_token"], ascending=[True, False, True], kind="stable")
    return ordered.groupby("_key", sort=False).head(limit).groupby("_key")["_token"].agg("^".join)


def build_profiles(feeds):
    frame = feeds.copy()
    frame["_key"] = frame["e_et"].astype(np.int64) * 10000000 + frame["u_userId"].astype(np.int64)
    frame["_clicked"] = frame["label"].eq(1).astype("int8")
    frame["_finished"] = frame["pro"].ge(80).astype("int8")
    frame["_liked"] = frame["cillabel"].eq(1).astype("int8")
    profile = frame.groupby("_key").agg(
        feed_day_impressions=("label", "size"), feed_day_clicks=("_clicked", "sum"),
        feed_day_progress_mean=("pro", "mean"), feed_day_progress_max=("pro", "max"),
        feed_day_finish_count=("_finished", "sum"), feed_day_topic_nunique=("i_cat", "nunique"),
        feed_day_source_nunique=("i_s_sourceId", "nunique"), feed_day_refresh_mean=("e_rn", "mean"),
        feed_day_position_mean=("e_po", "mean"), feed_day_like_count=("_liked", "sum"))
    prior = frame.groupby("e_et")["_clicked"].mean()
    days = (profile.index.to_numpy() // 10000000).astype(np.int64)
    profile["feed_day_ctr_smoothed"] = (profile.feed_day_clicks.to_numpy() + 20 * prior.reindex(days).to_numpy()) / (profile.feed_day_impressions.to_numpy() + 20)
    clicked = frame.loc[frame["_clicked"].eq(1)]
    positive = clicked.groupby("_key").agg(feed_day_click_progress_mean=("pro", "mean"),
                                           feed_day_clicked_topic_nunique=("i_cat", "nunique"),
                                           feed_day_clicked_source_nunique=("i_s_sourceId", "nunique"))
    profile = profile.join(positive).fillna(0)
    for source, target in [("u_phonePrice", DAILY_CATEGORIES[0]), ("u_browserLifeCycle", DAILY_CATEGORIES[1]), ("u_browserMode", DAILY_CATEGORIES[2])]:
        counts = token_counts(frame, source)
        profile[target] = top_tokens(counts, 1).reindex(profile.index).fillna("__MISSING__")
    for source, target, subset in [("u_newsCatInterests", "feed_long_interest", frame),
                                    ("u_newsCatDislike", "feed_dislike_topics", frame),
                                    ("i_cat", "feed_clicked_topics", clicked),
                                    ("u_click_ca2_news", "feed_click_ca2", frame),
                                    ("i_cat", "feed_exposed_topics", frame)]:
        counts = token_counts(subset, source)
        profile[target] = top_tokens(counts).reindex(profile.index).fillna("")
        if target in ["feed_long_interest", "feed_clicked_topics", "feed_exposed_topics"]:
            all_tokens = counts.groupby("_key")["_token"].agg("^".join)
            profile["_all_" + target] = all_tokens.reindex(profile.index).fillna("")
        if target == "feed_long_interest":
            profile["feed_day_long_interest_nunique"] = counts.groupby("_key").size().reindex(profile.index).fillna(0)
    return profile


def interest_overlap(short, clicked, exposed, longterm):
    tokens = lambda value: {token for token in str(value).split("^") if token and token not in ["nan", "None"]}
    current = tokens(short)
    if not current:
        return 0., 0., 0.
    return tuple(len(current & tokens(other)) / len(current) for other in [clicked, exposed, longterm])


def load_profiles(directory):
    path = directory / "daily_source_profiles.pkl"
    if path.exists():
        return pd.read_pickle(path)
    data = directory.parents[1] / "2026ml"
    frame = pd.concat([pd.read_csv(data / name, usecols=SOURCE_COLUMNS) for name in ["train_feeds.csv", "test_feeds.csv"]], ignore_index=True)
    profile = build_profiles(frame)
    profile.to_pickle(path)
    return profile
