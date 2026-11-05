"""80 fine-grained, same-day auxiliary news snapshot features; no model training.

Only advertisement input keys are read. News label (-1/1) and pro are permitted
auxiliary outcomes. A complete same-day snapshot is an OFFLINE assumption: news
outcomes can occur after an advertisement exposure, so this is not a claim of
online/pre-exposure availability or absence of temporal leakage.

Small synthetic tests (no real data read)::
    E:/Users/env/Machine_Learning/python.exe prepare_source_context.py --test-only

Inspect the input file plan without reading data rows::
    python prepare_source_context.py --plan-only

Default build uses train.csv + train_feeds.csv. --include-final additionally uses
final_test.csv + test_feeds.csv, appending advertisement rows in that order.
CSV chunks are spilled into user buckets. Each bucket is aggregated one key set
at a time; temporary and final NPYs are written through one-column memory maps.
The CLI never constructs the full dense feature matrix. Increase --buckets or
reduce --chunksize if a bucket or memory guard stops a build.
"""
from contextlib import contextmanager
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.dont_write_bytecode = True

import numpy as np
from numpy.lib.format import write_array_header_2_0
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ST = "u_newsCatInterestsST"
REFRESH = "u_refreshTimes"
LIFECYCLE = "u_feedLifeCycle"
DAY_KEYS = ["user_id", "pt_d"]
AD_COLUMNS = DAY_KEYS + [ST, REFRESH, LIFECYCLE]
SOURCE_RENAME = {"u_userId": "user_id", "e_et": "pt_d"}
SOURCE_METRICS = ["label", "pro", "i_cat", "e_rn", "e_po", "e_m", "i_dtype", "e_ch", "e_section"]
SOURCE_COLUMNS = AD_COLUMNS + SOURCE_METRICS
SOURCE_INPUT_COLUMNS = ["u_userId", "e_et", ST, REFRESH, LIFECYCLE] + SOURCE_METRICS
KEY_SETS = (
    ("user_day_st", DAY_KEYS + [ST]),
    ("user_day_refresh_lifecycle", DAY_KEYS + [REFRESH, LIFECYCLE]),
    ("user_day_st_refresh_lifecycle", AD_COLUMNS),
    ("user_day_st_refresh", DAY_KEYS + [ST, REFRESH]),
)
METRICS = (
    "impressions", "clicks", "ctr_smoothed", "pro_mean", "clicked_pro_mean",
    "finish_rate", "clicked_finish_rate", "i_cat_nunique", "e_rn_mean", "e_rn_std",
    "e_po_mean", "e_m_mean", "i_dtype_nunique", "e_ch_nunique", "e_m_nunique",
    "e_section_nunique", "e_rn_nunique", "impression_share", "click_share", "missing",
)
PRIOR = 0.1
KEY_STRENGTH = 20.0
DAY_STRENGTH = 20.0
FINISH_THRESHOLD = 80.0
ROW = "__advertisement_row"
FLOAT = np.dtype("<f4")
RULE = (
    "Offline complete same-day auxiliary source snapshot: supplied news label -1/1 and pro "
    "are inputs; advertisement labels are never loaded. Exact user/day/ST, "
    "user/day/refresh/lifecycle, user/day/ST/refresh/lifecycle, and user/day/ST/refresh "
    "matching; source u_userId/e_et renamed to user_id/pt_d. ST is the literal ordered "
    "u_newsCatInterestsST string, not a token-overlap key. Only same-user/day source "
    "outcomes enter CTR priors and shares. Same-day news may follow the advertisement "
    "exposure; no online pre-exposure availability or temporal leakage-free claim."
)


def feature_names():
    return ["source_context__" + name + "__" + metric
            for name, _ in KEY_SETS for metric in METRICS]


def _csv_dtypes(columns):
    return {name: (str if name == ST else "int64" if name in DAY_KEYS + ["u_userId", "e_et", ROW]
                   else "float64") for name in columns}


def _read_csv(path, columns, **kwargs):
    # Every advertisement read, including spilled chunks, uses this explicit allowlist.
    # keep_default_na=False preserves a literal interest string such as "NA".
    return pd.read_csv(path, usecols=list(columns), dtype=_csv_dtypes(columns),
                       keep_default_na=False, na_values=[""], **kwargs)


def _keys(frame):
    frame[ST] = frame[ST].fillna("").astype(str)
    for name in DAY_KEYS:
        values = pd.to_numeric(frame[name], errors="raise")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError("Missing or nonfinite identity/date key: " + name)
        if not (values == np.floor(values)).all():
            raise ValueError("Identity/date keys must be integers: " + name)
        frame[name] = values.astype("int64")
    for name in (REFRESH, LIFECYCLE):
        frame[name] = pd.to_numeric(frame[name], errors="raise").astype("float64")
        if np.isinf(frame[name].to_numpy()).any():
            raise ValueError("Infinite context key: " + name)
    return frame


def _advertisements(frame):
    # Do not access, copy, validate, or derive anything from frame['label'].
    return _keys(frame.loc[:, AD_COLUMNS].reset_index(drop=True).copy())


def _source(frame):
    selected = frame.rename(columns=SOURCE_RENAME)
    selected = _keys(selected.loc[:, SOURCE_COLUMNS].reset_index(drop=True).copy())
    for name in SOURCE_METRICS:
        selected[name] = pd.to_numeric(selected[name], errors="raise").astype("float64")
        if np.isinf(selected[name].to_numpy()).any():
            raise ValueError("Infinite source input: " + name)
    if not selected["label"].isin([-1, 1]).all():
        raise ValueError("Source labels must be -1/1")
    return selected


def _aligned_merge(requests, aggregates, keys):
    """Many-to-one lookup with an explicit positional row-alignment assertion."""
    lookup = requests.loc[:, keys].copy()
    lookup[ROW] = np.arange(len(requests), dtype=np.int64)
    result = lookup.merge(aggregates, on=keys, how="left", sort=False, validate="many_to_one")
    if not np.array_equal(result[ROW].to_numpy(), np.arange(len(requests))):
        result = result.sort_values(ROW, kind="stable").reset_index(drop=True)
    if len(result) != len(requests) or not np.array_equal(result[ROW], np.arange(len(requests))):
        raise AssertionError("Source lookup changed advertisement row order")
    return result


def iter_feature_columns(advertisements, source, *, prior=PRIOR,
                         key_strength=KEY_STRENGTH, day_strength=DAY_STRENGTH):
    """Yield (name, float32 vector) without constructing an N x 80 matrix.

    Missing exact matches have zero counts, descriptors, and shares, missing=1,
    and CTR backed off to the same user/day prior. If that parent is absent,
    CTR is the fixed prior. A matched group with no clicks has clicked progress
    and clicked finish rate zero. Other absent numeric measurements become zero.
    Empty ST and missing refresh/lifecycle are explicit matching key states;
    missing user/date identities are rejected. e_rn_std is sample std (ddof=1).
    """
    if not np.isfinite(prior) or not 0 < prior < 1:
        raise ValueError("prior must lie strictly between zero and one")
    if not all(np.isfinite(x) and x > 0 for x in (key_strength, day_strength)):
        raise ValueError("Smoothing strengths must be finite and positive")
    ads = _advertisements(advertisements)
    feeds = _source(source)
    feeds["_clicked"] = feeds["label"].eq(1).astype("float64")
    feeds["_clicked_pro"] = feeds["pro"].where(feeds["label"].eq(1))
    feeds["_finished"] = (feeds["label"].eq(1) & feeds["pro"].ge(FINISH_THRESHOLD)).astype("float64")
    parent = feeds.groupby(DAY_KEYS, sort=False, dropna=False).agg(
        day_impressions=("_clicked", "size"), day_clicks=("_clicked", "sum")).reset_index()
    aligned_parent = _aligned_merge(ads, parent, DAY_KEYS)
    day_n = aligned_parent["day_impressions"].fillna(0).to_numpy(dtype="float64")
    day_c = aligned_parent["day_clicks"].fillna(0).to_numpy(dtype="float64")
    day_ctr = (day_c + day_strength * prior) / (day_n + day_strength)
    del aligned_parent, parent

    for prefix, keys in KEY_SETS:
        aggregated = feeds.groupby(keys, sort=False, dropna=False).agg(
            impressions=("_clicked", "size"), clicks=("_clicked", "sum"),
            pro_mean=("pro", "mean"), clicked_pro_mean=("_clicked_pro", "mean"),
            finished=("_finished", "sum"), i_cat_nunique=("i_cat", "nunique"),
            e_rn_mean=("e_rn", "mean"), e_rn_std=("e_rn", "std"),
            e_po_mean=("e_po", "mean"), e_m_mean=("e_m", "mean"),
            i_dtype_nunique=("i_dtype", "nunique"), e_ch_nunique=("e_ch", "nunique"),
            e_m_nunique=("e_m", "nunique"), e_section_nunique=("e_section", "nunique"),
            e_rn_nunique=("e_rn", "nunique")).reset_index()
        matched = _aligned_merge(ads, aggregated, keys)
        del aggregated
        absent = matched["impressions"].isna().to_numpy()
        n = matched["impressions"].fillna(0).to_numpy(dtype="float64")
        c = matched["clicks"].fillna(0).to_numpy(dtype="float64")
        finished = matched["finished"].fillna(0).to_numpy(dtype="float64")
        derived = {
            "ctr_smoothed": (c + key_strength * day_ctr) / (n + key_strength),
            "finish_rate": finished / np.maximum(n, 1),
            "clicked_finish_rate": finished / np.maximum(c, 1),
            "impression_share": n / np.maximum(day_n, 1),
            "click_share": c / np.maximum(day_c, 1),
            "missing": absent,
        }
        for metric in METRICS:
            values = (np.asarray(derived[metric], dtype="float32") if metric in derived
                      else matched[metric].fillna(0).to_numpy(dtype="float32"))
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite source feature: " + prefix + "/" + metric)
            yield "source_context__" + prefix + "__" + metric, values
        del matched, derived, n, c, finished, absent


def build_features(advertisements, source, **kwargs):
    """Convenience for small tests; the production CLI uses the column iterator."""
    columns = list(iter_feature_columns(advertisements, source, **kwargs))
    return np.column_stack([values for _, values in columns]), [name for name, _ in columns]


class MemoryBudget:
    """Observe working-set/commit peaks, and refuse oversized bucket allocations."""
    def __init__(self, mib):
        if not 128 <= mib <= 1792:
            raise ValueError("memory-budget-mib must be between 128 and 1792 (<2 GiB)")
        self.limit = int(mib * 1024**2)
        self.peak_working_set = 0
        self.peak_private = 0

    def check(self, extra=0):
        if os.name == "nt":
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                        "PagefileUsage", "PeakPagefileUsage")]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            info = Counters()
            info.cb = ctypes.sizeof(info)
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb):
                raise ctypes.WinError(ctypes.get_last_error())
            current = max(info.WorkingSetSize, info.PagefileUsage)
            self.peak_working_set = max(self.peak_working_set, int(info.PeakWorkingSetSize))
            self.peak_private = max(self.peak_private, int(info.PeakPagefileUsage))
        else:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak = int(peak if sys.platform == "darwin" else peak * 1024)
            current = peak
            self.peak_working_set = max(self.peak_working_set, peak)
        if max(current + extra, self.peak_working_set, self.peak_private) > self.limit:
            raise MemoryError("Source context memory budget reached; increase --buckets or reduce --chunksize")

    def metadata(self):
        self.check()
        return {"budget_bytes": self.limit, "peak_working_set_bytes": self.peak_working_set,
                "peak_private_commit_bytes": self.peak_private}


def _new_column_npy(path, rows):
    """Create a standard Fortran-order NPY header without mapping the full matrix."""
    with path.open("xb") as stream:
        write_array_header_2_0(stream, {"descr": FLOAT.str, "fortran_order": True,
                                       "shape": (rows, len(feature_names()))})
        offset = stream.tell()
        stream.truncate(offset + rows * len(feature_names()) * FLOAT.itemsize)
    return offset


@contextmanager
def _column_map(path, rows, column, offset, mode="r+"):
    mapped = np.memmap(path, dtype=FLOAT, mode=mode, shape=(rows,),
                       offset=offset + column * rows * FLOAT.itemsize)
    try:
        yield mapped
        if mode != "r":
            mapped.flush()
    finally:
        mapped._mmap.close()


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_plan(args):
    advertisements = [args.data_dir / "train.csv"]
    sources = [args.data_dir / "train_feeds.csv"]
    if args.include_final:
        advertisements.append(args.data_dir / "final_test.csv")
        sources.append(args.data_dir / "test_feeds.csv")
    if not 1 <= args.buckets <= 4096 or not 1 <= args.chunksize <= 100000:
        raise ValueError("Require 1..4096 buckets and 1..100000 CSV rows per chunk")
    if not 1 <= args.max_bucket_mib <= 64:
        raise ValueError("max-bucket-mib must be between 1 and 64")
    MemoryBudget(args.memory_budget_mib).check()
    if args.output.suffix.lower() != ".npy":
        raise ValueError("output must have the .npy suffix")
    for path in advertisements:
        _read_csv(path, AD_COLUMNS, nrows=0)
    for path in sources:
        _read_csv(path, SOURCE_INPUT_COLUMNS, nrows=0)
    return advertisements, sources


def _spill(paths, directory, kind, args, budget, progress):
    row_offset = 0
    sections = []
    rows_per_bucket = np.zeros(args.buckets, dtype=np.int64)
    for path in paths:
        start = row_offset
        columns = AD_COLUMNS if kind == "ads" else SOURCE_INPUT_COLUMNS
        for chunk in _read_csv(path, columns, chunksize=args.chunksize):
            budget.check()
            if kind == "ads":
                chunk = _advertisements(chunk)
                chunk[ROW] = np.arange(row_offset, row_offset + len(chunk), dtype=np.int64)
            else:
                chunk = _source(chunk)
            routing = np.remainder(chunk["user_id"].to_numpy(), args.buckets)
            for bucket in np.unique(routing):
                selected = chunk.loc[routing == bucket]
                target = directory / f"{kind}_{bucket}.csv"
                selected.to_csv(target, mode="a", index=False, header=not target.exists())
                rows_per_bucket[bucket] += len(selected)
                if target.stat().st_size > args.max_bucket_mib * 1024**2:
                    raise MemoryError("Spilled bucket exceeds cap; increase --buckets: " + target.name)
            row_offset += len(chunk)
            budget.check()
            if progress and row_offset % (args.chunksize * 10) < len(chunk):
                progress(f"SOURCE_SPILL {kind} rows={row_offset}")
        sections.append({"file": str(path.resolve()), "start": start, "stop": row_offset})
    return sections, rows_per_bucket


def prepare(args, progress=None):
    """Chunked CSV -> per-key aggregates -> per-column memmaps, without training."""
    started = time.perf_counter()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Preserving existing output NPY/JSON: " + str(args.output))
    advertisements, sources = input_plan(args)
    budget = MemoryBudget(args.memory_budget_mib)
    names = feature_names()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".source_context_", dir=args.output.parent) as temporary:
        scratch = Path(temporary)
        ad_sections, ad_counts = _spill(advertisements, scratch, "ads", args, budget, progress)
        source_sections, source_counts = _spill(sources, scratch, "feeds", args, budget, progress)
        total = int(ad_counts.sum())
        if not total:
            raise ValueError("No advertisement rows")
        buckets = []
        for bucket in np.flatnonzero(ad_counts):
            ad_path = scratch / f"ads_{bucket}.csv"
            feed_path = scratch / f"feeds_{bucket}.csv"
            file_bytes = ad_path.stat().st_size + (feed_path.stat().st_size if feed_path.exists() else 0)
            # Reserve 24x CSV bytes plus 64 MiB for pandas parsing/groupby/merge
            # intermediates before reading either shard. Oversized input stops.
            budget.check(extra=file_bytes * 24 + 64 * 1024**2)
            ads = _read_csv(ad_path, AD_COLUMNS + [ROW])
            positions = ads.pop(ROW).to_numpy(dtype=np.int64)
            feeds = (_read_csv(feed_path, SOURCE_COLUMNS) if feed_path.exists()
                     else pd.DataFrame({name: pd.Series(dtype=_csv_dtypes([name])[name])
                                        for name in SOURCE_COLUMNS}))
            np.save(scratch / f"rows_{bucket}.npy", positions)
            bucket_path = scratch / f"values_{bucket}.npy"
            offset = _new_column_npy(bucket_path, len(ads))
            for column, (name, values) in enumerate(iter_feature_columns(ads, feeds)):
                if name != names[column]:
                    raise AssertionError("Feature schema changed")
                budget.check()
                with _column_map(bucket_path, len(ads), column, offset) as mapped:
                    mapped[:] = values
            buckets.append((int(bucket), len(ads), offset))
            if progress:
                progress(f"SOURCE_BUCKET {bucket + 1}/{args.buckets} ads={len(ads)} source={len(feeds)}")
            del ads, feeds, positions, values
            budget.check()

        staged = scratch / "complete.npy"
        final_offset = _new_column_npy(staged, total)
        for column in range(len(names)):
            budget.check(extra=total * FLOAT.itemsize)
            with _column_map(staged, total, column, final_offset) as output:
                for bucket, size, offset in buckets:
                    positions = np.load(scratch / f"rows_{bucket}.npy")
                    with _column_map(scratch / f"values_{bucket}.npy", size, column, offset, mode="r") as values:
                        output[positions] = values
            budget.check()
        metadata = {
            "shape": [total, len(names)], "dtype": "float32", "fortran_order": True,
            "features": names, "advertisement_labels_loaded": False, "source_outcomes_used": True,
            "rule": RULE, "keys": {name: keys for name, keys in KEY_SETS},
            "include_final": args.include_final, "advertisement_row_sections": ad_sections,
            "source_row_sections": source_sections, "source_rows": int(source_counts.sum()),
            "smoothing": {"initial_ctr": PRIOR, "user_day_strength": DAY_STRENGTH,
                          "key_strength": KEY_STRENGTH,
                          "formula": "day_ctr=(day_clicks+day_strength*initial_ctr)/(day_n+day_strength); key_ctr=(clicks+key_strength*day_ctr)/(n+key_strength)"},
            "missing_fallback": "Zero counts/descriptors/shares, missing=1; CTR backs off only to same-user/day, then fixed initial_ctr. No-click clicked means/rates are zero.",
            "shares": "key impressions / same-user-day impressions; key clicks / same-user-day clicks (zero when denominator is zero)",
            "completion": "label == 1 and pro >= 80; finish_rate per exposure, clicked_finish_rate per click",
            "dispersion": "e_rn sample standard deviation, ddof=1; singleton/absent -> 0",
            "key_nulls": "Empty ST and NaN refresh/lifecycle match their own missing state; identity/date cannot be missing. Literal ST strings including NA are preserved.",
            "numeric_codes_note": "Means of anonymous e_* codes are empirical descriptors, not asserted physical measurements.",
            "hash": _hash(staged), "hash_algorithm": "sha256 (complete NPY bytes)",
            "script_sha256": _hash(Path(__file__)), "memory": budget.metadata(),
            "buckets": args.buckets, "chunksize": args.chunksize,
            "seconds": time.perf_counter() - started,
        }
        staged_json = scratch / "complete.json"
        staged_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.output.exists() or args.output.with_suffix(".json").exists():
            raise FileExistsError("Output appeared during build; preserving it")
        staged.replace(args.output)
        staged_json.replace(args.output.with_suffix(".json"))
    return metadata


def _test_fixture():
    source = pd.DataFrame([
        [1, 20191116, "A^B", 1, 2, 1, 100, 10, 1, 1, 2, 1, 1, 0],
        [1, 20191116, "A^B", 1, 2, -1, 0, 20, 3, 3, 4, 2, 2, 1],
        [1, 20191116, "A^B", 2, 2, 1, 60, 10, 5, 5, 4, 1, 2, 1],
        [1, 20191116, "C", 1, 2, -1, 0, 30, 7, 1, 2, 1, 1, 0],
        [1, 20191116, "A^B", 1, 3, 1, 80, 40, 9, 9, 6, 3, 3, 1],
        [1, 20191115, "A^B", 1, 2, -1, 0, 10, 1, 1, 1, 1, 1, 0],
        [2, 20191116, "A^B", 1, 2, 1, 20, 20, 4, 1, 1, 1, 1, 1],
        [1, 20191117, "A^B", 1, 2, 1, 100, 30, 2, 2, 2, 1, 1, 1],
        [3, 20191116, "NA", 1, 2, 1, 90, 30, 2, 2, 2, 1, 1, 1],
        [1, 20191116, "", np.nan, np.nan, -1, 0, 10, 1, 1, 2, 1, 1, 0],
    ], columns=SOURCE_INPUT_COLUMNS)
    ads = pd.DataFrame([
        [1, 20191116, "A^B", 1, 2], [2, 20191116, "A^B", 1, 2],
        [1, 20191115, "A^B", 1, 2], [1, 20191116, "unseen", 1, 2],
        [4, 20191116, "A^B", 1, 2], [1, 20191117, "A^B", 1, 2],
        [1, 20191116, "A^B", 1, 3], [3, 20191116, "NA", 1, 2],
        [1, 20191116, "", np.nan, np.nan], [1, 20191116, "A^B", 1, 2],
    ], columns=AD_COLUMNS)
    ads.index = np.arange(len(ads)) * 7 + 101
    return ads, source


def _reference(ads, source):
    """Independent row-wise reference; no feature-builder or merge helpers."""
    feeds = source.rename(columns=SOURCE_RENAME)
    result = []
    for _, ad in ads.iterrows():
        day = feeds.loc[(feeds.user_id == ad.user_id) & (feeds.pt_d == ad.pt_d)]
        day_clicks = int(day.label.eq(1).sum())
        day_ctr = (day_clicks + DAY_STRENGTH * PRIOR) / (len(day) + DAY_STRENGTH)
        row = []
        for _, keys in KEY_SETS:
            selected = day
            for key in keys[2:]:
                selected = selected.loc[selected[key].isna() if pd.isna(ad[key])
                                        else selected[key].eq(ad[key])]
            clicked = selected.loc[selected.label.eq(1)]
            finished = int(clicked.pro.ge(FINISH_THRESHOLD).sum())
            values = {
                "impressions": len(selected), "clicks": len(clicked),
                "ctr_smoothed": (len(clicked) + KEY_STRENGTH * day_ctr) / (len(selected) + KEY_STRENGTH),
                "pro_mean": selected.pro.mean(), "clicked_pro_mean": clicked.pro.mean(),
                "finish_rate": finished / max(len(selected), 1),
                "clicked_finish_rate": finished / max(len(clicked), 1),
                "e_rn_std": selected.e_rn.std(ddof=1),
                "impression_share": len(selected) / max(len(day), 1),
                "click_share": len(clicked) / max(day_clicks, 1), "missing": int(not len(selected)),
            }
            for field in ("i_cat", "i_dtype", "e_ch", "e_m", "e_section", "e_rn"):
                values[field + "_nunique"] = selected[field].nunique()
            for field in ("e_rn", "e_po", "e_m"):
                values[field + "_mean"] = selected[field].mean()
            row.extend(0 if pd.isna(values[name]) else values[name] for name in METRICS)
        result.append(row)
    return np.asarray(result, dtype="float32")


def run_tests():
    import unittest
    from unittest.mock import patch

    class SourceContextTests(unittest.TestCase):
        def setUp(self):
            self.ads, self.source = _test_fixture()

        def test_reference_all_80_columns(self):
            actual, names = build_features(self.ads, self.source)
            self.assertEqual(actual.shape, (10, 80))
            self.assertEqual(names, feature_names())
            self.assertEqual(len(set(names)), 80)
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_allclose(actual, _reference(self.ads, self.source), rtol=1e-6, atol=1e-6)

        def test_advertisement_label_changes_are_irrelevant(self):
            original, _ = build_features(self.ads.assign(label=0), self.source)
            changed, _ = build_features(self.ads.assign(label="must-not-be-read"), self.source)
            without, _ = build_features(self.ads, self.source)
            np.testing.assert_array_equal(original, changed)
            np.testing.assert_array_equal(original, without)

        def test_user_and_day_boundaries_include_ctr_backoff(self):
            baseline, _ = build_features(self.ads, self.source)
            target = (self.ads.user_id == 1) & (self.ads.pt_d == 20191116)
            for other in (self.source.u_userId.ne(1), self.source.e_et.ne(20191116)):
                changed = self.source.copy()
                changed.loc[other, "label"] *= -1
                changed.loc[other, ["pro", "e_rn", "i_cat"]] = 999
                actual, _ = build_features(self.ads, changed)
                np.testing.assert_array_equal(actual[target], baseline[target])

        def test_row_permutation_duplicates_and_missing_fallback(self):
            baseline, names = build_features(self.ads, self.source)
            order = [9, 4, 1, 7, 8, 2, 5, 0, 6, 3]
            actual, _ = build_features(self.ads.iloc[order], self.source.iloc[::-1])
            np.testing.assert_allclose(actual, baseline[order], rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(baseline[0], baseline[9])
            prefix = "source_context__user_day_st_refresh_lifecycle__"
            self.assertEqual(baseline[3, names.index(prefix + "missing")], 1)
            self.assertEqual(baseline[3, names.index(prefix + "impressions")], 0)
            self.assertAlmostEqual(float(baseline[3, names.index(prefix + "ctr_smoothed")]), 5 / 26, places=6)
            self.assertAlmostEqual(float(baseline[4, names.index(prefix + "ctr_smoothed")]), PRIOR, places=6)
            self.assertAlmostEqual(float(baseline[0, names.index(prefix + "impression_share")]), 2 / 6, places=6)
            self.assertAlmostEqual(float(baseline[0, names.index(prefix + "click_share")]), 1 / 3, places=6)
            self.assertEqual(baseline[7, names.index(prefix + "missing")], 0)  # literal NA
            self.assertEqual(baseline[8, names.index(prefix + "missing")], 0)  # null key state

        def test_empty_source_and_invalid_outcomes(self):
            actual, names = build_features(self.ads, self.source.iloc[:0])
            for group in range(4):
                self.assertTrue((actual[:, group * 20 + METRICS.index("missing")] == 1).all())
                np.testing.assert_allclose(actual[:, group * 20 + METRICS.index("ctr_smoothed")], PRIOR)
            self.assertTrue(np.isfinite(actual).all())
            bad = self.source.copy()
            bad.loc[0, "label"] = 0
            with self.assertRaisesRegex(ValueError, "Source labels"):
                build_features(self.ads, bad)
            with self.assertRaisesRegex(ValueError, "Identity/date"):
                build_features(self.ads.assign(user_id=1.5), self.source)

        def test_source_outcomes_are_used_and_memory_guard_refuses_work(self):
            original, names = build_features(self.ads, self.source)
            source = self.source.copy()
            source.loc[0, "label"] = -1
            changed, _ = build_features(self.ads, source)
            for prefix, _ in KEY_SETS:
                column = names.index("source_context__" + prefix + "__clicks")
                self.assertEqual(changed[0, column], original[0, column] - 1)
            with self.assertRaises(MemoryError):
                MemoryBudget(1536).check(extra=2 * 1024**3)
            with self.assertRaises(ValueError):
                MemoryBudget(2048)

        def test_csv_allowlist_include_final_memmap_and_hash(self):
            original_reader = pd.read_csv
            ad_reads = []

            def checked_reader(path, *args, **kwargs):
                if Path(path).name in ("train.csv", "final_test.csv") or Path(path).name.startswith("ads_"):
                    self.assertIn("usecols", kwargs)
                    self.assertNotIn("label", kwargs["usecols"])
                    ad_reads.append(Path(path).name)
                return original_reader(path, *args, **kwargs)

            with tempfile.TemporaryDirectory() as directory, patch.object(pd, "read_csv", side_effect=checked_reader):
                folder = Path(directory)
                self.ads.assign(label=0).to_csv(folder / "train.csv", index=False)
                self.source.to_csv(folder / "train_feeds.csv", index=False)
                args = argparse.Namespace(data_dir=folder, output=folder / "default.npy",
                                          include_final=False, buckets=3, chunksize=3,
                                          memory_budget_mib=1536, max_bucket_mib=64)
                metadata = prepare(args)
                # Neither final file exists yet: default mode must never try to read it.
                loaded = np.load(args.output, mmap_mode="r")
                expected, _ = build_features(self.ads, self.source)
                np.testing.assert_allclose(loaded, expected, rtol=1e-6, atol=1e-6)
                self.assertTrue(loaded.flags.f_contiguous)
                self.assertFalse(metadata["advertisement_labels_loaded"])
                self.assertTrue(metadata["source_outcomes_used"])
                self.assertEqual(metadata["hash"], _hash(args.output))
                self.assertEqual(json.loads(args.output.with_suffix(".json").read_text(encoding="utf-8"))["features"], feature_names())
                self.assertEqual(metadata["advertisement_row_sections"][0]["stop"], len(self.ads))
                loaded._mmap.close()
                with self.assertRaises(FileExistsError):
                    prepare(args)
                self.ads.assign(label="unparseable-ad-target").to_csv(folder / "train.csv", index=False)
                args.output = folder / "changed_labels.npy"
                changed_metadata = prepare(args)
                changed = np.load(args.output)
                np.testing.assert_array_equal(changed, expected)
                self.assertEqual(changed_metadata["hash"], metadata["hash"])
                final_ads = self.ads.iloc[[0, 4]].copy()
                final_ads.iloc[1, final_ads.columns.get_loc("user_id")] = 99
                final_ads.to_csv(folder / "final_test.csv", index=False)
                test_source = self.source.iloc[[0, 0]].reset_index(drop=True)
                test_source.loc[1, "u_userId"] = 99
                test_source.to_csv(folder / "test_feeds.csv", index=False)
                args.include_final = True
                args.output = folder / "with_final.npy"
                final_meta = prepare(args)
                final_expected, _ = build_features(pd.concat([self.ads, final_ads]),
                                                   pd.concat([self.source, test_source]))
                np.testing.assert_allclose(np.load(args.output), final_expected, rtol=1e-6, atol=1e-6)
                self.assertEqual(final_meta["advertisement_row_sections"][1]["start"], len(self.ads))
                self.assertEqual(final_meta["source_rows"], len(self.source) + len(test_source))
                self.assertTrue(final_meta["include_final"])
                self.assertIn("final_test.csv", ad_reads)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SourceContextTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "2026ml")
    parser.add_argument("--output", type=Path, default=HERE / "source_context.npy")
    parser.add_argument("--include-final", action="store_true", help="Append final_test.csv and include test_feeds.csv")
    parser.add_argument("--buckets", type=int, default=64)
    parser.add_argument("--chunksize", type=int, default=25000)
    parser.add_argument("--memory-budget-mib", type=int, default=1536)
    parser.add_argument("--max-bucket-mib", type=int, default=32)
    parser.add_argument("--test-only", action="store_true", help="Synthetic tests only; no real dataset is read")
    parser.add_argument("--plan-only", action="store_true", help="Check headers and sizes only; do not read data rows")
    args = parser.parse_args()
    if args.test_only:
        run_tests()
        return
    advertisements, sources = input_plan(args)
    print("SOURCE_CONTEXT_PLAN " + json.dumps({
        "advertisements": [{"file": str(p), "bytes": p.stat().st_size} for p in advertisements],
        "sources": [{"file": str(p), "bytes": p.stat().st_size} for p in sources],
        "features": len(feature_names()), "include_final": args.include_final,
        "buckets": args.buckets, "chunksize": args.chunksize,
        "memory_budget_mib": args.memory_budget_mib, "max_bucket_mib": args.max_bucket_mib,
        "output": str(args.output), "plan_only": args.plan_only,
        "rule": RULE}, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    metadata = prepare(args, progress=lambda message: print(message, flush=True))
    print("SOURCE_CONTEXT_COMPLETE " + json.dumps({key: metadata[key] for key in (
        "shape", "dtype", "include_final", "advertisement_labels_loaded", "source_outcomes_used",
        "hash", "memory", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
