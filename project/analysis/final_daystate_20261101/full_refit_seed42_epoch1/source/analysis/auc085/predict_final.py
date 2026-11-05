"""Predict final ads with the verified best fixed ensemble and saved preprocessing.

Run with the existing Machine_Learning Python environment. No fitting or label
updates are performed. The official CSV follows the supplied baseline: id,pctr.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "2026ml"
BENCH = HERE.parent / "model_comparison"
sys.path.insert(0, str(BENCH))

from benchmark_core import category_key, encode_sequence, source_history_features
from bounded_id_models import BoundedIDDCN
from signed_collaborative_model import SignedCollaborativeTabM
from modern_architectures import member_probability
from source_profiles import (DAILY_CATEGORIES, DAILY_SEQUENCES, DAILY_NUMERIC,
                             LOG_STANDARD, UNIT100, interest_overlap, load_profiles)

CACHES = {
    "dcn": HERE / "cache_rich_stats_random20",
    "signed": HERE / "cache_rich_stats_random20_aligned",
}
MODELS = {
    "dcn": HERE / "results/random20_rich_oof_statistics/bounded/seed42",
    "signed": HERE / "results_modern/phase6_signed_contrast/signed/seed42",
}
SELECTION = "phase6_signed_contrast__replace_tabm32"
EXPECTED_ROWS = 404182


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


class SavedFeatureTransform:
    """Reproduce both saved feature pipelines without changing their fitted state."""

    def __init__(self):
        self.manifests = {name: read_json(path / "manifest.json") for name, path in CACHES.items()}
        self.manifest = self.manifests["dcn"]
        self.vocabs = read_json(CACHES["dcn"] / "vocabs.json")
        assert self.vocabs == read_json(CACHES["signed"] / "vocabs.json")
        for field in ["categorical_fields", "sequence_fields", "numeric_fields", "group_for_field", "vocabulary_sizes"]:
            assert self.manifest[field] == self.manifests["signed"][field]
        self.raw_categories = self.manifest["categorical_fields"][:24]
        self.raw_sequences = self.manifest["sequence_fields"][:7]
        self.read_columns = self.raw_categories + self.raw_sequences + ["pt_d", "log_id"]
        self.raw_vocab = {field: {int(raw): code for raw, code in self.vocabs[self.manifest["group_for_field"][field]].items()}
                          for field in self.raw_categories}
        self.sequence_cache = {field: {} for field in self.manifest["sequence_fields"]}
        self.membership_cache = {field: {} for field in self.raw_sequences[:6]}
        self.original_scaler = read_json(BENCH / "cache/manifest.json")["numeric_scaler"]
        self.profiles = load_profiles(HERE)
        self.past_feeds = pd.read_csv(DATA / "train_feeds.csv", usecols=["u_userId", "e_et", "label", "pro", "i_cat"], dtype="int32")
        self.past_table = pd.DataFrame(columns=range(7), dtype=np.float32)
        self.statistics = []
        for definition in self.manifest["target_stat_definitions"]:
            name = "__".join(definition["fields"])
            with np.load(CACHES["dcn"] / "target_stat_models" / (name + ".npz")) as model:
                self.statistics.append((definition["fields"], {key: model[key].copy() for key in model.files}))

    def sequences(self, values, field):
        cache = self.sequence_cache[field]
        vocabulary = self.vocabs[self.manifest["group_for_field"][field]]
        for value in values.unique():
            if value not in cache:
                cache[value] = encode_sequence(value, vocabulary, 5)
        return np.stack([cache[value] for value in values])

    def historical(self, frame, keys):
        requests = frame[["user_id", "pt_d"]].drop_duplicates()
        request_keys = requests["pt_d"].to_numpy(dtype=np.int64) * 10000000 + requests["user_id"].to_numpy(dtype=np.int64)
        missing = ~np.isin(request_keys, self.past_table.index.to_numpy())
        if missing.any():
            values = source_history_features(self.past_feeds, requests.loc[missing])
            self.past_table = pd.concat([self.past_table, pd.DataFrame(values, index=request_keys[missing])])
        return self.past_table.reindex(keys).to_numpy(dtype=np.float32)

    def transform(self, frame):
        size = len(frame)
        categories = np.empty((size, 27), dtype=np.int32)
        sequences = np.empty((size, 12, 5), dtype=np.int32)
        raw = np.zeros((size, 75), dtype=np.float32)
        for index, field in enumerate(self.raw_categories):
            categories[:, index] = frame[field].map(self.raw_vocab[field]).fillna(1).to_numpy(dtype=np.int32)
        for index, field in enumerate(self.raw_sequences):
            values = frame[field].fillna("")
            sequences[:, index] = self.sequences(values, field)
            raw[:, 6 + index] = np.count_nonzero(sequences[:, index], axis=1) / 5.
            if index < 6:
                memberships = self.membership_cache[field]
                for value in values.unique():
                    if value not in memberships:
                        memberships[value] = {int(token) for token in value.split("^") if token}
                current = ["task_id", "adv_prim_id", "spread_app_id"][index % 3]
                raw[:, index] = np.fromiter((int(target) in memberships[value] for value, target in zip(values, frame[current])), dtype=np.float32, count=size)
        keys = frame["pt_d"].to_numpy(dtype=np.int64) * 10000000 + frame["user_id"].to_numpy(dtype=np.int64)
        raw[:, 13:20] = self.historical(frame, keys)
        raw[:, 16] /= 100.
        for index in [13, 14, 17, 18, 19]:
            raw[:, index] = np.log1p(raw[:, index])
            # The random split cache was produced by undoing the original
            # historical standardization. Retain those float32 roundings.
            old = self.original_scaler[self.manifest["numeric_fields"][index]]
            raw[:, index] = (raw[:, index] - old["mean"]) / old["std"]
            raw[:, index] = raw[:, index] * old["std"] + old["mean"]
        aligned = self.profiles.reindex(keys)
        for index, field in enumerate(DAILY_CATEGORIES, start=24):
            values = aligned[field].fillna("__MISSING__").map(category_key)
            categories[:, index] = values.map(self.vocabs[field]).fillna(1).to_numpy(dtype=np.int32)
        for index, field in enumerate(DAILY_SEQUENCES, start=7):
            sequences[:, index] = self.sequences(aligned[field].fillna(""), field)
        for index, field in enumerate(DAILY_NUMERIC[:15], start=20):
            values = aligned[field].fillna(.1 if field == "feed_day_ctr_smoothed" else 0).to_numpy(dtype=np.float32)
            if field in UNIT100:
                values = values / 100.
            elif field in LOG_STANDARD:
                values = np.log1p(values)
            raw[:, index] = values
        short = frame["u_newsCatInterestsST"].fillna("").to_numpy()
        clicked = aligned["_all_feed_clicked_topics"].fillna("").to_numpy()
        exposed = aligned["_all_feed_exposed_topics"].fillna("").to_numpy()
        longterm = aligned["_all_feed_long_interest"].fillna("").to_numpy()
        overlaps = {}
        for index, (key, s, c, e, l) in enumerate(zip(keys, short, clicked, exposed, longterm)):
            lookup = (int(key), s)
            if lookup not in overlaps:
                overlaps[lookup] = interest_overlap(s, c, e, l)
            raw[index, 35:38] = overlaps[lookup]
        raw[:, 38] = aligned["feed_day_impressions"].isna().to_numpy()
        for index, (fields, model) in enumerate(self.statistics):
            lookup = np.zeros(size, dtype=np.int64)
            for field in fields:
                group = self.manifest["group_for_field"][field]
                column = self.manifest["categorical_fields"].index(field)
                lookup = lookup * self.manifest["vocabulary_sizes"][group] + categories[:, column]
            position = np.searchsorted(model["keys"], lookup)
            safe = np.minimum(position, len(model["keys"]) - 1)
            known = (position < len(model["keys"])) & (model["keys"][safe] == lookup)
            count, clicks = np.zeros(size, dtype=np.int64), np.zeros(size, dtype=np.float64)
            count[known], clicks[known] = model["counts"][safe[known]], model["clicks"][safe[known]]
            prior, alpha = float(model["prior"]), float(model["alpha"])
            probability = ((clicks + alpha * prior) / (count + alpha)).astype(np.float32)
            probability = np.clip(probability, 1e-6, 1 - 1e-6)
            raw[:, 39 + 2 * index] = np.log1p(count.astype(np.float32))
            raw[:, 40 + 2 * index] = np.log(probability / (1 - probability)) - np.log(prior / (1 - prior))
        numeric = {}
        for branch, manifest in self.manifests.items():
            numeric[branch] = raw.copy()
            for name, scale in manifest["numeric_scaler"].items():
                column = manifest["numeric_fields"].index(name)
                numeric[branch][:, column] = (numeric[branch][:, column] - scale["mean"]) / scale["std"]
            assert np.isfinite(numeric[branch]).all()
        for index, field in enumerate(self.manifest["categorical_fields"]):
            assert categories[:, index].min() >= 1
            assert categories[:, index].max() < self.manifest["vocabulary_sizes"][self.manifest["group_for_field"][field]]
        for index, field in enumerate(self.manifest["sequence_fields"]):
            assert sequences[:, index].min() >= 0
            assert sequences[:, index].max() < self.manifest["vocabulary_sizes"][self.manifest["group_for_field"][field]]
        return categories, sequences, numeric


def verify_transform(transformer):
    """Compare raw validation CSV rows against the features actually used in training."""
    valid = np.load(CACHES["dcn"] / "valid_indices.npy")
    labels = np.load(CACHES["dcn"] / "label.npy", mmap_mode="r")
    positive_positions = np.flatnonzero(labels[valid] == 1)
    selected = np.unique(np.concatenate([np.arange(2048), np.linspace(0, len(valid) - 1, 512, dtype=int),
                                        positive_positions[np.linspace(0, len(positive_positions) - 1, 128, dtype=int)]]))
    wanted = valid[selected]
    order = np.argsort(wanted)
    sorted_rows = wanted[order]
    pieces, offset = [], 0
    for chunk in pd.read_csv(DATA / "train.csv", usecols=transformer.read_columns,
                             dtype={name: str for name in transformer.raw_sequences}, chunksize=60000):
        within = sorted_rows[(sorted_rows >= offset) & (sorted_rows < offset + len(chunk))] - offset
        if len(within):
            pieces.append(chunk.iloc[within].copy())
        offset += len(chunk)
    raw = pd.concat(pieces, ignore_index=True).iloc[np.argsort(order)].reset_index(drop=True)
    assert len(raw) == len(wanted)
    np.testing.assert_array_equal(raw["log_id"], np.load(CACHES["dcn"] / "log_id.npy", mmap_mode="r")[wanted])
    categories, sequences, numeric = transformer.transform(raw)
    checks = {}
    for branch, cache in CACHES.items():
        np.testing.assert_array_equal(categories, np.load(cache / "categories.npy", mmap_mode="r")[wanted])
        np.testing.assert_array_equal(sequences, np.load(cache / "sequences.npy", mmap_mode="r")[wanted])
        expected = np.load(cache / "numeric.npy", mmap_mode="r")[wanted]
        difference = np.abs(numeric[branch] - expected)
        np.testing.assert_allclose(numeric[branch], expected, rtol=2e-6, atol=2e-6)
        checks[branch] = {"rows": len(wanted), "categories_exact": True, "sequences_exact": True,
                          "maximum_numeric_error": float(difference.max()),
                          "worst_numeric_field": transformer.manifest["numeric_fields"][int(np.unravel_index(difference.argmax(), difference.shape)[1])]}
    print("PREPROCESSING_VERIFIED " + json.dumps(checks), flush=True)
    return {"categories": categories[:2048], "sequences": sequences[:2048],
            "numeric": {name: values[:2048] for name, values in numeric.items()}}, checks


def predict(model, arrays, branch, batch_size, label):
    result = np.empty(len(arrays["categories"]), dtype=np.float32)
    last_update = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(result), batch_size):
            end = min(start + batch_size, len(result))
            inputs = (torch.as_tensor(np.array(arrays["categories"][start:end], dtype=np.int64), device="cuda"),
                      torch.as_tensor(np.array(arrays["sequences"][start:end], dtype=np.int64), device="cuda"),
                      torch.as_tensor(np.array(arrays["numeric"][branch][start:end]), device="cuda"))
            with torch.cuda.amp.autocast(enabled=branch == "signed"):
                logits = model(*inputs)
            probability = member_probability(logits) if branch == "signed" else torch.sigmoid(logits.float())
            result[start:end] = probability.cpu().numpy()
            if time.perf_counter() - last_update > 20 or end == len(result):
                print(f"PREDICT {label} {end:,}/{len(result):,}", flush=True)
                last_update = time.perf_counter()
    assert np.isfinite(result).all() and np.all((result >= 0) & (result <= 1))
    return result


def main():
    started = time.perf_counter()
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    official_path = output / "submission_best.csv"
    detail_path = output / "final_predictions_best_with_ids.csv"
    manifest_path = output / "final_predictions_best_manifest.json"
    if any(path.exists() for path in [official_path, detail_path, manifest_path]):
        raise FileExistsError("Preserving existing final prediction deliverables")
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the selected model's inference precision")
    verification = read_json(HERE / "phase6_full_verification.json")
    assert verification["status"] == "passed"
    selection = next(row for row in read_json(HERE / "results_signed_ensembles/metrics.json") if row["name"] == SELECTION)
    assert selection["auc"] == max(row["auc"] for row in read_json(HERE / "results_signed_ensembles/metrics.json"))
    transformer = SavedFeatureTransform()
    probe, feature_checks = verify_transform(transformer)
    cache = HERE / "final_prediction_cache"
    cache.mkdir(exist_ok=True)
    arrays = {
        "categories": open_memmap(cache / "categories.npy", mode="w+", shape=(EXPECTED_ROWS, 27), dtype="int32"),
        "sequences": open_memmap(cache / "sequences.npy", mode="w+", shape=(EXPECTED_ROWS, 12, 5), dtype="int32"),
        "numeric": {branch: open_memmap(cache / (branch + "_numeric.npy"), mode="w+", shape=(EXPECTED_ROWS, 75), dtype="float32") for branch in CACHES},
    }
    identifiers, offset = [], 0
    for frame in pd.read_csv(DATA / "final_test.csv", usecols=transformer.read_columns,
                             dtype={name: str for name in transformer.raw_sequences}, chunksize=20000):
        categories, sequences, numeric = transformer.transform(frame)
        end = offset + len(frame)
        arrays["categories"][offset:end], arrays["sequences"][offset:end] = categories, sequences
        for branch in CACHES:
            arrays["numeric"][branch][offset:end] = numeric[branch]
        identifiers.append(frame[["log_id", "user_id", "pt_d"]].copy())
        offset = end
        print(f"FINAL_FEATURES {offset:,}/{EXPECTED_ROWS:,}", flush=True)
    assert offset == EXPECTED_ROWS
    ids = pd.concat(identifiers, ignore_index=True)
    assert ids["pt_d"].eq(20191117).all()
    for value in [arrays["categories"], arrays["sequences"], *arrays["numeric"].values()]:
        value.flush()
    del transformer, identifiers, categories, sequences, numeric
    gc.collect()
    predictions, replay, checkpoints = {}, {}, {}
    builders = {"dcn": BoundedIDDCN, "signed": SignedCollaborativeTabM}
    for branch, directory in MODELS.items():
        print("LOADING " + branch, flush=True)
        checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
        manifest = checkpoint["manifest"]
        assert manifest == read_json(CACHES[branch] / "manifest.json")
        assert checkpoint["model_class"] == builders[branch].__name__
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"The \d+-th feature has just two bin edges.*", category=UserWarning)
            model = builders[branch](manifest["categorical_fields"], manifest["sequence_fields"], manifest["group_for_field"],
                                     manifest["vocabulary_sizes"], **checkpoint["model_options"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.cuda().eval()
        batch = 8192 if branch == "dcn" else 2048
        reproduced = predict(model, probe, branch, batch, "validation_probe_" + branch)
        expected = np.load(directory / "validation_prediction.npy")[:len(reproduced)]
        np.testing.assert_allclose(reproduced, expected, rtol=3e-3, atol=2e-6)
        replay[branch] = {"rows": len(reproduced), "maximum_probability_error": float(np.abs(reproduced - expected).max()),
                          "precision": "float32" if branch == "dcn" else checkpoint["inference_precision"]}
        print("CHECKPOINT_REPLAY " + json.dumps({branch: replay[branch]}), flush=True)
        predictions[branch] = predict(model, arrays, branch, batch, "final_" + branch)
        np.save(cache / (branch + "_prediction.npy"), predictions[branch])
        checkpoints[branch] = {"path": str((directory / "model.pt").resolve()), "sha256": digest(directory / "model.pt"),
                               "preprocessing_cache": str(CACHES[branch].resolve()), "validation_metrics": checkpoint["validation_metrics"]}
        del model, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    probability = .5 * predictions["dcn"] + .5 * predictions["signed"]
    assert probability.shape == (EXPECTED_ROWS,)
    assert np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1))
    official = pd.DataFrame({"id": np.arange(EXPECTED_ROWS), "pctr": probability})
    official.to_csv(official_path, index=False, float_format="%.10g")
    detail = official[["id"]].join(ids)
    detail["pctr"], detail["dcn_pctr"], detail["signed_pctr"] = probability, predictions["dcn"], predictions["signed"]
    detail.to_csv(detail_path, index=False, float_format="%.10g")
    # Read the actual exported files to catch serialization and row-order errors.
    exported = pd.read_csv(official_path)
    exported_detail = pd.read_csv(detail_path)
    raw_ids = pd.read_csv(DATA / "final_test.csv", usecols=["log_id", "user_id", "pt_d"])
    assert exported.columns.tolist() == ["id", "pctr"] and len(exported) == EXPECTED_ROWS
    np.testing.assert_array_equal(exported["id"], np.arange(EXPECTED_ROWS))
    np.testing.assert_allclose(exported["pctr"], probability, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(exported_detail[["log_id", "user_id", "pt_d"]], raw_ids[["log_id", "user_id", "pt_d"]])
    assert np.isfinite(exported["pctr"]).all() and exported["pctr"].between(0, 1).all()
    report = {
        "status": "passed", "model_selection": SELECTION, "validation_auc": selection["auc"],
        "validation_strategy": "stratified random 80/20, seed42; not a held-out future-day AUC",
        "ensemble": {"dcn": .5, "signed": .5}, "checkpoints": checkpoints,
        "rows": EXPECTED_ROWS, "test_day": 20191117, "final_test_auc": None,
        "final_test_auc_note": "No advertisement labels supplied; only predictions and file integrity verified",
        "preprocessing_verification": feature_checks, "checkpoint_replay": replay,
        "preprocessing_rule": "Saved vocabulary, scalers, training-only target statistics, and frozen graph buffers; no refitting on final",
        "source_rule": "Strict-past train_feeds features plus matched same-day test_feeds profiles supplied by the offline task",
        "same_day_source_assumption": "Full source-day auxiliary snapshot is available, matching the selected validation experiment",
        "submission_columns": ["id", "pctr"], "id_rule": "0-based original final_test.csv row number, not log_id",
        "raw_row_order_verified": True, "identifier_columns_verified": ["log_id", "user_id", "pt_d"],
        "probability": {"minimum": float(probability.min()), "maximum": float(probability.max()), "mean": float(probability.mean()),
                        "quantiles": {str(q): float(np.quantile(probability, q)) for q in [.01, .1, .5, .9, .99]}},
        "files": {"submission": str(official_path.resolve()), "detail": str(detail_path.resolve()),
                  "submission_sha256": digest(official_path), "detail_sha256": digest(detail_path)},
        "inputs": {name: {"bytes": (DATA / name).stat().st_size, "sha256": digest(DATA / name)}
                   for name in ["final_test.csv", "test_feeds.csv"]},
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(manifest_path, report)
    print("FINAL_COMPLETE " + json.dumps({"submission": str(official_path), "rows": EXPECTED_ROWS,
          "validation_auc": selection["auc"], "probability": report["probability"], "seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
