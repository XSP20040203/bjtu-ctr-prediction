"""Full-data refit and final export of the frozen protected-day reference.

Large row feature matrices are rebuilt in RAM. No final ad labels are read.
The one-epoch budget and 0.30 component replacement are fixed before fitting.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[variable] = "3"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / "analysis/daily_interest_din/full_refit_cache"
OOF = ROOT / "analysis/nextday_goal_20261029/full_best/oof"
ARCH = ROOT / "analysis/architecture_goal_20261029/full_refit/gated166"
DAY = ROOT / "analysis/userday_aux_20261101"
REFERENCE = DAY / "reference_day_state_protected_seed42_v1"
COMPACT = ROOT / "compact_final"
ORIGINAL = ROOT / "output/best_nextday_arch_v7"
RUN = HERE / "full_refit_seed42_epoch1"
OUTPUT = ROOT / "output/best_nextday_daystate_v8"
NTRAIN, NFINAL, FINAL_DAY = 2526320, 404182, 20191117
ORIGINAL_HASH = "c44a9e803b8b4566b9bcef4e1bf85181c2ee00eee71a9f84474b73c619c121b9"
for relative in ["analysis/model_comparison", "analysis/auc085",
                 "analysis/daily_interest_din", "analysis/strong_interest_control_20261031"]:
    sys.path.insert(0, str(ROOT / relative))
sys.path.insert(0, str(DAY))

import numpy as np
import pandas as pd
import torch
from torch import nn
from benchmark_core import source_history_features
from predict_final import SavedFeatureTransform
from target_statistics import oof_target_statistics
from prepare_exposure_features import build_exposure_features
from prepare_snapshot_features import build_features as build_snapshot_features
from features import read, write, digest, emit, standardize_architecture
from experiment import configure
from model import build as build_backbone
from day_context import build_context, training_targets
from day_model import auxiliary_loss
from protected_day_model import ProtectedUserDayCTR, clip_separately


def compact_array(name):
    return np.load(COMPACT / "arrays" / (name + ".npy"), mmap_mode="r", allow_pickle=False)


def array_digest(value):
    result = hashlib.sha256()
    for start in range(0, len(value), 16384):
        result.update(np.ascontiguousarray(value[start:start + 16384]).tobytes())
    return result.hexdigest()


def check_probability(value, count):
    if value.shape != (count,) or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise AssertionError("Invalid probability vector")


def selected_protocol():
    reference = read(REFERENCE / "manifest.json")
    if (reference["status"] != "verified_development_reference" or
            reference["family"] != "protected_day_aux_v1" or reference["mode"] != "both" or
            reference["recipe"] != "replacement" or reference["seed"] != 42):
        raise AssertionError("Unexpected selected development reference")
    for name, expected in reference["files_sha256"].items():
        if digest(REFERENCE / name) != expected:
            raise AssertionError("Frozen reference changed: " + name)
    evidence = {}
    source_paths = {Path(__file__), HERE / "test_contracts.py"}
    for day in [15, 16]:
        directory = DAY / f"runs/protected_day_aux_v1_day{day}_seed42"
        protocol = read(directory / "protocol.json")
        metrics = read(directory / "both/metrics.json")
        if metrics["status"] != "complete_and_verified" or protocol["aux_weight"] != .02:
            raise AssertionError("Selected development run not verified")
        for filename, expected in protocol["sources"].items():
            path = Path(filename)
            if digest(path) != expected:
                raise AssertionError("Selected source changed: " + str(path))
            source_paths.add(path)
        for filename, expected in metrics["files_sha256"].items():
            if digest(directory / "both" / filename) != expected:
                raise AssertionError("Selected checkpoint/prediction changed")
        evidence[str(20191100 + day)] = {
            "ensemble_scores": reference["development"][str(20191100 + day)],
            "component_best_epoch": metrics["best_checkpoint"]["fractional_epoch"],
            "protocol_sha256": digest(directory / "protocol.json"),
            "metrics_sha256": digest(directory / "both/metrics.json"),
        }
    for relative in [
        "analysis/strong_interest_control_20261031/model.py",
        "analysis/architecture_goal_20261029/interaction_models.py",
        "analysis/model_comparison/id_regularized_models.py",
        "analysis/model_comparison/mainstream_models.py",
        "analysis/model_comparison/benchmark_core.py",
        "analysis/auc085/predict_final.py", "analysis/auc085/source_profiles.py",
        "analysis/auc085/target_statistics.py",
        "analysis/daily_interest_din/prepare_exposure_features.py",
        "analysis/daily_interest_din/prepare_snapshot_features.py",
    ]:
        source_paths.add(ROOT / relative)
    if digest(ORIGINAL / "submission.csv") != ORIGINAL_HASH:
        raise AssertionError("Original v7 submission changed")
    compact = read(COMPACT / "manifest.json")
    for relative in ["arrays/categories.npy", "arrays/sequences.npy", "arrays/gate_oof500.npy",
                     "arrays/ids.npy", "reference/v7.npy", "reference/gated166.npy", "models/gated166.pt"]:
        record = compact["files"][relative]
        if digest(COMPACT / relative) != record["sha256"]:
            raise AssertionError("Compact input/reference changed: " + relative)
    for path, other in [(ARCH / "model.pt", COMPACT / "models/gated166.pt"),
                        (ARCH / "final_prediction.npy", COMPACT / "reference/gated166.npy"),
                        (ORIGINAL / "ensemble_final_prediction.npy", COMPACT / "reference/v7.npy")]:
        if digest(path) != digest(other):
            raise AssertionError("Original/compact assets disagree")
    return {
        "status": "fixed_before_training", "model": "ProtectedUserDayCTR", "seed": 42,
        "mode": "both", "epochs": 1, "batch_size": 1024, "learning_rate": .001,
        "weight_decay": .0001, "auxiliary_weight": .02,
        "gradient_clipping": "Backbone and heads separately, each maximum norm 5",
        "fixed_steps": math.ceil(NTRAIN / 1024),
        "budget_rule": "One complete epoch, selected endpoint on most recent development date 20191116",
        "ensemble_rule": "original_v7 + 0.30 * (protected_day_full - original_gated166_full)",
        "development": evidence, "reference_manifest_sha256": digest(REFERENCE / "manifest.json"),
        "train_rows": NTRAIN, "final_rows": NFINAL, "final_day": FINAL_DAY,
        "final_labels_read": False, "final_auc": None, "final_early_stopping": False,
        "availability": "Supplied complete same-day unlabeled exposures, news feedback and historical snapshots",
        "large_feature_matrices_saved": False,
        "original_v7_submission_sha256": ORIGINAL_HASH,
        "source_sha256": {str(path): digest(path) for path in sorted(source_paths)},
    }


class FullTransform(SavedFeatureTransform):
    """Replay the original full-refit first-39 pipeline, including its roundings."""
    def __init__(self, full_meta, requests):
        super().__init__()
        self.old_signed = copy.deepcopy(self.manifests["signed"])
        self.vocabs = read(CACHE / "vocabs.json")
        for branch in self.manifests.values():
            branch["vocabulary_sizes"] = full_meta["vocabulary_sizes"]
        self.raw_vocab = {
            field: {int(raw): code for raw, code in
                    self.vocabs[self.manifest["group_for_field"][field]].items()}
            for field in self.raw_categories
        }
        keys = requests.pt_d.to_numpy(dtype="int64") * 10000000 + requests.user_id.to_numpy(dtype="int64")
        self.past_table = pd.DataFrame(source_history_features(self.past_feeds, requests), index=keys)
        del self.past_feeds
        # These columns are replaced by full-data OOF values below; avoid random-split lookups.
        self.statistics = []


def raw_rows(columns, **kwargs):
    return pd.concat([pd.read_csv(ROOT / "2026ml" / name, usecols=columns, **kwargs)
                      for name in ["train.csv", "final_test.csv"]], ignore_index=True)


def prepare():
    started = time.perf_counter()
    full = read(CACHE / "manifest.json")
    oof_meta = read(OOF / "manifest.json")
    architecture = read(ARCH / "input_manifest.json")
    names = architecture["numeric_fields"]
    if len(names) != 166 or max(full["vocabulary_sizes"].values()) > 65536:
        raise AssertionError("Unexpected full encoded schema")
    for filename, expected in full["input_sha256"].items():
        if digest(ROOT / "2026ml" / filename) != expected:
            raise AssertionError("Raw dataset changed: " + filename)
    raw = raw_rows(["pt_d", "user_id", "task_id"], dtype="int32")
    if len(raw) != NTRAIN + NFINAL:
        raise AssertionError("Raw row count differs")
    for raw_name, cache_name in [("pt_d", "date"), ("user_id", "user_id")]:
        np.testing.assert_array_equal(raw[raw_name], np.load(CACHE / f"{cache_name}.npy"))
    if np.any(raw.pt_d.to_numpy()[:NTRAIN] >= FINAL_DAY) or not np.all(raw.pt_d.to_numpy()[NTRAIN:] == FINAL_DAY):
        raise AssertionError("Full training/final date boundary differs")
    transform = FullTransform(full, raw[["user_id", "pt_d"]].drop_duplicates())
    n = len(raw)
    categories = np.empty((n, 27), dtype="uint16")
    sequences = np.empty((n, 12, 5), dtype="uint16")
    numeric = np.empty((n, 166), dtype="float32")
    labels = np.full(n, -1, dtype="int8")
    cached_log_id = np.load(CACHE / "log_id.npy", mmap_mode="r")
    offset = 0
    for filename in ["train.csv", "final_test.csv"]:
        columns = transform.read_columns + (["label"] if filename == "train.csv" else [])
        for chunk in pd.read_csv(ROOT / "2026ml" / filename, usecols=columns,
                                 dtype={field: str for field in transform.raw_sequences}, chunksize=50000):
            cats, seqs, values = transform.transform(chunk)
            stop = offset + len(chunk)
            np.testing.assert_array_equal(chunk.log_id, cached_log_id[offset:stop])
            categories[offset:stop], sequences[offset:stop] = cats, seqs
            base = values["signed"][:, :39].copy()
            for name, scale in transform.old_signed["numeric_scaler"].items():
                column = transform.old_signed["numeric_fields"].index(name)
                if column < 39:
                    base[:, column] = base[:, column] * scale["std"] + scale["mean"]
            for name, scale in full["numeric_scaler"].items():
                column = full["numeric_fields"].index(name)
                if column < 39:
                    base[:, column] = (base[:, column] - scale["mean"]) / scale["std"]
            numeric[offset:stop, :39] = standardize_architecture(base, names[:39], architecture)
            if filename == "train.csv":
                labels[offset:stop] = chunk.label.to_numpy()
            offset = stop
            if offset % 400000 < len(chunk) or offset == n:
                emit("full_first39_encoded", rows=offset, total=n)
    np.testing.assert_array_equal(labels, np.load(CACHE / "label.npy"))
    if offset != n or not np.isin(labels[:NTRAIN], [0, 1]).all() or np.any(labels[NTRAIN:] != -1):
        raise AssertionError("Full label boundary failure")
    del transform, cats, seqs, values, base, chunk
    gc.collect()
    train = np.arange(NTRAIN, dtype="int32")
    folds = np.load(OOF / "target_stat_oof_fold.npy")
    if len(folds) != n or np.any(folds[:NTRAIN] < 0) or np.any(folds[NTRAIN:] != -1):
        raise AssertionError("OOF fold boundary failure")
    stat_checks = []
    for position, definition in enumerate(oof_meta["target_stat_definitions"]):
        keys = np.zeros(n, dtype="int64")
        for field in definition["fields"]:
            width = full["vocabulary_sizes"][full["group_for_field"][field]]
            keys = keys * width + categories[:, full["categorical_fields"].index(field)]
        encoded, fitted = oof_target_statistics(keys, labels, train, folds, definition["alpha"])
        path = OOF / "target_stat_models" / ("__".join(definition["fields"]) + ".npz")
        with np.load(path) as saved:
            for key in fitted:
                np.testing.assert_array_equal(fitted[key], saved[key])
        count = np.log1p(encoded[:, 0])
        probability = np.clip(encoded[:, 1], 1e-6, 1 - 1e-6)
        relative = np.log(probability / (1 - probability)) - np.log(fitted["prior"] / (1 - fitted["prior"]))
        for j, value in enumerate([count, relative]):
            column = 39 + 2 * position + j
            scale = oof_meta["numeric_scaler"][names[column]]
            scaled = (value - scale["mean"]) / scale["std"]
            outer = architecture["scalers"][names[column]]
            numeric[:, column] = np.clip((scaled.astype("float64") - outer["mean"]) / outer["std"], -8, 8)
        stat_checks.append({"fields": definition["fields"], "all_saved_statistics_exact": True})
        emit("full_oof_stat_rebuilt", group=position + 1, groups=18)
        del keys, encoded, fitted, count, probability, relative, value, scaled
    del folds
    gc.collect()
    fields = ["pt_d", "user_id", "task_id", "adv_id", "adv_prim_id",
              "spread_app_id", "app_second_class", "slot_id"]
    source = raw_rows(fields, dtype="int32")
    exposure, exposure_names = build_exposure_features(source)
    if exposure_names != names[75:118]:
        raise AssertionError("Exposure field order differs")
    for start in range(0, n, 20000):
        stop = min(start + 20000, n)
        numeric[start:stop, 75:118] = standardize_architecture(exposure[start:stop], exposure_names, architecture)
    del exposure, source
    gc.collect()
    emit("full_exposure_rebuilt", rows=n)
    fields = ["user_id", "pt_d", "task_id", "adv_prim_id", "spread_app_id"]
    fields += [f"ad_{action}_list_v00{i}" for action in ["click", "close"] for i in [1, 2, 3]]
    source = raw_rows(fields, dtype={field: "category" if "list" in field else "int32" for field in fields})
    snapshot, snapshot_names = build_snapshot_features(source)
    if snapshot_names != names[118:166]:
        raise AssertionError("Snapshot field order differs")
    for start in range(0, n, 20000):
        stop = min(start + 20000, n)
        numeric[start:stop, 118:166] = standardize_architecture(snapshot[start:stop], snapshot_names, architecture)
    del snapshot, source
    gc.collect()
    emit("full_snapshot_rebuilt", rows=n)
    final_numeric = compact_array("gate_oof500")
    np.testing.assert_array_equal(categories[NTRAIN:], compact_array("categories"))
    np.testing.assert_array_equal(sequences[NTRAIN:], compact_array("sequences"))
    maximum = 0.
    for start in range(0, n, 20000):
        block = numeric[start:start + 20000]
        if not np.isfinite(block).all() or np.any((block < -8) | (block > 8)):
            raise AssertionError("Invalid full numeric features")
    for start in range(0, NFINAL, 20000):
        stop = min(start + 20000, NFINAL)
        actual, expected = numeric[NTRAIN + start:NTRAIN + stop], final_numeric[start:stop, :166]
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    info = {"status": "complete_and_verified", "train_rows": NTRAIN, "final_rows": NFINAL,
            "numeric_fields": names, "final_categories_and_sequences_exact": True,
            "maximum_final_numeric_error": maximum, "target_stat_rebuild_checks": stat_checks,
            "numeric_sha256": array_digest(numeric), "categorical_sha256": array_digest(categories),
            "sequence_sha256": array_digest(sequences), "final_labels_read": False,
            "ram_only_large_matrices": True, "seconds": time.perf_counter() - started,
            "preprocessing_assets_sha256": {str(path): digest(path) for path in [
                CACHE / "manifest.json", CACHE / "vocabs.json", OOF / "manifest.json",
                OOF / "target_stat_oof_fold.npy", ARCH / "input_manifest.json"]}}
    write(RUN / "features.json", info)
    emit("full_features_verified", maximum_final_numeric_error=maximum, seconds=info["seconds"])
    return {"categories": categories, "sequences": sequences, "numeric": numeric,
            "label": labels, "meta": full, "raw": raw, "train": train, "names": names}


def inputs(data, rows):
    return tuple(torch.as_tensor(np.array(data[key][rows], dtype=numpy_dtype, copy=True, order="C"),
                                 dtype=dtype, device="cuda")
                 for key, dtype, numpy_dtype in [("categories", torch.int64, np.int64),
                                                ("sequences", torch.int64, np.int64),
                                                ("numeric", torch.float32, np.float32)])


def core_replay(data=None):
    saved = torch.load(ARCH / "model.pt", map_location="cpu", weights_only=False)
    if data is None:
        meta = read(CACHE / "manifest.json")
        data = {"meta": meta, "categories": compact_array("categories"),
                "sequences": compact_array("sequences"), "numeric": compact_array("gate_oof500")[:, :166]}
        offset = 0
    else:
        offset = NTRAIN
    prior = saved.get("initial_ctr", data["meta"]["train_clicks"] / NTRAIN)
    model = build_backbone(data["meta"], "control", prior).cuda()
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    prediction = np.empty(NFINAL, dtype="float32")
    with torch.no_grad():
        for start in range(0, NFINAL, 1024):
            stop = min(start + 1024, NFINAL)
            prediction[start:stop] = model(*inputs(data, slice(offset + start, offset + stop))).sigmoid().cpu().numpy()
    expected = np.load(ARCH / "final_prediction.npy")
    error = float(np.max(np.abs(prediction - expected)))
    if error > 2e-7:
        raise AssertionError(f"Original full core replay differs: {error}")
    if RUN.exists():
        write(RUN / "original_core_replay.json", {"status": "passed", "all_final_rows_replayed": True,
              "rows": NFINAL, "maximum_probability_error": error, "final_labels_read": False})
    emit("original_full_core_replayed", maximum_error=error, rows=NFINAL)
    del model, saved
    gc.collect()
    torch.cuda.empty_cache()


def fit(data, protocol):
    started = time.perf_counter()
    store = build_context(data["raw"], data["categories"], data["numeric"], data["names"], data["meta"], FINAL_DAY)
    if len(store["manifest"]["excluded_target_stat_columns"]) != 36:
        raise AssertionError("Day context target-stat exclusion differs")
    targets, prior = training_targets(store["keys"], store["rows"], data["label"], data["train"], FINAL_DAY)
    if np.any(targets[store["rows"][NTRAIN:]] != -1):
        raise AssertionError("Final auxiliary targets must be unknown")
    write(RUN / "context_manifest.json", store["manifest"])
    final_groups = np.flatnonzero(store["keys"][:, 0] == FINAL_DAY)
    final_rows = np.searchsorted(final_groups, store["rows"][NTRAIN:]).astype("int32")
    np.testing.assert_array_equal(final_groups[final_rows], store["rows"][NTRAIN:])
    np.savez_compressed(RUN / "final_context.npz", values=store["values"][final_groups], rows=final_rows,
                        keys=store["keys"][final_groups], sizes=store["sizes"][final_groups])
    contexts = torch.as_tensor(store["values"], device="cuda")
    day_rows, day_sizes = store["rows"], store["sizes"]
    emit("full_day_context_ready", groups=len(store["keys"]), width=contexts.shape[1], any_prior=prior)
    del store, data["raw"]
    gc.collect()
    configure(42)
    model = ProtectedUserDayCTR(data["meta"], float(data["label"][:NTRAIN].mean()), contexts, prior, use_context=True).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    order = np.random.default_rng(43).permutation(data["train"])
    steps = math.ceil(len(order) / 1024)
    if steps != protocol["fixed_steps"]:
        raise AssertionError("Fixed training budget differs")
    total, row_total, aux_total, seen = 0., 0., 0., 0
    history = []
    model.train()
    for step, start in enumerate(range(0, len(order), 1024), 1):
        rows = order[start:start + 1024]
        groups = day_rows[rows]
        target = torch.as_tensor(data["label"][rows], dtype=torch.float32, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        logits, auxiliary = model(*inputs(data, rows), torch.as_tensor(groups, dtype=torch.int64, device="cuda"))
        row_loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
        aux_loss = auxiliary_loss(auxiliary,
                                  torch.as_tensor(targets[groups], dtype=torch.float32, device="cuda"),
                                  torch.as_tensor(day_sizes[groups], dtype=torch.float32, device="cuda"))
        loss = row_loss + .02 * aux_loss
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite full refit loss")
        loss.backward()
        clip_separately(model, model.parameters(), 5.)
        optimizer.step()
        seen += len(rows)
        total += float(loss.detach()) * len(rows)
        row_total += float(row_loss.detach()) * len(rows)
        aux_total += float(aux_loss.detach()) * len(rows)
        if step % 300 == 0 or step == steps:
            record = {"step": step, "steps": steps, "rows_seen": seen,
                      "objective": total / seen, "row_bce": row_total / seen,
                      "auxiliary_bce": aux_total / seen, "seconds": time.perf_counter() - started}
            history.append(record)
            write(RUN / "training_history.json", history)
            emit("full_protected_day_training", **record)
    if seen != NTRAIN:
        raise AssertionError("Full epoch did not cover every training row")
    saved = {"state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
             "meta": data["meta"], "any_prior": prior, "mode": "both", "seed": 42,
             "steps": steps, "training_rows_seen": seen, "numeric_fields": data["names"],
             "context_width": contexts.shape[1], "protocol_sha256": digest(RUN / "protocol.json"),
             "feature_manifest_sha256": digest(RUN / "features.json"),
             "context_manifest_sha256": digest(RUN / "context_manifest.json"),
             "final_context_sha256": digest(RUN / "final_context.npz"), "final_labels_used": False}
    torch.save(saved, RUN / "model.pt")
    model.eval()
    prediction = np.empty(NFINAL, dtype="float32")
    with torch.no_grad():
        for start in range(0, NFINAL, 1024):
            stop = min(start + 1024, NFINAL)
            rows = slice(NTRAIN + start, NTRAIN + stop)
            logits, _ = model(*inputs(data, rows), torch.as_tensor(day_rows[rows], dtype=torch.int64, device="cuda"))
            prediction[start:stop] = logits.sigmoid().cpu().numpy()
    check_probability(prediction, NFINAL)
    np.save(RUN / "final_prediction.npy", prediction)
    del model, optimizer, saved, contexts
    gc.collect()
    torch.cuda.empty_cache()
    emit("full_fit_and_first_inference_complete", seconds=time.perf_counter() - started)


def replay_and_export():
    configure(42)
    protocol = read(RUN / "protocol.json")
    for filename, expected in protocol["source_sha256"].items():
        if digest(filename) != expected:
            raise AssertionError("Full-refit source changed")
    saved = torch.load(RUN / "model.pt", map_location="cpu", weights_only=False)
    for key, filename in [("protocol_sha256", "protocol.json"), ("feature_manifest_sha256", "features.json"),
                          ("context_manifest_sha256", "context_manifest.json"), ("final_context_sha256", "final_context.npz")]:
        if digest(RUN / filename) != saved[key]:
            raise AssertionError("Full refit checkpoint binding differs")
    with np.load(RUN / "final_context.npz") as context:
        contexts = torch.as_tensor(context["values"], device="cuda")
        groups = context["rows"].copy()
        keys = context["keys"].copy()
    arrays = {"categories": compact_array("categories"), "sequences": compact_array("sequences"),
              "numeric": compact_array("gate_oof500")[:, :166]}
    ids = compact_array("ids")
    np.testing.assert_array_equal(keys[groups, 1], ids[:, 1])
    np.testing.assert_array_equal(keys[groups, 0], ids[:, 2])
    model = ProtectedUserDayCTR(saved["meta"], saved["meta"]["train_clicks"] / NTRAIN,
                                contexts, saved["any_prior"], use_context=True).cuda()
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    prediction = np.empty(NFINAL, dtype="float32")
    with torch.no_grad():
        for start in range(0, NFINAL, 1024):
            stop = min(start + 1024, NFINAL)
            logits, _ = model(*inputs(arrays, slice(start, stop)),
                              torch.as_tensor(groups[start:stop], dtype=torch.int64, device="cuda"))
            prediction[start:stop] = logits.sigmoid().cpu().numpy()
    expected = np.load(RUN / "final_prediction.npy")
    error = float(np.max(np.abs(prediction - expected)))
    if error > 2e-7:
        raise AssertionError(f"Fresh model final replay differs: {error}")
    check_probability(prediction, NFINAL)
    v7 = np.load(ORIGINAL / "ensemble_final_prediction.npy").astype("float64")
    old = np.load(ARCH / "final_prediction.npy").astype("float64")
    final = v7 + .30 * (expected.astype("float64") - old)
    check_probability(final, NFINAL)
    raw = pd.read_csv(ROOT / "2026ml/final_test.csv", usecols=["log_id", "user_id", "pt_d"], dtype="int64")
    np.testing.assert_array_equal(raw[["log_id", "user_id", "pt_d"]], ids)
    OUTPUT.mkdir(exist_ok=True)
    submission = OUTPUT / "submission.csv"
    if not submission.exists():
        pd.DataFrame({"id": np.arange(NFINAL), "pctr": final}).to_csv(submission, index=False, float_format="%.10f")
        np.save(OUTPUT / "ensemble_final_prediction.npy", final)
    exported = pd.read_csv(submission, float_precision="round_trip")
    if exported.columns.tolist() != ["id", "pctr"] or len(exported) != NFINAL:
        raise AssertionError("Submission schema/row count differs")
    np.testing.assert_array_equal(exported.id.to_numpy(), np.arange(NFINAL))
    check_probability(exported.pctr.to_numpy(), NFINAL)
    rounding_error = float(np.max(np.abs(exported.pctr.to_numpy() - final)))
    if rounding_error > 5.00001e-11:
        raise AssertionError("Exported probabilities differ")
    if digest(ORIGINAL / "submission.csv") != ORIGINAL_HASH:
        raise AssertionError("Original v7 submission changed")
    verification = {
        "status": "complete_and_verified", "rows": NFINAL, "training_rows": NTRAIN,
        "training_rows_seen": saved["training_rows_seen"], "training_steps": saved["steps"],
        "fresh_model_reloaded": True, "all_final_rows_replayed": True,
        "independent_final_inputs": "Verified pre-existing compact final arrays and saved final-only day context",
        "maximum_reload_probability_error": error, "maximum_csv_rounding_error": rounding_error,
        "original_final_row_order_verified": True, "submission_columns": ["id", "pctr"],
        "id_rule": "Zero-based original final_test.csv row number",
        "duplicate_log_id_rows_retained": int(raw.log_id.duplicated().sum()),
        "all_probabilities_finite_and_in_unit_interval": True,
        "probability_min": float(final.min()), "probability_max": float(final.max()),
        "probability_mean": float(final.mean()), "development": protocol["development"],
        "ensemble_rule": protocol["ensemble_rule"], "final_labels_read": False, "final_auc": None,
        "original_v7_submission_unchanged": True, "original_v7_submission_sha256": ORIGINAL_HASH,
        "submission_sha256": digest(submission), "model_sha256": digest(RUN / "model.pt"),
        "prediction_sha256": digest(RUN / "final_prediction.npy"),
        "protocol_sha256": digest(RUN / "protocol.json"),
    }
    write(RUN / "verification.json", verification)
    write(OUTPUT / "verification.json", verification)
    write(OUTPUT / "manifest.json", {"status": "complete_and_verified", "protocol": protocol,
          "model_path": str(RUN / "model.pt"), "submission_path": str(submission),
          "submission_sha256": digest(submission), "verification": verification})
    print("FINAL_SUBMISSION_COMPLETE " + json.dumps(verification), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["run", "replay", "check-core"], default="run")
    args = parser.parse_args()
    if args.stage == "check-core":
        configure(42)
        core_replay()
        return
    if args.stage == "replay":
        replay_and_export()
        return
    if RUN.exists() or OUTPUT.exists():
        raise FileExistsError("Preserving existing full refit/output")
    protocol = selected_protocol()
    RUN.mkdir(parents=True)
    write(RUN / "protocol.json", protocol)
    snapshot = RUN / "source"
    snapshot.mkdir()
    for filename in protocol["source_sha256"]:
        source = Path(filename)
        target = snapshot / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    configure(42)
    data = prepare()
    core_replay(data)
    fit(data, protocol)
    del data
    gc.collect()
    torch.cuda.empty_cache()
    replay_and_export()


if __name__ == "__main__":
    main()
