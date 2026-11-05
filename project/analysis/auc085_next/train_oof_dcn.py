"""Train expanded offline grouped-OOF DCN (or optional CPU numeric CatBoost).

Example (training is explicit, never triggered by preparing statistics)::
    python train_oof_dcn.py --day 15 --tag goal085_day15_oof60 --stats oof_day15.npy

DCN: strict numeric[:39] + NEW OOF statistics + exposure43 + snapshot48,
optionally source_context80. The old numeric[39:75] is never accessed.
Tree: strict numeric[:39] + RAW new OOF statistics, no category CTR inputs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from prepare_oof_stats import (DAILY, HERE, PROTOCOL, array_digest, base_cache,
                               feature_names, normalize_day, sha256)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def materialize_numeric(output_path, strict, train, statistics, stat_names, extras=(), standardize=True,
                        chunk_rows=16384):
    """Only bounded row blocks and one train column are copied, never all numeric.

    extras is [(array, names), ...]. Original columns 39:75 are structurally
    unreachable here. Existing first39 values/scalers are retained as supplied.
    """
    sources = [(statistics, stat_names)] + list(extras)
    n = len(strict)
    if strict.shape[1] < 39 or any(len(array) != n or array.shape[1] != len(names) for array, names in sources):
        raise ValueError("Dense blocks do not align")
    width = 39 + sum(len(names) for _, names in sources)
    scalers = {}
    transforms = []
    for source, names in sources:
        means, stds = np.zeros(len(names), dtype=np.float64), np.ones(len(names), dtype=np.float64)
        for column, name in enumerate(names):
            if standardize:
                values = np.asarray(source[train, column], dtype=np.float64)
                means[column], stds[column] = float(values.mean()), max(float(values.std()), 1e-6)
                if not np.isfinite(means[column]) or not np.isfinite(stds[column]):
                    raise ValueError("Nonfinite training scaler: " + name)
                scalers[name] = {"mean": means[column], "std": stds[column]}
        transforms.append((means, stds))
    numeric = np.lib.format.open_memmap(output_path, mode="w+", dtype="float32", shape=(n, width))
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        numeric[start:end, :39] = strict[start:end, :39]
        offset = 39
        for (source, names), (means, stds) in zip(sources, transforms):
            block = np.asarray(source[start:end], dtype=np.float32)
            numeric[start:end, offset:offset + len(names)] = (block - means) / stds
            offset += len(names)
        if not np.isfinite(numeric[start:end]).all():
            raise ValueError("Nonfinite assembled numeric features")
    numeric.flush()
    return numeric, scalers


def original_first39_scalers(manifest):
    # Original binary indicators are retained verbatim and intentionally have
    # no scaler entry in historical manifests. Identity is their fitted map.
    return {name: manifest.get("numeric_scaler", {}).get(name, {"mean": 0., "std": 1.})
            for name in manifest["numeric_fields"][:39]}


def validate_statistics(path, day, dates, train, labels):
    meta = read(path.with_suffix(".json"))
    required = {"protocol": PROTOCOL, "status": "complete", "validation_day": day,
                "advertisement_validation_labels_used": False,
                "advertisement_validation_labels_loaded": False,
                "train_oof_allowed_across_train_dates": True, "per_row_strictly_historical": False,
                "duplicate_group_fold_check": "passed", "raw_predictor_field_count": 33, "fold_count": 5}
    for key, value in required.items():
        if meta.get(key) != value:
            raise ValueError("Stats protocol mismatch: " + key)
    if meta["features"] != feature_names():
        raise ValueError("Expanded OOF schema mismatch")
    checks = {"date_sha256": array_digest(np.asarray(dates)), "train_indices_sha256": array_digest(np.asarray(train, dtype=np.int64)),
              "train_label_sha256": array_digest(labels[train])}
    if any(meta.get(key) != value for key, value in checks.items()):
        raise ValueError("OOF cache does not match the dates, split or training labels")
    fold_path = path.with_suffix(".folds.npy")
    if sha256(fold_path) != meta["fold_sha256"]:
        raise ValueError("OOF fold cache checksum mismatch")
    stats = np.load(path, mmap_mode="r")
    if list(stats.shape) != meta["shape"] or len(stats) != len(dates) or stats.dtype != np.float32:
        raise ValueError("OOF array shape/dtype mismatch")
    return stats, meta


def train_dcn(args, output, base, manifest, numeric, names, protocol, scalers):
    import torch
    from torch import nn
    sys.path.insert(0, str(HERE.parent / "model_comparison"))
    from bounded_id_models import BoundedIDDCN
    sys.path.insert(0, str(HERE.parent / "user_personalization"))
    from evaluation import metrics
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    options = {"numeric_dim": len(names), "embedding_dim": 8, "hidden_dims": [128, 64],
               "dropout": .1, "unknown_dropout": .01, "cross_layers": 2,
               "group_norm_caps": {"user_id": .2, "ad_task": .2, "adv_id": .2}}
    train, valid = base["train_indices"], base["valid_indices"]
    prior = float(base["label"][train].mean())
    weighted_prior = args.pos_weight * prior / (1 - prior + args.pos_weight * prior)

    def new_model(initial_ctr):
        return BoundedIDDCN(manifest["categorical_fields"], manifest["sequence_fields"],
                            manifest["group_for_field"], manifest["vocabulary_sizes"],
                            initial_ctr=initial_ctr, **options).to(device)

    model = new_model(weighted_prior)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    positive_weight = torch.tensor(args.pos_weight, device=device)
    # Validation labels are first materialized here, AFTER feature assembly.
    y = base["label"][valid]
    batches = math.ceil(len(train) / args.batch_size)
    evalsteps = {math.ceil(batches * fraction) for fraction in (.25, .5, .75, 1.)}

    def inputs(rows):
        return (torch.as_tensor(base["categories"][rows].astype(np.int64), device=device),
                torch.as_tensor(base["sequences"][rows].astype(np.int64), device=device),
                torch.as_tensor(np.asarray(numeric[rows]), device=device))

    def predict():
        model.eval()
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(model(*inputs(valid[start:start + 4096])) - math.log(args.pos_weight))
                                   .cpu().numpy() for start in range(0, len(valid), 4096)])

    best, bad, history, record, stop = -1., 0, [], None, False
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = np.random.default_rng(args.seed + epoch).permutation(train)
        model.train()
        total, seen = 0., 0
        for step, start in enumerate(range(0, len(order), args.batch_size), 1):
            rows = order[start:start + args.batch_size]
            target = torch.as_tensor(base["label"][rows].astype(np.float32), device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(*inputs(rows)), target, pos_weight=positive_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite OOF DCN loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
            seen += len(rows)
            if step not in evalsteps:
                continue
            prediction = predict()
            score = metrics(y, prediction)
            entry = {"fractional_epoch": epoch - 1 + step / batches, "online_train_loss": total / seen, **score}
            history.append(entry)
            pd.DataFrame(history).to_csv(output / "history.csv", index=False)
            print(json.dumps({"tag": args.tag, **entry}), flush=True)
            if score["auc"] is None:
                raise ValueError("Both validation classes are required for AUC")
            if score["auc"] > best + 1e-5:
                best, bad = score["auc"], 0
                torch.save({"state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                            "manifest": manifest, "options": options, "numeric_fields": names, "scalers": scalers,
                            "protocol": protocol, "day": args.day, "pos_weight": args.pos_weight,
                            "seed": args.seed, "initial_ctr": weighted_prior}, output / "model.pt")
                np.save(output / "validation_prediction.npy", prediction)
                record = {**score, "best_fractional_epoch": entry["fractional_epoch"]}
            else:
                bad += 1
            model.train()
            if bad >= args.patience:
                stop = True
                break
        if stop:
            break
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=False)
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Fresh model reconstructs checkpoint architecture and verifies serialized state.
    model = new_model(checkpoint["initial_ctr"])
    model.load_state_dict(checkpoint["state_dict"])
    replay = predict()
    saved = np.load(output / "validation_prediction.npy")
    error = float(np.max(np.abs(replay - saved)))
    if error > 2e-7:
        raise AssertionError("DCN model reload prediction mismatch: " + str(error))
    record.update(maximum_reload_error=error, options=options, device=str(device),
                  training_seconds=time.perf_counter() - started, epochs_completed=epoch,
                  evaluations=len(history), model_sha256=sha256(output / "model.pt"))
    return record


def train_tree(args, output, base, numeric, names):
    from catboost import CatBoostClassifier, Pool
    sys.path.insert(0, str(HERE.parent / "user_personalization"))
    from evaluation import metrics
    train, valid = base["train_indices"], base["valid_indices"]
    # 39 + 121 dimensions only; indexed copies are disk maps written in blocks.
    pools = []
    for split, rows in (("train", train), ("valid", valid)):
        mapped = np.lib.format.open_memmap(output / (split + "_tree.npy"), mode="w+", dtype="float32",
                                          shape=(len(rows), len(names)))
        for start in range(0, len(rows), 16384):
            mapped[start:start + 16384] = numeric[rows[start:start + 16384]]
        mapped.flush()
        pools.append(Pool(mapped, base["label"][rows], feature_names=names))
        del mapped
    params = dict(iterations=args.tree_iterations, depth=7, learning_rate=.06, loss_function="Logloss", eval_metric="AUC",
                  random_seed=args.seed, thread_count=args.threads, l2_leaf_reg=10, task_type="CPU",
                  allow_writing_files=False, used_ram_limit="3gb")
    model = CatBoostClassifier(**params)
    started = time.perf_counter()
    model.fit(pools[0], eval_set=pools[1], early_stopping_rounds=100, use_best_model=True, verbose=100)
    prediction = model.predict_proba(pools[1])[:, 1]
    model.save_model(str(output / "model.cbm"))
    write(output / "history.json", model.get_evals_result())
    np.save(output / "validation_prediction.npy", prediction)
    loaded = CatBoostClassifier()
    loaded.load_model(str(output / "model.cbm"))
    replay = loaded.predict_proba(pools[1])[:, 1]
    error = float(np.max(np.abs(replay - prediction)))
    if error > 1e-12:
        raise AssertionError("CatBoost model reload mismatch")
    return {**metrics(base["label"][valid], prediction), "maximum_reload_error": error,
            "training_seconds": time.perf_counter() - started, "parameters": params,
            "best_iteration_zero_based": int(model.get_best_iteration()),
            "model_sha256": sha256(output / "model.cbm")}


def main(args):
    started = time.perf_counter()
    args.day = normalize_day(args.day)
    if not args.tag or Path(args.tag).name != args.tag or args.tag in (".", ".."):
        raise ValueError("--tag must be a single directory name")
    if not math.isfinite(args.pos_weight) or args.pos_weight <= 0:
        raise ValueError("--pos-weight must be finite and positive")
    output = DAILY / "dense_results" / args.tag
    if output.exists():
        raise FileExistsError("Preserving existing dense result")
    cache = base_cache(args.day)
    manifest = read(cache / "manifest.json")
    # Avoid load_arrays' irrelevant baseline/numeric maps; reuse categories and sequences.
    base = {name: np.load(cache / (name + ".npy"), mmap_mode="r") for name in
            ("categories", "sequences", "label", "date", "train_indices", "valid_indices", "log_id")}
    dates, train, valid = base["date"], base["train_indices"], base["valid_indices"]
    if not np.array_equal(train, np.flatnonzero(dates < args.day)) or not np.array_equal(valid, np.flatnonzero(dates == args.day)):
        raise ValueError("Whole-date outer split required")
    statistics, meta = validate_statistics(args.stats.resolve(), args.day, dates, train, base["label"])
    strict = np.load(DAILY / f"cache_day{args.day % 100}" / "numeric.npy", mmap_mode="r")
    names = manifest["numeric_fields"][:39] + meta["features"]
    extras, extra_files = [], []
    if args.model == "dcn":
        extra_files = [DAILY / "exposure_features.npy", DAILY / "snapshot_features.npy"]
        if args.source_context:
            extra_files.append(HERE / "source_context.npy")
        for path in extra_files:
            extra_meta = read(path.with_suffix(".json"))
            if extra_meta.get("advertisement_labels_loaded") is not False:
                raise ValueError("Dense extras must not load advertisement labels")
            array = np.load(path, mmap_mode="r")
            extras.append((array, extra_meta["features"]))
            names += extra_meta["features"]
    elif args.source_context:
        raise ValueError("--source-context is only supported for DCN")
    output.mkdir(parents=True)
    numeric, scalers = materialize_numeric(output / "numeric.npy", strict, train, statistics,
                                           meta["features"], extras, standardize=args.model == "dcn")
    protocol = {"protocol": PROTOCOL, "validation_day": args.day, "model": args.model,
                "outer_split": meta["outer_split"], "stats": str(args.stats.resolve()),
                "stats_metadata_sha256": sha256(args.stats.with_suffix(".json")),
                "stats_source_sha256": meta["source_sha256"], "cache": str(cache.resolve()),
                "strict_numeric": str((DAILY / f"cache_day{args.day % 100}" / "numeric.npy").resolve()),
                "old_target_statistics_columns_used": False, "old_target_statistics_replaced": "all original 39:75",
                "retained_original_numeric_slice": [0, 39], "extra_files": [str(path) for path in extra_files],
                "numeric_fields": names, "numeric_dim": len(names), "scalers": scalers,
                "original_first39_scalers": original_first39_scalers(manifest),
                "scaler_fit_rows": "outer training rows only", "train_oof_allowed_across_train_dates": True,
                "per_row_strictly_historical": False, "advertisement_validation_labels_used_in_features_or_optimizer_fit": False,
                "validation_labels_usage": "Metrics and early stopping/model selection only, after numeric preparation",
                "train_rows": len(train), "valid_rows": len(valid), "seed": args.seed,
                "max_epochs": args.epochs if args.model == "dcn" else None, "evaluation_fraction": .25,
                "patience": args.patience, "learning_rate": args.learning_rate, "pos_weight": args.pos_weight,
                "numeric_preparation_seconds": time.perf_counter() - started, "trainer_source_sha256": sha256(Path(__file__))}
    write(output / "protocol.json", protocol)
    record = (train_dcn(args, output, base, manifest, numeric, names, protocol, scalers) if args.model == "dcn"
              else train_tree(args, output, base, numeric, names))
    np.save(output / "validation_indices.npy", valid)
    np.save(output / "validation_log_id.npy", base["log_id"][valid])
    record.update(status="complete_and_verified", day=args.day, tag=args.tag, seed=args.seed,
                  model=args.model, numeric_dim=len(names), train_rows=len(train), valid_rows=len(valid),
                  protocol=PROTOCOL, protocol_file="protocol.json", seconds=time.perf_counter() - started,
                  final_labels_used=False)
    write(output / "metrics.json", record)
    print("OOF_TRAIN_COMPLETE " + json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", type=normalize_day, choices=(20191115, 20191116), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--model", choices=("dcn", "tree"), default="dcn")
    parser.add_argument("--source-context", action="store_true")
    parser.add_argument("--epochs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--pos-weight", type=float, default=1.)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--tree-iterations", type=int, default=1200)
    main(parser.parse_args())
