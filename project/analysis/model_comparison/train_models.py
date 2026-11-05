"""Train one comparison arm per process; all arms share the same validation rows."""
from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from benchmark_core import CTRModel

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"
SEED = 42


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_arrays(cache_dir=CACHE):
    names = ["categories", "sequences", "numeric", "baseline", "label", "date", "log_id", "train_indices", "valid_indices"]
    return {name: np.load(cache_dir / (name + ".npy"), mmap_mode="r") for name in names}


def metrics(labels, prediction):
    assert np.isfinite(prediction).all() and len(prediction) == len(labels)
    prediction = np.clip(np.asarray(prediction, dtype=np.float64), 1e-8, 1 - 1e-8)
    return {"auc": float(roc_auc_score(labels, prediction)), "logloss": float(log_loss(labels, prediction, labels=[0, 1])),
            "average_precision": float(average_precision_score(labels, prediction)), "mean_prediction": float(prediction.mean())}


def full_catboost_frame(arrays, manifest):
    columns = {}
    for index, name in enumerate(manifest["categorical_fields"]):
        group = manifest["group_for_field"][name]
        values = arrays["categories"][:, index]
        labels = [str(value) for value in range(manifest["vocabulary_sizes"][group])]
        columns[name] = pd.Categorical.from_codes(values, categories=labels)
    print("Building canonical list categories for CatBoost (same unordered tokens as the neural model)", flush=True)
    for index, name in enumerate(manifest["sequence_fields"]):
        sorted_values = np.sort(arrays["sequences"][:, index], axis=1)
        combinations, inverse = np.unique(sorted_values, axis=0, return_inverse=True)
        # This is a deterministic representation of already encoded input, not
        # target encoding. CatBoost never fits on validation labels.
        labels = ["^".join(str(int(token)) for token in row) for row in combinations]
        columns[name] = pd.Categorical.from_codes(inverse.astype(np.int32), categories=labels)
        print(f"  {name}: {len(combinations):,} list patterns", flush=True)
        del sorted_values, inverse, combinations
    for index, name in enumerate(manifest["numeric_fields"]):
        columns[name] = np.asarray(arrays["numeric"][:, index])
    return pd.DataFrame(columns)


def train_catboost(kind, arrays, manifest, output):
    from catboost import CatBoostClassifier, Pool
    train_idx, valid_idx = arrays["train_indices"], arrays["valid_indices"]
    started = time.perf_counter()
    if kind == "catboost_original":
        train_pool = Pool(arrays["baseline"][train_idx], arrays["label"][train_idx], feature_names=manifest["baseline_fields"])
        valid_pool = Pool(arrays["baseline"][valid_idx], arrays["label"][valid_idx], feature_names=manifest["baseline_fields"])
        feature_note = "Notebook's original 11 numeric inputs, including log_id; no source features. Shared benchmark hyperparameters, not an exact replay of defaults."
    else:
        frame = full_catboost_frame(arrays, manifest)
        categorical = manifest["categorical_fields"] + manifest["sequence_fields"]
        train_pool = Pool(frame.iloc[train_idx], arrays["label"][train_idx], cat_features=categorical)
        valid_pool = Pool(frame.iloc[valid_idx], arrays["label"][valid_idx], cat_features=categorical)
        del frame
        gc.collect()
        feature_note = "24 category inputs + 7 canonical unordered-list categories + the same 20 numeric inputs as the neural model."
    preparation_seconds = time.perf_counter() - started
    params = dict(iterations=1000, depth=6, learning_rate=.08, loss_function="Logloss", eval_metric="AUC",
                  random_seed=SEED, thread_count=6, l2_leaf_reg=5, task_type="CPU", allow_writing_files=False,
                  max_ctr_complexity=1, one_hot_max_size=10,
                  used_ram_limit="3gb")
    model = CatBoostClassifier(**params)
    print(f"Training {kind}; pool preparation {preparation_seconds:.1f}s", flush=True)
    started = time.perf_counter()
    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=100, use_best_model=True, verbose=100)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = model.predict_proba(valid_pool)[:, 1]
    inference_seconds = time.perf_counter() - started
    model.save_model(str(output / "model.cbm"))
    write_json(output / "evaluation_history.json", model.get_evals_result())
    importance = pd.DataFrame({"feature": model.feature_names_, "importance": model.get_feature_importance()}).sort_values("importance", ascending=False)
    importance.to_csv(output / "feature_importance.csv", index=False)
    return prediction, {"device": "CPU (6 threads)", "best_iteration_zero_based": int(model.get_best_iteration()),
                        "trees": int(model.tree_count_), "parameters": params, "feature_note": feature_note,
                        "input_preparation_seconds": preparation_seconds, "training_seconds": train_seconds,
                        "inference_seconds": inference_seconds}


def train_neural(arrays, manifest, output, epochs, batch_size, model_builder=None, model_options=None, seed=SEED):
    import torch
    from torch import nn
    torch.set_num_threads(6)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_idx, valid_idx = arrays["train_indices"], arrays["valid_indices"]
    initial_ctr = float(arrays["label"][train_idx].mean())
    builder = model_builder or CTRModel
    options = {"embedding_dim": 16, "hidden_dims": [256, 128, 64], "dropout": .1, "unknown_dropout": .01}
    options.update(model_options or {})
    model = builder(manifest["categorical_fields"], manifest["sequence_fields"], manifest["group_for_field"], manifest["vocabulary_sizes"], initial_ctr=initial_ctr, **options).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=1, min_lr=.00001)
    criterion = nn.BCEWithLogitsLoss()
    parameter_count = sum(value.numel() for value in model.parameters())
    print(f"Training {builder.__name__} on {device}; parameters={parameter_count:,}; batch_size={batch_size}; seed={seed}", flush=True)

    def batch(indices):
        return (torch.as_tensor(arrays["categories"][indices].astype(np.int64), device=device),
                torch.as_tensor(arrays["sequences"][indices].astype(np.int64), device=device),
                torch.as_tensor(np.array(arrays["numeric"][indices]), device=device))

    def predict(indices):
        model.eval()
        prediction = []
        with torch.no_grad():
            for start in range(0, len(indices), 8192):
                prediction.append(torch.sigmoid(model(*batch(indices[start:start + 8192]))).cpu().numpy())
        return np.concatenate(prediction)

    history = []
    best_auc, best_epoch, bad_epochs = -1., 0, 0
    best_prediction = None
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        shuffled = np.random.default_rng(seed + epoch).permutation(train_idx)
        total_loss = 0.
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start:start + batch_size]
            target = torch.as_tensor(arrays["label"][indices].astype(np.float32), device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*batch(indices)), target)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite neural training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
            if (start // batch_size) % 250 == 0:
                print(f"  epoch {epoch}, trained {start:,}/{len(shuffled):,} rows", flush=True)
        prediction = predict(valid_idx)
        result = metrics(arrays["label"][valid_idx], prediction)
        entry = {"epoch": epoch, "train_logloss": total_loss / len(train_idx), **result,
                 "epoch_seconds": time.perf_counter() - epoch_started, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(entry)
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        write_json(output / "live_status.json", entry)
        print(json.dumps(entry), flush=True)
        if result["auc"] > best_auc + .00001:
            best_auc, best_epoch, bad_epochs = result["auc"], epoch, 0
            best_prediction = prediction.copy()
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            torch.save({"state_dict": state, "manifest": manifest, "epoch": epoch, "validation_metrics": result,
                        "model_class": builder.__name__, "model_options": options, "seed": seed}, output / "model.pt")
        else:
            bad_epochs += 1
        scheduler.step(result["auc"])
        if bad_epochs >= 3:
            print(f"Early stopping after epoch {epoch}; best epoch {best_epoch}", flush=True)
            break
    train_seconds = time.perf_counter() - started
    assert best_prediction is not None
    return best_prediction, {"device": str(device) + (" / " + torch.cuda.get_device_name(0) if device.type == "cuda" else ""),
                             "best_epoch": best_epoch, "epochs_completed": len(history), "parameter_count": parameter_count,
                             "training_seconds": train_seconds, "inference_seconds": None,
                             "feature_note": "24 category embeddings + 7 masked-mean pooled lists + 20 shared numeric features; historical labels are source-domain observations only.",
                             "model_class": builder.__name__, "model_options": options,
                             "parameters": {**options, "optimizer": "AdamW", "learning_rate": .001,
                                            "weight_decay": .00001, "batch_size": batch_size, "max_epochs": epochs,
                                            "early_stopping_patience": 3, "loss": "unweighted BCEWithLogitsLoss", "seed": seed}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["catboost_original", "catboost_full", "neural"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    output = RESULTS / args.model
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    arrays = load_arrays()
    if args.model == "neural":
        prediction, metadata = train_neural(arrays, manifest, output, args.epochs, args.batch_size)
    else:
        prediction, metadata = train_catboost(args.model, arrays, manifest, output)
    valid_idx = arrays["valid_indices"]
    result = {"model": args.model, **metrics(arrays["label"][valid_idx], prediction), **metadata,
              "train_rows": manifest["train_rows"], "valid_rows": manifest["valid_rows"],
              "validation_date": "2019-11-16", "validation_clicks": manifest["valid_clicks"], "seed": SEED,
              "python": platform.python_version()}
    np.save(output / "validation_prediction.npy", prediction)
    write_json(output / "metrics.json", result)
    print("FINAL_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
