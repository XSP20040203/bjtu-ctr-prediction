"""Two-epoch controlled diagnosis with quarter-epoch ranking measurements."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from mainstream_models import mainstream_definition
from train_models import load_arrays, metrics, write_json

HERE = Path(__file__).resolve().parent
OUT = HERE / "auc_diagnosis"


def positive_pair_ranks(labels, prediction):
    """For each positive, fraction of all negatives it outranks (ties half)."""
    labels = np.asarray(labels)
    prediction = np.asarray(prediction)
    negatives = np.sort(prediction[labels == 0])
    positives = prediction[labels == 1]
    left = np.searchsorted(negatives, positives, side="left")
    right = np.searchsorted(negatives, positives, side="right")
    return (left + .5 * (right - left)) / len(negatives)


def prediction_summary(labels, prediction):
    result = metrics(labels, prediction)
    probability = np.clip(prediction.astype(np.float64), 1e-8, 1 - 1e-8)
    positive = labels == 1
    result.update(positive_logloss=float(-np.log(probability[positive]).mean()),
                  negative_logloss=float(-np.log1p(-probability[~positive]).mean()),
                  positive_mean_probability=float(probability[positive].mean()),
                  negative_mean_probability=float(probability[~positive].mean()))
    return result


def category_batch(categories, indices, user_column=None):
    values = categories[indices].astype(np.int64)
    if user_column is not None:
        values[:, user_column] = 1
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=HERE / "cache")
    parser.add_argument("--mask-user-id", action="store_true")
    args = parser.parse_args()
    tag = f"lr{args.learning_rate:g}".replace(".", "p")
    if args.mask_user_id:
        tag += "_no_user_id"
    output = OUT / tag
    output.mkdir(parents=True, exist_ok=True)
    if (output / "history.csv").exists():
        raise FileExistsError(f"Preserving diagnosis: {output}")
    torch.set_num_threads(6)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    arrays = load_arrays(args.cache_dir)
    train, valid = arrays["train_indices"], arrays["valid_indices"]
    assert arrays["date"][train].max() < manifest["validation_day"]
    assert np.all(arrays["date"][valid] == manifest["validation_day"])
    sample = np.sort(np.random.default_rng(20261027).choice(train, size=min(200000, len(train)), replace=False))
    np.save(output / "train_probe_indices.npy", sample)
    builder, options = mainstream_definition("dcnv2")
    model = builder(manifest["categorical_fields"], manifest["sequence_fields"],
                    manifest["group_for_field"], manifest["vocabulary_sizes"],
                    initial_ctr=float(arrays["label"][train].mean()), **options).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=1, min_lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    y_valid, y_probe = arrays["label"][valid], arrays["label"][sample]
    masks = {}
    user_column = manifest["categorical_fields"].index("user_id") if args.mask_user_id else None
    for name in ["user_id", "task_id", "adv_id"]:
        column = manifest["categorical_fields"].index(name)
        size = manifest["vocabulary_sizes"][manifest["group_for_field"][name]]
        positive_count = np.bincount(arrays["categories"][train, column],
                                     weights=arrays["label"][train], minlength=size)
        counts = positive_count[arrays["categories"][valid, column]]
        masks[name + "__zero_train_click"] = counts == 0
        masks[name + "__has_train_click"] = counts > 0

    def batch(indices):
        return (torch.as_tensor(category_batch(arrays["categories"], indices, user_column), device=device),
                torch.as_tensor(arrays["sequences"][indices].astype(np.int64), device=device),
                torch.as_tensor(np.array(arrays["numeric"][indices]), device=device))

    def predict(indices):
        model.eval()
        with torch.no_grad():
            result = np.concatenate([torch.sigmoid(model(*batch(indices[start:start + 8192]))).cpu().numpy()
                                     for start in range(0, len(indices), 8192)])
        model.train()
        return result

    history, slices = [], []
    best_auc = -1.
    started = time.perf_counter()
    batch_size = 2048
    batches = math.ceil(len(train) / batch_size)
    eval_steps = {math.ceil(batches * fraction) for fraction in [.25, .5, .75, 1.]}
    protocol = {"seed": 42, "model": "DCNV2Model", "learning_rate": args.learning_rate,
                "mask_user_id": args.mask_user_id,
                "epochs": args.epochs, "batch_size": batch_size, "train_probe_rows": len(sample),
                "train_probe_note": "Fixed sample drawn from optimization training data; not an independent holdout.",
                "validation_day": manifest["validation_day"], "model_options": options,
                "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in [HERE / "mainstream_models.py", HERE / "train_models.py", args.cache_dir / "manifest.json"]},
                "single_variable_comparison": ("Direct user_id input is replaced by UNK in both training and prediction; other features, initialization and settings are fixed. Selected after observing the diagnostic user slices." if args.mask_user_id else "Only initial learning rate changes; architecture, row shuffle, features and seed held fixed."),
                "scope": "Exploratory validation-selected checkpoints, not a new independent final-test score."}
    write_json(output / "protocol.json", protocol)
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = np.random.default_rng(42 + epoch).permutation(train)
        running_loss = 0.
        seen = 0
        for step, start in enumerate(range(0, len(shuffled), batch_size), start=1):
            indices = shuffled[start:start + batch_size]
            target = torch.as_tensor(arrays["label"][indices].astype(np.float32), device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*batch(indices)), target)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite diagnosis loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            running_loss += float(loss.detach()) * len(indices)
            seen += len(indices)
            if step not in eval_steps:
                continue
            fraction = epoch - 1 + step / batches
            checkpoint_name = f"epoch{fraction:.2f}"
            valid_prediction = predict(valid)
            probe_prediction = predict(sample)
            valid_result = prediction_summary(y_valid, valid_prediction)
            probe_result = prediction_summary(y_probe, probe_prediction)
            pair_ranks = positive_pair_ranks(y_valid, valid_prediction)
            assert abs(pair_ranks.mean() - valid_result["auc"]) < 1e-10
            row = {"fractional_epoch": fraction, "epoch": epoch, "step_in_epoch": step,
                   "online_train_logloss_so_far": running_loss / seen,
                   **{"valid_" + key: value for key, value in valid_result.items()},
                   **{"train_probe_" + key: value for key, value in probe_result.items()},
                   "elapsed_seconds": time.perf_counter() - started}
            for name, table in model.encoder.embeddings.items():
                if name in ["user_id", "ad_task", "adv_id"]:
                    row[name + "_embedding_mean_l2"] = float(table.weight[2:].norm(dim=-1).mean().detach())
            history.append(row)
            pd.DataFrame(history).to_csv(output / "history.csv", index=False)
            np.save(output / (checkpoint_name + "_valid_prediction.npy"), valid_prediction)
            np.save(output / (checkpoint_name + "_train_probe_prediction.npy"), probe_prediction)
            for name, mask in masks.items():
                positive_mask = mask[y_valid == 1]
                sliced = prediction_summary(y_valid[mask], valid_prediction[mask])
                slices.append({"fractional_epoch": fraction, "slice": name, "rows": int(mask.sum()),
                               "clicks": int(y_valid[mask].sum()), "actual_ctr": float(y_valid[mask].mean()),
                               "positive_pair_rank_vs_all_negatives": float(pair_ranks[positive_mask].mean()),
                               "positive_weight": float(positive_mask.mean()), **sliced})
            pd.DataFrame(slices).to_csv(output / "slices.csv", index=False)
            if valid_result["auc"] > best_auc:
                best_auc = valid_result["auc"]
                state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                torch.save({"state_dict": state, "manifest": manifest, "model_class": builder.__name__,
                            "model_options": options, "fractional_epoch": fraction, "validation_metrics": valid_result,
                            "seed": 42, "learning_rate": args.learning_rate}, output / "best_model.pt")
                write_json(output / "best_metrics.json", {**valid_result, "fractional_epoch": fraction,
                                                            "learning_rate": args.learning_rate,
                                                            "mask_user_id": args.mask_user_id})
            print(json.dumps({key: row[key] for key in ["fractional_epoch", "valid_auc", "valid_logloss",
                                                       "train_probe_auc", "train_probe_logloss"]}), flush=True)
        scheduler.step(history[-1]["valid_auc"])
    if args.learning_rate == .001 and not args.mask_user_id:
        original = pd.read_csv(HERE / "results_mainstream/20191116/dcnv2/seed42/training_history.csv")
        for epoch in range(1, args.epochs + 1):
            reproduced = next(row for row in history if row["fractional_epoch"] == epoch)
            assert abs(reproduced["valid_auc"] - original.iloc[epoch - 1]["auc"]) < 1e-7
        print("Original two-epoch AUC trajectory reproduced.", flush=True)
    write_json(output / "completion.json", {"status": "complete", "measurements": len(history),
                                              "training_seconds": time.perf_counter() - started, "best_auc": best_auc})


if __name__ == "__main__":
    main()
