"""Train one frozen ID-regularization candidate with quarter-epoch validation."""
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

from id_regularized_models import IDRegularizedDCN
from train_models import load_arrays, metrics, write_json

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_id_regularization"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--cache-dir", type=Path, default=HERE / "cache")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    protocol_path = HERE / "id_regularization_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rates = protocol["candidates"][args.variant]
    manifest = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    day = manifest["validation_day"]
    output = OUT / str(day) / args.variant / f"seed{args.seed}"
    if (output / "model.pt").exists():
        raise FileExistsError(f"Preserving existing experiment {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays = load_arrays(args.cache_dir)
    train, valid = arrays["train_indices"], arrays["valid_indices"]
    y_valid = arrays["label"][valid]
    assert arrays["date"][train].max() < day and np.all(arrays["date"][valid] == day)
    options = {"embedding_dim": 8, "hidden_dims": [128, 64], "dropout": .1,
               "unknown_dropout": .01, "cross_layers": 2, "group_dropout_rates": rates}
    model = IDRegularizedDCN(manifest["categorical_fields"], manifest["sequence_fields"], manifest["group_for_field"],
                             manifest["vocabulary_sizes"], initial_ctr=float(arrays["label"][train].mean()), **options).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=1, min_lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    batch_size = protocol["batch_size"]
    batches = math.ceil(len(train) / batch_size)
    eval_steps = {math.ceil(batches * fraction) for fraction in [.25, .5, .75, 1.]}
    new_tasks = ~np.isin(arrays["baseline"][valid, 8], np.unique(arrays["baseline"][train, 8]))
    user_column = manifest["categorical_fields"].index("user_id")
    user_clicks = np.bincount(arrays["categories"][train, user_column], weights=arrays["label"][train],
                              minlength=manifest["vocabulary_sizes"]["user_id"])
    cold_users = user_clicks[arrays["categories"][valid, user_column]] == 0
    write_json(output / "provenance.json", {"cache_dir": str(args.cache_dir.resolve()),
               "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [protocol_path, HERE / "id_regularized_models.py", HERE / "mainstream_models.py",
                                       HERE / "run_id_regularization.py", args.cache_dir / "manifest.json",
                                       args.cache_dir / "train_indices.npy", args.cache_dir / "valid_indices.npy"]}})

    def batch(indices):
        return (torch.as_tensor(arrays["categories"][indices].astype(np.int64), device=device),
                torch.as_tensor(arrays["sequences"][indices].astype(np.int64), device=device),
                torch.as_tensor(np.array(arrays["numeric"][indices]), device=device))

    def predict():
        model.eval()
        with torch.no_grad():
            prediction = np.concatenate([torch.sigmoid(model(*batch(valid[start:start + 8192]))).cpu().numpy()
                                         for start in range(0, len(valid), 8192)])
        model.train()
        return prediction

    best_auc, bad_checkpoints = -1., 0
    history = []
    started = time.perf_counter()
    print(f"START_ID_REGULARIZATION {args.variant} day={day} seed={args.seed} rates={rates}", flush=True)
    stop = False
    best_record = None
    for epoch in range(1, protocol["max_epochs"] + 1):
        shuffled = np.random.default_rng(args.seed + epoch).permutation(train)
        model.train()
        running_loss, seen = 0., 0
        for step, start in enumerate(range(0, len(shuffled), batch_size), start=1):
            indices = shuffled[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            target = torch.as_tensor(arrays["label"][indices].astype(np.float32), device=device)
            loss = criterion(model(*batch(indices)), target)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite optimization loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            running_loss += float(loss.detach()) * len(indices)
            seen += len(indices)
            if step not in eval_steps:
                continue
            prediction = predict()
            result = metrics(y_valid, prediction)
            entry = {"fractional_epoch": epoch - 1 + step / batches, "epoch": epoch, **result,
                     "online_train_logloss": running_loss / seen,
                     "new_task_auc": metrics(y_valid[new_tasks], prediction[new_tasks])["auc"],
                     "zero_train_click_user_mean_prediction": float(prediction[cold_users].mean()),
                     "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.perf_counter() - started}
            history.append(entry)
            pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
            print(json.dumps({"variant": args.variant, "day": day, "seed": args.seed, **entry}), flush=True)
            if result["auc"] > best_auc + 1e-5:
                best_auc, bad_checkpoints = result["auc"], 0
                state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_record = {"variant": args.variant, "model_class": "IDRegularizedDCN", "seed": args.seed,
                               "validation_day": day, "cache_dir": str(args.cache_dir.resolve()),
                               "model_options": options, "best_fractional_epoch": entry["fractional_epoch"],
                               "train_rows": len(train), "valid_rows": len(valid), "valid_clicks": int(y_valid.sum()),
                               "parameter_count": sum(p.numel() for p in model.parameters()),
                               "new_task_metrics": metrics(y_valid[new_tasks], prediction[new_tasks]),
                               "zero_train_click_user_mean_prediction": entry["zero_train_click_user_mean_prediction"],
                               "zero_train_click_user_actual_ctr": float(y_valid[cold_users].mean()), **result}
                torch.save({"state_dict": state, "manifest": manifest, "model_options": options,
                            "model_class": "IDRegularizedDCN", "seed": args.seed,
                            "fractional_epoch": entry["fractional_epoch"], "validation_metrics": result}, output / "model.pt")
                np.save(output / "validation_prediction.npy", prediction)
                write_json(output / "best_so_far.json", best_record)
            else:
                bad_checkpoints += 1
            if bad_checkpoints >= 4:
                stop = True
                break
        scheduler.step(history[-1]["auc"])
        if stop:
            break
    best_record.update(training_seconds=time.perf_counter() - started, checkpoints_completed=len(history),
                       final_fractional_epoch=history[-1]["fractional_epoch"], final_auc=history[-1]["auc"],
                       first_epoch_auc=next((row["auc"] for row in history if row["fractional_epoch"] == 1), None),
                       second_epoch_auc=next((row["auc"] for row in history if row["fractional_epoch"] == 2), None))
    write_json(output / "metrics.json", best_record)
    print("FINAL_ID_REGULARIZATION " + json.dumps(best_record), flush=True)


if __name__ == "__main__":
    main()
