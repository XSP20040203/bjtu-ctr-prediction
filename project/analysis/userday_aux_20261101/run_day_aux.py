"""Matched strong-backbone context/any-click auxiliary 2x2 experiment."""
from __future__ import annotations

import argparse
import gc
import hashlib
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = ROOT / "analysis/learned_audience_20261101"
sys.path.insert(0, str(ROOT / "analysis/strong_interest_control_20261031"))
from features import prepare, read, write, emit, digest, ARCH
from experiment import configure, state_digest, scores
sys.path.insert(0, str(REFERENCE))
from freeze_reference import freeze
from day_context import build_context, training_targets
from day_model import UserDayCTR, auxiliary_loss
import numpy as np
import pandas as pd
import torch
from torch import nn

MODES = ["base", "context", "aux", "both"]


def batch(data, rows):
    return tuple(torch.as_tensor(np.ascontiguousarray(values), dtype=dtype, device="cuda")
                 for values, dtype in [
                     (data["categories"][rows], torch.int64), (data["sequences"][rows], torch.int64),
                     (data["numeric"][rows], torch.float32), (data["day_rows"][rows], torch.int64)])


def predict(model, data, rows):
    model.eval()
    prediction = np.empty(len(rows), dtype="float32")
    any_prediction = np.empty(len(rows), dtype="float32")
    with torch.no_grad():
        for start in range(0, len(rows), 1024):
            stop = min(start + 1024, len(rows))
            logit, auxiliary = model(*batch(data, rows[start:stop]))
            prediction[start:stop] = logit.sigmoid().cpu().numpy()
            any_prediction[start:stop] = auxiliary.sigmoid().cpu().numpy()
    if not np.isfinite(prediction).all() or not np.isfinite(any_prediction).all():
        raise AssertionError("Nonfinite user-day prediction")
    return prediction, any_prediction


def train(args, data, contexts, out, mode):
    directory = out / mode
    directory.mkdir()
    configure(args.seed)
    model = UserDayCTR(data["meta"], float(data["label"][data["train"]].mean()),
                       contexts, data["any_prior"], use_context=mode in ["context", "both"]).cuda()
    initial_backbone = state_digest(model.backbone.state_dict())
    initial_full = state_digest(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    order = np.random.default_rng(args.seed + 1).permutation(data["train"])
    order_hash = hashlib.sha256(order.tobytes()).hexdigest()
    steps = math.ceil(len(order) / 1024)
    checks = {math.ceil(steps * fraction) for fraction in [.25, .5, .75, 1.]}
    use_auxiliary = mode in ["aux", "both"]
    total, row_total, auxiliary_total, seen = 0., 0., 0., 0
    history, best = [], -1.
    started = time.perf_counter()
    model.train()
    emit("userday_start", day=args.day, mode=mode,
         parameters=sum(value.numel() for value in model.parameters()))
    for step, start in enumerate(range(0, len(order), 1024), 1):
        rows = order[start:start + 1024]
        target = torch.as_tensor(data["label"][rows], dtype=torch.float32, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        logits, auxiliary = model(*batch(data, rows))
        row_loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
        aux_loss = row_loss.new_zeros(())
        if use_auxiliary:
            groups = data["day_rows"][rows]
            group_target = torch.as_tensor(data["any_targets"][groups], dtype=torch.float32, device="cuda")
            group_size = torch.as_tensor(data["day_sizes"][groups], dtype=torch.float32, device="cuda")
            aux_loss = auxiliary_loss(auxiliary, group_target, group_size)
        loss = row_loss + args.aux_weight * aux_loss
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite user-day loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        total += float(loss.detach()) * len(rows)
        row_total += float(row_loss.detach()) * len(rows)
        auxiliary_total += float(aux_loss.detach()) * len(rows)
        seen += len(rows)
        if step not in checks:
            continue
        prediction, any_prediction = predict(model, data, data["valid"])
        record = {"fractional_epoch": step / steps,
                  **scores(data["label"][data["valid"]], prediction),
                  "train_objective": total / seen, "train_row_bce": row_total / seen,
                  "train_auxiliary_bce": auxiliary_total / seen,
                  "seconds": time.perf_counter() - started}
        history.append(record)
        write(directory / "history.json", history)
        emit("userday_progress", day=args.day, mode=mode, **record)
        checkpoint = {
            "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            "meta": data["meta"], "mode": mode, "seed": args.seed,
            "any_prior": data["any_prior"], "context_width": contexts.shape[1],
            "record": record, "context_manifest_sha256": digest(out / "context_manifest.json"),
        }
        if record["auc"] > best:
            best = record["auc"]
            torch.save(checkpoint, directory / "best_model.pt")
            np.save(directory / "best_prediction.npy", prediction)
            np.save(directory / "best_any_prediction.npy", any_prediction)
        if step == steps:
            torch.save(checkpoint, directory / "final_model.pt")
            np.save(directory / "final_prediction.npy", prediction)
            np.save(directory / "final_any_prediction.npy", any_prediction)
        model.train()
    errors = {}
    for endpoint in ["best", "final"]:
        checkpoint = torch.load(directory / f"{endpoint}_model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        replay, any_replay = predict(model, data, data["valid"])
        expected = np.load(directory / f"{endpoint}_prediction.npy")
        expected_any = np.load(directory / f"{endpoint}_any_prediction.npy")
        errors[endpoint] = float(np.max(np.abs(replay - expected)))
        errors[endpoint + "_auxiliary"] = float(np.max(np.abs(any_replay - expected_any)))
        if max(errors.values()) > 2e-7:
            raise AssertionError("User-day checkpoint replay differs")
    suffix = args.day % 100
    y = data["label"][data["valid"]]
    selected = np.load(directory / "best_prediction.npy").astype("float64")
    reference = np.load(REFERENCE / f"reference/day{suffix}_prediction.npy")
    v7 = np.load(ARCH / f"fixed_gated20_allocated10_core30_day{suffix}.npy").astype("float64")
    old_core = np.load(ARCH / f"results/gated166_day{suffix}/validation_prediction.npy")
    known = np.isin(data["task_id"][data["valid"]], np.unique(data["task_id"][data["train"]]))
    result = {
        "status": "complete_and_verified", "day": args.day, "mode": mode,
        "fixed_endpoint": history[-1], "best_checkpoint": max(history, key=lambda row: row["auc"]),
        "initial_backbone_sha256": initial_backbone, "initial_full_weights_sha256": initial_full,
        "row_order_sha256": order_hash, "reload_errors": errors,
        "v7_replace30": scores(y, v7 + .3 * (selected - old_core)),
        "candidate_blend10": scores(y, .9 * reference + .1 * selected),
        "slices": {name: scores(y[mask], selected[mask]) for name, mask in [("seen", known), ("cold", ~known)]},
        "seconds": time.perf_counter() - started,
        "files_sha256": {name: digest(directory / name) for name in [
            "best_model.pt", "final_model.pt", "best_prediction.npy", "final_prediction.npy",
            "best_any_prediction.npy", "final_any_prediction.npy"]},
    }
    np.save(directory / "valid_indices.npy", data["global_rows"][data["valid"]])
    write(directory / "metrics.json", result)
    emit("userday_complete", day=args.day, mode=mode, fixed=result["fixed_endpoint"]["auc"],
         best=best, replace30=result["v7_replace30"]["auc"], blend10=result["candidate_blend10"]["auc"])
    del model, optimizer, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(args):
    freeze()
    out = HERE / "runs" / f"{args.tag}_day{args.day % 100}_seed{args.seed}"
    if out.exists():
        raise FileExistsError("Preserving user-day run")
    out.mkdir(parents=True)
    configure(args.seed)
    write(out / "protocol.json", {
        "day": args.day, "seed": args.seed, "modes": args.modes,
        "backbone": "Original GatedCross166; one original forward per batch",
        "epochs": 1, "batch_size": 1024, "learning_rate": .001,
        "aux_weight": args.aux_weight, "context_delta_cap": 2.,
        "loss": "Row BCE plus optional inverse-group-size weighted training any-click BCE",
        "context": "Input-only daily mean/max, categorical distributions and strictly prior three-day exposures",
        "excluded_from_pooling": "Every target_stat__ field; no row predictions pooled",
        "auxiliary_targets": "Only outer-training advertisement labels; missing validation targets remain -1",
        "metric": "Original full advertisement-row AUC; user-day AUC is diagnostic only",
        "primary": "Fixed full epoch; same four equal checks are secondary",
        "sources": {str(HERE / name): digest(HERE / name) for name in [
            "run_day_aux.py", "day_context.py", "day_model.py", "test_day_context.py"]},
        "final_test_read": False, "large_row_feature_matrices_saved": False,
    })
    data = prepare(args.day, out)
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["pt_d", "user_id", "task_id"], dtype="int32")
    raw = raw.iloc[data["global_rows"]].reset_index(drop=True)
    store = build_context(raw, data["categories"], data["numeric"],
                           data["feature_manifest"]["numeric_fields"], data["meta"], args.day)
    if len(store["manifest"]["excluded_target_stat_columns"]) != 36:
        raise AssertionError("Original target-stat exclusion contract changed")
    targets, prior = training_targets(store["keys"], store["rows"], data["label"], data["train"], args.day)
    if not np.all(targets[store["rows"][data["valid"]]] == -1):
        raise AssertionError("Validation target entered auxiliary target table")
    write(out / "context_manifest.json", store["manifest"])
    np.savez_compressed(out / "context.npz", keys=store["keys"], sizes=store["sizes"], values=store["values"])
    np.save(out / "training_any_targets.npy", targets)
    write(out / "auxiliary_manifest.json", {"training_groups": int((targets >= 0).sum()),
                                            "prior": prior, "validation_target_sentinel": -1,
                                            "targets_in_model_inputs": False})
    data["day_rows"], data["day_sizes"] = store["rows"], store["sizes"]
    data["any_targets"], data["any_prior"] = targets, prior
    contexts = torch.as_tensor(store["values"], device="cuda")
    emit("userday_context_ready", day=args.day, groups=len(store["keys"]),
         width=store["values"].shape[1], excluded_target_columns=36, training_any_prior=prior)
    del raw, store
    gc.collect()
    records = [train(args, data, contexts, out, mode) for mode in args.modes]
    for name in ["initial_backbone_sha256", "initial_full_weights_sha256", "row_order_sha256"]:
        if len({record[name] for record in records}) != 1:
            raise AssertionError("User-day matched arms differ: " + name)
    write(out / "comparison.json", {"status": "complete_and_verified", "records": records})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, choices=[20191115, 20191116], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="day_aux_v1")
    parser.add_argument("--aux-weight", type=float, default=.02)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    args = parser.parse_args()
    if args.aux_weight < 0:
        raise ValueError("Nonnegative auxiliary loss weight required")
    run(args)
