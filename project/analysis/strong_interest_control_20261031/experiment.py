"""Replay and compare matched original-protocol GatedCross interest controls."""
from __future__ import annotations

import argparse
import gc
import hashlib
import math
from pathlib import Path
import time

from features import HERE, ROOT, ARCH, prepare, read, write, emit, digest
from model import build, NEWS_FIELDS, QUERY_FIELDS
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score

SETTINGS = {
    "seed": 42, "epochs": 1, "batch_size": 1024, "learning_rate": .001,
    "weight_decay": .0001, "max_grad_norm": 5.,
    "checks": [.25, .5, .75, 1.], "numeric_width": 166,
    "primary": "Fixed complete one-epoch matched comparison",
    "secondary": "Best of the same four quarter-epoch checks, matching old selection",
    "promotion_min_auc_gain_both_days": .0005,
    "promotion_max_logloss_worsening_both_days": .0001,
    "confirmation_seed": 2026,
}


def state_digest(state):
    result = hashlib.sha256()
    for name, value in state.items():
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def configure(seed):
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.42)


def batch(data, rows):
    return tuple(torch.as_tensor(np.ascontiguousarray(data[name][rows]),
                                 dtype=torch.float32 if name == "numeric" else torch.int64,
                                 device="cuda")
                 for name in ["categories", "sequences", "numeric"])


def predict(model, data, rows):
    model.eval()
    output = np.empty(len(rows), dtype="float32")
    with torch.no_grad():
        for start in range(0, len(rows), SETTINGS["batch_size"]):
            stop = min(start + SETTINGS["batch_size"], len(rows))
            output[start:stop] = model(*batch(data, rows[start:stop])).sigmoid().cpu().numpy()
    if not np.isfinite(output).all():
        raise AssertionError("Nonfinite prediction")
    return output


def scores(y, prediction):
    return {"auc": float(roc_auc_score(y, prediction)),
            "logloss": float(log_loss(y, prediction)),
            "average_precision": float(average_precision_score(y, prediction)),
            "mean_probability": float(prediction.mean())}


def sources():
    paths = [HERE / name for name in ["features.py", "model.py", "experiment.py"]]
    paths += [ARCH / "interaction_models.py"]
    paths += [ROOT / "analysis/model_comparison" / name for name in
              ["id_regularized_models.py", "mainstream_models.py", "benchmark_core.py"]]
    return {str(path): digest(path) for path in paths}


def slices(data, prediction):
    rows = data["valid"]
    labels = data["label"][rows]
    known = np.isin(data["task_id"][rows], np.unique(data["task_id"][data["train"]]))
    return {name: {"rows": int(mask.sum()), **scores(labels[mask], prediction[mask])}
            for name, mask in [("seen_task", known), ("new_task", ~known)]}


def replay(day, data, out):
    directory = ARCH / f"results/gated166_day{day % 100}"
    checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    if checkpoint["manifest"] != data["meta"]:
        raise AssertionError("Original checkpoint manifest differs")
    model = build(data["meta"], "control", float(data["label"][data["train"]].mean()))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.cuda()
    p = predict(model, data, data["valid"])
    expected = np.load(directory / "validation_prediction.npy")
    error = float(np.max(np.abs(p - expected)))
    actual = scores(data["label"][data["valid"]], p)
    if error > 2e-7:
        raise AssertionError(f"Original checkpoint replay failed: maximum error={error}")
    result = {"status": "passed", "day": day, "rows": len(p), **actual,
              "maximum_probability_error": error,
              "original_checkpoint_sha256": digest(directory / "model.pt"),
              "prediction_sha256": digest(directory / "validation_prediction.npy")}
    write(out / "original_replay.json", result)
    emit("original_strong_replay", **result)
    del model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()


def train(day, data, out, mode, seed):
    directory = out / mode
    if directory.exists():
        raise FileExistsError("Preserving completed/partial arm: " + str(directory))
    configure(seed)
    ytrain = data["label"][data["train"]]
    model = build(data["meta"], mode, float(ytrain.mean())).cuda()
    initial = {key: value for key, value in model.state_dict().items()
               if "interest_" not in key}
    initial_hash = state_digest(initial)
    initial_branch = {key: value.detach().cpu().clone() for key, value in
                      model.state_dict().items() if "interest_" in key}
    del initial
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS["learning_rate"],
                                 weight_decay=SETTINGS["weight_decay"])
    directory.mkdir(parents=True)
    np.save(directory / "valid_indices.npy", data["global_rows"][data["valid"]])
    history, best = [], -1.
    started = time.perf_counter()
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(np.arange(len(data["train"])))
    order_hash = hashlib.sha256(order.tobytes()).hexdigest()
    batches = math.ceil(len(order) / SETTINGS["batch_size"])
    checks = {math.ceil(batches * fraction) for fraction in SETTINGS["checks"]}
    total, seen = 0., 0
    model.train()
    emit("train_start", day=day, mode=mode, seed=seed, train_rows=len(order),
         parameters=sum(p.numel() for p in model.parameters()))
    for step, start in enumerate(range(0, len(order), SETTINGS["batch_size"]), 1):
        rows = data["train"][order[start:start + SETTINGS["batch_size"]]]
        target = torch.as_tensor(data["label"][rows].astype("float32"), device="cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(*batch(data, rows)), target)
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), SETTINGS["max_grad_norm"])
        optimizer.step()
        total += float(loss.detach()) * len(rows)
        seen += len(rows)
        if step not in checks:
            continue
        p = predict(model, data, data["valid"])
        record = {"fractional_epoch": step / batches, **scores(data["label"][data["valid"]], p),
                  "train_loss": total / seen, "seconds": time.perf_counter() - started}
        history.append(record)
        write(directory / "history.json", history)
        emit("train_progress", day=day, mode=mode, seed=seed, **record)
        checkpoint = {
            "state_dict": {key: value.detach().cpu().clone()
                           for key, value in model.state_dict().items()},
            "meta": data["meta"], "mode": mode, "seed": seed, "record": record,
            "source_sha256": sources(),
        }
        if record["auc"] > best + 1e-5:
            best = record["auc"]
            torch.save(checkpoint, directory / "best_model.pt")
            np.save(directory / "best_prediction.npy", p)
        if step == batches:
            torch.save(checkpoint, directory / "final_model.pt")
            np.save(directory / "final_prediction.npy", p)
        model.train()
    errors = {}
    for tag in ["best", "final"]:
        checkpoint = torch.load(directory / f"{tag}_model.pt", map_location="cpu",
                                weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        reproduced = predict(model, data, data["valid"])
        expected = np.load(directory / f"{tag}_prediction.npy")
        errors[tag] = float(np.max(np.abs(reproduced - expected)))
        if errors[tag] > 2e-7:
            raise AssertionError("New checkpoint replay differs")
    branch_change = {
        key: float((model.state_dict()[key].detach().cpu() - value).norm())
        for key, value in initial_branch.items()
    }
    result = {
        "status": "complete_and_verified", "day": day, "mode": mode, "seed": seed,
        "fixed_epochs_completed": 1, "train_rows": len(data["train"]),
        "valid_rows": len(data["valid"]), "valid_clicks": int(data["label"][data["valid"]].sum()),
        "initial_backbone_sha256": initial_hash, "batch_order_sha256": order_hash,
        "scores": {key: history[-1][key] for key in scores(data["label"][data["valid"]], expected)},
        "best_scores": max(history, key=lambda row: row["auc"]),
        "slices": slices(data, np.load(directory / "final_prediction.npy")),
        "reload_errors": errors, "branch_parameter_update_norm": branch_change,
        "source_sha256": sources(),
        "feature_manifest_sha256": digest(out / "feature_manifest.json"),
        "seconds": time.perf_counter() - started,
    }
    result["files_sha256"] = {
        name: digest(directory / name) for name in
        ["best_model.pt", "final_model.pt", "best_prediction.npy", "final_prediction.npy",
         "valid_indices.npy", "history.json"]}
    write(directory / "metrics.json", result)
    emit("arm_complete", day=day, mode=mode, seed=seed, scores=result["scores"],
         seconds=result["seconds"])
    del model, optimizer, checkpoint
    gc.collect()
    torch.cuda.empty_cache()


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA needed for original precision replay")
    configure(args.seed)
    out = HERE / "runs" / f"day{args.day % 100}_seed{args.seed}"
    if out.exists():
        raise FileExistsError("Preserving experiment directory")
    out.mkdir(parents=True)
    protocol = {
        **SETTINGS, "seed": args.seed, "day": args.day,
        "backbone": "Original GatedCrossCTR, categories27 + sequences12x5 + numeric166",
        "input_availability": "Original full supplied same-day source, snapshots and unlabeled exposures",
        "change": "Only six news-interest fields use candidate-conditioned residual attention pooling",
        "query_fields": QUERY_FIELDS, "interest_fields": NEWS_FIELDS,
        "support": "n/(n+3); original mean plus supported attentive-minus-mean residual",
        "initial_attention": "Zero output scores give exactly uniform pooling at initialization",
        "validation": "Two previously reused whole-day development dates; no blind-test claim",
        "optimizer_validation_labels_used": False, "final_test_read": False,
        "ensemble_comparison": "Replace only v7's .30 original gated166 component",
        "promotion_rule": "Both fixed-budget AUC gains>=.0005, logloss worsening<=.0001; then seed2026 confirmation",
        "source_sha256": sources(),
    }
    write(out / "protocol.json", protocol)
    data = prepare(args.day, out)
    replay(args.day, data, out)
    if args.replay_only:
        return
    for mode in ["control", "attention"]:
        train(args.day, data, out, mode, args.seed)
    first, second = [read(out / mode / "metrics.json") for mode in ["control", "attention"]]
    for key in ["initial_backbone_sha256", "batch_order_sha256"]:
        if first[key] != second[key]:
            raise AssertionError("Matched control differs: " + key)
    emit("matched_day_complete", day=args.day, seed=args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, choices=[20191115, 20191116], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay-only", action="store_true")
    run(parser.parse_args())
