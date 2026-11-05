"""Controlled rare-click weighting and learning-rate decay on the verified backbone.

All advertisement fitting labels precede the complete evaluation day. The
evaluation day is a repeatedly used development set, never a blind test.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis/strong_interest_control_20261031"))
from features import prepare, read, write, emit, digest, ARCH
from experiment import configure, state_digest, scores
sys.path.insert(0, str(ROOT / "analysis/userday_aux_20261101"))
from day_context import build_context, training_targets
from day_model import auxiliary_loss
from protected_day_model import ProtectedUserDayCTR, clip_separately
sys.path.insert(0, str(ROOT / "analysis"))
from auc85_reference import capture_current_reference, load_reference
import numpy as np
import pandas as pd
import torch
from torch import nn

SETTINGS = {
    "positive_weights": [1., 5., 20.],
    "learning_rates": [.001, .0002, .00005],
    "batch_size": 1024,
    "weight_decay": .0001,
    "auxiliary_weight": .02,
    "ema_decay": .99,
    "ema_start_fraction": .5,
    "quarter_epoch_checks": [.25, .5, .75, 1.],
    "seed": 42,
    "blend_weights": [.25, .5, .75],
}


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def batch(data, rows, device="cuda"):
    return tuple(torch.as_tensor(np.ascontiguousarray(data[name][rows]), dtype=dtype,
                                 device=device) for name, dtype in [
        ("categories", torch.int64), ("sequences", torch.int64),
        ("numeric", torch.float32), ("day_rows", torch.int64)])


def corrected_probability(logits, positive_weight):
    return torch.sigmoid(logits - math.log(positive_weight))


def predict(model, data, rows, weight):
    model.eval()
    prediction = np.empty(len(rows), dtype="float32")
    with torch.no_grad():
        for start in range(0, len(rows), SETTINGS["batch_size"]):
            chosen = rows[start:start + SETTINGS["batch_size"]]
            logits, _ = model(*batch(data, chosen))
            prediction[start:start + len(chosen)] = corrected_probability(logits, weight).cpu().numpy()
    if not np.isfinite(prediction).all():
        raise AssertionError("Nonfinite prediction")
    return prediction


def prepare_inputs(day, out):
    data = prepare(day, out)
    raw = pd.read_csv(ROOT / "2026ml/train.csv", usecols=["pt_d", "user_id", "task_id", "label"],
                      dtype="int32").iloc[data["global_rows"]].reset_index(drop=True)
    np.testing.assert_array_equal(data["label"], raw.label.to_numpy())
    np.testing.assert_array_equal(data["train"], np.flatnonzero(raw.pt_d.to_numpy() < day))
    np.testing.assert_array_equal(data["valid"], np.flatnonzero(raw.pt_d.to_numpy() == day))
    data["evaluation_labels"] = raw.label.to_numpy()[data["valid"]].copy()
    # All functions in the training loop see -1 in held-day label positions.
    data["label"] = np.array(data["label"], copy=True)
    data["label"][data["valid"]] = -1
    context = build_context(raw.drop(columns="label"), data["categories"], data["numeric"],
                            data["feature_manifest"]["numeric_fields"], data["meta"], day)
    targets, prior = training_targets(context["keys"], context["rows"], data["label"], data["train"], day)
    if np.any(targets[context["rows"][data["valid"]]] != -1):
        raise AssertionError("Evaluation outcomes entered auxiliary training targets")
    data.update(day_rows=context["rows"], day_sizes=context["sizes"], any_targets=targets, any_prior=prior)
    write(out / "context_manifest.json", context["manifest"])
    np.savez_compressed(out / "context.npz", keys=context["keys"], values=context["values"],
                        sizes=context["sizes"], any_targets=targets)
    np.save(out / "valid_indices.npy", data["global_rows"][data["valid"]])
    write(out / "input_contract.json", {
        "fit_max_date": int(raw.pt_d.iloc[data["train"]].max()), "evaluation_date": day,
        "fit_rows": len(data["train"]), "evaluation_rows": len(data["valid"]),
        "fit_clicks": int(data["label"][data["train"]].sum()),
        "evaluation_clicks": int(data["evaluation_labels"].sum()),
        "fit_indices_sha256": array_hash(data["global_rows"][data["train"]]),
        "evaluation_indices_sha256": array_hash(data["global_rows"][data["valid"]]),
        "evaluation_labels_masked_from_training": True,
        "context_sha256": array_hash(context["values"]),
    })
    return data, torch.as_tensor(context["values"], device="cuda")


def combined_scores(y, p, reference, v7, old_core):
    answer = {"single": scores(y, p), "core_replacement_030": scores(y, v7 + .3 * (p - old_core))}
    for alpha in SETTINGS["blend_weights"]:
        answer[f"reference_blend_{alpha:g}"] = scores(y, (1-alpha)*reference + alpha*p)
    return answer


def make_model(data, contexts, weight):
    prior = float(data["label"][data["train"]].mean())
    effective_prior = weight * prior / (1 - prior + weight * prior)
    return ProtectedUserDayCTR(data["meta"], effective_prior, contexts, data["any_prior"],
                               use_context=True).cuda()


def run_arm(args, data, contexts, out, weight, reference, v7, old_core):
    directory = out / f"positive_weight_{weight:g}"
    directory.mkdir()
    configure(SETTINGS["seed"])
    model = make_model(data, contexts, weight)
    ema = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS["learning_rates"][0],
                                  weight_decay=SETTINGS["weight_decay"])
    y = data["evaluation_labels"]
    prior = float(data["label"][data["train"]].mean())
    normalizer = 1 + (weight - 1) * prior
    history, best = [], {"raw": -1., "ema": -1.}
    started = time.perf_counter()
    initial = state_digest(model.state_dict())
    orders = []
    steps = math.ceil(len(data["train"]) / SETTINGS["batch_size"])
    checks = {math.ceil(steps*f) for f in SETTINGS["quarter_epoch_checks"]}
    ema_updates = 0
    emit("weighted_arm_start", day=args.day, weight=weight, parameters=sum(p.numel() for p in model.parameters()))
    for epoch, learning_rate in enumerate(SETTINGS["learning_rates"]):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        order = np.random.default_rng(SETTINGS["seed"] + 1 + epoch).permutation(data["train"])
        orders.append(array_hash(order))
        totals, seen = np.zeros(3), 0
        model.train()
        for step, start in enumerate(range(0, len(order), SETTINGS["batch_size"]), 1):
            rows = order[start:start + SETTINGS["batch_size"]]
            labels = data["label"][rows]
            if not np.isin(labels, [0, 1]).all():
                raise AssertionError("Unknown labels reached optimization")
            target = torch.as_tensor(labels, dtype=torch.float32, device="cuda")
            optimizer.zero_grad(set_to_none=True)
            logits, auxiliary = model(*batch(data, rows))
            row_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, target, pos_weight=target.new_tensor(weight)) / normalizer
            groups = data["day_rows"][rows]
            group_target = torch.as_tensor(data["any_targets"][groups], dtype=torch.float32, device="cuda")
            group_size = torch.as_tensor(data["day_sizes"][groups], dtype=torch.float32, device="cuda")
            aux_loss = auxiliary_loss(auxiliary, group_target, group_size)
            loss = row_loss + SETTINGS["auxiliary_weight"] * aux_loss
            if not torch.isfinite(loss):
                raise AssertionError("Nonfinite objective")
            loss.backward()
            clip_separately(model, model.parameters(), 5.)
            optimizer.step()
            if epoch > 0 or step / steps >= SETTINGS["ema_start_fraction"]:
                with torch.no_grad():
                    decay = SETTINGS["ema_decay"] if ema_updates else 0.
                    for p_ema, p in zip(ema.parameters(), model.parameters()):
                        p_ema.lerp_(p, 1-decay)
                ema_updates += 1
            totals += np.array([float(loss.detach()), float(row_loss.detach()), float(aux_loss.detach())]) * len(rows)
            seen += len(rows)
            if step not in checks:
                continue
            for endpoint, current in [("raw", model)] + ([("ema", ema)] if ema_updates else []):
                prediction = predict(current, data, data["valid"], weight)
                metrics = combined_scores(y, prediction, reference, v7, old_core)
                record = {"epoch": epoch + step / steps, "endpoint": endpoint,
                          "learning_rate": learning_rate, "training_losses": (totals/seen).tolist(),
                          "scores": metrics, "seconds": time.perf_counter()-started}
                history.append(record)
                checkpoint = {"state_dict": {n: p.detach().cpu().clone() for n, p in current.state_dict().items()},
                              "meta": data["meta"], "positive_weight": weight, "record": record,
                              "any_prior": data["any_prior"], "protocol_sha256": digest(out / "protocol.json")}
                if metrics["single"]["auc"] > best[endpoint]:
                    best[endpoint] = metrics["single"]["auc"]
                    torch.save(checkpoint, directory / f"{endpoint}_best.pt")
                    np.save(directory / f"{endpoint}_best_prediction.npy", prediction)
                if epoch == 0 and step == steps and endpoint == "raw":
                    torch.save(checkpoint, directory / "epoch1.pt")
                    np.save(directory / "epoch1_prediction.npy", prediction)
                write(directory / "history.json", history)
                emit("weighted_progress", day=args.day, weight=weight, endpoint=endpoint,
                     epoch=record["epoch"], auc=metrics["single"]["auc"],
                     replacement_auc=metrics["core_replacement_030"]["auc"],
                     best_blend_auc=max(m["auc"] for m in metrics.values()), seconds=record["seconds"])
            model.train()
    reload_errors = {}
    for endpoint in ["raw", "ema"]:
        checkpoint = torch.load(directory / f"{endpoint}_best.pt", map_location="cpu", weights_only=False)
        fresh = make_model(data, contexts, weight)
        fresh.load_state_dict(checkpoint["state_dict"], strict=True)
        replay = predict(fresh, data, data["valid"], weight)
        expected = np.load(directory / f"{endpoint}_best_prediction.npy")
        reload_errors[endpoint] = float(np.max(np.abs(replay - expected)))
        if reload_errors[endpoint] > 2e-7:
            raise AssertionError("Fresh checkpoint replay mismatch")
        train_prediction = predict(fresh, data, data["train"], weight)
        write(directory / f"{endpoint}_train_metrics.json", scores(data["label"][data["train"]], train_prediction))
        del fresh, train_prediction
    if weight == 1.:
        old = np.load(ROOT / f"analysis/userday_aux_20261101/runs/protected_day_aux_v1_day{args.day % 100}_seed42/both/best_prediction.npy")
        baseline_error = float(np.max(np.abs(old - np.load(directory / "epoch1_prediction.npy"))))
    else:
        baseline_error = None
    summary = {"weight": weight, "initial_state_sha256": initial, "orders_sha256": orders,
               "best_single_auc": best, "reload_errors": reload_errors,
               "unweighted_epoch1_vs_saved_baseline_error": baseline_error,
               "seconds": time.perf_counter()-started,
               "checkpoint_sha256": {p.name: digest(p) for p in directory.glob("*.pt")}}
    write(directory / "summary.json", summary)
    del model, ema, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def run(args):
    out = HERE / "runs" / f"weighted_v1_day{args.day % 100}"
    if out.exists():
        raise FileExistsError("Preserve existing experiment; choose a new version")
    out.mkdir(parents=True)
    reference_spec = capture_current_reference()
    protocol = {"status": "fixed_before_training", "target_real16_auc": .82, "day": args.day,
                "settings": SETTINGS, "reference": reference_spec,
                "fitting_ad_dates": f"strictly before {args.day}",
                "held_day_labels_used_for_optimization": False,
                "held_day_is_reused_development_not_blind_test": True,
                "label_free_same_day_exposures_and_supplied_news_allowed": True,
                "raw_datasets_sha256": {p.name: digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
                "source_sha256": {str(p): digest(p) for p in [Path(__file__),
                    ROOT / "analysis/strong_interest_control_20261031/features.py",
                    ROOT / "analysis/userday_aux_20261101/day_context.py",
                    ROOT / "analysis/userday_aux_20261101/protected_day_model.py"]},
                "probability_correction": "sigmoid(weighted_logit - log(positive_weight))",
                "best_checkpoint_rule": "maximum full-day single-model development AUC per arm and endpoint"}
    write(out / "protocol.json", protocol)
    configure(SETTINGS["seed"])
    data, contexts = prepare_inputs(args.day, out)
    reference, indices, _ = load_reference(args.day, reference_spec)
    np.testing.assert_array_equal(indices, data["global_rows"][data["valid"]])
    v7 = np.load(ARCH / f"fixed_gated20_allocated10_core30_day{args.day % 100}.npy").astype("float64")
    old_core = np.load(ARCH / f"results/gated166_day{args.day % 100}/validation_prediction.npy")
    write(out / "reference_metrics.json", scores(data["evaluation_labels"], reference))
    records = [run_arm(args, data, contexts, out, w, reference, v7, old_core) for w in SETTINGS["positive_weights"]]
    final_hashes = {p.name: digest(p) for p in (ROOT / "2026ml").glob("*.csv")}
    if final_hashes != protocol["raw_datasets_sha256"]:
        raise AssertionError("Original datasets changed")
    write(out / "completion.json", {"status": "training_and_reload_complete", "records": records,
          "original_datasets_unchanged": True, "independent_verification_pending": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=int, choices=[20191115, 20191116], default=20191116)
    run(parser.parse_args())
