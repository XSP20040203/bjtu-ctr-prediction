"""Reduce conflicting-duplicate gradient noise without editing original data."""
import copy
import gc
import math
import time

from train_weighted import (HERE, ROOT, ARCH, SETTINGS, digest, read, write, emit, array_hash,
                            configure, prepare_inputs, make_model, batch, predict, combined_scores,
                            capture_current_reference, load_reference, scores, clip_separately)
from duplicate_targets import duplicate_targets
import numpy as np
import torch
from torch import nn


def main():
    day = 20191116
    out = HERE / "runs/duplicate_mean_v1_day16"
    if out.exists():
        raise FileExistsError("Preserve duplicate experiment")
    out.mkdir(parents=True)
    settings = dict(SETTINGS, positive_weights=[1.])
    protocol = {"status": "fixed_before_training", "day": day, "settings": settings,
                "reference": capture_current_reference(), "target_real16_auc": .82,
                "original_datasets_changed": False,
                "training_target": "Observed-label mean within exactly identical pre16 model inputs; all rows retained",
                "full_deterministic_BCE_objective_preserved": True,
                "held_day_labels_used_for_training": False, "development_day_reused": True,
                "hard_label_comparator": "weighted_v1_day16/positive_weight_1",
                "raw_datasets_sha256": {p.name: digest(p) for p in (ROOT / "2026ml").glob("*.csv")},
                "source_sha256": {str(p): digest(p) for p in [
                    HERE / "train_duplicate.py", HERE / "duplicate_targets.py", HERE / "train_weighted.py"]}}
    write(out / "protocol.json", protocol)
    configure(SETTINGS["seed"])
    data, contexts = prepare_inputs(day, out)
    soft, grouping = duplicate_targets(data)
    if not np.all(soft[data["valid"]] == -1):
        raise AssertionError("Held targets were averaged")
    grouping["target_array_sha256"] = array_hash(soft)
    write(out / "duplicate_groups.json", grouping)
    emit("duplicate_targets_ready", **grouping)
    del grouping
    gc.collect()
    reference, indices, _ = load_reference(day, protocol["reference"])
    np.testing.assert_array_equal(indices, data["global_rows"][data["valid"]])
    v7 = np.load(ARCH / "fixed_gated20_allocated10_core30_day16.npy").astype("float64")
    old_core = np.load(ARCH / "results/gated166_day16/validation_prediction.npy")
    configure(SETTINGS["seed"])
    model = make_model(data, contexts, 1.)
    ema = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    directory = out / "positive_weight_1"
    directory.mkdir()
    best = {"raw": -1., "ema": -1.}
    history = []
    started = time.perf_counter()
    steps = math.ceil(len(data["train"])/SETTINGS["batch_size"])
    checks = {math.ceil(steps*f) for f in SETTINGS["quarter_epoch_checks"]}
    ema_updates = 0
    orders = []
    for epoch, lr in enumerate(SETTINGS["learning_rates"]):
        for group in optimizer.param_groups:
            group["lr"] = lr
        order = np.random.default_rng(SETTINGS["seed"]+1+epoch).permutation(data["train"])
        orders.append(array_hash(order))
        model.train()
        for step, start in enumerate(range(0, len(order), SETTINGS["batch_size"]), 1):
            rows = order[start:start+SETTINGS["batch_size"]]
            target = torch.as_tensor(soft[rows], dtype=torch.float32, device="cuda")
            if bool(torch.any((target < 0) | (target > 1))):
                raise AssertionError("Unknown training target")
            optimizer.zero_grad(set_to_none=True)
            logits, auxiliary = model(*batch(data, rows))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            groups = data["day_rows"][rows]
            targets = torch.as_tensor(data["any_targets"][groups], dtype=torch.float32, device="cuda")
            sizes = torch.as_tensor(data["day_sizes"][groups], dtype=torch.float32, device="cuda")
            from day_model import auxiliary_loss
            loss = loss + SETTINGS["auxiliary_weight"]*auxiliary_loss(auxiliary, targets, sizes)
            loss.backward()
            clip_separately(model, model.parameters(), 5.)
            optimizer.step()
            if epoch > 0 or step/steps >= SETTINGS["ema_start_fraction"]:
                with torch.no_grad():
                    decay = SETTINGS["ema_decay"] if ema_updates else 0.
                    for a, b in zip(ema.parameters(), model.parameters()):
                        a.lerp_(b, 1-decay)
                ema_updates += 1
            if step not in checks:
                continue
            for endpoint, current in [("raw", model)] + ([("ema", ema)] if ema_updates else []):
                p = predict(current, data, data["valid"], 1.)
                metrics = combined_scores(data["evaluation_labels"], p, reference, v7, old_core)
                record = {"epoch": epoch+step/steps, "endpoint": endpoint, "learning_rate": lr,
                          "scores": metrics, "seconds": time.perf_counter()-started}
                history.append(record)
                if metrics["single"]["auc"] > best[endpoint]:
                    best[endpoint] = metrics["single"]["auc"]
                    torch.save({"state_dict": {n: v.detach().cpu().clone() for n,v in current.state_dict().items()},
                                "meta": data["meta"], "positive_weight": 1., "record": record,
                                "any_prior": data["any_prior"], "protocol_sha256": digest(out / "protocol.json")},
                               directory / f"{endpoint}_best.pt")
                    np.save(directory / f"{endpoint}_best_prediction.npy", p)
                emit("duplicate_progress", endpoint=endpoint, epoch=record["epoch"],
                     auc=metrics["single"]["auc"], best_blend_auc=max(m["auc"] for m in metrics.values()))
                write(directory / "history.json", history)
            model.train()
    errors = {}
    for endpoint in ["raw", "ema"]:
        checkpoint = torch.load(directory / f"{endpoint}_best.pt", map_location="cpu", weights_only=False)
        fresh = make_model(data, contexts, 1.)
        fresh.load_state_dict(checkpoint["state_dict"], strict=True)
        p = predict(fresh, data, data["valid"], 1.)
        errors[endpoint] = float(np.max(np.abs(p-np.load(directory / f"{endpoint}_best_prediction.npy"))))
        if errors[endpoint] > 2e-7:
            raise AssertionError("Reload mismatch")
        ptrain = predict(fresh, data, data["train"], 1.)
        write(directory / f"{endpoint}_train_metrics.json", scores(data["label"][data["train"]], ptrain))
        del fresh
    if protocol["raw_datasets_sha256"] != {p.name: digest(p) for p in (ROOT / "2026ml").glob("*.csv")}:
        raise AssertionError("Original datasets changed")
    write(directory / "summary.json", {"best_single_auc": best, "reload_errors": errors, "orders_sha256": orders})
    write(out / "completion.json", {"status": "training_and_reload_complete", "best": best,
                                     "original_datasets_unchanged": True, "independent_verification_pending": True})


if __name__ == "__main__":
    main()
