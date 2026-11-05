"""Continue the verified row network with same-user and same-user-day pairs."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from train_weighted import (ROOT, HERE, ARCH, configure, prepare_inputs, make_model,
                            batch, predict, digest, write, array_hash, clip_separately)
from recent_finetune import checked_source, reference_inputs, SOURCE, WEIGHT
from label_boundary_audit import signed_labels
from verify_stacker import rank_sum_auc
from pairwise_neural_objective import class_pools
from matched_pair_objective import make_pool, sample, objective

OUT = HERE / "runs/rank_matched_v1_day16"
BASE = HERE / "runs/previous_day_residual_v1"
ARMS = [{"name": "user_t1", "scope": "user", "temperature": 1.},
        {"name": "userday_t1", "scope": "userday", "temperature": 1.},
        {"name": "user_t05", "scope": "user", "temperature": .5}]
POWERS = [.125, .25, .5, 1.]
SEED = 8163
STEPS_PER_EPOCH = 512
RATES = [.00003, .00001]


def dependencies():
    return [Path(__file__), HERE/"pairwise_neural_objective.py", HERE/"matched_pair_objective.py", HERE/"train_weighted.py",
            HERE/"recent_finetune.py", HERE/"label_boundary_audit.py",
            HERE/"verify_stacker.py", ROOT/"analysis/strong_interest_control_20261031/features.py",
            ROOT/"analysis/strong_interest_control_20261031/model.py",
            ROOT/"analysis/userday_aux_20261101/day_context.py",
            ROOT/"analysis/userday_aux_20261101/day_model.py",
            ROOT/"analysis/userday_aux_20261101/protected_day_model.py",
            SOURCE/"protocol.json", SOURCE/"verification.json", BASE/"verification.json",
            BASE/"prediction.npy"]


def candidates(p, initial, reference, v7, old_core, best_base):
    yield "single", p.astype("float64")
    yield "core_replacement_030", v7 + .3*(p-old_core)
    for power in [.25, .5, .75]:
        yield f"reference_blend_{power:g}", (1.-power)*reference+power*p
    delta = logit(np.clip(p, 1e-6, 1.-1e-6))-logit(np.clip(initial, 1e-6, 1.-1e-6))
    base_logit = logit(np.clip(best_base, 1e-6, 1.-1e-6))
    for power in POWERS:
        yield f"rank_delta_{power:g}", expit(base_logit+power*delta)


def select_recipe(y, p, initial, reference, v7, old_core, best_base):
    scores = {name: rank_sum_auc(y, values) for name, values in
              candidates(p, initial, reference, v7, old_core, best_base)}
    name = max(scores, key=scores.get)
    return name, scores


def save(model, target, record, p):
    torch.save({"state_dict": {n: v.detach().cpu().clone() for n, v in model.state_dict().items()},
                "record": record, "protocol_sha256": digest(OUT/"protocol.json")}, target/"best.pt")
    np.save(target/"raw_prediction.npy", p)


def inputs(directory):
    data, contexts = prepare_inputs(20191116, directory)
    data["label"] = signed_labels(data["label"], data["train"], data["valid"])
    raw = pd.read_csv(ROOT/"2026ml/train.csv", usecols=["pt_d"], dtype="int32")
    dates = raw.pt_d.to_numpy()[data["global_rows"]]
    positive, negative = class_pools(data["label"], data["train"], dates)
    if not np.all(data["label"][data["valid"]] == -1):
        raise AssertionError("Held labels must be masked")
    return data, contexts, dates, positive, negative


def main():
    if OUT.exists():
        raise FileExistsError("Preserve ranking continuation")
    source = checked_source()
    base_verification = json.loads((BASE/"verification.json").read_text(encoding="utf8"))
    if base_verification["status"] != "passed":
        raise AssertionError("Published baseline must be independently verified")
    source_protocol = json.loads((SOURCE/"protocol.json").read_text(encoding="utf8"))
    OUT.mkdir(parents=True)
    protocol = {"source_checkpoint": str(source), "source_checkpoint_sha256": digest(source),
                "source_files_sha256": {str(p): digest(p) for p in dependencies()},
                "raw_datasets_sha256": source_protocol["raw_datasets_sha256"],
                "reference": source_protocol["reference"], "arms": ARMS,
                "steps_per_epoch": STEPS_PER_EPOCH, "learning_rates": RATES,
                "positive_batch": 64, "matched_negative_batch": 448, "uniform_negative_batch": 512, "seed": SEED,
                "delta_powers": POWERS, "positive_weight": WEIGHT,
                "objective": "Within-user or within-user-day softplus pair loss; quarter-weight BCE regularizer uses matched-eligible positives and uniform negatives; no auxiliary objective",
                "sampling": "Uniform eligible positive rows,7 matched negatives each,512 global negatives; only strictly pre16 rows",
                "optimizer": "Fresh AdamW weight_decay0.0001; separate backbone/head clip5",
                "selection": "Initial and four checks per epoch; best fixed recipe on reused full16 development day",
                "evaluation16_labels_fit": False, "development_day_reused": True}
    for name, expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT/"2026ml"/name) != expected:
            raise AssertionError("Original dataset changed")
    write(OUT/"protocol.json", protocol)
    configure(SEED)
    data, contexts, dates, positive, negative = inputs(OUT)
    contract = {"positive_indices_sha256": array_hash(data["global_rows"][positive]),
                "negative_indices_sha256": array_hash(data["global_rows"][negative]),
                "positive_rows": len(positive), "negative_rows": len(negative),
                "maximum_fit_date": int(dates[data["train"]].max()),
                "held_label_dtype": str(data["label"].dtype), "held_label_values": [-1]}
    write(OUT/"training_contract.json", contract)
    prior = len(positive)/(len(positive)+len(negative))
    reference, rows, v7, old_core = reference_inputs(protocol)
    np.testing.assert_array_equal(rows, data["global_rows"][data["valid"]])
    initial = np.load(SOURCE/"positive_weight_5/ema_best_prediction.npy")
    best_base = np.load(BASE/"prediction.npy")
    if rank_sum_auc(data["evaluation_labels"], best_base) != base_verification["auc"]:
        raise AssertionError("Baseline score differs")
    source_state = torch.load(source, map_location="cpu", weights_only=False)["state_dict"]
    reports = []
    for arm in ARMS:
        target = OUT/arm["name"]
        target.mkdir()
        configure(SEED)
        model = make_model(data, contexts, WEIGHT)
        model.load_state_dict(source_state, strict=True)
        replay = predict(model, data, data["valid"], WEIGHT)
        np.testing.assert_allclose(replay, initial, rtol=0, atol=2e-7)
        recipe, metrics = select_recipe(data["evaluation_labels"], replay, initial,
                                        reference, v7, old_core, best_base)
        best = metrics[recipe]
        record = {"epoch": 0., "recipe": recipe, "metrics": metrics}
        save(model, target, record, replay)
        history, sample_hashes = [record], []
        optimizer = torch.optim.AdamW(model.parameters(), lr=RATES[0], weight_decay=.0001)
        rng = np.random.default_rng(SEED+1)
        pool = make_pool(data["label"], data["train"], dates, data["user_id"], arm["scope"])
        for epoch, lr in enumerate(RATES):
            for group in optimizer.param_groups:
                group["lr"] = lr
            totals = np.zeros(3)
            samples = []
            model.train()
            for step in range(1, STEPS_PER_EPOCH+1):
                selected = sample(pool, rng, 64, 7, 512)
                samples.append(selected)
                if np.any(dates[selected] >= 20191116):
                    raise AssertionError("Held day reached optimizer")
                np.testing.assert_array_equal(data["label"][selected[:64]], np.ones(64))
                np.testing.assert_array_equal(data["label"][selected[64:]], np.zeros(960))
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(*batch(data, selected))
                loss, pair, bce = objective(logits, 64, 7, prior, arm["temperature"])
                if not bool(torch.isfinite(loss)):
                    raise AssertionError("Nonfinite ranking objective")
                loss.backward()
                clip_separately(model, model.parameters(), 5.)
                optimizer.step()
                totals += [float(loss.detach()), float(pair.detach()), float(bce.detach())]
                if step % 128:
                    continue
                p = predict(model, data, data["valid"], WEIGHT)
                recipe, metrics = select_recipe(data["evaluation_labels"], p, initial,
                                                reference, v7, old_core, best_base)
                record = {"epoch": epoch+step/STEPS_PER_EPOCH, "recipe": recipe,
                          "metrics": metrics, "training_loss_means": (totals/step).tolist()}
                history.append(record)
                if metrics[recipe] > best:
                    best = metrics[recipe]
                    save(model, target, record, p)
                write(target/"history.json", history)
                print(json.dumps({"arm": arm["name"], "epoch": record["epoch"],
                                  "recipe": recipe, "auc": metrics[recipe], "best": best}), flush=True)
                model.train()
            sample_hashes.append(array_hash(np.concatenate(samples)))
        write(target/"sampling_contract.json", {"epoch_rows_sha256": sample_hashes})
        reports.append({"name": arm["name"], "best_auc": best,
                        "checkpoint_sha256": digest(target/"best.pt")})
        del model, optimizer, samples, replay
        gc.collect()
        torch.cuda.empty_cache()
    write(OUT/"completion.json", {"status": "trained", "arms": reports,
                                 "independent_verification_pending": True})


def verify():
    protocol = json.loads((OUT/"protocol.json").read_text(encoding="utf8"))
    completion = json.loads((OUT/"completion.json").read_text(encoding="utf8"))
    contract = json.loads((OUT/"training_contract.json").read_text(encoding="utf8"))
    for p, expected in protocol["source_files_sha256"].items():
        if digest(p) != expected:
            raise AssertionError("Ranking dependency changed")
    for name, expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT/"2026ml"/name) != expected:
            raise AssertionError("Original dataset changed")
    if digest(protocol["source_checkpoint"]) != protocol["source_checkpoint_sha256"]:
        raise AssertionError("Initial model changed")
    configure(SEED)
    rebuild = OUT/"independent_rebuild"
    rebuild.mkdir()
    data, contexts, dates, positive, negative = inputs(rebuild)
    if (contract["positive_indices_sha256"] != array_hash(data["global_rows"][positive])
            or contract["negative_indices_sha256"] != array_hash(data["global_rows"][negative])
            or contract["maximum_fit_date"] >= 20191116):
        raise AssertionError("Training pool reconstruction differs")
    expected_samples = {}
    for spec in ARMS:
        rng = np.random.default_rng(SEED+1)
        pool = make_pool(data["label"], data["train"], dates, data["user_id"], spec["scope"])
        expected_samples[spec["name"]] = [array_hash(np.concatenate([
            sample(pool, rng, 64, 7, 512) for _ in range(STEPS_PER_EPOCH)])) for _ in RATES]
    raw = pd.read_csv(ROOT/"2026ml/train.csv", usecols=["pt_d", "label"], dtype="int32")
    rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191116)
    y = raw.label.to_numpy()[rows]
    np.testing.assert_array_equal(rows, np.load(OUT/"valid_indices.npy"))
    np.testing.assert_array_equal(y, data["evaluation_labels"])
    if len(y) != 421983 or int(y.sum()) != 3724:
        raise AssertionError("Complete16 cohort changed")
    reference, refrows, v7, old_core = reference_inputs(protocol)
    np.testing.assert_array_equal(rows, refrows)
    initial = np.load(SOURCE/"positive_weight_5/ema_best_prediction.npy")
    best_base = np.load(BASE/"prediction.npy")
    records, predictions = [], []
    for arm in completion["arms"]:
        target = OUT/arm["name"]
        sampling = json.loads((target/"sampling_contract.json").read_text(encoding="utf8"))
        if expected_samples[arm["name"]] != sampling["epoch_rows_sha256"]:
            raise AssertionError("Independent sampling replay differs")
        if digest(target/"best.pt") != arm["checkpoint_sha256"]:
            raise AssertionError("Ranking checkpoint changed")
        checkpoint = torch.load(target/"best.pt", map_location="cpu", weights_only=False)
        if checkpoint["protocol_sha256"] != digest(OUT/"protocol.json"):
            raise AssertionError("Wrong ranking protocol")
        model = make_model(data, contexts, WEIGHT)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        p = predict(model, data, data["valid"], WEIGHT)
        error = float(np.max(np.abs(p-np.load(target/"raw_prediction.npy"))))
        if error > 2e-7:
            raise AssertionError("Ranking model replay differs")
        record = checkpoint["record"]
        selected = dict(candidates(p, initial, reference, v7, old_core, best_base))[record["recipe"]]
        score = rank_sum_auc(y, selected)
        if abs(score-arm["best_auc"]) > 2e-8:
            raise AssertionError("Independent full16 score differs")
        records.append(dict(arm, auc=score, recipe=record["recipe"], epoch=record["epoch"], replay_error=error))
        predictions.append(selected)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    best = int(np.argmax([r["auc"] for r in records]))
    np.save(OUT/"prediction.npy", predictions[best])
    result = {"status": "passed", "auc": records[best]["auc"], "records": records,
              "rows": len(y), "clicks": int(y.sum()), "original_datasets_unchanged": True,
              "training_pools_and_sampling_independently_rebuilt": True,
              "evaluation16_labels_fit": False, "development_day_reused": True,
              "target_met": records[best]["auc"] >= .82}
    write(OUT/"verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    verify() if parser.parse_args().verify else main()
