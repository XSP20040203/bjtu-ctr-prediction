"""Train shared metadata input adapters while protecting the best TabM32 backbone."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model_comparison"))
from train_models import load_arrays, metrics, write_json
from hierarchical_id_model import HierarchicalIDTabM, fit_exposure_support
from modern_architectures import member_probability
from protected_adapters import initialize_protected_adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--tag", default="phase4_protected_adapter")
    args = parser.parse_args()
    protocol = json.loads((HERE / "phase4_protected_adapter_protocol.json").read_text(encoding="utf-8"))
    schedule = protocol["optimization"]
    output = HERE / "results_modern" / args.tag / "hierarchical" / "seed42"
    if output.exists():
        raise FileExistsError(f"Preserving protected adapter experiment {output}")
    source_record = json.loads((args.teacher_dir / "metrics.json").read_text(encoding="utf-8"))
    cache = Path(source_record["cache_dir"])
    teacher = torch.load(args.teacher_dir / "model.pt", map_location="cpu", weights_only=False)
    manifest = teacher["manifest"]
    assert manifest == json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    arrays = load_arrays(cache)
    train, valid = arrays["train_indices"], arrays["valid_indices"]
    assert not np.intersect1d(train, valid).size
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    filenames = ["modern_architectures.py", "hierarchical_id_model.py", "protected_adapters.py", "train_protected_adapter.py", "phase4_protected_adapter_protocol.json"]
    digests = {}
    for name in filenames:
        source = HERE / name
        shutil.copy2(source, snapshot / name)
        digests[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(output / "provenance.json", {"snapshot_sha256": digests, "tabm_version": importlib.metadata.version("tabm"),
                                          "rtdl_num_embeddings_version": importlib.metadata.version("rtdl_num_embeddings")})
    if (args.teacher_dir / "numeric_bin_fit_indices.npy").exists():
        shutil.copy2(args.teacher_dir / "numeric_bin_fit_indices.npy", output / "numeric_bin_fit_indices.npy")
    torch.set_num_threads(6)
    torch.manual_seed(schedule["seed"])
    np.random.seed(schedule["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for precision-matched teacher adaptation")
    torch.cuda.manual_seed_all(schedule["seed"])
    device = torch.device("cuda")
    options = dict(teacher["model_options"])
    options.update(hierarchy_mode="metadata", prior_width=schedule["prior_width"], support_temperature=200., clip_numeric=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"The \d+-th feature has just two bin edges.*", category=UserWarning)
        model = HierarchicalIDTabM(manifest["categorical_fields"], manifest["sequence_fields"], manifest["group_for_field"],
                                  manifest["vocabulary_sizes"], **options)
    protected = initialize_protected_adapter(model, teacher["state_dict"])
    support = fit_exposure_support(arrays["categories"], train, manifest)
    for field, count in support.items():
        getattr(model, "support_counts_" + field).copy_(torch.as_tensor(count, dtype=torch.float32))
    model.to(device)
    np.savez(output / "training_exposure_support.npz", **support)
    write_json(output / "support_initialization.json", {"fitting_rule": "Advertisement training current-field X only; no label or validation input counts",
        "train_indices_sha256": hashlib.sha256((cache / "train_indices.npy").read_bytes()).hexdigest(),
        "support_sha256": {field: hashlib.sha256(count.tobytes()).hexdigest() for field, count in support.items()},
        "shared_prior_fields": model.hierarchy_priors, "initial_temperature": 200., "mode": "metadata"})
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=schedule["learning_rate"], weight_decay=schedule["weight_decay"])
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.BCEWithLogitsLoss()
    inference_batch = min(8192, max(512, 65536 // options["k"]))

    def batch(rows):
        return (torch.as_tensor(arrays["categories"][rows].astype(np.int64), device=device),
                torch.as_tensor(arrays["sequences"][rows].astype(np.int64), device=device),
                torch.as_tensor(np.array(arrays["numeric"][rows]), device=device))

    def predict():
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(valid), inference_batch):
                with torch.cuda.amp.autocast():
                    logits = model(*batch(valid[start:start + inference_batch]))
                predictions.append(member_probability(logits).cpu().numpy())
        return np.concatenate(predictions)

    def assert_base_unchanged():
        current = model.state_dict()
        for name, original in teacher["state_dict"].items():
            assert torch.equal(current[name].detach().cpu(), original), f"Protected base changed: {name}"

    started = time.perf_counter()
    initial_prediction = predict()
    saved_teacher_prediction = np.load(args.teacher_dir / "validation_prediction.npy")
    np.testing.assert_allclose(initial_prediction, saved_teacher_prediction, rtol=3e-3, atol=2e-6)
    initial_metrics = metrics(arrays["label"][valid], initial_prediction)
    assert abs(initial_metrics["auc"] - source_record["auc"]) < 2e-5
    protected.update(teacher_dir=str(args.teacher_dir.resolve()), teacher_checkpoint_sha256=hashlib.sha256((args.teacher_dir / "model.pt").read_bytes()).hexdigest(),
                     initial_teacher_replay_max_error=float(np.max(np.abs(initial_prediction - saved_teacher_prediction))),
                     initial_teacher_replay_auc=initial_metrics["auc"], frozen_function_eval_mode=True)
    write_json(output / "protected_initialization.json", protected)
    history, best_auc, bad_evaluations = [], -1., 0
    record = None

    def evaluate(epoch, train_logloss, prediction=None):
        nonlocal best_auc, bad_evaluations, record
        prediction = predict() if prediction is None else prediction
        score = metrics(arrays["label"][valid], prediction)
        assert_base_unchanged()
        history.append({"epoch": epoch, "train_logloss": train_logloss, **score, "elapsed_seconds": time.perf_counter() - started,
                        "learning_rate": schedule["learning_rate"]})
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(json.dumps({"adapter_epoch": epoch, **score}), flush=True)
        if score["auc"] > best_auc + 1e-5:
            best_auc, bad_evaluations = score["auc"], 0
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            torch.save({"state_dict": state, "manifest": manifest, "model_class": "HierarchicalIDTabM", "model_options": options,
                        "epoch": epoch, "validation_metrics": score, "seed": 42, "ensemble_members": options["k"],
                        "inference_precision": "CUDA float16 autocast; probability mean in float32",
                        "training_regime": "protected_backbone_shared_input_adapters"}, output / "model.pt")
            np.save(output / "validation_prediction.npy", prediction)
            record = {**score, "family": "hierarchical", "model_class": "HierarchicalIDTabM", "model_options": options,
                      "seed": 42, "cache_dir": str(cache.resolve()), "validation_day": manifest["validation_day"],
                      "validation_strategy": manifest.get("validation_strategy", "random20"), "best_epoch": epoch,
                      "train_rows": len(train), "valid_rows": len(valid), "parameter_count": sum(p.numel() for p in model.parameters()),
                      "trainable_parameter_count": sum(p.numel() for p in parameters), "batch_size": schedule["batch_size"],
                      "amp": True, "ensemble_members": options["k"], "auxiliary_weight": 0., "source_cache_dir": None,
                      "ranking_weight": 0., "teacher_dir": str(args.teacher_dir.resolve()),
                      "training_regime": "protected_backbone_shared_input_adapters"}
            write_json(output / "best_so_far.json", record)
        else:
            bad_evaluations += 1

    print("PROTECTED_ADAPTER_START " + json.dumps(protected), flush=True)
    evaluate(0., None, initial_prediction)
    completed_epoch = 0.
    for epoch in range(1, schedule["max_epochs"] + 1):
        order = np.random.default_rng(42 + epoch).permutation(train)
        steps = (len(order) + schedule["batch_size"] - 1) // schedule["batch_size"]
        milestones = {int(np.ceil(steps * fraction)): fraction for fraction in [.25, .5, .75, 1.]}
        total_loss, seen = 0., 0
        for step, start in enumerate(range(0, len(order), schedule["batch_size"]), 1):
            rows = order[start:start + schedule["batch_size"]]
            target = torch.as_tensor(arrays["label"][rows].astype(np.float32), device=device)
            optimizer.zero_grad(set_to_none=True)
            # Keep the protected function deterministic while differentiating
            # through it to the newly introduced metadata networks.
            model.eval()
            with torch.cuda.amp.autocast():
                logits = model(*batch(rows))
                loss = criterion(logits.float(), target[:, None].expand_as(logits))
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite protected-adapter loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 5.)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(rows)
            seen += len(rows)
            if step % 250 == 0:
                print(f"  adapter epoch={epoch} rows={seen}/{len(order)}", flush=True)
            if step in milestones:
                completed_epoch = epoch - 1 + milestones[step]
                evaluate(completed_epoch, total_loss / seen)
                if bad_evaluations >= schedule["early_stop_evaluations"]:
                    break
        if bad_evaluations >= schedule["early_stop_evaluations"]:
            break
    assert_base_unchanged()
    record.update(epochs_completed=completed_epoch, training_seconds=time.perf_counter() - started,
                  architecture_source="Bounded-PLE official TabM32 with protected trained base and trainable shared metadata input priors",
                  scope="Reused held-out advertisement validation; final test unscored")
    write_json(output / "metrics.json", record)
    write_json(output / "protected_base_audit.json", {"status": "passed", "teacher_tensors_checked": len(teacher["state_dict"]),
        "all_teacher_tensors_unchanged": True, "only_prior_encoders_trainable": all(name.startswith("prior_encoders.") for name, p in model.named_parameters() if p.requires_grad),
        "initial_teacher_replay_max_error": protected["initial_teacher_replay_max_error"], "best_epoch": record["best_epoch"]})
    print("PROTECTED_ADAPTER_RESULT " + json.dumps({key: record[key] for key in ["auc", "best_epoch", "training_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
