"""Train controlled collaborative attention with cached, protected TabM32 logits."""
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
from collaborative_model import CollaborativeAttentionTabM, initialize_collaborative_adapter
from signed_collaborative_model import SignedCollaborativeTabM, initialize_signed_adapter
from collaborative_memory import MEMORY_ARRAYS
from modern_architectures import member_probability
from cache_collaborative_base import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["none", "neighbors", "degrees", "mean", "positive", "contrast"], required=True)
    parser.add_argument("--tag")
    parser.add_argument("--memory-dir", type=Path, default=HERE / "collaborative_memory_random20_k16")
    parser.add_argument("--base-cache", type=Path, default=HERE / "collaborative_base_tabm32")
    parser.add_argument("--protocol", type=Path, default=HERE / "phase5_collaborative_protocol.json")
    parser.add_argument("--teacher-dir", type=Path)
    parser.add_argument("--negative-memory", type=Path, default=HERE / "negative_collaborative_memory_random20_k16")
    parser.add_argument("--learn-queries", action="store_true")
    args = parser.parse_args()
    is_signed = args.mode in ["positive", "contrast"]
    if args.learn_queries and not is_signed:
        raise ValueError("Independent query adaptation requires the signed expert")
    family = "signed" if is_signed else "collaborative"
    regime = "protected_signed_collaborative" if is_signed else "protected_collaborative_attention"
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    schedule = protocol["optimization"]
    base_meta = json.loads((args.base_cache / "manifest.json").read_text(encoding="utf-8"))
    memory_meta = json.loads((args.memory_dir / "manifest.json").read_text(encoding="utf-8"))
    assert base_meta["status"] == memory_meta["status"] == "complete"
    assert memory_meta["max_neighbors"] == schedule["max_neighbors"]
    assert not memory_meta["advertisement_validation_labels_used"]
    cache = Path(base_meta["cache_dir"])
    teacher_directory = args.teacher_dir or (HERE / protocol["teacher"] if is_signed else Path(base_meta["teacher_dir"]))
    assert base_meta["cache_manifest_sha256"] == memory_meta["cache_manifest_sha256"] == digest(cache / "manifest.json")
    assert base_meta["teacher_checkpoint_sha256"] == digest(Path(base_meta["teacher_dir"]) / "model.pt")
    if is_signed:
        positive_initial = json.loads((teacher_directory / "collaborative_initialization.json").read_text(encoding="utf-8"))
        assert positive_initial["teacher_checkpoint_sha256"] == base_meta["teacher_checkpoint_sha256"]
        assert positive_initial["memory_manifest_sha256"] == digest(args.memory_dir / "manifest.json")
        negative_meta = json.loads((args.negative_memory / "manifest.json").read_text(encoding="utf-8"))
        assert negative_meta["status"] == "complete" and not negative_meta["advertisement_validation_labels_used"]
        assert negative_meta["positive_memory_manifest_sha256"] == digest(args.memory_dir / "manifest.json")
        assert json.loads((args.negative_memory / "audit.json").read_text(encoding="utf-8"))["status"] == "passed"
        for name, expected in negative_meta["array_sha256"].items():
            assert digest(args.negative_memory / (name + ".npy")) == expected
    assert base_meta["base_logits_sha256"] == digest(args.base_cache / "base_logits.npy")
    for name, expected in memory_meta["array_sha256"].items():
        assert digest(args.memory_dir / (name + ".npy")) == expected
    default_tag = ("phase6_signed_" if is_signed else "phase5_collaborative_") + args.mode
    if args.learn_queries:
        default_tag += "_learned_queries"
    output = HERE / "results_modern" / (args.tag or default_tag) / family / "seed42"
    if output.exists():
        raise FileExistsError(f"Preserving collaborative experiment {output}")
    arrays = load_arrays(cache)
    train, valid = arrays["train_indices"], arrays["valid_indices"]
    assert not np.intersect1d(train, valid).size
    assert digest(cache / "train_indices.npy") == memory_meta["train_indices_sha256"] == base_meta["train_indices_sha256"]
    assert hashlib.sha256(np.asarray(arrays["label"][train]).tobytes()).hexdigest() == memory_meta["train_label_sha256"]
    if is_signed:
        assert negative_meta["train_label_sha256"] == memory_meta["train_label_sha256"]
        assert negative_meta["train_indices_sha256"] == memory_meta["train_indices_sha256"]
    teacher = torch.load(teacher_directory / "model.pt", map_location="cpu", weights_only=False)
    manifest = teacher["manifest"]
    assert manifest == json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    names = ["modern_architectures.py", "collaborative_memory.py", "collaborative_model.py", "train_collaborative.py"]
    if is_signed:
        names += ["signed_collaborative_model.py", "negative_collaboration.py"]
    hashes = {}
    for name in names:
        shutil.copy2(HERE / name, snapshot / name)
        hashes[name] = digest(HERE / name)
    shutil.copy2(args.protocol, snapshot / args.protocol.name)
    hashes[args.protocol.name] = digest(args.protocol)
    write_json(output / "provenance.json", {"snapshot_sha256": hashes, "tabm_version": importlib.metadata.version("tabm"),
        "rtdl_num_embeddings_version": importlib.metadata.version("rtdl_num_embeddings")})
    shutil.copy2(teacher_directory / "numeric_bin_fit_indices.npy", output / "numeric_bin_fit_indices.npy")
    torch.set_num_threads(6)
    torch.manual_seed(42)
    np.random.seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA precision matching is required")
    torch.cuda.manual_seed_all(42)
    options = dict(teacher["model_options"])
    options.update(memory_mode=options["memory_mode"] if is_signed else args.mode, max_neighbors=memory_meta["max_neighbors"], positive_edge_count=memory_meta["positive_edges"],
                   metadata_fields=memory_meta["metadata_fields"], source_dim=memory_meta["source_dimension"], graph_dim=schedule["graph_dim"])
    if is_signed:
        options.update(signed_mode=args.mode, negative_edge_count=negative_meta["negative_edges"], learn_queries=args.learn_queries)
    builder = SignedCollaborativeTabM if is_signed else CollaborativeAttentionTabM
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"The \d+-th feature has just two bin edges.*", category=UserWarning)
        model = builder(manifest["categorical_fields"], manifest["sequence_fields"], manifest["group_for_field"],
                                          manifest["vocabulary_sizes"], **options)
    initializer = initialize_signed_adapter if is_signed else initialize_collaborative_adapter
    initialized = initializer(model, teacher["state_dict"])
    memory_arrays = {name: np.load(args.memory_dir / (name + ".npy")) for name in MEMORY_ARRAYS}
    model.memory.install(memory_arrays)
    model.source_user_vectors.copy_(torch.from_numpy(np.load(args.memory_dir / "source_user_vectors.npy")))
    model.source_positive_count.copy_(torch.from_numpy(np.load(args.memory_dir / "source_positive_count.npy")))
    negative_arrays = None
    if is_signed:
        negative_arrays = {name: np.load(args.negative_memory / (name + ".npy")) for name in MEMORY_ARRAYS}
        model.negative_memory.install(negative_arrays)
    model.cuda()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=schedule["learning_rate"], weight_decay=schedule["weight_decay"])
    criterion, scaler = nn.BCEWithLogitsLoss(), torch.cuda.amp.GradScaler()
    cached_logits = np.load(args.base_cache / "base_logits.npy", mmap_mode="r")
    inference_batch = base_meta["inference_batch_size"]

    def inputs(rows):
        return (torch.as_tensor(arrays["categories"][rows].astype(np.int64), device="cuda"),
                torch.as_tensor(arrays["sequences"][rows].astype(np.int64), device="cuda"),
                torch.as_tensor(np.array(cached_logits[rows]), device="cuda"))

    def predict():
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(valid), inference_batch):
                with torch.cuda.amp.autocast():
                    logits = model.forward_with_base(*inputs(valid[start:start + inference_batch]))
                predictions.append(member_probability(logits).cpu().numpy())
        return np.concatenate(predictions)

    def check_frozen():
        state = model.state_dict()
        for name, original in teacher["state_dict"].items():
            assert torch.equal(state[name].detach().cpu(), original), f"Protected teacher tensor changed: {name}"
        for name, original in memory_arrays.items():
            np.testing.assert_array_equal(state["memory." + name].cpu().numpy(), original)
        if negative_arrays is not None:
            for name, original in negative_arrays.items():
                np.testing.assert_array_equal(state["negative_memory." + name].cpu().numpy(), original)
        for name in ["source_user_vectors", "source_positive_count"]:
            np.testing.assert_array_equal(state[name].cpu().numpy(), np.load(args.memory_dir / (name + ".npy")))

    started = time.perf_counter()
    initial_prediction = predict()
    saved_teacher = np.load(teacher_directory / "validation_prediction.npy")
    np.testing.assert_allclose(initial_prediction, saved_teacher, rtol=3e-3, atol=2e-6)
    initialized.update(teacher_dir=str(teacher_directory), teacher_checkpoint_sha256=digest(teacher_directory / "model.pt"),
        base_teacher_checkpoint_sha256=base_meta["teacher_checkpoint_sha256"],
        memory_dir=str(args.memory_dir.resolve()), memory_manifest_sha256=digest(args.memory_dir / "manifest.json"),
        base_cache_dir=str(args.base_cache.resolve()), base_manifest_sha256=digest(args.base_cache / "manifest.json"),
        initial_teacher_replay_max_error=float(np.max(np.abs(initial_prediction - saved_teacher))), mode=args.mode)
    if is_signed:
        initialized.update(negative_memory_dir=str(args.negative_memory.resolve()),
                           negative_memory_manifest_sha256=digest(args.negative_memory / "manifest.json"))
    write_json(output / "collaborative_initialization.json", initialized)
    history, best_auc, bad_evaluations, record = [], -1., 0, None

    def evaluate(epoch, train_loss, prediction=None):
        nonlocal best_auc, bad_evaluations, record
        prediction = predict() if prediction is None else prediction
        result = metrics(arrays["label"][valid], prediction)
        check_frozen()
        history.append({"epoch": epoch, "train_logloss": train_loss, **result, "elapsed_seconds": time.perf_counter() - started,
                        "learning_rate": schedule["learning_rate"]})
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(json.dumps({"mode": args.mode, "epoch": epoch, **result}), flush=True)
        if result["auc"] > best_auc + 1e-5:
            best_auc, bad_evaluations = result["auc"], 0
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            torch.save({"state_dict": state, "manifest": manifest, "model_class": builder.__name__, "model_options": options,
                "epoch": epoch, "validation_metrics": result, "seed": 42, "ensemble_members": options["k"],
                "inference_precision": "CUDA float16 autocast; residual and probability mean float32", "training_regime": regime}, output / "model.pt")
            np.save(output / "validation_prediction.npy", prediction)
            record = {**result, "family": family, "model_class": builder.__name__, "model_options": options,
                "seed": 42, "cache_dir": str(cache), "validation_day": manifest["validation_day"], "validation_strategy": manifest.get("validation_strategy", "random20"),
                "best_epoch": epoch, "train_rows": len(train), "valid_rows": len(valid), "parameter_count": sum(p.numel() for p in model.parameters()),
                "trainable_parameter_count": sum(p.numel() for p in parameters), "batch_size": schedule["batch_size"], "amp": True,
                "ensemble_members": options["k"], "source_cache_dir": None, "auxiliary_weight": 0., "teacher_dir": str(teacher_directory),
                "memory_dir": str(args.memory_dir.resolve()), "training_regime": regime,
                "negative_memory_dir": str(args.negative_memory.resolve()) if is_signed else None}
            write_json(output / "best_so_far.json", record)
        else:
            bad_evaluations += 1

    print("COLLABORATIVE_START " + json.dumps(initialized), flush=True)
    evaluate(0., None, initial_prediction)
    completed = 0.
    for epoch in range(1, schedule["max_epochs"] + 1):
        order = np.random.default_rng(42 + epoch).permutation(train)
        n_steps = (len(order) + schedule["batch_size"] - 1) // schedule["batch_size"]
        milestones = {int(np.ceil(n_steps * fraction)): fraction for fraction in [.25, .5, .75, 1.]}
        total, seen = 0., 0
        model.train()
        for step, start in enumerate(range(0, len(order), schedule["batch_size"]), 1):
            rows = order[start:start + schedule["batch_size"]]
            target = torch.as_tensor(arrays["label"][rows].astype(np.float32), device="cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model.forward_with_base(*inputs(rows))
                loss = criterion(logits.float(), target[:, None].expand_as(logits))
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite collaborative objective")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 5.)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(rows)
            seen += len(rows)
            if step in milestones:
                completed = epoch - 1 + milestones[step]
                evaluate(completed, total / seen)
                model.train()
                if bad_evaluations >= schedule["early_stop_evaluations"]:
                    break
        if bad_evaluations >= schedule["early_stop_evaluations"]:
            break
    record.update(epochs_completed=completed, training_seconds=time.perf_counter() - started,
        architecture_source=("Protected positive collaborative model plus a comparative clicked/never-clicked neighbor expert" if is_signed else "Protected official TabM32 plus source-anchored, query-excluded collaborative user/task neighbor attention"),
        scope="Reused held-out advertisement validation; final test remains unscored")
    write_json(output / "metrics.json", record)
    write_json(output / "collaborative_frozen_audit.json", {"status": "passed", "teacher_tensors_unchanged": True,
        "memory_and_source_buffers_unchanged": True, "initial_teacher_replay_max_error": initialized["initial_teacher_replay_max_error"],
        "best_epoch": record["best_epoch"]})
    print("COLLABORATIVE_RESULT " + json.dumps({key: record[key] for key in ["auc", "best_epoch", "training_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
