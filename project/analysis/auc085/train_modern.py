"""Train modern tabular architectures on the audited CTR caches."""
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
BENCH = HERE.parent / "model_comparison"
sys.path.insert(0, str(BENCH))
from train_models import load_arrays, metrics, write_json
from modern_architectures import TabMFieldModel, ClippedTabMFieldModel, FieldTransformerModel, fit_numeric_bins, member_probability
from cross_domain_model import CrossDomainGatedTabM
from interaction_architectures import InteractionExpertTabM
from attentive_transfer import AttentiveCrossDomainTabM
from graph_interest_model import GraphInterestTabM
from ad_history_model import AdHistoryTabM
from ranking_objective import member_ranking_loss
from hierarchical_id_model import HierarchicalIDTabM, fit_exposure_support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--family", choices=["tabm", "tabm_ple", "tabm_ple_clip", "tabm_cin", "transformer", "joint", "joint_attention", "graph", "ad_history", "hierarchical"], required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--aux-weight", type=float, default=.05)
    parser.add_argument("--numeric-embedding", choices=["raw", "ple"], default="raw")
    parser.add_argument("--ensemble-members", type=int, default=8)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--id-cap", type=float, default=.2)
    parser.add_argument("--num-embedding-dim", type=int, default=8)
    parser.add_argument("--graph-teacher", type=Path)
    parser.add_argument("--graph-fine-tune", action="store_true")
    parser.add_argument("--history-mode", choices=["mean", "query", "position"], default="position")
    parser.add_argument("--rank-weight", type=float, default=0.)
    parser.add_argument("--rank-negatives", type=int, default=128)
    parser.add_argument("--hierarchy-mode", choices=["raw", "metadata", "shrink", "shared"], default="shrink")
    parser.add_argument("--support-temperature", type=float, default=200.)
    parser.add_argument("--prior-width", type=int, default=32)
    args = parser.parse_args()
    if args.rank_weight < 0 or args.rank_negatives < 1:
        raise ValueError("Ranking weight must be nonnegative and negative cap positive")
    output = HERE / "results_modern" / args.tag / args.family / f"seed{args.seed}"
    if (output / "model.pt").exists():
        raise FileExistsError(f"Preserving existing model: {output}")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    provenance = {}
    for path in [HERE / "modern_architectures.py", HERE / "cross_domain_model.py", HERE / "attentive_transfer.py", HERE / "interaction_architectures.py", HERE / "graph_interest_model.py", HERE / "source_graph.py", HERE / "ad_history_model.py", HERE / "ranking_objective.py", HERE / "hierarchical_id_model.py", HERE / "train_modern.py", HERE / "phase2_protocol.json", HERE / "phase3_ad_history_protocol.json", HERE / "phase3_ranking_protocol.json", HERE / "phase4_hierarchy_protocol.json"]:
        shutil.copy2(path, snapshot / path.name)
        provenance[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output / "provenance.json", {"snapshot_sha256": provenance, "tabm_version": importlib.metadata.version("tabm"),
               "rtdl_num_embeddings_version": importlib.metadata.version("rtdl_num_embeddings")})
    torch.set_num_threads(6)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    arrays = load_arrays(args.cache_dir)
    manifest = json.loads((args.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    train, valid = arrays["train_indices"], arrays["valid_indices"]
    assert not np.intersect1d(train, valid).size
    if manifest["validation_day"] is not None:
        assert arrays["date"][train].max() < manifest["validation_day"]
        assert np.all(arrays["date"][valid] == manifest["validation_day"])
    options = {"numeric_dim": len(manifest["numeric_fields"]), "dropout": .1, "unknown_dropout": .01}
    if args.family == "transformer":
        builder = FieldTransformerModel
        options.update(embedding_dim=16, d_token=32, n_layers=2, n_heads=4)
        batch_size = args.batch_size or 512
    else:
        builder = TabMFieldModel
        if args.family == "tabm_ple_clip":
            builder = ClippedTabMFieldModel
        if args.family == "tabm_cin":
            builder = InteractionExpertTabM
            options.update(cin_widths=[32, 32])
        if args.family == "ad_history":
            builder = AdHistoryTabM
            options.update(history_mode=args.history_mode, history_max_slots=arrays["sequences"].shape[-1])
        if args.family == "hierarchical":
            builder = HierarchicalIDTabM
            options.update(hierarchy_mode=args.hierarchy_mode, support_temperature=args.support_temperature,
                           prior_width=args.prior_width, clip_numeric=True)
        options.update(embedding_dim=args.embedding_dim, n_blocks=3, d_block=args.width, k=args.ensemble_members)
        options["group_norm_caps"] = {name: args.id_cap for name in ["user_id", "ad_task", "adv_id"]} if args.id_cap > 0 else {}
        batch_size = args.batch_size or 2048
        if args.family in ["tabm_ple", "tabm_ple_clip"] or ((args.family.startswith("joint") or args.family == "hierarchical") and args.numeric_embedding == "ple"):
            fitting = np.sort(np.random.default_rng(20261027).choice(train, min(200000, len(train)), replace=False))
            np.save(output / "numeric_bin_fit_indices.npy", fitting)
            options.update(numeric_bin_edges=fit_numeric_bins(arrays["numeric"][fitting], 12), numeric_embedding_dim=args.num_embedding_dim)
    source_arrays = None
    graph_metadata = None
    if args.family == "graph":
        if args.graph_teacher is None:
            raise ValueError("A completed source graph teacher is required")
        graph_metadata = json.loads((args.graph_teacher / "manifest.json").read_text(encoding="utf-8"))
        assert graph_metadata["status"] == "complete" and not graph_metadata["advertisement_labels_used"]
        assert graph_metadata["source_cutoff_day"] <= max(arrays["date"][train].max(), arrays["date"][valid].max())
        options.update(graph_dim=graph_metadata["dimension"], protected_source=not args.graph_fine_tune)
        builder = GraphInterestTabM
    if args.family.startswith("joint"):
        if args.source_cache is None or args.aux_weight < 0:
            raise ValueError("Joint learning requires a matching source cache and nonnegative source loss weight")
        source_schema = json.loads((args.source_cache / "manifest.json").read_text(encoding="utf-8"))
        assert source_schema["ad_manifest_sha256"] == hashlib.sha256((args.cache_dir / "manifest.json").read_bytes()).hexdigest()
        source_arrays = {name: np.load(args.source_cache / (name + ".npy"), mmap_mode="r")
                         for name in ["categories", "sequences", "numeric", "label", "date"]}
        assert source_arrays["date"].max() <= max(arrays["date"][train].max(), arrays["date"][valid].max())
        options.update(source_schema=source_schema, profile_dim=64)
        builder = AttentiveCrossDomainTabM if args.family == "joint_attention" else CrossDomainGatedTabM
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"The \d+-th feature has just two bin edges.*", category=UserWarning)
        model = builder(manifest["categorical_fields"], manifest["sequence_fields"], manifest["group_for_field"],
                        manifest["vocabulary_sizes"], initial_ctr=float(arrays["label"][train].mean()), **options).to(device)
    if graph_metadata is not None:
        raw_ids = np.load(args.graph_teacher / "user_raw_ids.npy")
        vectors = np.load(args.graph_teacher / "user_vectors.npy")
        source_count = np.load(args.graph_teacher / "user_positive_count.npy")
        raw_to_graph = {str(int(raw)): index for index, raw in enumerate(raw_ids)}
        vocabulary = json.loads((args.cache_dir / "vocabs.json").read_text(encoding="utf-8"))["user_id"]
        mapped = 0
        with torch.no_grad():
            for raw, encoded in vocabulary.items():
                if raw in raw_to_graph:
                    graph_row = raw_to_graph[raw]
                    model.source_user_vectors[encoded].copy_(torch.as_tensor(vectors[graph_row], device=device))
                    model.source_positive_count[encoded] = int(source_count[graph_row])
                    mapped += 1
        assert mapped > 0
        write_json(output / "graph_initialization.json", {"teacher_dir": str(args.graph_teacher.resolve()),
                   "teacher_manifest_sha256": hashlib.sha256((args.graph_teacher / "manifest.json").read_bytes()).hexdigest(),
                   "mapped_target_users": mapped, "protected_source": not args.graph_fine_tune})
    if args.family == "hierarchical":
        support = fit_exposure_support(arrays["categories"], train, manifest)
        with torch.no_grad():
            for field, count in support.items():
                getattr(model, "support_counts_" + field).copy_(torch.as_tensor(count, dtype=torch.float32, device=device))
        np.savez(output / "training_exposure_support.npz", **support)
        write_json(output / "support_initialization.json", {
            "fitting_rule": "Advertisement training current-field X only; no label or validation input counts",
            "train_indices_sha256": hashlib.sha256((args.cache_dir / "train_indices.npy").read_bytes()).hexdigest(),
            "support_sha256": {field: hashlib.sha256(count.tobytes()).hexdigest() for field, count in support.items()},
            "shared_prior_fields": model.hierarchy_priors,
            "initial_temperature": args.support_temperature,
            "mode": args.hierarchy_mode})
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=1, min_lr=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.BCEWithLogitsLoss()

    def batch(indices):
        return (torch.as_tensor(arrays["categories"][indices].astype(np.int64), device=device),
                torch.as_tensor(arrays["sequences"][indices].astype(np.int64), device=device),
                torch.as_tensor(np.array(arrays["numeric"][indices]), device=device))

    def predict():
        model.eval()
        predictions = []
        inference_batch = 1024 if args.family == "transformer" else min(8192, max(512, 65536 // options.get("k", 8)))
        with torch.no_grad():
            for start in range(0, len(valid), inference_batch):
                with torch.cuda.amp.autocast(enabled=amp):
                    logits = model(*batch(valid[start:start + inference_batch]))
                predictions.append(member_probability(logits).cpu().numpy())
        return np.concatenate(predictions)

    best_auc, bad_epochs = -1., 0
    history = []
    started = time.perf_counter()
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"MODERN_START {args.family} parameters={parameter_count} batch={batch_size} amp={amp} aux={args.aux_weight if source_arrays is not None else 0}", flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        shuffled = np.random.default_rng(args.seed + epoch).permutation(train)
        total_loss = 0.
        total_objective, total_source_loss, source_steps, total_rank_loss = 0., 0., 0, 0.
        if source_arrays is not None:
            source_rng = np.random.default_rng(args.seed + 10000 + epoch)
            source_order = source_rng.permutation(len(source_arrays["label"]))
            source_pointer = 0
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start:start + batch_size]
            target = torch.as_tensor(arrays["label"][indices].astype(np.float32), device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(*batch(indices))
                ad_loss = criterion(logits.float(), target[:, None].expand_as(logits))
                source_loss = ad_loss.new_zeros(())
                if source_arrays is not None and args.aux_weight > 0:
                    if source_pointer + 768 > len(source_order):
                        source_order = source_rng.permutation(len(source_order))
                        source_pointer = 0
                    source_indices = source_order[source_pointer:source_pointer + 768]
                    source_pointer += len(source_indices)
                    source_batch = (torch.as_tensor(source_arrays["categories"][source_indices].astype(np.int64), device=device),
                                    torch.as_tensor(source_arrays["sequences"][source_indices].astype(np.int64), device=device),
                                    torch.as_tensor(np.array(source_arrays["numeric"][source_indices]), device=device))
                    source_target = torch.as_tensor(source_arrays["label"][source_indices].astype(np.float32), device=device)
                    # Keep source dropout randomness from changing the ad dropout stream.
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if amp else []):
                        torch.manual_seed(args.seed + epoch * 100000 + start // batch_size + 10000000)
                        source_logits = model.forward_source(*source_batch)
                        source_loss = criterion(source_logits.float(), source_target)
                    total_source_loss += float(source_loss.detach())
                    source_steps += 1
                rank_loss = member_ranking_loss(logits, target, args.rank_negatives) if args.rank_weight else ad_loss.new_zeros(())
                loss = ad_loss + args.rank_weight * rank_loss + (args.aux_weight * source_loss if source_arrays is not None else 0.)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite modern-model loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(ad_loss.detach()) * len(indices)
            total_objective += float(loss.detach()) * len(indices)
            total_rank_loss += float(rank_loss.detach()) * len(indices)
            if (start // batch_size) % 250 == 0:
                print(f"  epoch={epoch} rows={start}/{len(train)}", flush=True)
        prediction = predict()
        result = metrics(arrays["label"][valid], prediction)
        entry = {"epoch": epoch, "train_logloss": total_loss / len(train), **result,
                 "train_objective": total_objective / len(train),
                 "source_train_logloss": total_source_loss / source_steps if source_steps else None,
                 "train_ranking_loss": total_rank_loss / len(train),
                 "epoch_seconds": time.perf_counter() - epoch_started,
                 "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(entry)
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(json.dumps(entry), flush=True)
        if result["auc"] > best_auc + 1e-5:
            best_auc, bad_epochs = result["auc"], 0
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            checkpoint = {"state_dict": state, "manifest": manifest, "model_class": builder.__name__, "model_options": options,
                          "epoch": epoch, "validation_metrics": result, "seed": args.seed,
                          "inference_precision": "CUDA float16 autocast; probability mean in float32" if amp else "float32",
                          "ensemble_members": options.get("k", 1)}
            checkpoint["auxiliary_weight"] = args.aux_weight if source_arrays is not None else 0.
            checkpoint.update(ranking_weight=args.rank_weight, ranking_negative_cap=args.rank_negatives)
            torch.save(checkpoint, output / "model.pt")
            np.save(output / "validation_prediction.npy", prediction)
            record = {**result, "family": args.family, "model_class": builder.__name__, "model_options": options,
                      "seed": args.seed, "cache_dir": str(args.cache_dir.resolve()), "validation_day": manifest["validation_day"],
                      "validation_strategy": manifest.get("validation_strategy", "time16"), "best_epoch": epoch,
                      "train_rows": len(train), "valid_rows": len(valid), "parameter_count": parameter_count,
                      "batch_size": batch_size, "amp": amp, "ensemble_members": options.get("k", 1)}
            record.update(auxiliary_weight=args.aux_weight if source_arrays is not None else 0.,
                          ranking_weight=args.rank_weight, ranking_negative_cap=args.rank_negatives,
                          source_cache_dir=str(args.source_cache.resolve()) if args.source_cache is not None else None)
            if graph_metadata is not None:
                record.update(graph_teacher_dir=str(args.graph_teacher.resolve()), protected_source=not args.graph_fine_tune,
                              trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad))
            write_json(output / "best_so_far.json", record)
        else:
            bad_epochs += 1
        scheduler.step(result["auc"])
        if bad_epochs >= 3:
            break
    record.update(epochs_completed=len(history), training_seconds=time.perf_counter() - started,
                  architecture_source=("Official TabM with shared metadata priors and exposure-supported current user/task/material ID residuals" if args.family == "hierarchical" else "Official TabM plus candidate-conditioned ad click/close pooling and optional supplied-slot matching" if args.family == "ad_history" else "Source-only typed graph preference pretraining plus source user representation replacement and gated ad-query graph matching in official TabM" if args.family == "graph" else "Official TabM plus shared item-conditioned cross-domain interest attention, gated transfer and separate task heads" if args.family == "joint_attention" else "Official TabM plus explicit compressed high-order interactions and a learned mixture of deep, interaction and linear experts" if args.family == "tabm_cin" else "Official TabM backbone plus a shared source/ad user-interest tower, gated transfer and separate prediction heads" if args.family == "joint" else "Official TabM backbone with CTR feature adapter" if args.family.startswith("tabm") else "Custom compact field-token Transformer"),
                  scope="Held-out advertisement validation; no claim about final-test AUC")
    write_json(output / "metrics.json", record)
    print("MODERN_RESULT " + json.dumps({key: record[key] for key in ["family", "auc", "logloss", "best_epoch", "epochs_completed", "training_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
