"""Verify completed modern-architecture runs, including source and bin provenance."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model_comparison"))
from train_models import load_arrays, metrics, write_json
from modern_architectures import (TabMFieldModel, ClippedTabMFieldModel, FieldTransformerModel,
                                  member_probability, fit_numeric_bins)
from cross_domain_model import CrossDomainGatedTabM
from attentive_transfer import AttentiveCrossDomainTabM
from interaction_architectures import InteractionExpertTabM
from graph_interest_model import GraphInterestTabM
from ad_history_model import AdHistoryTabM
from hierarchical_id_model import HierarchicalIDTabM
from collaborative_model import CollaborativeAttentionTabM
from collaborative_memory import MEMORY_ARRAYS
from signed_collaborative_model import SignedCollaborativeTabM
from align_frequency_features import training_input_frequency

BUILDERS = {cls.__name__: cls for cls in [TabMFieldModel, ClippedTabMFieldModel, FieldTransformerModel,
                                        CrossDomainGatedTabM, AttentiveCrossDomainTabM, InteractionExpertTabM,
                                        GraphInterestTabM, AdHistoryTabM, HierarchicalIDTabM, CollaborativeAttentionTabM, SignedCollaborativeTabM]}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_aligned_cache(cache):
    manifest = load(cache / "manifest.json")
    category = np.load(cache / "categories.npy", mmap_mode="r")
    numeric = np.load(cache / "numeric.npy", mmap_mode="r")
    train = np.load(cache / "train_indices.npy")
    base = np.load(Path(manifest["base_cache"]) / "numeric.npy", mmap_mode="r")
    for definition in manifest["target_stat_definitions"]:
        keys = np.zeros(len(category), dtype=np.int64)
        for field in definition["fields"]:
            column = manifest["categorical_fields"].index(field)
            size = manifest["vocabulary_sizes"][manifest["group_for_field"][field]]
            keys = keys * size + category[:, column]
        full_count = training_input_frequency(keys, train)
        name = "target_stat__" + "__".join(definition["fields"])
        count_column = manifest["numeric_fields"].index(name + "__log_count")
        parameters = manifest["numeric_scaler"][name + "__log_count"]
        expected = (np.log1p(full_count.astype(np.float32)) - parameters["mean"]) / parameters["std"]
        np.testing.assert_allclose(numeric[:, count_column], expected, rtol=2e-5, atol=2e-5)
        rate_column = manifest["numeric_fields"].index(name + "__relative_logit")
        np.testing.assert_array_equal(numeric[:, rate_column], base[:, rate_column])
    return {"cache": str(cache), "full_training_input_counts_recomputed": True,
            "cross_fitted_target_rate_columns_preserved_exactly": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-gpu", action="store_true")
    parser.add_argument("--tag-prefix", default="random20_", help="Select experiments without mixing research phases")
    parser.add_argument("--report-prefix", help="Output prefix, for example phase3_graph_full")
    args = parser.parse_args()
    if args.full_gpu and not torch.cuda.is_available():
        raise RuntimeError("Matching CUDA autocast is required for full precision-matched replay")
    torch.set_num_threads(4)
    records, source_checked, bin_checked = [], set(), set()
    aligned_audits = {}
    paths = sorted((HERE / "results_modern").glob("*/*/seed*/metrics.json"))
    paths = [path for path in paths if path.parts[-4].startswith(args.tag_prefix)]
    if not paths:
        raise ValueError(f"No completed runs match {args.tag_prefix!r}")
    for path in paths:
        record = load(path)
        cache = Path(record["cache_dir"])
        arrays = load_arrays(cache)
        train, valid = arrays["train_indices"], arrays["valid_indices"]
        assert not np.intersect1d(train, valid).size
        prediction = np.load(path.parent / "validation_prediction.npy")
        assert prediction.shape == (len(valid),)
        recomputed = metrics(arrays["label"][valid], prediction)
        for key, value in recomputed.items():
            assert abs(record[key] - value) < 1e-10, (path, key)
        history = pd.read_csv(path.parent / "training_history.csv")
        best = history.loc[history.epoch.eq(record["best_epoch"])].iloc[0]
        assert abs(best.auc - record["auc"]) < 1e-10
        provenance = load(path.parent / "provenance.json")
        for filename, digest in provenance["snapshot_sha256"].items():
            assert hashlib.sha256((path.parent / "source_snapshot" / filename).read_bytes()).hexdigest() == digest
        checkpoint = torch.load(path.parent / "model.pt", map_location="cpu", weights_only=False)
        manifest, options = checkpoint["manifest"], checkpoint["model_options"]
        assert manifest == load(cache / "manifest.json")
        assert checkpoint["epoch"] == record["best_epoch"]
        if checkpoint["model_class"] in ["CollaborativeAttentionTabM", "SignedCollaborativeTabM"]:
            initialization = load(path.parent / "collaborative_initialization.json")
            teacher_directory = Path(initialization["teacher_dir"])
            assert initialization["teacher_checkpoint_sha256"] == hashlib.sha256((teacher_directory / "model.pt").read_bytes()).hexdigest()
            teacher = torch.load(teacher_directory / "model.pt", map_location="cpu", weights_only=False)
            assert teacher["manifest"] == manifest
            for name, value in teacher["state_dict"].items():
                assert torch.equal(value, checkpoint["state_dict"][name]), f"Protected teacher tensor changed: {name}"
            memory_directory = Path(initialization["memory_dir"])
            assert initialization["memory_manifest_sha256"] == hashlib.sha256((memory_directory / "manifest.json").read_bytes()).hexdigest()
            memory_audit = load(memory_directory / "audit.json")
            assert memory_audit["status"] == "passed" and Path(memory_audit["memory_dir"]) == memory_directory
            for name in MEMORY_ARRAYS:
                np.testing.assert_array_equal(checkpoint["state_dict"]["memory." + name].numpy(), np.load(memory_directory / (name + ".npy")))
            for name in ["source_user_vectors", "source_positive_count"]:
                np.testing.assert_array_equal(checkpoint["state_dict"][name].numpy(), np.load(memory_directory / (name + ".npy")))
            base_directory = Path(initialization["base_cache_dir"])
            assert initialization["base_manifest_sha256"] == hashlib.sha256((base_directory / "manifest.json").read_bytes()).hexdigest()
            if checkpoint["model_class"] == "SignedCollaborativeTabM":
                negative_directory = Path(initialization["negative_memory_dir"])
                assert initialization["negative_memory_manifest_sha256"] == hashlib.sha256((negative_directory / "manifest.json").read_bytes()).hexdigest()
                negative_info = load(negative_directory / "manifest.json")
                assert not negative_info["advertisement_validation_labels_used"]
                assert load(negative_directory / "audit.json")["status"] == "passed"
                for name in MEMORY_ARRAYS:
                    np.testing.assert_array_equal(checkpoint["state_dict"]["negative_memory." + name].numpy(), np.load(negative_directory / (name + ".npy")))
                user_column, task_column = [manifest["categorical_fields"].index(name) for name in ["user_id", "task_id"]]
                user = arrays["categories"][train, user_column].astype(np.int64)
                task = arrays["categories"][train, task_column].astype(np.int64)
                task_size = manifest["vocabulary_sizes"][manifest["group_for_field"]["task_id"]]
                pair = user * task_size + task
                expected_negative = np.setdiff1d(np.unique(pair[arrays["label"][train] == 0]), np.unique(pair[arrays["label"][train] == 1]))
                saved_negative = checkpoint["state_dict"]["negative_memory.positive_keys"].numpy()
                np.testing.assert_array_equal(saved_negative, expected_negative)
                np.testing.assert_array_equal(checkpoint["state_dict"]["negative_memory.user_degree"].numpy(), np.bincount(expected_negative // task_size, minlength=len(checkpoint["state_dict"]["negative_memory.user_degree"])))
                np.testing.assert_array_equal(checkpoint["state_dict"]["negative_memory.task_degree"].numpy(), np.bincount(expected_negative % task_size, minlength=task_size))
                np.testing.assert_array_equal(checkpoint["state_dict"]["negative_memory.task_metadata"].numpy(), checkpoint["state_dict"]["memory.task_metadata"].numpy())
            del teacher
        if checkpoint["model_class"] == "HierarchicalIDTabM":
            support = np.load(path.parent / "training_exposure_support.npz")
            initialization = load(path.parent / "support_initialization.json")
            assert initialization["train_indices_sha256"] == hashlib.sha256((cache / "train_indices.npy").read_bytes()).hexdigest()
            for field in support.files:
                column = manifest["categorical_fields"].index(field)
                size = manifest["vocabulary_sizes"][manifest["group_for_field"][field]]
                count = np.bincount(arrays["categories"][train, column], minlength=size).astype(np.int64)
                count[:2] = 0
                np.testing.assert_array_equal(support[field], count)
                assert initialization["support_sha256"][field] == hashlib.sha256(count.tobytes()).hexdigest()
                np.testing.assert_array_equal(checkpoint["state_dict"]["support_counts_" + field].numpy(), count.astype(np.float32))
            if record.get("training_regime") == "protected_backbone_shared_input_adapters":
                initialization = load(path.parent / "protected_initialization.json")
                teacher_directory = Path(initialization["teacher_dir"])
                assert initialization["teacher_checkpoint_sha256"] == hashlib.sha256((teacher_directory / "model.pt").read_bytes()).hexdigest()
                teacher = torch.load(teacher_directory / "model.pt", map_location="cpu", weights_only=False)
                assert teacher["manifest"] == manifest
                for name, value in teacher["state_dict"].items():
                    assert torch.equal(value, checkpoint["state_dict"][name]), f"Protected teacher tensor changed: {name}"
                if record["best_epoch"] > 0:
                    assert sum(value.abs().sum().item() for name, value in checkpoint["state_dict"].items()
                               if name.startswith("prior_encoders.") and (name.endswith(".2.weight") or name.endswith(".2.bias"))) > 0
                del teacher
        if options.get("numeric_bin_edges") is not None:
            fit_path = path.parent / "numeric_bin_fit_indices.npy"
            fitting = np.load(fit_path)
            assert np.isin(fitting, train).all() and not np.intersect1d(fitting, valid).size
            key = (str(cache), hashlib.sha256(fit_path.read_bytes()).hexdigest())
            if key not in bin_checked:
                expected_bins = fit_numeric_bins(arrays["numeric"][fitting], 12)
                assert expected_bins == options["numeric_bin_edges"]
                bin_checked.add(key)
        source_directory = record.get("source_cache_dir")
        if source_directory is not None and source_directory not in source_checked:
            source = Path(source_directory)
            schema = load(source / "manifest.json")
            assert schema["ad_manifest_sha256"] == hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest()
            fields = schema["categorical_fields"] + schema["sequence_fields"] + schema["numeric_fields"]
            assert not {"label", "pro", "cillabel"}.intersection(name.lower() for name in fields)
            raw = pd.read_csv(HERE.parents[1] / "2026ml/train_feeds.csv", usecols=["e_et", "label"])
            raw = raw.loc[raw.e_et.le(schema["latest_day"])]
            np.testing.assert_array_equal(np.load(source / "label.npy"), raw.label.eq(1).to_numpy(dtype=np.uint8))
            assert np.load(source / "date.npy").max() <= max(arrays["date"][train].max(), arrays["date"][valid].max())
            source_checked.add(source_directory)
        if "frequency_alignment" in manifest and str(cache) not in aligned_audits:
            aligned_audits[str(cache)] = audit_aligned_cache(cache)
        device = torch.device("cuda" if args.full_gpu else "cpu")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"The \d+-th feature has just two bin edges.*", category=UserWarning)
            model = BUILDERS[checkpoint["model_class"]](manifest["categorical_fields"], manifest["sequence_fields"],
                    manifest["group_for_field"], manifest["vocabulary_sizes"], **options).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        selection = np.arange(len(valid)) if args.full_gpu else np.linspace(0, len(valid) - 1, 128, dtype=int)
        batch_size = (1024 if record["family"] == "transformer" else min(8192, max(512, 65536 // options.get("k", 8)))) if args.full_gpu else 128
        reproduced = []
        with torch.no_grad():
            for start in range(0, len(selection), batch_size):
                rows = valid[selection[start:start + batch_size]]
                inputs = (torch.as_tensor(arrays["categories"][rows].astype(np.int64), device=device),
                          torch.as_tensor(arrays["sequences"][rows].astype(np.int64), device=device),
                          torch.as_tensor(np.array(arrays["numeric"][rows]), device=device))
                with torch.cuda.amp.autocast(enabled=args.full_gpu):
                    logits = model(*inputs)
                reproduced.append(member_probability(logits).cpu().numpy())
        reproduced = np.concatenate(reproduced)
        expected = prediction[selection]
        error = float(np.max(np.abs(reproduced - expected)))
        if args.full_gpu:
            np.testing.assert_allclose(reproduced, expected, rtol=3e-3, atol=2e-6)
            replay_auc = metrics(arrays["label"][valid], reproduced)["auc"]
            assert abs(replay_auc - record["auc"]) < 2e-5
            probe_status = "full_precision_matched_replay_passed"
        else:
            close = np.allclose(reproduced, expected, rtol=.02, atol=3e-6)
            probe_status = "cpu_probe_close_to_amp" if close else "needs_precision_matched_gpu_replay"
            replay_auc = None
        record.update(experiment=path.parts[-4], results_dir=str(path.parent), replay_mode="full_gpu" if args.full_gpu else "cpu_probe",
                      replay_max_error=error, replay_auc=replay_auc, replay_status=probe_status)
        records.append(record)
        print(json.dumps({"experiment": record["experiment"], "family": record["family"], "auc": record["auc"],
                          "replay_status": probe_status, "max_probability_error": error}), flush=True)
        del model, checkpoint, inputs
        gc.collect()
        if args.full_gpu:
            torch.cuda.empty_cache()
    label = args.report_prefix or ("phase2_full" if args.full_gpu else "phase2_probe")
    if Path(label).name != label:
        raise ValueError("Report prefix must be a filename prefix")
    write_json(HERE / (label + "_verified_results.json"), records)
    write_json(HERE / (label + "_verification.json"), {"status": "passed" if args.full_gpu else "saved_metrics_verified_with_cpu_precision_probes",
               "completed_runs_verified": len(records), "all_saved_predictions_rescored": True,
               "checkpoint_replay": "all validation rows with original AMP precision" if args.full_gpu else "128-row CPU probe; see per-run precision flags",
               "source_caches_audited": len(source_checked), "numeric_bin_fit_sets_audited": len(bin_checked),
               "aligned_cache_audits": list(aligned_audits.values()),
               "auc_above_085_found": any(record["auc"] > .85 for record in records),
               "goal_complete": False})


if __name__ == "__main__":
    main()
