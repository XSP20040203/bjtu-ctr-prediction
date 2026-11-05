"""Cache a frozen teacher's logits, preserving canonical validation batch layout."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model_comparison"))
from train_models import load_arrays, metrics
from modern_architectures import ClippedTabMFieldModel, member_probability


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", type=Path, default=HERE / "results_modern/random20_architecture_capacity32/tabm_ple_clip/seed42")
    parser.add_argument("--output-dir", type=Path, default=HERE / "collaborative_base_tabm32")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving frozen-base cache {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA precision matching is required")
    torch.set_num_threads(6)
    checkpoint = torch.load(args.teacher_dir / "model.pt", map_location="cpu", weights_only=False)
    record = json.loads((args.teacher_dir / "metrics.json").read_text(encoding="utf-8"))
    assert checkpoint["model_class"] == "ClippedTabMFieldModel"
    cache = Path(record["cache_dir"])
    arrays = load_arrays(cache)
    manifest, options = checkpoint["manifest"], checkpoint["model_options"]
    assert manifest == json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    train, valid = arrays["train_indices"], arrays["valid_indices"]
    assert not np.intersect1d(train, valid).size
    assert np.unique(np.r_[train, valid]).size == manifest["rows"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"The \d+-th feature has just two bin edges.*", category=UserWarning)
        model = ClippedTabMFieldModel(manifest["categorical_fields"], manifest["sequence_fields"], manifest["group_for_field"],
            manifest["vocabulary_sizes"], **options).cuda()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    args.output_dir.mkdir(parents=True)
    logits = np.lib.format.open_memmap(args.output_dir / "base_logits.npy", mode="w+", dtype=np.float32,
                                       shape=(manifest["rows"], options["k"]))
    batch_size = min(8192, max(512, 65536 // options["k"]))
    with torch.no_grad():
        for partition, indices in [("train", train), ("validation", valid)]:
            for start in range(0, len(indices), batch_size):
                rows = indices[start:start + batch_size]
                inputs = (torch.as_tensor(arrays["categories"][rows].astype(np.int64), device="cuda"),
                          torch.as_tensor(arrays["sequences"][rows].astype(np.int64), device="cuda"),
                          torch.as_tensor(np.array(arrays["numeric"][rows]), device="cuda"))
                with torch.cuda.amp.autocast():
                    value = model(*inputs)
                logits[rows] = value.float().cpu().numpy()
                if start % (batch_size * 250) == 0:
                    print(f"BASE_CACHE {partition} rows={start}/{len(indices)}", flush=True)
    logits.flush()
    prediction = []
    with torch.no_grad():
        for start in range(0, len(valid), batch_size):
            rows = valid[start:start + batch_size]
            prediction.append(member_probability(torch.as_tensor(np.array(logits[rows]), device="cuda")).cpu().numpy())
    prediction = np.concatenate(prediction)
    saved = np.load(args.teacher_dir / "validation_prediction.npy")
    np.testing.assert_allclose(prediction, saved, rtol=3e-3, atol=2e-6)
    score = metrics(arrays["label"][valid], prediction)
    assert abs(score["auc"] - record["auc"]) < 2e-5
    metadata = {"status": "complete", "teacher_dir": str(args.teacher_dir.resolve()), "cache_dir": str(cache.resolve()),
                "teacher_checkpoint_sha256": digest(args.teacher_dir / "model.pt"),
                "cache_manifest_sha256": digest(cache / "manifest.json"),
                "train_indices_sha256": digest(cache / "train_indices.npy"), "valid_indices_sha256": digest(cache / "valid_indices.npy"),
                "base_logits_sha256": digest(args.output_dir / "base_logits.npy"), "ensemble_members": options["k"],
                "dtype": "float32, no downcast beyond original inference", "inference_batch_size": batch_size,
                "validation_replay_max_probability_error": float(np.max(np.abs(prediction - saved))), "validation_auc": score["auc"],
                "scope": "Frozen teacher inference only; labels used only for validation scoring, never input fitting"}
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
