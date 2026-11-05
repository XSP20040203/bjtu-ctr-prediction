"""Replay the legacy float32 DCN member before relying on a mixed-family ensemble."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model_comparison"))
from train_models import load_arrays, metrics, write_json
from bounded_id_models import BoundedIDDCN


def main():
    torch.set_num_threads(4)
    directory = HERE / "results/random20_rich_oof_statistics/bounded/seed42"
    record = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    arrays = load_arrays(Path(record["cache_dir"]))
    valid = arrays["valid_indices"]
    checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    assert checkpoint["model_class"] == "BoundedIDDCN"
    manifest = checkpoint["manifest"]
    model = BoundedIDDCN(manifest["categorical_fields"], manifest["sequence_fields"], manifest["group_for_field"],
                        manifest["vocabulary_sizes"], **checkpoint["model_options"]).cuda()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(valid), 8192):
            rows = valid[start:start + 8192]
            logits = model(torch.as_tensor(arrays["categories"][rows].astype(np.int64), device="cuda"),
                           torch.as_tensor(arrays["sequences"][rows].astype(np.int64), device="cuda"),
                           torch.as_tensor(np.array(arrays["numeric"][rows]), device="cuda"))
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    reproduced = np.concatenate(predictions)
    saved = np.load(directory / "validation_prediction.npy")
    np.testing.assert_allclose(reproduced, saved, rtol=2e-5, atol=1e-7)
    score = metrics(arrays["label"][valid], reproduced)
    assert abs(score["auc"] - record["auc"]) < 1e-10
    modern = json.loads((HERE / "phase2_full_verified_results.json").read_text(encoding="utf-8"))
    assert all(row["replay_status"] == "full_precision_matched_replay_passed" for row in modern)
    assert all(row["replay_max_error"] == 0. for row in modern)
    result = {"status": "passed", "rows": len(valid), "reference_dcn_auc": score["auc"],
              "maximum_probability_error": float(np.max(np.abs(reproduced - saved))),
              "reference_precision": "float32 matching original inference", "modern_precision": "full CUDA AMP replay",
              "ensemble_weights": "Fixed before scoring and reconstructed from matching validation rows in build_phase2_report.py"}
    write_json(HERE / "phase2_ensemble_verification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
