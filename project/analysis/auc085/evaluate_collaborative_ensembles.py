"""Evaluate only the predeclared equal-weight collaborative architecture blends."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model_comparison"))
from train_models import metrics


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification-prefix", default="phase5_full")
    parser.add_argument("--reference-prediction", type=Path, default=HERE / "results_architecture_ensembles/dcn_tabm32_validation_prediction.npy")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results_collaborative_ensembles")
    args = parser.parse_args()
    verification = load(HERE / (args.verification_prefix + "_verification.json"))
    assert verification["status"] == "passed"
    records = load(HERE / (args.verification_prefix + "_verified_results.json"))
    cache = Path(records[0]["cache_dir"])
    valid = np.load(cache / "valid_indices.npy")
    target = np.load(cache / "label.npy", mmap_mode="r")[valid]
    dcn_dir = HERE / "results/random20_rich_oof_statistics/bounded/seed42"
    dcn_record = load(dcn_dir / "metrics.json")
    np.testing.assert_array_equal(valid, np.load(Path(dcn_record["cache_dir"]) / "valid_indices.npy"))
    dcn = np.load(dcn_dir / "validation_prediction.npy").astype(np.float64)
    previous = np.load(args.reference_prediction).astype(np.float64)
    previous_score = metrics(target, previous)
    output = args.output_dir.resolve()
    output.mkdir(exist_ok=True)
    result = []
    for record in records:
        if record["model_options"]["memory_mode"] == "none":
            continue
        assert record["replay_status"] == "full_precision_matched_replay_passed"
        np.testing.assert_array_equal(valid, np.load(Path(record["cache_dir"]) / "valid_indices.npy"))
        value = np.load(Path(record["results_dir"]) / "validation_prediction.npy").astype(np.float64)
        assert abs(metrics(target, value)["auc"] - record["auc"]) < 1e-10
        for name, baseline in [("replace_tabm32", dcn), ("combine_previous_best", previous)]:
            prediction = .5 * baseline + .5 * value
            label = record["experiment"] + "__" + name
            scores = metrics(target, prediction)
            np.save(output / (label + ".npy"), prediction)
            result.append({"name": label, **scores, "model_dir": record["results_dir"],
                "max_neighbors": record["model_options"]["max_neighbors"], "combination": name,
                "weights": {"baseline": .5, "collaborative": .5}, "weight_fitting": "none; fixed before training",
                "baseline_prediction_file": str((dcn_dir / "validation_prediction.npy").resolve() if name == "replace_tabm32" else args.reference_prediction.resolve()),
                "validated_rows": len(valid), "prediction_file": str(output / (label + ".npy")),
                "auc_change_from_previous_best": scores["auc"] - previous_score["auc"]})
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"previous_best_auc": previous_score["auc"], "fixed_combinations": [{key: row[key] for key in ["name", "auc", "auc_change_from_previous_best"]} for row in result]}), flush=True)


if __name__ == "__main__":
    main()
