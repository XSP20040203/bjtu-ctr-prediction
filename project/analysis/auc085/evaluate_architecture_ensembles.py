"""Evaluate predeclared fixed-probability mixtures without fitting blend weights."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model_comparison"))
from train_models import metrics, write_json


def main():
    protocol = json.loads((HERE / "architecture_ensemble_protocol.json").read_text(encoding="utf-8"))
    probabilities, indices, labels = {}, None, None
    member_paths = {}
    for name, relative in protocol["members"].items():
        directory = HERE / relative
        record = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        cache = Path(record["cache_dir"])
        valid = np.load(cache / "valid_indices.npy")
        target = np.load(cache / "label.npy", mmap_mode="r")[valid]
        if indices is None:
            indices, labels = valid, target
        else:
            np.testing.assert_array_equal(indices, valid)
            np.testing.assert_array_equal(labels, target)
        prediction = np.load(directory / "validation_prediction.npy")
        assert prediction.shape == (len(valid),) and np.isfinite(prediction).all()
        assert abs(metrics(target, prediction)["auc"] - record["auc"]) < 1e-10
        probabilities[name] = prediction
        member_paths[name] = str(directory)
    output = HERE / "results_architecture_ensembles"
    output.mkdir(exist_ok=True)
    records = []
    for name, weights in protocol["combinations"].items():
        assert abs(sum(weights.values()) - 1.) < 1e-12 and all(weight >= 0 for weight in weights.values())
        prediction = sum(probabilities[member].astype(np.float64) * weight for member, weight in weights.items())
        assert ((prediction >= 0) & (prediction <= 1)).all()
        np.save(output / (name + "_validation_prediction.npy"), prediction)
        result = {"name": name, "weights": weights, "members": {key: member_paths[key] for key in weights},
                  **metrics(labels, prediction), "validation_strategy": "random20", "seed": 42,
                  "weight_fitting": "none; weights fixed before scoring", "validated_rows": len(labels)}
        records.append(result)
    write_json(output / "metrics.json", records)
    pd.DataFrame([{key: row[key] for key in ["name", "auc", "logloss", "average_precision", "mean_prediction"]} for row in records]).to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
    print(pd.read_csv(output / "comparison.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
