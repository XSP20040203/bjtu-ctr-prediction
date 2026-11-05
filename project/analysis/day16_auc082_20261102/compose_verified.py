"""Evaluate fixed arithmetic combinations of independently verified predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from fit_stacker import HERE, ROOT, digest, write
from verify_stacker import rank_sum_auc

OUT = HERE/"runs/compose_verified_v1"
BASE = "rank_finetune_v1_day16"
ANCHOR = "previous_day_residual_v1"
DELTA_SOURCES = ["global_row_context_v1", "row_context_residual_v1", "context_scale_rank_v1",
                 "joint_group_calibration_v1", "neural_userday_v1", "catboost_daily_history_v1"]
WEIGHTS = [.25, .5, 1.]


def sources():
    return [BASE, ANCHOR]+DELTA_SOURCES


def evaluate(verifying):
    raw = pd.read_csv(ROOT/"2026ml/train.csv", usecols=["pt_d", "label"], dtype="int32")
    rows = np.flatnonzero(raw.pt_d.to_numpy() == 20191116)
    y = raw.label.to_numpy()[rows]
    if len(rows) != 421983 or int(y.sum()) != 3724:
        raise AssertionError("Complete16 cohort changed")
    protocol = json.loads((OUT/"protocol.json").read_text(encoding="utf8"))
    for p, expected in protocol["inputs_sha256"].items():
        if digest(p) != expected:
            raise AssertionError("Verified source changed")
    for name, expected in protocol["raw_datasets_sha256"].items():
        if digest(ROOT/"2026ml"/name) != expected:
            raise AssertionError("Original dataset changed")
    values = {}
    for name in sources():
        directory = HERE/"runs"/name
        verified = json.loads((directory/"verification.json").read_text(encoding="utf8"))
        p = np.load(directory/"prediction.npy")
        if verified["status"] != "passed" or abs(rank_sum_auc(y, p)-verified["auc"]) > 1e-12:
            raise AssertionError("Source verification differs")
        np.testing.assert_array_equal(rows, np.load(directory/"valid_indices.npy"))
        values[name] = logit(p.clip(1e-6, 1.-1e-6))
    records = [{"source": BASE, "weight": 0., "auc": rank_sum_auc(y, expit(values[BASE]))}]
    predictions = [expit(values[BASE])]
    for source in DELTA_SOURCES:
        delta = values[source]-values[ANCHOR]
        for weight in WEIGHTS:
            p = expit(values[BASE]+weight*delta)
            records.append({"source": source, "weight": weight, "auc": rank_sum_auc(y, p)})
            predictions.append(p)
    best = int(np.argmax([r["auc"] for r in records]))
    p = predictions[best]
    error = 0.
    if verifying:
        error = float(np.max(np.abs(p-np.load(OUT/"prediction.npy"))))
        np.testing.assert_array_equal(rows, np.load(OUT/"valid_indices.npy"))
        if error > 1e-12:
            raise AssertionError("Arithmetic replay differs")
    else:
        np.save(OUT/"prediction.npy", p)
        np.save(OUT/"valid_indices.npy", rows)
    result = {"status": "passed" if verifying else "evaluated", "auc": records[best]["auc"],
              "best": records[best], "records": records, "rows": len(rows), "clicks": int(y.sum()),
              "maximum_replay_error": error, "original_datasets_unchanged": True,
              "evaluation16_labels_fit": False, "development16_model_selection": True,
              "target_met": records[best]["auc"] >= .82}
    write(OUT/("verification.json" if verifying else "result.json"), result)
    print(json.dumps({k:v for k,v in result.items() if k != "records"}, indent=2), flush=True)


def main():
    if OUT.exists():
        raise FileExistsError("Preserving arithmetic comparison")
    paths = [Path(__file__)]
    for name in sources():
        paths += [HERE/"runs"/name/file for file in ["verification.json", "prediction.npy", "valid_indices.npy"]]
    expected = json.loads((HERE/"runs/weighted_v1_day16/protocol.json").read_text(encoding="utf8"))["raw_datasets_sha256"]
    OUT.mkdir(parents=True)
    write(OUT/"protocol.json", {"inputs_sha256": {str(p): digest(p) for p in paths},
          "raw_datasets_sha256": expected, "base": BASE, "anchor": ANCHOR,
          "delta_sources": DELTA_SOURCES, "weights": WEIGHTS,
          "formula": "sigmoid(logit(base) + weight * (logit(source)-logit(anchor)))",
          "evaluation16_labels_fit": False, "development16_model_selection": True,
          "selection": "Fixed arithmetic recipes declared before16 scoring; no unbiased score claim"})
    evaluate(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    evaluate(True) if parser.parse_args().verify else main()
