"""Describe where the retained fixed blend improves on the same held-out rows."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_collaborative import metrics

HERE = Path(__file__).resolve().parent


def main():
    ensembles = json.loads((HERE / "results_collaborative_ensembles/metrics.json").read_text(encoding="utf-8"))
    best = max(ensembles, key=lambda row: row["auc"])
    record = json.loads((Path(best["model_dir"]) / "metrics.json").read_text(encoding="utf-8"))
    cache, memory = Path(record["cache_dir"]), Path(record["memory_dir"])
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    memory_meta = json.loads((memory / "manifest.json").read_text(encoding="utf-8"))
    valid = np.load(cache / "valid_indices.npy")
    category = np.load(cache / "categories.npy", mmap_mode="r")
    target = np.load(cache / "label.npy", mmap_mode="r")[valid]
    dates = np.load(cache / "date.npy", mmap_mode="r")[valid]
    user = category[valid, manifest["categorical_fields"].index("user_id")].astype(np.int64)
    task = category[valid, manifest["categorical_fields"].index("task_id")].astype(np.int64)
    keys = np.load(memory / "positive_keys.npy")
    query_keys = user * memory_meta["n_tasks"] + task
    positions = np.searchsorted(keys, query_keys)
    present = (positions < len(keys)) & (keys[np.minimum(positions, len(keys) - 1)] == query_keys)
    user_degree = np.load(memory / "user_degree.npy")[user] - present
    task_degree = np.load(memory / "task_degree.npy")[task] - present
    old = np.load(HERE / "results_architecture_ensembles/dcn_tabm32_validation_prediction.npy")
    new = np.load(best["prediction_file"])
    groups = [("全部验证样本", np.ones(len(valid), dtype=bool))]
    groups += [("两侧都有其他点击关系", (user_degree > 0) & (task_degree > 0)),
               ("只有用户点过其他广告", (user_degree > 0) & (task_degree == 0)),
               ("只有广告有其他点击用户", (user_degree == 0) & (task_degree > 0)),
               ("两侧都无其他点击关系", (user_degree == 0) & (task_degree == 0))]
    groups += [(str(int(day)), dates == day) for day in np.unique(dates)]
    result = []
    for name, mask in groups:
        clicks, rows = int(target[mask].sum()), int(mask.sum())
        enough = clicks >= 20 and rows - clicks >= 20
        before = metrics(target[mask], old[mask])["auc"] if enough else None
        after = metrics(target[mask], new[mask])["auc"] if enough else None
        result.append({"分组": name, "验证样本数": rows, "点击数": clicks, "旧组合AUC": before, "新组合AUC": after,
                       "AUC变化": after - before if enough else None,
                       "说明": "同一随机验证中的描述性分组" if enough else "点击或未点击样本少于20，不给AUC"})
    total_pairs = int(target.sum()) * int((target == 0).sum())
    relation_groups = groups[1:5]
    pair_matrix, before_total, after_total, within_delta, cross_delta = [], 0., 0., 0., 0.
    for i, (positive_name, positive_mask) in enumerate(relation_groups):
        positive = positive_mask & (target == 1)
        for j, (negative_name, negative_mask) in enumerate(relation_groups):
            negative = negative_mask & (target == 0)
            weight = int(positive.sum()) * int(negative.sum()) / total_pairs
            scores = []
            for probabilities in [old, new]:
                negative_values = np.sort(probabilities[negative])
                positives = probabilities[positive]
                left = np.searchsorted(negative_values, positives, side="left")
                right = np.searchsorted(negative_values, positives, side="right")
                scores.append(float(((left + right) / (2. * len(negative_values))).mean()))
            before_total += weight * scores[0]
            after_total += weight * scores[1]
            change = weight * (scores[1] - scores[0])
            if i == j:
                within_delta += change
            else:
                cross_delta += change
            pair_matrix.append({"positive_group": positive_name, "negative_group": negative_name,
                                "pair_weight": weight, "old_pair_auc": scores[0], "new_pair_auc": scores[1],
                                "weighted_auc_change": change})
    assert abs(before_total - result[0]["旧组合AUC"]) < 1e-12
    assert abs(after_total - result[0]["新组合AUC"]) < 1e-12
    pd.DataFrame(result).to_csv(HERE / "phase5_uplift_slices.csv", index=False, encoding="utf-8-sig")
    output = {"scope": "Descriptive slices of the reused random holdout; date slices are not prospective temporal tests; full-holdout AUC remains the goal metric",
              "best_fixed_blend": best["name"], "slices": result,
              "pair_decomposition": {"old_auc_reconstructed": before_total, "new_auc_reconstructed": after_total,
                                     "within_group_weighted_change": within_delta, "cross_group_weighted_change": cross_delta,
                                     "matrix": pair_matrix}}
    (HERE / "phase5_uplift_slices.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
