"""Describe the random-holdout/future-day gap without using hidden final labels."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE / "cache_rich_stats_random20_aligned"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def membership(values, observed):
    positions = np.searchsorted(observed, values)
    safe = np.minimum(positions, len(observed) - 1)
    return (positions < len(observed)) & (observed[safe] == values)


def main():
    manifest = load_json(BASE / "manifest.json")
    categories = np.load(BASE / "categories.npy", mmap_mode="r")
    numeric = np.load(BASE / "numeric.npy", mmap_mode="r")
    labels = np.load(BASE / "label.npy", mmap_mode="r")
    dates = np.load(BASE / "date.npy", mmap_mode="r")
    train, valid = [np.load(BASE / (name + ".npy")) for name in ["train_indices", "valid_indices"]]
    final_cat = np.load(HERE / "final_prediction_cache/categories.npy", mmap_mode="r")
    final_num = np.load(HERE / "final_prediction_cache/signed_numeric.npy", mmap_mode="r")
    best = np.load(HERE / "results_signed_ensembles/phase6_signed_contrast__replace_tabm32.npy")
    auc = float(roc_auc_score(labels[valid], best))
    user_reported_final_auc = .74404
    checkpoint = torch.load(HERE / "results_modern/phase6_signed_contrast/signed/seed42/model.pt", map_location="cpu", weights_only=False)
    result = {
        "scope": "Descriptive audit; no training or final labels accessed; user-reported external score is unverified locally",
        "final_score": {"auc": user_reported_final_auc, "provenance": "User reported final score in this chat", "labels_available": False},
        "random_validation_auc": auc, "auc_gap": auc - user_reported_final_auc,
        "rows": {"train": len(train), "valid": len(valid), "final": len(final_cat)},
        "positive_examples": {"all_labeled_ads": int(labels.sum()), "training": int(labels[train].sum()), "valid": int(labels[valid].sum())},
        "training_click_fraction": float(labels[train].mean()),
        "fraction_labeled_rows_used_for_model_weights": len(train) / len(labels),
        "test_day": 20191117,
    }
    result["current_model_random_validation_by_date"] = []
    for day in np.unique(dates[valid]):
        mask = dates[valid] == day
        target = labels[valid][mask]
        result["current_model_random_validation_by_date"].append({"date": int(day), "rows": int(mask.sum()), "clicks": int(target.sum()),
            "auc": float(roc_auc_score(target, best[mask])) if len(np.unique(target)) == 2 else None,
            "scope": "Date slice from random holdout, not a chronological validation"})
    result["unknown_current_category_fraction"] = {}
    for field in ["user_id", "task_id", "adv_id", "adv_prim_id", "spread_app_id", "device_name", "slot_id"]:
        column = manifest["categorical_fields"].index(field)
        result["unknown_current_category_fraction"][field] = {"random_valid": float((categories[valid, column] == 1).mean()),
                                                            "final": float((final_cat[:, column] == 1).mean())}
    user_col, task_col = [manifest["categorical_fields"].index(field) for field in ["user_id", "task_id"]]
    n_tasks = manifest["vocabulary_sizes"][manifest["group_for_field"]["task_id"]]
    train_pair = categories[train, user_col].astype(np.int64) * n_tasks + categories[train, task_col]
    known_pairs = np.unique(train_pair)
    result["unseen_user_task_pair_fraction"] = {}
    result["query_excluded_positive_graph_support"] = {}
    state = checkpoint["state_dict"]
    positive_edges = state["memory.positive_keys"].numpy()
    user_degree, task_degree = [state["memory." + field].numpy() for field in ["user_degree", "task_degree"]]
    for name, cats in [("random_valid", categories[valid]), ("final", final_cat)]:
        users, tasks = cats[:, user_col].astype(np.int64), cats[:, task_col].astype(np.int64)
        packed = users * n_tasks + tasks
        result["unseen_user_task_pair_fraction"][name] = float((~membership(packed, known_pairs)).mean())
        present = membership(packed, positive_edges) & (users >= 2) & (tasks >= 2)
        user_supported, task_supported = (user_degree[users] - present) > 0, (task_degree[tasks] - present) > 0
        result["query_excluded_positive_graph_support"][name] = {
            "both_sides": float((user_supported & task_supported).mean()),
            "user_only": float((user_supported & ~task_supported).mean()),
            "task_only": float((~user_supported & task_supported).mean()),
            "neither_side": float((~user_supported & ~task_supported).mean())}
    hashes = np.load(HERE / "raw_predictor_hash.npy", mmap_mode="r")
    result["random_valid_exact_raw_predictor_overlap_with_training"] = float(membership(hashes[valid], np.unique(hashes[train])).mean())
    result["feature_shift"] = []
    edges = checkpoint["model_options"]["numeric_bin_edges"]
    for column, field in enumerate(manifest["numeric_fields"]):
        train_values, final_values = numeric[train, column], final_num[:, column]
        mean, std = float(train_values.mean()), float(train_values.std())
        result["feature_shift"].append({"field": field, "train_mean": mean, "final_mean": float(final_values.mean()),
            "mean_shift_in_train_std": (float(final_values.mean()) - mean) / max(std, 1e-6),
            "final_fraction_clipped_to_train_bin_edges": float(((final_values < edges[column][0]) | (final_values > edges[column][-1])).mean())})
    result["feature_shift"].sort(key=lambda row: abs(row["mean_shift_in_train_std"]), reverse=True)
    active_column = manifest["numeric_fields"].index("feed_active_days")
    active_scale = manifest["numeric_scaler"]["feed_active_days"]
    train_active = np.rint(np.expm1(numeric[train, active_column] * active_scale["std"] + active_scale["mean"]))
    final_active = np.rint(np.expm1(final_num[:, active_column] * active_scale["std"] + active_scale["mean"]))
    values, counts = np.unique(final_active, return_counts=True)
    result["cumulative_active_days_saturation"] = {
        "training_maximum_raw_days": int(train_active.max()),
        "final_raw_days_counts": {str(int(value)): int(count) for value, count in zip(values, counts)},
        "signed_branch_upper_raw_days": float(np.expm1(edges[active_column][-1] * active_scale["std"] + active_scale["mean"])),
        "random_valid_fraction_clipped": float((numeric[valid, active_column] > edges[active_column][-1]).mean()),
        "final_fraction_clipped": float((final_num[:, active_column] > edges[active_column][-1]).mean()),
        "applies_to_ensemble_member": "Signed TabM (weight 0.5); DCN has no numeric clipping",
        "interpretation": "Cumulative days grow with the observation window; no causal share of the final AUC gap identified",
    }
    with (HERE / "results_modern/random20_architecture_capacity32/tabm_ple_clip/seed42/training_history.csv").open(encoding="utf-8", newline="") as handle:
        result["tabm32_training_history"] = [{key: float(row[key]) for key in ["epoch", "train_logloss", "auc"]} for row in csv.DictReader(handle)]
    for phase in ["results", "results_modern"]:
        result[phase + "_completed_experiments"] = len(list((HERE / phase).glob("*/*/seed*/metrics.json")))
    result["previous_collaborative_uplift_decomposition"] = {key: value for key, value in load_json(HERE / "phase5_uplift_slices.json")["pair_decomposition"].items() if key != "matrix"}
    path = HERE / "final_generalization_diagnosis.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# final 评分 0.74404 的模型局限诊断

用户反馈的 final AUC 为 {user_reported_final_auc:.5f}；对应提交文件为 output/submission_best.csv。
随机验证 AUC 为 {auc:.8f}，差值 {auc - user_reported_final_auc:.8f}。final 标签不在本地，外部成绩按用户反馈记录，不能逐行分析 final 错误或计算误差置信区间。

## 已确认的输入和评估差异

| 指标 | 随机验证集 | final |
|---|---:|---:|
| 用户编号不在训练词表 | 0.00% | 0.00% |
| 广告任务编号不在训练词表 | 0.04% | 11.38% |
| 广告编号 adv_id 不在训练词表 | 0.06% | 11.80% |
| 用户与广告任务的组合未在训练曝光里出现 | 15.67% | 47.30% |
| 用户和广告两侧均有其他训练点击关系 | 77.60% | 62.70% |

随机划分将各天的记录混合后拆成 80% 训练和 20% 验证，很多熟悉用户与广告同时存在于两边。53.83% 验证行的原始预测字段哈希在训练部分也出现过；这不是把验证标签直接输入模型的证据，但这种划分没有模拟预测新一天。final 为 2019-11-17，广告更新及用户与广告的新组合明显更多。当前最佳组合在随机验证内的 2019-11-16 切片 AUC 也只有 0.76027；此切片仍不是严格的时间留出实验。

## 具体的特征局限

资讯累计活跃天数随着数据覆盖的天数增长。训练时最多 5 天，final 的 379,665 行是 6 天，占 93.93%。Signed TabM 分支将数值截到训练分箱范围，因而把这部分 6 天记录压到约 5 天的上限；随机验证没有这种超范围情况。该分支权重为 50%，DCN 分支没有这项截断。累计资讯曝光、点击数的分布也明显变化。这证实了特征随观测窗口漂移，但不能据此估算它单独造成多少 AUC 损失。

新广告编号通常进入统一的未知类别，广告历史统计不足，点击图的广告邻居也缺失。模型还保留广告主、应用类别等属性，但现有共享表示能弥补多少，需要做严格跨天消融。当前多数历史列表采用均值池化，源域主题取前 5 个，完整频次和当前广告条件下的兴趣差异没有被充分保留。

## 为什么继续扩大模型提分有限

1. 增大的架构仍然使用同一套输入；输入缺失与时间变化并不会因模型更大自动消失。
2. 训练正例只有 17,083 条，占 0.845%。对大量广告编号和关系分支来说，可用点击证据较少。这不是 AUC 的理论上限。
3. TabM32 主干第 2 轮随机验证 AUC 约 0.77013，第 3 轮约 0.68125，而训练损失继续降低，存在明显过拟合。提交使用保存的最佳检查点，没有使用后续掉分轮次。
4. 上一轮协同组合从 0.773998 提升到 0.776748，正负配对分解显示，同一关系可用性组内的加权变化为 -0.00006893，组间变化为 +0.00281816。那一轮的增益主要是不同人群间的排序调整，同组内区分未改善；不能把此结论推广为所有优化都无效。
5. 多个实验持续复用同一个随机验证集，存在模型选择过度适应该验证集的风险。当前只有一次用户反馈的 final 总分，尚不能确定哪条分支在 final 拖累，也没有旧模型的 final 对照来量化实际提升。
6. 本次按已验证的检查点直接预测，广告权重、目标统计和广告图只使用了 80% 的带标签训练记录；剩余 20% 保留作验证，没有做全量广告重训。

## 下一轮优先次序

先以 2019-11-15、2019-11-16 的先前日期训练、次日验证作为滚动评估，重新拟合每一折的词表、缩放、目标统计和图，比较 DCN、Signed 分支及组合。之后优先试固定时间窗口、日均活跃率和曝光归一化特征，并为新广告加强广告主/应用类别等共享属性路径。用跨天结果决定训练轮数、正则和分支权重，再使用全部带标签广告重新拟合完整流水线，生成下一份 final 提交。

以上是可核查的局限与实验方向，不是对 0.03295 分差的因果分摊，也不保证修改后能达到 0.85。

详细数值：final_generalization_diagnosis.json。复现命令：用 Machine_Learning 环境运行 analysis/auc085/diagnose_final_generalization.py。
"""
    (HERE / "final泛化差距与模型局限诊断.md").write_text(report, encoding="utf-8")
    display = {key: value for key, value in result.items() if key != "feature_shift"}
    display["top_feature_shifts"] = result["feature_shift"][:8]
    print(json.dumps(display, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
