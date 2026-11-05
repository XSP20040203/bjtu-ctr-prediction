"""Verify the diagnostic predictions and explain the first/second-epoch gap."""
import base64
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch

from diagnose_auc import category_batch, positive_pair_ranks, prediction_summary
from mainstream_models import mainstream_definition
from train_models import load_arrays, write_json

HERE = Path(__file__).resolve().parent
OUT = HERE / "auc_diagnosis"
ARMS = {"lr0p001": "原设置", "lr0p0003": "只降低学习率", "lr0p001_no_user_id": "只移除用户ID"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    torch.set_num_threads(4)
    arrays = load_arrays()
    valid, train = arrays["valid_indices"], arrays["train_indices"]
    labels = arrays["label"][valid]
    assert arrays["date"][train].max() < 20191116
    assert np.all(arrays["date"][valid] == 20191116)
    histories, comparisons, errors = {}, [], []
    reference_sample = None
    for tag, name in ARMS.items():
        directory = OUT / tag
        assert read_json(directory / "completion.json")["status"] == "complete"
        protocol = read_json(directory / "protocol.json")
        for filename, digest in protocol["sha256"].items():
            path = HERE / ("cache/" + filename if filename == "manifest.json" else filename)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        history = pd.read_csv(directory / "history.csv")
        assert len(history) == 8
        histories[tag] = history
        sample = np.load(directory / "train_probe_indices.npy")
        assert not np.intersect1d(sample, valid).size
        if reference_sample is None:
            reference_sample = sample
        else:
            np.testing.assert_array_equal(sample, reference_sample)
        for row in history.to_dict("records"):
            fraction = row["fractional_epoch"]
            for split, positions, prefix in [("valid", valid, "valid_"), ("train_probe", sample, "train_probe_")]:
                prediction = np.load(directory / f"epoch{fraction:.2f}_{split}_prediction.npy")
                assert prediction.shape == (len(positions),) and np.isfinite(prediction).all()
                result = prediction_summary(arrays["label"][positions], prediction)
                for key, value in result.items():
                    assert abs(value - row[prefix + key]) < 1e-10
        checkpoint = torch.load(directory / "best_model.pt", map_location="cpu", weights_only=False)
        manifest = checkpoint["manifest"]
        builder, options = mainstream_definition("dcnv2")
        model = builder(manifest["categorical_fields"], manifest["sequence_fields"], manifest["group_for_field"],
                        manifest["vocabulary_sizes"], **options)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        mask_column = manifest["categorical_fields"].index("user_id") if protocol.get("mask_user_id") else None
        chosen = np.linspace(0, len(valid) - 1, 144, dtype=int)
        indices = valid[chosen]
        with torch.no_grad():
            prediction = torch.sigmoid(model(torch.from_numpy(category_batch(arrays["categories"], indices, mask_column)),
                                             torch.from_numpy(arrays["sequences"][indices].astype(np.int64)),
                                             torch.from_numpy(np.array(arrays["numeric"][indices])))).numpy()
        saved = np.load(directory / f"epoch{checkpoint['fractional_epoch']:.2f}_valid_prediction.npy")[chosen]
        np.testing.assert_allclose(prediction, saved, rtol=3e-5, atol=2e-7)
        errors.append(float(np.max(np.abs(prediction - saved))))
        first = history.loc[history.fractional_epoch.eq(1)].iloc[0]
        second = history.loc[history.fractional_epoch.eq(2)].iloc[0]
        best = history.loc[history.valid_auc.idxmax()]
        comparisons.append({"设置": name, "第一轮AUC": first.valid_auc, "第二轮AUC": second.valid_auc,
                            "第二轮变化": second.valid_auc - first.valid_auc,
                            "本次最佳AUC": best.valid_auc, "最佳时点（轮）": best.fractional_epoch})
    original = histories["lr0p001"]
    previous = pd.read_csv(HERE / "results_mainstream/20191116/dcnv2/seed42/training_history.csv")
    for epoch in [1, 2]:
        assert abs(original.loc[original.fractional_epoch.eq(epoch), "valid_auc"].iloc[0] - previous.loc[previous.epoch.eq(epoch), "auc"].iloc[0]) < 1e-7
    slices = pd.read_csv(OUT / "lr0p001/slices.csv")
    audit = []
    for fraction in [1., 2.]:
        p = np.load(OUT / "lr0p001" / f"epoch{fraction:.2f}_valid_prediction.npy")
        ranks = positive_pair_ranks(labels, p)
        assert abs(ranks.mean() - original.loc[original.fractional_epoch.eq(fraction), "valid_auc"].iloc[0]) < 1e-10
        audit.append(slices.loc[slices.fractional_epoch.eq(fraction) & slices['slice'].eq("user_id__zero_train_click")].iloc[0].to_dict())
    contributions = []
    for key in ["user_id__zero_train_click", "user_id__has_train_click"]:
        pair = slices.loc[slices['slice'].eq(key) & slices.fractional_epoch.isin([1., 2.])].sort_values("fractional_epoch")
        a, b = pair.iloc[0], pair.iloc[1]
        contributions.append({"group": key, "positive_weight": a.positive_weight,
                              "weighted_auc_change": float((b.positive_pair_rank_vs_all_negatives - a.positive_pair_rank_vs_all_negatives) * a.positive_weight)})
    observed_drop = comparisons[0]["第二轮变化"]
    assert abs(sum(row["weighted_auc_change"] for row in contributions) - observed_drop) < 1e-10
    write_json(OUT / "rank_loss_decomposition.json", contributions)
    pd.DataFrame(comparisons).to_csv(OUT / "diagnostic_comparison.csv", index=False, encoding="utf-8-sig")

    font = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained")
    axes[0].plot(original.fractional_epoch, original.train_probe_auc, marker="o", label="训练样本AUC", color="#2a7d80")
    axes[0].plot(original.fractional_epoch, original.valid_auc, marker="o", label="下一天验证AUC", color="#c16c3c")
    axes[0].set(title="原设置：训练越来越好，验证越来越差", xlabel="已训练轮次", ylabel="AUC")
    for tag, color in zip(ARMS, ["#c16c3c", "#5379aa", "#2a7d80"]):
        history = histories[tag]
        axes[1].plot(history.fractional_epoch, history.valid_auc, marker="o", label=ARMS[tag], color=color)
    axes[1].set(title="控制变量：降学习率、去用户ID仍未解决", xlabel="已训练轮次", ylabel="验证AUC")
    for ax in axes:
        ax.axvline(1, color="#9ba7b0", linestyle="--", linewidth=1)
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
    fig.suptitle("AUC为什么第二轮下降：真实诊断训练", fontsize=17)
    figure = OUT / "AUC下降诊断曲线.png"
    fig.savefig(figure, dpi=170)
    plt.close(fig)

    evidence = read_json(OUT / "data_evidence.json")
    insights = [
        "已经复现原训练轨迹：原设置第1轮AUC=0.725099，第2轮=0.675134。训练样本AUC从0.780261升到0.836337。这里的训练AUC来自固定20万条参与训练的样本，只用于观察拟合程度。",
        "第一轮0.25、0.50、0.75、1.00轮的验证AUC持续改善；第二轮刚走完四分之一就降到0.682832。因此，在这次原设置的检查点上，问题集中出现在重新学习同一批样本之后。",
        f"训练用户中{evidence['user_id']['train_ids_at_most_2_click_fraction']:.2%}只有0～2次点击，中位数仅1次；训练广告任务中{evidence['task_id']['train_ids_zero_click_fraction']:.2%}从未被点。编号与点击倾向之间存在大量不确定估计。",
        "训练零点击用户在验证日有68,667条展示、816次点击，实际点击率1.188%。模型给他们的平均概率却从第1轮0.688%降到第2轮0.210%。以前没点过，不等于明天不点。",
        f"把全部点击样本按用户训练期是否有点击分组，零点击用户组贡献的AUC变化为{contributions[0]['weighted_auc_change']:+.6f}，有点击组为{contributions[1]['weighted_auc_change']:+.6f}，相加等于总体变化{observed_drop:+.6f}。这是与所有未点击样本比较的精确分解，不是简单平均两组AUC。",
        "这说明本次下降主要集中在训练零点击用户的排序上，但不能单凭分组定位就断言只有user_id这个输入负责。设备、历史、广告等相关字段也可能携带用户身份线索。",
        "学习率从0.001降到0.0003，仍然在第二轮下降，最佳AUC也未提高。只移除用户ID减轻了下降，但第一轮峰值同样未超过原设置；不能把降低过拟合程度直接等同于已经提分。",
        "训练代码每轮都打乱样本，验证时使用eval模式，概率按验证行顺序对齐。原设置前两轮分数已重现，当前证据更支持跨日泛化恶化，而非AUC计算或Dropout模式错误。",
        "训练优化的是二分类交叉熵，验证AUC衡量排序；训练损失降低并不保证下一天的排序提高。单纯更改点击判定阈值不会提高AUC。",
        "本轮新增实验只覆盖2019-11-16、种子42；每四分之一轮选最佳点仍是在使用验证标签，不能当作独立测试。去用户ID实验是在观察切片之后追加的探索性对照。"
    ]
    next_steps = [
        "先保留当前0.725099的最佳权重作为对照。不要为了训练损失更低而强行训练更多轮。将验证和保存频率改为每四分之一轮，检查其他日期是否有更早的峰值。",
        "优先试针对用户/任务/素材ID的更强正则或随机屏蔽，而不是直接永久删除所有ID。当前对照表明，身份信息有用，但过度依赖有害；适合分别测试屏蔽比例、Embedding约束及较小维度。",
        "对照去掉任务ID、素材ID及两者组合，区分用户记忆与广告记忆各自的影响；保持其他参数不变。未做这些实验前，不将其列为已确认原因。",
        "将累计资讯行为改为最近1天/3天的统计进行单独对照。已有累计曝光中位数训练23、验证77的变化，但尚未证明它造成当前掉分。所有窗口仍只使用广告日期之前的数据。",
        "任何候选都要复核11月15日、16日及多个随机种子，比较平均表现和波动。提高本次单日验证峰值还不等于稳定提高最终未知标签测试集AUC。"
    ]
    frame = pd.DataFrame(comparisons)
    for column in ["第一轮AUC", "第二轮AUC", "第二轮变化", "本次最佳AUC"]:
        frame[column] = frame[column].map(lambda value: f"{value:.6f}")
    format_bullets = lambda texts: "".join("<li>" + html.escape(text) + "</li>" for text in texts)
    image_src = base64.b64encode(figure.read_bytes()).decode("ascii")
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AUC下降原因诊断</title>
<style>body{{background:#f3f6f8;color:#213547;font:15px/1.9 "Microsoft YaHei",sans-serif;margin:0}}main{{max-width:1150px;margin:auto;padding:28px}}section{{background:white;padding:24px;margin:22px 0;border:1px solid #dce5e9;border-radius:12px;overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}td,th{{padding:10px;border-bottom:1px solid #e1e8ed;text-align:left}}th{{color:#127d80}}img{{width:100%}}li{{margin:10px 0}}.lead{{font-size:19px;color:#127d80}}</style>
<main><h1>AUC为什么下一轮反而下降</h1><p class="lead">已复现：模型对训练样本越来越熟，但对下一天的排序变差；主要跌幅集中在训练期零点击的用户。</p>
<section><h2>三组真实对照</h2>{frame.to_html(index=False,border=0)}<p>降低学习率、直接删除用户ID，都没有提高本次最佳AUC。去用户ID仅减轻了后续下降。</p></section>
<section><img src="data:image/png;base64,{image_src}" alt="训练验证AUC及控制变量曲线"></section><section><h2>证据与解释</h2><ul>{format_bullets(insights)}</ul></section><section><h2>提高AUC的下一步</h2><ol>{format_bullets(next_steps)}</ol></section>
<section><h2>资料与核验</h2><p>CTR研究中有专门讨论的单轮过拟合（one-epoch phenomenon）：Zhang等，2022，Towards Understanding the Overfitting Phenomenon of Deep Click-Through Rate Prediction Models，arXiv:2209.06053。本报告借助其提出相关假设，不据此断言唯一机制。AUC和交叉熵定义参考scikit-learn官方文档。</p><p>全部24个检查点的验证预测和固定训练样本预测已重新计算指标，3个最佳权重已重新加载。原始数据、模型和此前结果保留。代码：diagnose_auc.py；数值分解：rank_loss_decomposition.json。</p></section></main></html>'''
    (OUT / "AUC下降原因诊断.html").write_text(document, encoding="utf-8")
    markdown = ["# AUC下降原因诊断", "", "## 三组对照", "", frame.to_string(index=False), "", "## 证据与解释", ""]
    markdown += ["- " + text for text in insights]
    markdown += ["", "## 提高AUC的下一步", ""] + [f"{i}. {text}" for i, text in enumerate(next_steps, 1)]
    (OUT / "AUC下降原因诊断.md").write_text("\n".join(markdown), encoding="utf-8")
    write_json(OUT / "verification.json", {"status": "passed", "runs": 3, "measurement_points": 24,
               "original_two_epochs_reproduced": True, "prediction_arrays_rescored": 48,
               "best_checkpoints_reloaded": 3, "max_reload_error": max(errors),
               "same_train_probe_checked": True, "positive_rank_decomposition_exact": True,
               "validation_day": 20191116, "seed": 42, "final_test_used": False})
    print(frame.to_string(index=False))
    print(json.dumps(contributions, ensure_ascii=False))
    print("VERIFIED_DIAGNOSIS " + str(OUT / "AUC下降原因诊断.html"))


if __name__ == "__main__":
    main()
