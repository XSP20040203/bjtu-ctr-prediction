"""Report completed architecture comparisons and preserve the unachieved target."""
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def model_name(record):
    family, experiment = record["family"], record["experiment"]
    if "capacity32" in experiment:
        return "TabM32：扩大集成与表示维度"
    if "architecture_aligned" in experiment:
        return "TabM8：频次对齐＋有界数值嵌入"
    if family == "joint_attention":
        return "TabM＋物品条件兴趣注意力" + ("＋资讯辅助任务" if record["auxiliary_weight"] else "（广告任务对照）")
    if family == "joint":
        return "TabM＋跨域门控" + ("＋资讯辅助任务" if record["auxiliary_weight"] else "（广告任务对照）")
    return {"tabm": "TabM8：官方骨干适配", "tabm_ple": "TabM8＋分段数值嵌入", "tabm_ple_clip": "TabM8＋有界数值嵌入",
            "transformer": "字段 Transformer", "tabm_cin": "TabM＋二/三阶交互专家"}[family]


def main():
    verification = load(HERE / "phase2_full_verification.json")
    assert verification["status"] == "passed"
    assert load(HERE / "phase2_ensemble_verification.json")["status"] == "passed"
    records = load(HERE / "phase2_full_verified_results.json")
    assert len(records) == 11
    baseline = load(HERE / "results/random20_rich_oof_statistics/bounded/seed42/metrics.json")
    ensembles = load(HERE / "results_architecture_ensembles/metrics.json")
    protocol = load(HERE / "architecture_ensemble_protocol.json")
    labels = None
    canonical = None
    for record in ensembles:
        assert record["weights"] == protocol["combinations"][record["name"]]
        expected = None
        for member, weight in record["weights"].items():
            directory = HERE / protocol["members"][member]
            metrics = load(directory / "metrics.json")
            cache = Path(metrics["cache_dir"])
            valid = np.load(cache / "valid_indices.npy")
            if canonical is None:
                canonical, labels = valid, np.load(cache / "label.npy", mmap_mode="r")[valid]
            else:
                np.testing.assert_array_equal(canonical, valid)
            value = np.load(directory / "validation_prediction.npy").astype(np.float64) * weight
            expected = value if expected is None else expected + value
        saved = np.load(HERE / "results_architecture_ensembles" / (record["name"] + "_validation_prediction.npy"))
        np.testing.assert_array_equal(expected, saved)
        from sklearn.metrics import roc_auc_score
        assert abs(roc_auc_score(labels, saved) - record["auc"]) < 1e-10
    best_model = max(records, key=lambda row: row["auc"])
    best_ensemble = max(ensembles, key=lambda row: row["auc"])
    headline = f"当前单架构最高AUC={best_model['auc']:.6f}；固定权重多架构集成最高={best_ensemble['auc']:.6f}。0.85目标尚未达到。"
    names = {"dcn_tabm32": "DCN＋TabM32（各50%）", "tabm32_attention": "TabM32＋兴趣注意力（各50%）",
             "three_architectures": "DCN＋TabM32＋兴趣注意力（等权）", "four_architectures": "四种结构等权集成"}
    architecture_table = pd.DataFrame([{"架构": model_name(row), "AUC ↑": f"{row['auc']:.6f}",
                        "相对上阶段DCN": f"{row['auc'] - baseline['auc']:+.6f}", "Logloss ↓": f"{row['logloss']:.6f}",
                        "参数量": f"{row['parameter_count']:,}", "最佳轮次": row["best_epoch"]} for row in records])
    ensemble_table = pd.DataFrame([{"结构组合": names[row["name"]], "AUC ↑": f"{row['auc']:.6f}",
                                  "相对上阶段DCN": f"{row['auc'] - baseline['auc']:+.6f}",
                                  "Logloss ↓": f"{row['logloss']:.6f}"} for row in ensembles])
    architecture_table.to_csv(HERE / "phase2_architecture_comparison.csv", index=False, encoding="utf-8-sig")
    ensemble_table.to_csv(HERE / "phase2_ensemble_comparison.csv", index=False, encoding="utf-8-sig")
    notes = [
        "目标已按用户最新要求改为：主要优化先进模型架构，以未参与训练的广告验证AUC超过0.85为标准。最终测试评分仍暂缓；训练集AUC不能替代验证AUC。",
        f"上阶段同一随机划分的DCN基线为{baseline['auc']:.6f}。本轮11个已训练架构及4个预先固定权重组合全部保留，最高集成相对该基线变化{best_ensemble['auc'] - baseline['auc']:+.6f}。",
        "TabM骨干使用作者发布的tabm 0.0.3包，属于ICLR 2025的参数高效集成路线。本项目为高基数类别编号和短历史列表添加CTR输入适配；成绩不等于原论文公开基准的复现。",
        "32成员版本采用384宽度、三层骨干、16维类别/数值表示。其他8成员版本通常为256宽度；字段Transformer为两层、四头、32维token。大模型和小模型的batch与表示维度不同，不能将所有差异归因于某一单层。",
        "跨域架构实现了共享用户/兴趣表示、独立广告与资讯预测头、门控迁移，以及物品条件的兴趣注意力。源码和测试证明源域监督能更新共享表示；这些组合在本轮没有超过最强TabM。",
        "资讯辅助任务不接受其自身label、pro、cillabel作为输入。物品条件注意力版本还去掉资讯文档ID，降低直接记忆文档的倾向。广告侧仍可使用任务提供的源域反馈画像，这是离线源域快照可用的设定。",
        "数值分段编码曾在部分频次统计上出现严重外推：22.49%的验证行在广告位频次上超过训练边界，最大约933个末端分箱宽度。后续版本限制编码边界，并把X-only曝光频次与标签交叉拟合的点击率分开。它修复了表示稳定性问题，不能直接宣称达到0.85。",
        "四种固定权重集成先声明权重再评分，没有根据验证标签训练融合权重。最佳组合是50% DCN概率加50% TabM32概率；这属于可复算的多架构集成，不是声称发表了新的SOTA模型。",
        "所有模型沿用同一组505,264条随机验证索引；其中53.83%的原始特征在训练集中存在完全相同的记录。该评估更接近重复场景，不能据此保证未来日期或新的测试数据表现。",
        "当前结果仍来自种子42和反复使用的验证划分。尚未达到0.85，也尚未完成本轮最佳配置的其他随机划分确认。所有目标达成标记保持false。",
        "下一阶段将检查更强的源域兴趣表征：源域用户—资讯—类别—实体的图编码与偏好排序预训练，并对比保护用户表征、减少广告标签直接改写用户ID的梯度路由。此方向目前仅完成源字段可用性审计，未训练或宣称任何新分数。"
    ]
    font = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(), "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    chosen = sorted(records, key=lambda row: row["auc"], reverse=True)[:6]
    axes[0].barh(np.arange(len(chosen)), [row["auc"] - .5 for row in chosen], left=.5, color="#317c9d")
    axes[0].set_yticks(np.arange(len(chosen)), [model_name(row) for row in chosen])
    axes[0].invert_yaxis()
    axes[0].set(title="已完成架构中最高的六组", xlabel="随机验证AUC", xlim=(.5, .87))
    axes[0].axvline(.85, linestyle="--", color="#b75242")
    axes[0].axvline(baseline["auc"], linestyle=":", color="#657681")
    for index, row in enumerate(chosen):
        axes[0].text(row["auc"] + .003, index, f"{row['auc']:.4f}", va="center")
    for record in records:
        if record["experiment"] in ["random20_architecture_capacity32", "random20_attentive_transfer_aux005"]:
            h = pd.read_csv(Path(record["results_dir"]) / "training_history.csv")
            axes[1].plot(h.epoch, h.auc, marker="o", label=model_name(record))
    axes[1].axhline(baseline["auc"], color="#657681", linestyle=":", label="上阶段DCN最佳")
    axes[1].set(title="先进骨干仍然存在后续掉分", xlabel="训练轮次", ylabel="验证AUC")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle("以架构为主的第二阶段：当前最高集成AUC约0.7740", fontsize=18)
    figure = HERE / "phase2_架构对比.png"
    fig.savefig(figure, dpi=170)
    plt.close(fig)
    source_image = base64.b64encode(figure.read_bytes()).decode("ascii")
    bullets = "".join("<li>" + html.escape(text) + "</li>" for text in notes)
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>先进架构优化：第二阶段实测</title>
<style>body{{margin:0;background:#f3f6f8;color:#213547;font:15px/1.9 "Microsoft YaHei",sans-serif}}main{{max-width:1250px;margin:auto;padding:28px}}section{{background:white;border:1px solid #dce5e9;border-radius:12px;padding:24px;margin:22px 0;overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}td,th{{padding:10px;border-bottom:1px solid #e1e8ed;text-align:left}}th{{color:#147e80}}li{{margin:10px 0}}img{{width:100%}}.lead{{font-size:20px;color:#147e80}}</style>
<main><h1>先进架构优化：第二阶段实测</h1><p class="lead">{headline}</p><section><h2>11个已完成的架构版本</h2>{architecture_table.to_html(index=False,border=0)}</section>
<section><h2>4个预先固定权重的架构集成</h2>{ensemble_table.to_html(index=False,border=0)}</section><section><img src="data:image/png;base64,{source_image}" alt="架构成绩和训练稳定性"></section>
<section><h2>大白话说明与证据边界</h2><ul>{bullets}</ul></section><section><h2>当前最高分结构</h2><p>同一条广告分别经过DCN显式交叉分支与32成员TabM分支，得到两组点击概率；按50%＋50%固定权重平均输出。TabM各成员共享大部分权重，并使用独立缩放和输出头。</p><p>主模型定义及权重路径见architecture_ensemble_protocol.json；保存预测已按原验证行索引重新对齐并复算。</p></section>
<section><h2>参考实现与核验</h2><p>TabM：Yandex Research官方实现，ICLR 2025；CIN：xDeepFM，KDD 2018，arXiv:1803.05170。字段Transformer和跨域模块为本项目改造，不冒充原论文官方成绩。</p><p>11组最佳检查点已在原CUDA混合精度设置下重放全部验证样本，概率与保存结果完全一致。源域标签、数值分箱拟合索引和频次对齐也已核验，见phase2_full_verification.json。</p></section></main></html>'''
    (HERE / "先进架构优化_第二阶段报告.html").write_text(document, encoding="utf-8")
    (HERE / "先进架构优化_第二阶段报告.md").write_text("# 先进架构优化：第二阶段\n\n" + headline + "\n\n" + architecture_table.to_string(index=False) + "\n\n" + ensemble_table.to_string(index=False) + "\n\n" + "\n".join("- " + text for text in notes), encoding="utf-8")
    state = load(HERE / "optimization_state.json")
    state.update(phase="phase2_modern_architecture_benchmark_completed_target_not_reached", goal_achieved=False,
                 best_random_validation_auc=max(state["best_random_validation_auc"], best_ensemble["auc"]),
                 best_single_modern_validation_auc=best_model["auc"], best_ensemble_definition=best_ensemble,
                 phase2_verified_modern_models=len(records), phase2_verified_fixed_ensembles=len(ensembles),
                 live_tool_sessions={}, last_goal_turn_classification="progress",
                 missing_completion_evidence="No held-out advertisement validation AUC above 0.85; best architecture candidates need further improvement and another split/seed confirmation",
                 next_actions=["Implement and test typed source graph preference pretraining and protected user-interest transfer", "Compare against the same TabM and DCN controls on unchanged validation indices", "Use pairwise ranking only as a secondary controlled objective on modern architecture", "Replay winning model and confirm on another split/seed before any 0.85 completion claim"])
    (HERE / "optimization_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(headline)
    print(HERE / "先进架构优化_第二阶段报告.html")


if __name__ == "__main__":
    main()
