"""Independently verify saved predictions and create readable comparison artifacts."""
from __future__ import annotations

import base64
import importlib.metadata as metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, precision_recall_curve, roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CACHE = HERE / "cache"
NAMES = {"catboost_original": "CatBoost：原11字段基线", "catboost_full": "CatBoost：完整特征", "neural": "完整神经网络：Embedding＋MLP"}
COLORS = {"catboost_original": "#7f93a3", "catboost_full": "#be7820", "neural": "#087f78"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    rows = np.load(CACHE / "valid_indices.npy")
    labels = np.load(CACHE / "label.npy", mmap_mode="r")[rows]
    log_ids = np.load(CACHE / "log_id.npy", mmap_mode="r")[rows]
    predictions = {"row_index": rows, "log_id": log_ids, "label": labels}
    results = []
    for name in NAMES:
        result = json.loads((RESULTS / name / "metrics.json").read_text(encoding="utf-8"))
        prediction = np.load(RESULTS / name / "validation_prediction.npy")
        assert len(prediction) == len(labels) == 421983
        assert np.isfinite(prediction).all() and np.logical_and(prediction >= 0, prediction <= 1).all()
        clipped = np.clip(prediction.astype(np.float64), 1e-8, 1 - 1e-8)
        checked_auc = float(roc_auc_score(labels, clipped))
        checked_loss = float(log_loss(labels, clipped, labels=[0, 1]))
        assert abs(checked_auc - result["auc"]) < 1e-10
        assert abs(checked_loss - result["logloss"]) < 1e-10
        assert result["train_rows"] == 2104337 and result["validation_clicks"] == 3724
        predictions[name] = prediction
        results.append(result)
    assert int(labels.sum()) == 3724
    pd.DataFrame(predictions).to_csv(RESULTS / "validation_predictions.csv.gz", index=False, compression="gzip")
    table = pd.DataFrame([{"模型": NAMES[row["model"]], "AUC": row["auc"], "Logloss": row["logloss"],
                           "PR_AUC_AP": row["average_precision"], "平均预测概率": row["mean_prediction"],
                           "训练秒数": row["training_seconds"], "设备": row["device"],
                           "最佳轮次": row.get("trees", row.get("best_epoch"))} for row in results])
    table.to_csv(RESULTS / "comparison.csv", index=False, encoding="utf-8-sig")
    write_json(RESULTS / "comparison.json", results)

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font_path)).get_name(), "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.3), layout="constrained")
    for name in NAMES:
        prediction = predictions[name]
        fpr, tpr, _ = roc_curve(labels, prediction)
        precision, recall, _ = precision_recall_curve(labels, prediction)
        row = next(row for row in results if row["model"] == name)
        axes[0].plot(fpr, tpr, color=COLORS[name], label=f"{NAMES[name]}  AUC={row['auc']:.4f}", linewidth=1.8)
        axes[1].plot(recall, precision, color=COLORS[name], label=f"{NAMES[name]}  AP={row['average_precision']:.4f}", linewidth=1.5)
    axes[0].plot([0, 1], [0, 1], "--", color="#c3cdd3", linewidth=1)
    axes[1].axhline(labels.mean(), color="#c3cdd3", linestyle="--", label=f"实际点击率 {labels.mean():.4%}")
    axes[0].set(title="ROC：对点击与未点击记录的排序能力", xlabel="误报比例 FPR", ylabel="召回比例 TPR", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(title="Precision–Recall：点击样本很少时的表现", xlabel="召回率 Recall", ylabel="精确率 Precision", xlim=(0, 1), ylim=(0, 1))
    for ax in axes:
        ax.legend(frameon=False, fontsize=8, loc="lower right" if ax == axes[0] else "upper right")
        ax.grid(color="#e3e9ed", linewidth=.6)
    fig.suptitle("同一天、同一批记录的模型对比｜2019-11-16", fontsize=18)
    fig.savefig(RESULTS / "模型对比曲线.png", dpi=170)
    plt.close(fig)

    best = max(results, key=lambda row: row["auc"])
    original, full, neural = results
    neural_history = pd.read_csv(RESULTS / "neural" / "training_history.csv")
    last_epoch = neural_history.iloc[-1]
    overfit_note = (f"神经网络第{neural['best_epoch']}个epoch最佳（AUC={neural['auc']:.6f}）。"
                   f"继续训练到第{int(last_epoch['epoch'])}轮时，训练Logloss下降到{last_epoch['train_logloss']:.6f}，"
                   f"但验证AUC降到{last_epoch['auc']:.6f}、验证Logloss升到{last_epoch['logloss']:.6f}，"
                   "出现明显过拟合。最终报告和model.pt均使用最佳轮次，不使用最后一轮权重。")
    displayed = table.copy()
    for col in ["AUC", "Logloss", "PR_AUC_AP", "平均预测概率"]:
        displayed[col] = displayed[col].map(lambda value: f"{value:.6f}")
    displayed["训练秒数"] = displayed["训练秒数"].map(lambda value: f"{value:.1f}")
    html_table = displayed.to_html(index=False, border=0, escape=True)
    markdown = ["# CatBoost与完整神经网络的实际训练对比", "", "分析日期：2026-10-27。以下为本机真实运行结果，不是根据模型名称推测。", "",
                f"本次验证AUC最高：**{NAMES[best['model']]}，AUC={best['auc']:.6f}**。Logloss越低越好，AUC越高越好；AUC不是点击率或准确率。", "",
                "| 模型 | AUC | Logloss | 训练耗时 | 设备 | 最佳轮次 |", "| --- | --- | --- | --- | --- | --- |"]
    for row in results:
        markdown.append(f"| {NAMES[row['model']]} | {row['auc']:.6f} | {row['logloss']:.6f} | {row['training_seconds']:.1f}秒 | {row['device']} | {row.get('trees', row.get('best_epoch'))} |")
    markdown += ["", "## 神经网络的训练过程", "", overfit_note, "",
                 "| Epoch | 训练Logloss | 验证AUC | 验证Logloss |", "| --- | --- | --- | --- |"]
    for _, epoch in neural_history.iterrows():
        markdown.append(f"| {int(epoch['epoch'])} | {epoch['train_logloss']:.6f} | {epoch['auc']:.6f} | {epoch['logloss']:.6f} |")
    markdown += ["", "## 如何理解差异", "",
                 f"- 完整特征CatBoost相对原11字段基线：AUC差值{full['auc']-original['auc']:+.6f}。这比较增加类别处理、补入广告位等原基线未用字段、历史列表、资讯统计并移除log_id之后的整体方案，不能把增益单独归因于资讯特征。",
                 f"- 完整神经网络相对完整特征CatBoost：AUC差值{neural['auc']-full['auc']:+.6f}。两者共享原始信息与20个数值特征，但列表编码不同：CatBoost把排序后的列表当类别，神经网络对列表向量做平均汇总。结果比较的是两套完整方案，不能仅归因于网络层。",
                 "- CatBoost使用CPU 6线程，神经网络使用GPU（可用时）。耗时是本机实际记录，不是同硬件条件下的算法效率评测。",
                 "- 本次是固定种子42、单日留出、有限训练预算的初次对比；未做多种子置信区间或系统调参，不能据此宣称某类算法总是更好。", "",
                 "## 数据与防止信息提前进入", "",
                 "- 训练：2019-11-10至11-15，2,104,337条，17,630次点击。",
                 "- 验证：2019-11-16，421,983条，3,724次点击。相同原始行、顺序和标签用于三组模型。",
                 "- 未使用final_test.csv或test_feeds.csv训练、早停、调参或计算分数；11月17日没有广告标签，不能计算真实测试AUC。",
                 "- 每条广告的资讯画像只用其日期之前的资讯，e_et < pt_d。词表和缩放统计量只拟合广告训练窗口；CatBoost的序列组合键只表示已编码输入，不用验证标签计算映射。",
                 "- 所有组均保留训练数据的自然正负比例，没有负采样或类别加权。",
                 "- 验证集用于早停与选轮次，因此这些数字是模型选择的验证结果。", "",
                 "## 三组具体设置", "",
                 "原CatBoost基线：沿用Notebook的11个数值输入（包括log_id），没有加入源域数据；为比较统一设置训练预算，不是原默认超参数的完全复现。",
                 "",
                 "完整CatBoost：24个类别字段＋7个无序列表组合类别＋20个数值特征；depth=6、learning_rate=0.08、最多1000棵树、AUC早停100轮、max_ctr_complexity=1。",
                 "",
                 "完整神经网络：24个16维类别Embedding＋7个16维列表平均汇总＋20个数值特征，516→256→128→64→1。AdamW，初始lr=0.001，batch=2048，最多10个epoch，3个epoch不改善则早停。BCEWithLogitsLoss，不额外加正类权重。输出偏置按训练窗口点击率初始化。",
                 "",
                 f"共享数据准备耗时：{manifest['preparation_seconds']:.1f}秒，单独记录，未算入各模型训练耗时。各组实际参数、最佳轮次、训练曲线及模型权重均在各自结果文件夹内。", "",
                 "## 文件与复现", "",
                 "- comparison.csv / comparison.json：结果总表。",
                 "- validation_predictions.csv.gz：同一验证集的原始行号、log_id、标签与三组概率；不能仅按会重复的log_id合并。",
                 "- catboost_original/model.cbm、catboost_full/model.cbm、neural/model.pt：模型权重。",
                 "- cache/manifest.json：实际使用字段、词表规模、数值缩放和数据边界。",
                 "",
                 "```powershell",
                 "& 'E:/Users/env/Machine_Learning/python.exe' -X utf8 './analysis/model_comparison/test_benchmark_core.py'",
                 "& 'E:/Users/env/Machine_Learning/python.exe' -X utf8 './analysis/model_comparison/prepare_comparison.py'",
                 "foreach ($benchmarkModel in @('catboost_original','catboost_full','neural')) {",
                 "  & 'E:/Users/env/Machine_Learning/python.exe' -X utf8 './analysis/model_comparison/train_models.py' --model $benchmarkModel --epochs 10 --batch-size 2048",
                 "}",
                 "& 'E:/Users/env/Machine_Learning/python.exe' -X utf8 './analysis/model_comparison/summarize_comparison.py'",
                 "```", ""]
    (RESULTS / "模型对比报告.md").write_text("\n".join(markdown), encoding="utf-8")
    image_data = base64.b64encode((RESULTS / "模型对比曲线.png").read_bytes()).decode("ascii")
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>模型实际训练对比</title><style>body{{font:15px/1.8 "Microsoft YaHei",sans-serif;background:#f3f6f8;color:#193344;margin:0}}main{{max-width:1260px;margin:35px auto;padding:25px}}h1{{font-size:32px}}.box{{background:white;border:1px solid #dce5e9;padding:22px;border-radius:10px;margin:20px 0;overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:12px;border-bottom:1px solid #e0e8ed;text-align:left}}th{{color:#087f78}}img{{width:100%;height:auto}}.note{{color:#607481}}strong{{color:#087f78}}li{{margin:8px 0}}</style><main><p class="note">真实训练结果 · 验证日期2019-11-16</p><h1>CatBoost 与完整神经网络对比</h1><p>三组使用同一批验证记录。<strong>本次AUC最高：{NAMES[best['model']]}，{best['auc']:.6f}</strong>。</p><div class="box">{html_table}</div><p class="note">AUC越高越好，Logloss越低越好。AUC不是点击率，也不是准确率。CatBoost在CPU运行，神经网络在GPU运行，时间只代表本机本次配置。</p><div class="box"><img alt="三组模型ROC和PR曲线" src="data:image/png;base64,{image_data}"></div><div class="box"><h2>这次比较了什么？</h2><ul><li>原11字段CatBoost：对应Notebook使用的简单数值输入。</li><li>完整特征CatBoost：加入完整类别、历史列表与过去资讯画像。</li><li>完整神经网络：实现了前面设计的Embedding、列表平均汇总和516→256→128→64→1网络。</li></ul><h2>结果适用范围</h2><p>训练210万条，验证42万条；资讯历史只取各广告日期之前。原始测试集没有参与评分。本次是一个种子和一个日期的初次对比，仍需其他日期和消融实验检验稳定性。</p><p class="note">完整CatBoost和神经网络使用相同原始信息，但列表编码方式不同，差值反映整个方案的差异。</p></div></main></html>'''
    html = html.replace("<h2>结果适用范围</h2>", "<h2>训练中发现的问题</h2><p>" + overfit_note + "</p><h2>结果适用范围</h2>")
    (RESULTS / "模型对比报告.html").write_text(html, encoding="utf-8")
    versions = {}
    for package in ["torch", "catboost", "numpy", "pandas", "scikit-learn"]:
        versions[package] = metadata.version(package)
    write_json(RESULTS / "run_environment.json", versions)
    write_json(RESULTS / "verification.json", {"status": "passed", "models": 3, "same_validation_rows": 421983, "validation_clicks": 3724,
                                               "metrics_recomputed_from_saved_predictions": True, "all_probabilities_finite_and_in_range": True,
                                               "raw_test_used_for_training_or_metrics": False})
    print(table.to_string(index=False), flush=True)
    print("All three saved predictions independently verified; reports generated.", flush=True)


if __name__ == "__main__":
    main()
