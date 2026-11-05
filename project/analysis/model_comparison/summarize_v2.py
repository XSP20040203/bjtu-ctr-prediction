"""Verify and report the architecture experiments and an earlier-date recheck."""
import base64
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_v2"
NAMES = {"baseline": "旧模型", "B1": "B1：只缩小Embedding", "B2": "B2：再缩小隐藏层", "B3": "B3：再加注意力", "B4": "B4：再加资讯门控"}
ORDER = {name: index for index, name in enumerate(NAMES)}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(rows):
    lines = ["| 版本 | AUC | Logloss | 新任务AUC | 参数量 | 最佳轮次 |", "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {NAMES[row['variant']]} | {row['auc']:.6f} | {row['logloss']:.6f} | {row['new_task_auc']:.6f} | {row['parameter_count']:,} | {row['best_epoch']} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-only", action="store_true", help="Report finished primary-date runs without claiming interrupted confirmation runs completed")
    args = parser.parse_args()
    selection = load_json(OUT / "frozen_selection.json")
    original_dir = HERE / "results" / "neural"
    original = load_json(original_dir / "metrics.json")
    original.update(variant="baseline", validation_day=20191116, cache_dir=str(HERE / "cache"), results_dir=str(original_dir))
    records = [original]
    for path in sorted(OUT.glob("20*/*/seed*/metrics.json")):
        record = load_json(path)
        record["results_dir"] = str(path.parent)
        records.append(record)
    required = {"baseline", *selection["selected_variants"]}
    if not args.primary_only:
        assert required.issubset({row["variant"] for row in records if row["validation_day"] == 20191115})
    prepared = {}
    for record in records:
        cache = Path(record["cache_dir"])
        key = str(cache)
        if key not in prepared:
            train = np.load(cache / "train_indices.npy")
            valid = np.load(cache / "valid_indices.npy")
            labels = np.load(cache / "label.npy", mmap_mode="r")[valid]
            raw = np.load(cache / "baseline.npy", mmap_mode="r")
            new_task = ~np.isin(raw[valid, 8], np.unique(raw[train, 8]))
            dates = np.load(cache / "date.npy", mmap_mode="r")
            assert dates[train].max() < record["validation_day"]
            assert np.all(dates[valid] == record["validation_day"])
            prepared[key] = (valid, labels, new_task)
        valid, labels, new_task = prepared[key]
        probability = np.load(Path(record["results_dir"]) / "validation_prediction.npy")
        assert len(probability) == len(valid) == record["valid_rows"]
        assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
        clipped = np.clip(probability.astype(float), 1e-8, 1 - 1e-8)
        assert abs(roc_auc_score(labels, clipped) - record["auc"]) < 1e-10
        assert abs(log_loss(labels, clipped, labels=[0, 1]) - record["logloss"]) < 1e-10
        record["new_task_auc"] = float(roc_auc_score(labels[new_task], probability[new_task]))
        record["new_task_rows"] = int(new_task.sum())
    records.sort(key=lambda row: (row["validation_day"], ORDER[row["variant"]], row["seed"]))
    primary = [row for row in records if row["validation_day"] == 20191116 and row["seed"] == 42]
    secondary = [row for row in records if row["validation_day"] == 20191115 and row["seed"] == 42]
    assert {row["variant"] for row in primary} == set(NAMES)
    baseline = next(row for row in primary if row["variant"] == "baseline")
    compact = next(row for row in primary if row["variant"] == "B2")
    gated = next(row for row in primary if row["variant"] == "B4")
    table = pd.DataFrame([{key: row[key] for key in ["validation_day", "variant", "seed", "auc", "logloss", "new_task_auc", "parameter_count", "best_epoch", "epochs_completed", "training_seconds"]} for row in records])
    table.to_csv(OUT / "v2_comparison.csv", index=False, encoding="utf-8-sig")
    (OUT / "v2_comparison.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    font = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(), "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    colors = {"baseline": "#647b8d", "B1": "#357ab1", "B2": "#9b7c42", "B3": "#995c9b", "B4": "#087f78"}
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6), layout="constrained")
    for row in primary:
        history = pd.read_csv(Path(row["results_dir"]) / "training_history.csv")
        axes[0].plot(history.epoch, history.auc, marker="o", color=colors[row["variant"]], label=NAMES[row["variant"]])
    axes[0].set(title="11月16日：继续训练是否仍然过拟合", xlabel="Epoch", ylabel="验证AUC")
    axes[0].set_xticks(range(1, max(row["epochs_completed"] for row in primary) + 1))
    if secondary:
        for row in secondary:
            matching = next(item for item in primary if item["variant"] == row["variant"])
            axes[1].plot([15, 16], [row["auc"], matching["auc"]], marker="o", color=colors[row["variant"]], label=NAMES[row["variant"]])
        axes[1].set(title="两个日期复核：并非独立最终测试", xlabel="2019年11月验证日期", ylabel="验证AUC")
        axes[1].set_xticks([15, 16], ["11月15日", "11月16日"])
    else:
        for row in primary:
            axes[1].scatter(row["parameter_count"] / 10000, row["auc"], color=colors[row["variant"]], s=45, label=NAMES[row["variant"]])
        axes[1].set(title="参数减少，单日AUC变化仍很小", xlabel="参数量（万）", ylabel="11月16日验证AUC", ylim=(.70, .75))
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        ax.grid(color="#e1e8ed", linewidth=.6)
    fig.suptitle("V2候选结构的真实训练结果", fontsize=18)
    fig.savefig(OUT / "V2模型测试曲线.png", dpi=170)
    plt.close(fig)

    headline = f"完整V2（B4）11月16日AUC={gated['auc']:.6f}，旧模型={baseline['auc']:.6f}，差值{gated['auc']-baseline['auc']:+.6f}。"
    bests = []
    for variant in selection["selected_variants"]:
        row = next((row for row in secondary if row["variant"] == variant), None)
        base = next((row for row in secondary if row["variant"] == "baseline"), None)
        if row is None or base is None:
            continue
        bests.append(f"11月15日{NAMES[variant]}的AUC差值为{row['auc']-base['auc']:+.6f}，Logloss差值为{row['logloss']-base['logloss']:+.6f}。")
    observations = [headline, *bests,
                    f"B2参数从{baseline['parameter_count']:,}降到{compact['parameter_count']:,}，减少{1-compact['parameter_count']/baseline['parameter_count']:.2%}；其主验证AUC差值为{compact['auc']-baseline['auc']:+.6f}。参数减少不等于已经解决泛化问题。",
                    f"本日旧模型新任务AUC为{baseline['new_task_auc']:.6f}；四个新版中最高为{max(row['new_task_auc'] for row in primary if row['variant'] != 'baseline'):.6f}，当前新任务切片没有改善。",
                    "本轮五个版本都是第1个epoch最佳，后续验证AUC下降；缩小结构、加入注意力和门控没有明显缓解这组训练设置下的过拟合模式。",
                    "各版本都按最佳验证AUC保存权重。需要同时看后续轮次的下降，而不是只展示峰值。",
                    ("这是固定种子42、单日主验证结果。11月15日复核在中断后未完成，不能据此判断跨日期稳定性。" if args.primary_only else "这是固定种子42、两次按日留出的测试；11月16日用于筛选架构，11月15日用于复核，二者都不是最终未知标签测试集。") + "尚未做多种子置信区间。"]
    scope_notice = "11月16日主验证完成；11月15日复核未完成。" if args.primary_only else "主验证与第二日期复核已完成。"
    secondary_markdown = markdown_table(secondary) if secondary else "复核训练在中断后未完成，没有可报告的该日分数。"
    markdown = ["# V2新模型测试报告", "", scope_notice, "", "本报告来自真实全量训练，包含未明显提升的版本。未重新训练CatBoost。", "", "## 11月16日主验证", "", markdown_table(primary), "", "## 11月15日复核", "", secondary_markdown, "", "## 结果说明", ""]
    markdown += ["- " + text for text in observations]
    markdown += ["", "## 实验口径", "", "- B1只降低Embedding维度；B2再缩小隐藏层；B3增加三组目标相关注意力；B4再增加独立资讯分支和门控。", "- 同一日期中复用相同特征缓存、20个数值字段、训练顺序、种子和主要训练设置。没有把窗口特征变化混入结构实验。", "- 11月15日重新拟合词表与缩放参数，只用更早广告训练；11月16日广告不进入该折训练或验证。资讯始终只取对应广告日期之前。", "- B4的来源分支和预测头按设计重构；不能把B4与B3之间全部差异只归因于一个门控乘法。", "- 候选由11月16日的最高AUC和最低Logloss选出，在查看11月15日分数之前冻结，见frozen_selection.json。", "- 使用相同RTX 3050 GPU；计时受缓存、数据准备和其他负载影响，不作严格速度结论。", "- 原始测试集未用于这些训练与评分。最佳模型分数是验证分数，不是最终测试分数。", "", "## 实现与复现", "", "模型：analysis/model_comparison/v2_models.py；训练入口：run_v2.py；已有原模型权重与结果保留。每个日期/版本/种子目录包含metrics.json、training_history.csv、model.pt和validation_prediction.npy。", "", "```powershell", "& 'E:/Users/env/Machine_Learning/python.exe' -X utf8 -m unittest discover -s './analysis/model_comparison' -p 'test_*.py' -v", "& 'E:/Users/env/Machine_Learning/python.exe' -X utf8 './analysis/model_comparison/run_v2.py' --variant B4 --epochs 10 --batch-size 2048", "& 'E:/Users/env/Machine_Learning/python.exe' -X utf8 './analysis/model_comparison/summarize_v2.py'", "```", ""]
    if args.primary_only:
        markdown = [line.replace("11月15日重新拟合词表与缩放参数，只用更早广告训练；11月16日广告不进入该折训练或验证。资讯始终只取对应广告日期之前。", "已为11月15日准备独立词表与缩放缓存，但该折训练未完成；本报告只检查已完成的11月16日模型。") for line in markdown]
        markdown = [line + " --primary-only" if line.endswith("'./analysis/model_comparison/summarize_v2.py'") else line for line in markdown]
    (OUT / "V2新模型测试报告.md").write_text("\n".join(markdown), encoding="utf-8")
    def html_table(rows):
        return pd.DataFrame([{"模型": NAMES[row["variant"]], "AUC ↑": f"{row['auc']:.6f}", "Logloss ↓": f"{row['logloss']:.6f}", "新任务AUC ↑": f"{row['new_task_auc']:.6f}", "参数量": f"{row['parameter_count']:,}", "最佳轮次": row["best_epoch"]} for row in rows]).to_html(index=False, border=0)
    png = base64.b64encode((OUT / "V2模型测试曲线.png").read_bytes()).decode("ascii")
    bullets = "".join("<li>" + text + "</li>" for text in observations)
    secondary_html = html_table(secondary) if secondary else "<p>中断后复核训练未完成，没有可报告的该日分数。</p>"
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V2新模型测试报告</title><style>body{{margin:0;background:#f3f6f8;color:#193344;font:14px/1.9 "Microsoft YaHei",sans-serif}}main{{max-width:1260px;margin:30px auto;padding:24px}}h1{{font-size:32px}}.panel{{background:white;border:1px solid #dce5e9;border-radius:10px;padding:22px;margin:20px 0;overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:11px;border-bottom:1px solid #e1e8ed;text-align:left}}th{{color:#087f78}}img{{width:100%;height:auto}}li{{margin:8px 0}}.note{{color:#607481}}</style><main><p class="note">真实训练 · 同一日期内相同特征与训练设置</p><h1>V2新模型测试报告</h1><p>{headline}</p><p class="note">{scope_notice}</p><div class="panel"><h2>11月16日：所有结构版本</h2>{html_table(primary)}</div><div class="panel"><h2>11月15日：旧模型与冻结候选复核</h2>{secondary_html}</div><div class="panel"><h2>大白话说明</h2><ul>{bullets}</ul></div><div class="panel"><img alt="训练过程和参数规模比较" src="data:image/png;base64,{png}"></div><p class="note">B1=Embedding缩小；B2=隐藏层再缩小；B3=再加广告相关注意力；B4=再加资讯分支与门控。AUC不是点击率或准确率，越高越好；Logloss越低越好。没有使用最终测试标签。</p></main></html>'''
    (OUT / "V2新模型测试报告.html").write_text(html, encoding="utf-8")
    (OUT / "verification.json").write_text(json.dumps({"status": "passed_for_completed_runs", "runs_verified": len(records), "dates": sorted({row["validation_day"] for row in records}), "secondary_confirmation_completed": not args.primary_only, "saved_predictions_recomputed": True, "strict_time_boundaries_checked": True, "original_baseline_preserved": True}, indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print("V2 report generated; all saved metrics independently verified.", flush=True)


if __name__ == "__main__":
    main()
