"""Rescore saved predictions, reload checkpoints and produce a Chinese report."""
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

from mainstream_models import mainstream_definition
from train_models import load_arrays, metrics, write_json

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_mainstream"
NAMES = {"mlp": "普通MLP（B2）", "deepfm": "DeepFM", "dcnv2": "DCN-v2"}
DAYS = [20191115, 20191116]
COLORS = {"mlp": "#687b8d", "deepfm": "#127d80", "dcnv2": "#c37534"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_run(name, day):
    directory = (HERE / "results_v2" / str(day) / "B2" / "seed42") if (name, day) == ("mlp", 20191116) else OUT / str(day) / name / "seed42"
    record = read_json(directory / "metrics.json")
    cache = Path(record["cache_dir"])
    arrays = load_arrays(cache)
    train, valid = arrays["train_indices"], arrays["valid_indices"]
    assert arrays["date"][train].max() < day
    assert np.all(arrays["date"][valid] == day)
    assert not np.intersect1d(train, valid).size
    prediction = np.load(directory / "validation_prediction.npy")
    assert prediction.shape == (len(valid),)
    assert np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
    recomputed = metrics(arrays["label"][valid], prediction)
    for key, value in recomputed.items():
        assert abs(value - record[key]) < 1e-10, (name, day, key)
    new_task = ~np.isin(arrays["baseline"][valid, 8], np.unique(arrays["baseline"][train, 8]))
    new_metrics = metrics(arrays["label"][valid][new_task], prediction[new_task])
    assert abs(new_metrics["auc"] - record["new_task_metrics"]["auc"]) < 1e-10
    history = pd.read_csv(directory / "training_history.csv")
    best = history.loc[history["epoch"].eq(record["best_epoch"])].iloc[0]
    assert abs(float(best["auc"]) - record["auc"]) < 1e-10
    checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    manifest = checkpoint["manifest"]
    assert manifest["validation_day"] == day
    assert manifest == read_json(cache / "manifest.json")
    assert checkpoint["epoch"] == record["best_epoch"]
    assert checkpoint["seed"] == 42
    builder, _ = mainstream_definition(name)
    assert checkpoint["model_class"] == builder.__name__
    model = builder(manifest["categorical_fields"], manifest["sequence_fields"],
                    manifest["group_for_field"], manifest["vocabulary_sizes"],
                    **checkpoint["model_options"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    # Include both seen and new tasks rather than just the first rows.
    chosen = np.unique(np.concatenate([np.linspace(0, len(valid) - 1, 128, dtype=int),
                                      np.flatnonzero(new_task)[:16]]))
    indices = valid[chosen]
    with torch.no_grad():
        restored = torch.sigmoid(model(torch.from_numpy(arrays["categories"][indices].astype(np.int64)),
                                       torch.from_numpy(arrays["sequences"][indices].astype(np.int64)),
                                       torch.from_numpy(np.array(arrays["numeric"][indices])))).numpy()
    error = float(np.max(np.abs(restored - prediction[chosen])))
    np.testing.assert_allclose(restored, prediction[chosen], rtol=3e-5, atol=2e-7)
    provenance_path = directory / "provenance.json"
    if provenance_path.exists():
        provenance = read_json(provenance_path)
        for filename, digest in provenance["sha256"].items():
            path = (cache / filename) if filename.endswith(".npy") or filename == "manifest.json" else HERE / filename
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    record.update(model=name, results_dir=str(directory), validation_day=day,
                  new_task_auc=new_metrics["auc"], actual_ctr=float(arrays["label"][valid].mean()),
                  reload_max_abs_error=error, reload_rows=len(chosen))
    return record


def result_frame(records, day):
    rows = [r for r in records if r["validation_day"] == day]
    control = next(r for r in rows if r["model"] == "mlp")
    return pd.DataFrame([{"模型": NAMES[r["model"]], "AUC ↑": f"{r['auc']:.6f}",
                          "比普通MLP变化": f"{r['auc'] - control['auc']:+.6f}",
                          "Logloss ↓": f"{r['logloss']:.6f}",
                          "新广告任务AUC ↑": f"{r['new_task_auc']:.6f}",
                          "参数量": f"{r['parameter_count']:,}",
                          "最佳轮次": r["best_epoch"], "实际训练轮次": r["epochs_completed"]} for r in rows])


def main():
    torch.set_num_threads(4)
    records = [verify_run(name, day) for day in DAYS for name in NAMES]
    write_json(OUT / "comparison.json", records)
    columns = ["validation_day", "model", "seed", "auc", "logloss", "average_precision", "new_task_auc",
               "parameter_count", "best_epoch", "epochs_completed", "training_seconds", "mean_prediction", "actual_ctr"]
    table = pd.DataFrame([{key: r[key] for key in columns} for r in records])
    table.to_csv(OUT / "comparison.csv", index=False, encoding="utf-8-sig")
    by_run = {(r["model"], r["validation_day"]): r for r in records}
    deltas = {name: [by_run[name, day]["auc"] - by_run["mlp", day]["auc"] for day in DAYS]
              for name in ("deepfm", "dcnv2")}
    statements = []
    for name, values in deltas.items():
        interpretation = "两个日期都超过普通MLP" if min(values) > 0 else "两个日期都低于普通MLP" if max(values) < 0 else "两个日期的胜负不一致"
        statements.append(f"{NAMES[name]}：11月15日AUC变化{values[0]:+.6f}，11月16日变化{values[1]:+.6f}；{interpretation}。")
    best_mean = max(NAMES, key=lambda name: np.mean([by_run[name, day]["auc"] for day in DAYS]))
    if best_mean == "mlp":
        recommendation = "本轮普通MLP的两日平均AUC最高，暂时保留它作基础。增加特征交叉层并没有带来稳定收益。"
    elif min(deltas[best_mean]) > 0:
        recommendation = f"本轮{NAMES[best_mean]}在两个日期都领先普通MLP，可以优先作为下一步改造的基础；仍需多随机种子复核。"
    else:
        recommendation = f"本轮{NAMES[best_mean]}的两日平均AUC最高，但跨日胜负不一致，暂时不能认为它稳定优于普通MLP。"
    best_epochs = [r["best_epoch"] for r in records]
    if all(epoch == 1 for epoch in best_epochs):
        statements.append("六次实验都是第1轮最好，继续训练时验证效果下降。加入两两组合或交叉层，仍未消除现有的过拟合现象。")
    else:
        statements.append("最佳轮次和完整训练曲线见下方；保存的是最佳验证AUC对应权重，不能只看最后一轮。")
    statements.append("新广告任务AUC单独衡量训练广告中没出现过的任务；这个切片比总体分数更能提醒我们对新广告的适应情况。")
    statements.append("这里只运行随机种子42。两个日期的训练窗口有重叠，而且各自用于选择最佳轮次，不能把这当作独立最终测试或显著性证明。")
    statements.append("DeepFM与DCN-v2都沿用相同原始输入信息，但其内部表示和参数量不同；这轮没有针对每个架构单独调参，不能据此宣称某个架构普遍更强。")

    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    for name in NAMES:
        axes[0, 0].plot([15, 16], [by_run[name, day]["auc"] for day in DAYS],
                        marker="o", color=COLORS[name], label=NAMES[name])
        axes[0, 1].plot([15, 16], [by_run[name, day]["new_task_auc"] for day in DAYS],
                        marker="o", color=COLORS[name], label=NAMES[name])
    for ax, title in zip(axes[0], ["同一天内比较：总体验证AUC", "训练广告中没见过的新任务：AUC"]):
        ax.set(title=title, xlabel="2019年11月验证日期", ylabel="AUC（越高越好）")
        ax.set_xticks([15, 16], ["11月15日", "11月16日"])
    for ax, day in zip(axes[1], DAYS):
        for name in NAMES:
            history = pd.read_csv(Path(by_run[name, day]["results_dir"]) / "training_history.csv")
            ax.plot(history.epoch, history.auc, marker="o", color=COLORS[name], label=NAMES[name])
        ax.set(title=f"11月{day % 100}日：多训练几轮会怎样", xlabel="训练轮次", ylabel="验证AUC")
        ax.set_xticks(range(1, max(by_run[name, day]["epochs_completed"] for name in NAMES) + 1))
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle("DeepFM / DCN-v2：全量训练、两个日期验证", fontsize=19)
    figure = OUT / "主流模型对比曲线.png"
    fig.savefig(figure, dpi=170)
    plt.close(fig)

    historical = []
    for name, directory in [("原始完整神经网络", HERE / "results/neural"),
                            ("B1：缩小Embedding", HERE / "results_v2/20191116/B1/seed42"),
                            ("B4：注意力＋资讯门控", HERE / "results_v2/20191116/B4/seed42"),
                            ("全特征CatBoost", HERE / "results/catboost_full")]:
        r = read_json(directory / "metrics.json")
        historical.append({"历史参照模型": name, "11月16日AUC": f"{r['auc']:.6f}", "Logloss": f"{r['logloss']:.6f}"})
    historical_table = pd.DataFrame(historical)
    scope = ["每条广告展示预测点击概率；AUC越高越好，Logloss越低越好。AUC不是准确率，也不是点击率。",
             "11月16日：此前2,104,337条广告训练，当日421,983条验证；11月15日：此前1,679,557条训练，当日424,780条验证。",
             "输入：24个类别、7组短历史列表、20个数值特征。词表与缩放只拟合各折训练窗口，资讯仅来自广告日期之前。",
             "普通MLP和两种新结构均使用Embedding=8、隐藏层128→64、AdamW学习率0.001、batch=2048，最多10轮，3轮无改善停止。",
             "11月16日普通MLP复用已完成且设置相同的B2；其余五次为本轮新训练。原有模型文件保留。",
             "DeepFM=一次项＋FM两两组合＋深层网络，FM与深层网络共享类别和数值字段Embedding；历史列表按平均向量作为一个字段。",
             "DCN-v2=两层完整矩阵交叉网络，与深层网络并行后拼接预测；没有启用可选的专家混合模块。",
             "这是适配本数据字段的实现，并非复刻原论文数据或成绩。未使用无广告标签的最终测试集评分。"]
    bullets = lambda texts: "".join("<li>" + html.escape(text) + "</li>" for text in texts)
    imagesrc = base64.b64encode(figure.read_bytes()).decode("ascii")
    tables_html = "".join(f'<section><h2>2019年11月{day % 100}日</h2>{result_frame(records, day).to_html(index=False, border=0)}</section>' for day in DAYS[::-1])
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>主流模型测试报告</title>
<style>body{{margin:0;background:#f3f6f8;color:#213547;font:15px/1.9 "Microsoft YaHei",sans-serif}}main{{max-width:1200px;margin:auto;padding:28px}}h1{{font-size:32px}}h2{{font-size:22px}}section{{background:white;border:1px solid #dde6ea;border-radius:12px;padding:24px;margin:22px 0;overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #e4eaee}}th{{color:#127d80}}li{{margin:8px 0}}img{{width:100%;height:auto}}.note{{color:#607586}}.lead{{font-size:19px;background:#e8f3f1;padding:20px;border-radius:10px}}</style></head>
<body><main><p class="note">真实全量训练 · 相同信息输入 · 按日期留出 · 固定随机种子42</p><h1>主流模型测试报告</h1><p class="lead">{html.escape(recommendation)}</p>
{tables_html}<section><h2>大白话：这次测出了什么</h2><ul>{bullets(statements)}</ul></section><section><h2>训练过程</h2><img src="data:image/png;base64,{imagesrc}" alt="两天总体验证、新广告和每轮训练曲线"></section>
<section><h2>此前模型参照（只完成11月16日）</h2>{historical_table.to_html(index=False, border=0)}<p class="note">历史参照未在本轮重训，结构规模与输入编码可能不同，不能与两日平均分混排。</p></section>
<section><h2>数据与测试方法</h2><ul>{bullets(scope)}</ul></section><p class="note">已重新计算全部保存预测的指标，并加载六个最佳权重，各抽取128条均匀样本及最多16条新任务样本复算概率。原始结果见同目录comparison.csv、各次metrics.json和training_history.csv。</p></main></body></html>'''
    (OUT / "主流模型测试报告.html").write_text(document, encoding="utf-8")
    markdown = ["# 主流模型测试报告", "", recommendation, ""]
    for day in DAYS[::-1]:
        frame = result_frame(records, day)
        markdown += [f"## 2019年11月{day % 100}日", "", "| " + " | ".join(frame.columns) + " |",
                     "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
        markdown += ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)]
        markdown.append("")
    markdown += ["## 大白话说明", ""] + ["- " + text for text in statements]
    markdown += ["", "## 实验口径", ""] + ["- " + text for text in scope]
    markdown += ["", "## 结构参考", "", "- DeepFM：Guo等，2017，arXiv:1703.04247。", "- DCN-v2：Wang等，2021，arXiv:2008.13535。", "",
                 "复现入口：run_mainstream.py；固定方案：mainstream_protocol.json；核验和报告：summarize_mainstream.py。"]
    (OUT / "主流模型测试报告.md").write_text("\n".join(markdown), encoding="utf-8")
    write_json(OUT / "verification.json", {"status": "passed", "runs_verified": len(records),
               "saved_predictions_rescored": True, "all_six_checkpoints_reloaded": True,
               "time_boundaries_and_no_overlap_checked": True, "new_run_provenance_checked": True,
               "seed": 42, "dates": DAYS, "final_unlabeled_test_used": False,
               "maximum_reload_probability_error": max(r["reload_max_abs_error"] for r in records)})
    print(table.to_string(index=False), flush=True)
    print(recommendation, flush=True)
    print("VERIFIED_REPORT " + str(OUT / "主流模型测试报告.html"), flush=True)


if __name__ == "__main__":
    main()
