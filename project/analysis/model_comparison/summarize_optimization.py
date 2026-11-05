"""Independently verify all optimization runs and compare the frozen candidate."""
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

from bounded_id_models import BoundedIDDCN
from id_regularized_models import IDRegularizedDCN
from mainstream_models import DCNV2Model
from train_models import load_arrays, metrics, write_json

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_optimization"
CELLS = [(20191115, 42), (20191116, 42), (20191115, 2026), (20191116, 2026)]
NAMES = {"baseline": "原DCN-v2", "U20": "用户编号屏蔽20%", "U50": "用户编号屏蔽50%",
         "UA20": "用户/任务/素材各屏蔽20%", "UA50": "用户50%，任务/素材30%",
         "window3": "资讯换最近3天", "CU10": "用户向量强度≤0.1", "CU20": "用户向量强度≤0.2",
         "CUA20": "用户/任务/素材向量强度≤0.2"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_run(directory, variant, family, original=False):
    record = read_json(directory / "metrics.json")
    cache = Path(record["cache_dir"])
    arrays = load_arrays(cache)
    train, valid = arrays["train_indices"], arrays["valid_indices"]
    day = record.get("validation_day", 20191116)
    assert arrays["date"][train].max() < day and np.all(arrays["date"][valid] == day)
    assert not np.intersect1d(train, valid).size
    prediction = np.load(directory / "validation_prediction.npy")
    assert prediction.shape == (len(valid),)
    assert np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
    result = metrics(arrays["label"][valid], prediction)
    for key, value in result.items():
        assert abs(value - record[key]) < 1e-10, (directory, key)
    new_task = ~np.isin(arrays["baseline"][valid, 8], np.unique(arrays["baseline"][train, 8]))
    new_metrics = metrics(arrays["label"][valid][new_task], prediction[new_task])
    assert abs(new_metrics["auc"] - record["new_task_metrics"]["auc"]) < 1e-10
    checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    manifest = checkpoint["manifest"]
    assert manifest == read_json(cache / "manifest.json")
    builders = {"DCNV2Model": DCNV2Model, "IDRegularizedDCN": IDRegularizedDCN, "BoundedIDDCN": BoundedIDDCN}
    builder = builders[checkpoint["model_class"]]
    model = builder(manifest["categorical_fields"], manifest["sequence_fields"], manifest["group_for_field"],
                    manifest["vocabulary_sizes"], **checkpoint["model_options"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    chosen = np.unique(np.concatenate([np.linspace(0, len(valid) - 1, 128, dtype=int), np.flatnonzero(new_task)[:16]]))
    indices = valid[chosen]
    with torch.no_grad():
        restored = torch.sigmoid(model(torch.from_numpy(arrays["categories"][indices].astype(np.int64)),
                                       torch.from_numpy(arrays["sequences"][indices].astype(np.int64)),
                                       torch.from_numpy(np.array(arrays["numeric"][indices])))).numpy()
    np.testing.assert_allclose(restored, prediction[chosen], rtol=3e-5, atol=2e-7)
    error = float(np.max(np.abs(restored - prediction[chosen])))
    if (directory / "provenance.json").exists():
        for filename, digest in read_json(directory / "provenance.json")["sha256"].items():
            path = cache / filename if filename.endswith(".npy") or filename == "manifest.json" else HERE / filename
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, (directory, filename)
    if original:
        history = pd.read_csv(HERE / "auc_diagnosis/lr0p001/history.csv").rename(columns={"valid_auc": "auc", "valid_logloss": "logloss"})
        record.update(best_fractional_epoch=float(record["best_epoch"]), first_epoch_auc=float(history.loc[history.fractional_epoch.eq(1), "auc"].iloc[0]),
                      second_epoch_auc=float(history.loc[history.fractional_epoch.eq(2), "auc"].iloc[0]))
    else:
        history = pd.read_csv(directory / "training_history.csv")
        selected = history.loc[np.isclose(history.fractional_epoch, record["best_fractional_epoch"])].iloc[0]
        assert abs(selected["auc"] - record["auc"]) < 1e-10
        assert abs(checkpoint["fractional_epoch"] - record["best_fractional_epoch"]) < 1e-12
    user_column = manifest["categorical_fields"].index("user_id")
    user_clicks = np.bincount(arrays["categories"][train, user_column], weights=arrays["label"][train],
                              minlength=manifest["vocabulary_sizes"]["user_id"])
    zero_user = user_clicks[arrays["categories"][valid, user_column]] == 0
    record.update(variant=variant, family=family, results_dir=str(directory), validation_day=day,
                  zero_train_click_user_mean_prediction=float(prediction[zero_user].mean()),
                  zero_train_click_user_actual_ctr=float(arrays["label"][valid][zero_user].mean()),
                  new_task_auc=new_metrics["auc"], actual_ctr=float(arrays["label"][valid].mean()),
                  reload_max_error=error, history=history.to_dict("records"))
    return record


def main():
    torch.set_num_threads(4)
    selection = read_json(OUT / "frozen_selection.json")
    choice = selection["selected"]
    assert selection["contender"] and not selection["confirmation_scores_used"]
    records = {}
    baseline_primary_dir = HERE / "results_mainstream/20191116/dcnv2/seed42"
    baseline_primary = verify_run(baseline_primary_dir, "baseline", "baseline", original=True)
    records[str(baseline_primary_dir)] = baseline_primary
    screen = [baseline_primary]
    for row in selection["all_screening_candidates"]:
        directory = Path(row["results_dir"])
        record = verify_run(directory, row["variant"], row["family"])
        records[str(directory)] = record
        screen.append(record)
    comparisons = []
    for day, seed in CELLS:
        baseline_dir = baseline_primary_dir if (day, seed) == (20191116, 42) else HERE / "results_id_regularization" / str(day) / "baseline" / f"seed{seed}"
        candidate_dir = HERE / choice["results_root"] / str(day) / choice["variant"] / f"seed{seed}"
        for directory, variant, family in [(baseline_dir, "baseline", "baseline"), (candidate_dir, choice["variant"], choice["family"])]:
            if str(directory) not in records:
                records[str(directory)] = verify_run(directory, variant, family)
        baseline, candidate = records[str(baseline_dir)], records[str(candidate_dir)]
        comparisons.append({"validation_day": day, "seed": seed, "baseline_auc": baseline["auc"],
                            "candidate_auc": candidate["auc"], "auc_delta": candidate["auc"] - baseline["auc"],
                            "baseline_logloss": baseline["logloss"], "candidate_logloss": candidate["logloss"],
                            "baseline_new_task_auc": baseline["new_task_auc"], "candidate_new_task_auc": candidate["new_task_auc"],
                            "baseline_best_epoch": baseline["best_fractional_epoch"], "candidate_best_epoch": candidate["best_fractional_epoch"],
                            "baseline_second_epoch_auc": baseline.get("second_epoch_auc"), "candidate_second_epoch_auc": candidate.get("second_epoch_auc"),
                            "screening_cell": (day, seed) == (20191116, 42)})
    paired = pd.DataFrame(comparisons)
    confirmation = paired.loc[~paired.screening_cell]
    wins = int((paired.auc_delta > 0).sum())
    mean_gain = float(paired.auc_delta.mean())
    confirmation_gain = float(confirmation.auc_delta.mean())
    mean_baseline, mean_candidate = float(paired.baseline_auc.mean()), float(paired.candidate_auc.mean())
    headline = f"编号向量约束：四组验证平均AUC从{mean_baseline:.6f}到{mean_candidate:.6f}（{mean_gain:+.6f}），{wins}/4组提高。"
    conclusion = "可以作为下一轮改造的基础。" if confirmation_gain > 0 and int((confirmation.auc_delta > 0).sum()) >= 2 else "复核没有支持稳定替换原模型，保留为实验候选。"
    primary_candidate = records[str(HERE / choice["results_root"] / "20191116" / choice["variant"] / "seed42")]
    write_json(OUT / "verified_runs.json", list(records.values()))
    paired.to_csv(OUT / "paired_comparison.csv", index=False, encoding="utf-8-sig")
    screening_rows = [{"方案": NAMES["window3" if r["family"] == "window3" else r["variant"]],
                       "AUC ↑": f"{r['auc']:.6f}", "相对原模型": f"{r['auc'] - baseline_primary['auc']:+.6f}",
                       "Logloss ↓": f"{r['logloss']:.6f}", "最佳轮次": f"{r['best_fractional_epoch']:.2f}",
                       "第二轮AUC": "未到第2轮即早停" if r.get("second_epoch_auc") is None else f"{r['second_epoch_auc']:.6f}"} for r in screen]
    screening_frame = pd.DataFrame(screening_rows)
    screening_frame.to_csv(OUT / "screening_comparison.csv", index=False, encoding="utf-8-sig")
    display = pd.DataFrame([{"验证日期": str(r["validation_day"]), "随机种子": r["seed"],
                             "原模型AUC": f"{r['baseline_auc']:.6f}", "优化AUC": f"{r['candidate_auc']:.6f}",
                             "变化": f"{r['auc_delta']:+.6f}", "原Logloss": f"{r['baseline_logloss']:.6f}",
                             "优化Logloss": f"{r['candidate_logloss']:.6f}"} for r in comparisons])
    insights = [headline + conclusion,
                f"主筛选日2019-11-16、种子42：AUC从{baseline_primary['auc']:.6f}到{primary_candidate['auc']:.6f}。",
                f"排除用于选方案的主筛选格后，另外三组复核的平均AUC变化为{confirmation_gain:+.6f}。候选在看到这些复核分数前已冻结。",
                "优化保留了原DCN-v2的Embedding=8、两层交叉网络、128→64深层网络，以及相同输入信息。它限制用户、任务、素材编号的内部向量强度，保留向量方向；0.2不是点击概率，也不是原始编号数值。",
                "约束也应用于与任务共享向量空间的点击/关闭历史字段。它在训练和预测时都生效，新增的随机屏蔽方案仅在训练时生效。",
                "四组随机屏蔽和最近3天资讯统计没有超过原来的主验证AUC；完整失败结果保留在筛选表中。抑制过拟合不自动等于提高峰值AUC。",
                f"主验证第2轮：原模型{baseline_primary.get('second_epoch_auc', float('nan')):.6f}，约束模型{primary_candidate.get('second_epoch_auc', float('nan')):.6f}。下降已减轻，但仍然需要按最佳验证分数保存权重。",
                "本轮统一每四分之一轮检查验证集。11月15日原模型的种子42最优点出现在约0.75轮，因此本轮对照AUC会高于旧版只在整轮检查时的0.721911；这是验证频率变化，不能把这部分算成新结构的收益。",
                "四组平均是两个日期×两个种子的算术均值。训练窗口重叠，验证日期也曾用于之前实验，这些结果不构成独立最终测试或统计显著性证明。原始无广告标签测试集未参与选型或评分。"]
    font = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(), "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), layout="constrained")
    for record, name, color in [(baseline_primary, "原DCN-v2", "#9a7048"), (primary_candidate, "编号向量约束", "#137e80")]:
        history = pd.DataFrame(record["history"])
        axes[0].plot(history.fractional_epoch, history.auc, marker="o", color=color, label=name)
    axes[0].set(title="11月16日、种子42：后续掉分减轻", xlabel="训练轮次", ylabel="验证AUC")
    positions = np.arange(len(paired))
    axes[1].plot(positions, paired.baseline_auc, marker="o", label="原DCN-v2", color="#9a7048")
    axes[1].plot(positions, paired.candidate_auc, marker="o", label="编号向量约束", color="#137e80")
    axes[1].set_xticks(positions, [f"11月{r['validation_day'] % 100}日\n种子{r['seed']}" for r in comparisons])
    axes[1].set(title="固定候选后，跨日期和随机种子复核", ylabel="最佳验证AUC")
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
    fig.suptitle("AUC优化：真实筛选与四组对照", fontsize=18)
    figure = OUT / "AUC优化对比曲线.png"
    fig.savefig(figure, dpi=170)
    plt.close(fig)
    image_src = base64.b64encode(figure.read_bytes()).decode("ascii")
    bullets = "".join("<li>" + html.escape(text) + "</li>" for text in insights)
    checkpoint_path = Path(primary_candidate["results_dir"]) / "model.pt"
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AUC优化与复核报告</title>
<style>body{{margin:0;background:#f3f6f8;color:#213547;font:15px/1.9 "Microsoft YaHei",sans-serif}}main{{max-width:1230px;margin:auto;padding:28px}}section{{background:white;border:1px solid #dce5e9;border-radius:12px;padding:24px;margin:22px 0;overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:10px;border-bottom:1px solid #e1e8ed;text-align:left}}th{{color:#137e80}}img{{width:100%}}li{{margin:10px 0}}.lead{{padding:20px;background:#e5f2ef;border-radius:10px;font-size:19px}}code{{overflow-wrap:anywhere}}</style>
<main><h1>AUC优化与复核报告</h1><p class="lead">{html.escape(headline + conclusion)}</p><section><h2>固定方案：两个日期、两个随机种子</h2>{display.to_html(index=False,border=0)}</section>
<section><img src="data:image/png;base64,{image_src}" alt="过拟合曲线和四组复核结果"></section><section><h2>大白话说明</h2><ul>{bullets}</ul></section>
<section><h2>所有主筛选结果</h2>{screening_frame.to_html(index=False,border=0)}</section><section><h2>结构与权重</h2><p>类别编号 → Embedding；历史列表 → 平均向量；用户/任务/素材字段向量限制强度≤0.2 → 与数值特征拼接 → 两层CrossNet和128→64神经网络并行 → Sigmoid点击概率。</p><p>参数量与原结构相同，主验证为{primary_candidate['parameter_count']:,}个。保留原模型作为对照。</p><p>主验证候选检查点：<code>{html.escape(str(checkpoint_path))}</code></p><p>加载时须使用checkpoint内model_options构造BoundedIDDCN；约束是预测结构的一部分，不能换成原DCNV2Model直接预测。</p></section>
<p>已重新计算全部{len(records)}组保存预测的指标，并重新加载全部最佳检查点复算概率。详细表格见同目录paired_comparison.csv和screening_comparison.csv；核验见verification.json。</p></main></html>'''
    (OUT / "AUC优化与复核报告.html").write_text(document, encoding="utf-8")
    markdown = ["# AUC优化与复核报告", "", headline + conclusion, "", "## 四组复核", "", display.to_string(index=False), "", "## 说明", ""]
    markdown += ["- " + text for text in insights]
    markdown += ["", "## 全部筛选结果", "", screening_frame.to_string(index=False), "", "主验证检查点：" + str(checkpoint_path)]
    (OUT / "AUC优化与复核报告.md").write_text("\n".join(markdown), encoding="utf-8")
    write_json(OUT / "verification.json", {"status": "passed", "runs_verified": len(records), "paired_cells": len(comparisons),
               "all_predictions_rescored": True, "all_checkpoints_reloaded": True, "time_boundaries_checked": True,
               "source_provenance_checked": True, "maximum_reload_probability_error": max(r["reload_max_error"] for r in records.values()),
               "candidate_frozen_before_confirmation": True, "mean_auc_delta": mean_gain,
               "confirmation_mean_auc_delta": confirmation_gain, "auc_wins": wins,
               "seeds": [42, 2026], "validation_days": [20191115, 20191116], "final_test_used": False})
    print(display.to_string(index=False), flush=True)
    print(headline + conclusion, flush=True)
    print("VERIFIED_OPTIMIZATION_REPORT " + str(OUT / "AUC优化与复核报告.html"), flush=True)


if __name__ == "__main__":
    main()
