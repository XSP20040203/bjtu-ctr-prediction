"""Summarize replay-verified source graphs and candidate-conditioned histories."""
import base64
import html
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model_comparison"))
from train_models import metrics


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


NAMES = {
    "phase3_graph_control": "图模型对照：普通TabM",
    "phase3_graph_frozen": "图兴趣＋固定资讯表示",
    "phase3_graph_finetune": "图兴趣＋允许广告微调",
    "phase3_history_mean": "历史对照：平均处理",
    "phase3_history_query": "历史：按当前广告加权",
    "phase3_history_position": "历史：加权＋原表位置",
    "phase3_ranking_position": "位置历史＋辅助排序目标",
}


def main():
    tests = load(HERE / "phase3_test_verification.json")
    assert tests["status"] == "passed" and tests["tests_run"] == 40
    for name, expected_hash in tests["source_sha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected_hash
    source_audit = load(HERE / "phase3_graph_source_audit.json")
    assert source_audit["status"] == "passed"
    records = []
    for prefix, expected_count in [("phase3_graph_full", 3), ("phase3_history_full", 3), ("phase3_ranking_full", 1)]:
        verification = load(HERE / (prefix + "_verification.json"))
        assert verification["status"] == "passed" and verification["completed_runs_verified"] == expected_count
        checked = load(HERE / (prefix + "_verified_results.json"))
        assert all(row["replay_status"] == "full_precision_matched_replay_passed" for row in checked)
        records.extend(checked)
    assert set(row["experiment"] for row in records) == set(NAMES)
    by_tag = {row["experiment"]: row for row in records}
    cache = Path(records[0]["cache_dir"])
    valid = np.load(cache / "valid_indices.npy")
    target = np.load(cache / "label.npy", mmap_mode="r")[valid]
    for row in records:
        np.testing.assert_array_equal(valid, np.load(Path(row["cache_dir"]) / "valid_indices.npy"))
    previous = next(row for row in load(HERE / "results_architecture_ensembles/metrics.json") if row["name"] == "dcn_tabm32")
    reference = np.load(HERE / "results_architecture_ensembles/dcn_tabm32_validation_prediction.npy")
    assert abs(metrics(target, reference)["auc"] - previous["auc"]) < 1e-10
    position = np.load(Path(by_tag["phase3_history_position"]["results_dir"]) / "validation_prediction.npy")
    # This fixed combination was declared in phase3_ad_history_protocol.json
    # before any of the three history models were trained.
    blended = .5 * reference.astype(np.float64) + .5 * position.astype(np.float64)
    blend_metrics = metrics(target, blended)
    np.save(HERE / "phase3_position_fixed_blend_validation_prediction.npy", blended)
    blend_record = {"name": "existing_best_plus_position", **blend_metrics,
                    "weights": {"previous_dcn_tabm32": .5, "position_history": .5},
                    "validation_rows": len(valid), "weight_fitting": "none; fixed in protocol before training",
                    "scope": "Reused random validation; final test unscored"}
    (HERE / "phase3_fixed_blend_metrics.json").write_text(json.dumps(blend_record, indent=2), encoding="utf-8")
    table = []
    for row in records:
        reference_tag = "phase3_graph_control" if row["experiment"].startswith("phase3_graph") else "phase3_history_mean"
        if row["experiment"] == "phase3_ranking_position":
            reference_tag = "phase3_history_position"
        table.append({"架构": NAMES[row["experiment"]], "验证AUC": f"{row['auc']:.6f}",
                      "相对本组对照": f"{row['auc'] - by_tag[reference_tag]['auc']:+.6f}",
                      "验证Logloss": f"{row['logloss']:.6f}", "最佳轮次": row["best_epoch"], "参数量": f"{row['parameter_count']:,}"})
    frame = pd.DataFrame(table)
    frame.to_csv(HERE / "phase3_architecture_comparison.csv", index=False, encoding="utf-8-sig")
    best_new = max(records, key=lambda row: row["auc"])
    best_auc = max(previous["auc"], blend_metrics["auc"], best_new["auc"])
    graph_best = max(row["auc"] for row in records if row["family"] == "graph")
    history_delta = by_tag["phase3_history_position"]["auc"] - by_tag["phase3_history_mean"]["auc"]
    audit = load(HERE / "phase3_ad_history_audit.json")
    history_table = pd.DataFrame([{"历史记录": {"task_id": "点过当前广告任务", "adv_prim_id": "点过当前广告主", "spread_app_id": "点过当前应用"}[row["current_field"]],
                                  "训练集中匹配的占比": f"{row['match_fraction']:.2%}", "匹配时点击率": f"{row['ctr_if_match']:.2%}",
                                  "不匹配时点击率": f"{row['ctr_if_no_match']:.2%}"} for row in audit["fields"][:3]])
    teacher = load(HERE / "source_graph_teacher_day16/manifest.json")
    graph = load(Path(teacher["source_graph_dir"]) / "manifest.json")
    plain = [
        f"本轮实际训练并核验了7组广告模型。当前全部保留方案中的最高随机验证AUC为{best_auc:.6f}，0.85目标仍未达到。最终测试集没有广告点击答案，按你的要求暂不评分。",
        "train_feeds.csv参与了建模：先根据用户点过的资讯、资讯类别和共享实体训练图兴趣表示，再把这个表示交给广告预测模型。资讯点击答案与广告点击答案属于两个任务；资讯训练损失下降不能当作广告AUC提升。",
        f"图中包含{len(graph['vocabularies']['users']):,}个用户、{len(graph['vocabularies']['documents']):,}篇资讯、{len(graph['vocabularies']['categories']):,}个类别和{len(graph['vocabularies']['entities']):,}个共享实体。图预训练仅使用提供的资讯记录，没有使用广告验证标签。",
        f"固定资讯表示和允许广告微调两种图模型的最高AUC为{graph_best:.6f}，普通TabM对照为{by_tag['phase3_graph_control']['auc']:.6f}。本轮没有观察到图兴趣带来的峰值提升。",
        "新的广告历史模块让模型结合当前广告，挑选过去点过或关闭过的广告、广告主和应用记录；另一个版本还学习原表每个位置的作用。并不强行把关闭行为解释成负偏好，也不假定列表是按时间排序的。",
        f"位置版本相对同一模块的平均处理对照，AUC变化为{history_delta:+.6f}。这是同一份验证集上的观察，不能把很小的差异直接解释为稳定收益。",
        "平均、按广告加权、保留位置三组分配相同的参数，初始化时都等价于平均处理，沿用同一训练顺序、学习率和停止条件；这样比较更能反映新信息路径的作用。",
        f"在位置历史模型结构不变的情况下，还增加了一组辅助排序训练：除了判断点没点，还训练点击样本的分数高于未点击样本。该组AUC为{by_tag['phase3_ranking_position']['auc']:.6f}，相对同结构普通训练变化{by_tag['phase3_ranking_position']['auc'] - by_tag['phase3_history_position']['auc']:+.6f}。它属于训练目标对照，不是另一种新骨干。",
        f"按训练前声明的固定权重，将原最佳组合与位置模型各取50%概率，AUC为{blend_metrics['auc']:.6f}；原最佳组合为{previous['auc']:.6f}。没有拿验证标签拟合融合权重。",
        "本轮训练样本为2,021,056条，正样本17,083条；验证为505,264条，正样本4,271条。广告点击非常稀少，大量编号能得到的正反馈很少；复杂结构仍可能先记住训练里的偶然组合。",
        "同一模型后续训练损失下降、验证AUC下降，表示它更会解释已经见过的数据，却更不会排列没参与训练的样本。每组均保存验证AUC最高的一轮；资讯图表示固定后，后续掉分可能减缓，但峰值未提高。",
        "所有新模型均重新加载权重，对全部验证样本按原CUDA混合精度推理，核验保存概率与指标。源图还从原始资讯CSV独立重建，并检查负例排除了用户已经点过的资讯。完整40项行为测试通过，包括空历史、位置区分、梯度、训练/验证隔离和模型重载。",
        "这些结果属于反复使用的80%/20%随机验证；其中约53.83%的验证行在训练中有相同原始特征。资讯画像和图使用任务提供的截至11月16日的离线反馈快照，不能据此宣称在线曝光前可用，也不能代替未来日期测试。",
        "下一步优先测试稀疏广告/素材编号的共享元信息表示与可收缩残差，以及仅由广告训练点击构建的协同结构。达到0.85后仍需另一划分或种子确认。"
    ]
    font = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(), "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    labels = [NAMES[row["experiment"]] for row in records] + ["原最佳DCN＋TabM32", "原最佳＋位置历史（固定融合）"]
    scores = [row["auc"] for row in records] + [previous["auc"], blend_metrics["auc"]]
    axes[0].barh(np.arange(len(scores)), np.array(scores) - .7, left=.7,
                 color=["#54858c"] * 3 + ["#7c71a8"] * 3 + ["#579869", "#bd8540", "#ad6251"])
    axes[0].set_yticks(np.arange(len(scores)), labels)
    axes[0].invert_yaxis()
    axes[0].set(title="同一随机验证集上的实测", xlabel="验证AUC（横轴从0.70开始）", xlim=(.7, .86))
    axes[0].axvline(.85, color="#a34439", linestyle="--", label="目标0.85")
    for index, value in enumerate(scores):
        axes[0].text(value + .002, index, f"{value:.4f}", va="center", fontsize=9)
    for row in records:
        training = pd.read_csv(Path(row["results_dir"]) / "training_history.csv")
        axes[1].plot(training.epoch, training.auc, marker="o", label=NAMES[row["experiment"]])
    axes[1].set(title="每轮验证AUC：保存峰值轮次", xlabel="训练轮次", ylabel="验证AUC")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=.18)
    fig.suptitle(f"第三阶段：源域图兴趣与广告历史建模；当前最高 {best_auc:.4f}", fontsize=17)
    figure = HERE / "phase3_图兴趣与广告历史.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    encoded = base64.b64encode(figure.read_bytes()).decode("ascii")
    notes = "".join("<li>" + html.escape(text) + "</li>" for text in plain)
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第三阶段：图兴趣与广告历史实测</title>
<style>body{{margin:0;background:#f4f6f8;color:#263c4a;font:15px/1.9 "Microsoft YaHei",sans-serif}}main{{max-width:1280px;margin:auto;padding:28px}}section{{background:white;border:1px solid #dce5e9;border-radius:12px;padding:24px;margin:20px 0;overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #e3e9ee}}th{{color:#1c6f75}}li{{margin:12px 0}}img{{width:100%;height:auto}}.lead{{font-size:20px;color:#1c6f75}}</style>
<main><h1>图兴趣与广告历史：第三阶段实测</h1><p class="lead">当前最高验证AUC：{best_auc:.6f}；目标0.85尚未达到。最终测试评分暂缓。</p>
<section><h2>7个已训练、已重放核验的模型</h2>{frame.to_html(index=False,border=0)}<p>图兴趣、历史处理分别有匹配对照；排序目标与位置历史普通训练比较。所有模型使用同一批广告训练/验证索引。</p></section>
<section><h2>图表：有没有提高，后面有没有掉分</h2><img src="data:image/png;base64,{encoded}" alt="模型AUC比较和训练轮次曲线"></section>
<section><h2>先看数据：点击历史有没有信号</h2>{history_table.to_html(index=False,border=0)}<p>表中仅使用训练样本计算。差异表示相关性，不能直接证明因果关系。</p></section>
<section><h2>用大白话理解这轮结果</h2><ul>{notes}</ul></section>
<section><h2>可复查文件</h2><p>phase3_graph_source_audit.json：原始资讯图重建与表示检查。phase3_graph_full_verification.json、phase3_history_full_verification.json、phase3_ranking_full_verification.json：全部验证样本重放。phase3_ad_history_protocol.json、phase3_ranking_protocol.json：训练前固定的比较设置。phase3_architecture_comparison.csv：本轮模型成绩。</p></section></main></html>'''
    (HERE / "先进架构优化_第三阶段报告.html").write_text(document, encoding="utf-8")
    markdown = "# 第三阶段：图兴趣与广告历史\n\n" + "\n\n".join(plain[:2]) + "\n\n" + frame.to_string(index=False) + "\n\n" + "\n\n".join(plain[2:])
    (HERE / "先进架构优化_第三阶段报告.md").write_text(markdown, encoding="utf-8")
    state = load(HERE / "optimization_state.json")
    state.update(phase="phase3_graph_and_history_completed_target_not_reached", goal_achieved=False,
                 best_random_validation_auc=best_auc, phase3_verified_models=len(records),
                 phase3_behavior_tests_total=40,
                 phase3_fixed_blend=blend_record, phase3_graph_source_audit="phase3_graph_source_audit.json",
                 phase3_best_new_model={key: best_new[key] for key in ["experiment", "auc", "results_dir"]},
                 live_tool_sessions={}, last_goal_turn_classification="progress",
                 next_actions=["Implement metadata priors and frequency-shrunk high-cardinality item residuals on the same modern backbone",
                               "Consider ad-training-click-only collaborative graphs with self-edge exclusion for training predictions",
                               "Replay any winner and confirm AUC above 0.85 on another split or seed before completion"])
    state.pop("phase3_new_behavior_tests_passed", None)
    state.pop("phase3_followup", None)
    if best_auc == previous["auc"]:
        state["best_random_results_dir"] = str((HERE / "results_architecture_ensembles").resolve())
        best_definition = previous
    elif best_auc == blend_metrics["auc"]:
        state["best_random_results_dir"] = str(HERE)
        state["best_ensemble_definition"] = blend_record
        best_definition = {**blend_record, "prediction_file": str(HERE / "phase3_position_fixed_blend_validation_prediction.npy"),
                           "previous_ensemble": load(HERE / "architecture_ensemble_protocol.json"),
                           "position_model_dir": by_tag["phase3_history_position"]["results_dir"]}
    else:
        state["best_random_results_dir"] = best_new["results_dir"]
        best_definition = {key: best_new[key] for key in ["experiment", "auc", "results_dir", "model_class", "model_options"]}
    state["best_single_modern_validation_auc"] = max(state.get("best_single_modern_validation_auc", 0.), best_new["auc"])
    (HERE / "phase3_best_model_selection.json").write_text(json.dumps(best_definition, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "optimization_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    plan = load(HERE / "phase3_graph_plan.json")
    plan.update(status="implemented_trained_and_full_validation_replayed", current_goal_achieved=False,
                graph_results={tag: by_tag[tag]["auc"] for tag in NAMES if tag.startswith("phase3_graph")},
                conclusion="Source graph transfers source-trained preferences, but peak held-out ad AUC did not exceed the matched graph-free control")
    (HERE / "phase3_graph_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_retained_auc": best_auc, "best_new_model_auc": best_new["auc"],
                      "fixed_position_blend_auc": blend_metrics["auc"], "history_position_delta": history_delta,
                      "goal_achieved": False, "report": str(HERE / "先进架构优化_第三阶段报告.html")}), flush=True)


if __name__ == "__main__":
    main()
