"""Report replay-verified collaborative controls and fixed architecture blends."""
import base64
import hashlib
import html
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


def name(row):
    mode = row["model_options"]["memory_mode"]
    if mode == "none":
        return "不使用邻居信息（对照）"
    if mode == "degrees":
        return "只看邻居数量"
    if mode == "mean":
        return "邻居兴趣平均汇总"
    return f"邻居注意力（最多{row['model_options']['max_neighbors']}个）"


def main():
    verification = load(HERE / "phase5_full_verification.json")
    assert verification["status"] == "passed" and verification["completed_runs_verified"] == 5
    records = load(HERE / "phase5_full_verified_results.json")
    assert all(row["replay_max_error"] == 0. for row in records)
    tests = load(HERE / "phase5_test_verification.json")
    assert tests["status"] == "passed" and tests["tests_run"] == 64
    for filename, expected in tests["source_sha256"].items():
        assert hashlib.sha256((HERE / filename).read_bytes()).hexdigest() == expected
    by_mode = {(row["model_options"]["memory_mode"], row["model_options"]["max_neighbors"]): row for row in records}
    control, degree, mean = [by_mode[(mode, 16)] for mode in ["none", "degrees", "mean"]]
    attention = by_mode[("neighbors", 16)]
    cache = Path(control["cache_dir"])
    valid = np.load(cache / "valid_indices.npy")
    target = np.load(cache / "label.npy", mmap_mode="r")[valid]
    for row in records:
        np.testing.assert_array_equal(valid, np.load(Path(row["cache_dir"]) / "valid_indices.npy"))
        saved = np.load(Path(row["results_dir"]) / "validation_prediction.npy")
        assert abs(metrics(target, saved)["auc"] - row["auc"]) < 1e-10
    previous = next(row for row in load(HERE / "results_architecture_ensembles/metrics.json") if row["name"] == "dcn_tabm32")
    previous_prediction = np.load(HERE / "results_architecture_ensembles/dcn_tabm32_validation_prediction.npy")
    assert abs(metrics(target, previous_prediction)["auc"] - previous["auc"]) < 1e-10
    ensembles = load(HERE / "results_collaborative_ensembles/metrics.json")
    assert len(ensembles) == 8
    dcn_directory = HERE / "results/random20_rich_oof_statistics/bounded/seed42"
    dcn_prediction = np.load(dcn_directory / "validation_prediction.npy").astype(np.float64)
    for row in ensembles:
        model_prediction = np.load(Path(row["model_dir"]) / "validation_prediction.npy").astype(np.float64)
        baseline = dcn_prediction if row["combination"] == "replace_tabm32" else previous_prediction
        expected = .5 * baseline + .5 * model_prediction
        saved = np.load(row["prediction_file"])
        np.testing.assert_array_equal(expected, saved)
        assert abs(metrics(target, saved)["auc"] - row["auc"]) < 1e-10
    best_single = max(records, key=lambda row: row["auc"])
    best_blend = max(ensembles, key=lambda row: row["auc"])
    candidates = [(previous["auc"], "previous", previous), (best_single["auc"], "single", best_single), (best_blend["auc"], "blend", best_blend)]
    best_auc, best_kind, best_record = max(candidates, key=lambda item: item[0])
    table = pd.DataFrame([{"处理方式": name(row), "单模型验证AUC": f"{row['auc']:.6f}",
                          "相对无邻居对照": f"{row['auc'] - control['auc']:+.6f}", "最佳训练轮次": row["best_epoch"],
                          "新增可训练参数": f"{row['trainable_parameter_count']:,}"} for row in records])
    table.to_csv(HERE / "phase5_architecture_comparison.csv", index=False, encoding="utf-8-sig")
    lookup = {row["results_dir"]: row for row in records}
    blend_table = pd.DataFrame([{"固定组合": ("DCN＋" if row["combination"] == "replace_tabm32" else "旧组合＋") + name(lookup[row["model_dir"]]),
                                "验证AUC": f"{row['auc']:.6f}", "相对旧组合": f"{row['auc'] - previous['auc']:+.6f}"} for row in ensembles])
    blend_table.to_csv(HERE / "phase5_ensemble_comparison.csv", index=False, encoding="utf-8-sig")
    source_initial = load(Path(attention["results_dir"]) / "collaborative_initialization.json")
    memories = [load(HERE / (f"collaborative_memory_random20_k{k}/audit.json")) for k in [16, 64]]
    assert all(row["status"] == "passed" for row in memories)
    gain = best_auc - previous["auc"]
    slices = load(HERE / "phase5_uplift_slices.json")
    assert slices["best_fixed_blend"] == best_blend["name"]
    pair_changes = slices["pair_decomposition"]
    slice_table = pd.DataFrame(slices["slices"])
    for column in ["旧组合AUC", "新组合AUC", "AUC变化"]:
        slice_table[column] = slice_table[column].map(lambda value: "—" if pd.isna(value) else f"{value:.6f}")
    notes = [
        f"本轮最高随机验证AUC为{best_auc:.6f}，旧组合为{previous['auc']:.6f}，变化{gain:+.6f}。0.85目标尚未达到，最终测试评分仍暂缓。",
        "新结构增加两条判断路径：当前用户像不像其他点过这条广告的人；当前广告像不像这个用户点过的其他广告。用户相似性先借助资讯点击训练得到的兴趣表示，再由广告训练数据学习如何用于当前任务。",
        "原TabM32模型的参数保持原值，新分支只训练65,577个参数，再根据当前样本学习给原有分数增加多少修正。全部模型保留训练前的第0轮候选，因此继续训练没有改善时会保留原模型。",
        f"没有邻居的对照为{control['auc']:.6f}，16邻居注意力为{attention['auc']:.6f}，相对变化{attention['auc'] - control['auc']:+.6f}。两组具有相同参数量、初始化、训练顺序和原模型，比较的是新加入的协同信息。",
        f"只看调整后的邻居数量为{degree['auc']:.6f}，邻居兴趣平均汇总为{mean['auc']:.6f}。16邻居注意力相对这两组分别变化{attention['auc'] - degree['auc']:+.6f}和{attention['auc'] - mean['auc']:+.6f}；收益来源应结合这些对照判断，不能把所有提升都归因于注意力。",
        "扩大到64个邻居还同时改变了按邻居上限归一化的支持特征，所以这是邻域配置的比较。更大的邻域不能自动视为更好的模型；本报告保留实际结果。",
        "所有协同关系仅由2,021,056条广告训练记录中的点击标签构成，共15,293条不同的用户—广告关系。广告元信息原型只使用训练输入，资讯兴趣来自之前已核验的源域模型。验证广告点击答案没有参与这些输入的拟合。",
        "预测某条记录时，会从两个方向剔除当前用户与当前广告这一对，也减去它在点击次数里的贡献。剔除发生在限制邻居数量之前，避免排在截断边界的当前点击暗中改变其他邻居。",
        "两种邻域都已独立重建并逐一检查16,285个查询，覆盖全部训练正关系、负关系样本和未知编号。把当前标签翻转后，返回的其他邻居和调整后的次数保持不变的检查也已通过。",
        "同一个用户可能既点过又没点过同一广告。图关系提供偏好线索，不等于每次曝光的点击答案；本轮不把关系记忆或训练AUC当成验证成绩。",
        "八种融合都采用训练前声明的50%＋50%概率平均，没有用验证标签训练权重。旧组合指此前的DCN＋原TabM32，AUC为0.773998。所有组合都列在表中，避免只展示最好的一行。",
        "62项原有及新行为检查加入两个消融检查后，共64项测试通过。所有5组保留模型都从原始输入重放全部505,264条验证记录，概率与保存结果完全一致。原主干参数、协同关系和冻结资讯向量也逐项核验。",
        "核验时发现，原始推理和缓存推理的数组排列不同，会使均值的浮点舍入差异被半精度门控放大。已通过可复现的GPU检查定位并统一排列；原有训练缓存已经采用该排列，所以保留了训练权重，只修复推理的一致性。",
        "当前结果仍来自反复使用的80%/20%随机验证，约53.83%的验证行在训练中有相同原始特征。新结果不能直接代表新日期或最终测试表现；达到0.85后仍需新划分或种子确认。资讯反馈沿用离线任务已提供快照的假设。",
        f"进一步按有无其他点击关系分组，组内排序对整体AUC变化的贡献为{pair_changes['within_group_weighted_change']:+.6f}，不同组之间排序的贡献为{pair_changes['cross_group_weighted_change']:+.6f}。两者之和精确还原总体变化；目前主要改善了跨组排序，组内区分能力仍需提高。整体AUC包含跨组的点击/未点击比较，不能简单平均各组内部AUC。",
        "下一步优先比较点击邻居与未点击邻居的差别，让模型判断用户更接近哪类人；所有新关系仍必须来自广告训练标签，并在两个方向排除当前查询及其次数影响。此方向尚未训练或评分。"
    ]
    font = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(), "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
    names = [name(row) for row in records] + ["旧DCN＋TabM32组合", "本轮最高固定组合"]
    scores = [row["auc"] for row in records] + [previous["auc"], best_blend["auc"]]
    axes[0].barh(np.arange(len(scores)), np.array(scores) - .7, left=.7, color=["#517e89"] * len(records) + ["#9b8b72", "#bc7c3f"])
    axes[0].set_yticks(np.arange(len(scores)), names)
    axes[0].invert_yaxis()
    axes[0].set(title="单模型与固定架构组合", xlabel="验证AUC（横轴从0.70开始）", xlim=(.7, .86))
    axes[0].axvline(.85, color="#a6483c", linestyle="--")
    for index, value in enumerate(scores):
        axes[0].text(value + .002, index, f"{value:.4f}", va="center", fontsize=9)
    for row in records:
        history = pd.read_csv(Path(row["results_dir"]) / "training_history.csv")
        axes[1].plot(history.epoch, history.auc, marker="o", markersize=3, label=name(row))
    axes[1].set(title="新增分支训练过程：比较各自峰值", xlabel="累计训练轮次", ylabel="验证AUC")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=.18)
    fig.suptitle(f"第五阶段：协同兴趣建模；最高固定组合AUC {best_auc:.4f}", fontsize=18)
    figure = HERE / "phase5_协同兴趣对比.png"
    fig.savefig(figure, dpi=165)
    plt.close(fig)
    encoded = base64.b64encode(figure.read_bytes()).decode("ascii")
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第五阶段：协同兴趣模型实测</title>
<style>body{{margin:0;background:#f3f6f8;color:#243b49;font:15px/1.9 "Microsoft YaHei",sans-serif}}main{{max-width:1350px;margin:auto;padding:26px}}section{{background:#fff;border:1px solid #dce5e9;border-radius:12px;padding:24px;margin:20px 0;overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:10px;border-bottom:1px solid #e3e9ee;text-align:left}}th{{color:#206f75}}li{{margin:12px 0}}img{{width:100%;height:auto}}.lead{{font-size:21px;color:#176d74}}</style>
<main><h1>协同兴趣模型：第五阶段实测</h1><p class="lead">当前最高验证AUC：{best_auc:.6f}；相对旧组合{gain:+.6f}；目标0.85尚未达到。</p>
<section><h2>先用大白话理解模型</h2><p>原模型根据广告、用户和已有历史给出判断；新分支再看“其他点过这条广告的人”和“这个用户点过的其他广告”，学习是否需要修正原判断。最后与DCN按预先确定的相同比例组合。</p><p>预测当前记录时，会先剔除它自己对应的点击关系和次数贡献，再查其他人的偏好。</p></section>
<section><h2>5组单模型对照</h2>{table.to_html(index=False,border=0)}</section>
<section><h2>8组训练前固定权重的组合</h2>{blend_table.to_html(index=False,border=0)}<p>全部权重为50%＋50%；旧组合为此前的DCN＋原TabM32。</p></section>
<section><h2>成绩与训练曲线</h2><img src="data:image/png;base64,{encoded}" alt="协同模型AUC对比和训练曲线"></section>
<section><h2>分组诊断：哪里提高，哪里还没提高</h2><p>所有分组仍属于同一份随机验证。按日期分组也不等于用过去预测未来。</p><details><summary>展开查看点击关系与日期分组</summary>{slice_table.to_html(index=False,border=0)}</details><p>组内排序贡献：{pair_changes['within_group_weighted_change']:+.6f}；跨组排序贡献：{pair_changes['cross_group_weighted_change']:+.6f}。两者共同构成总体AUC变化。</p></section>
<section><h2>结果、方法与核验说明</h2><ul>{''.join('<li>'+html.escape(text)+'</li>' for text in notes)}</ul></section>
<section><h2>可复查证据</h2><p>phase5_full_verification.json：完整原始输入重放。各协同关系目录的audit.json：训练标签来源、全部正关系查询排除与资讯用户映射。phase5_test_verification.json：64项测试。phase5_numerical_alignment_fix.json：推理排列差异定位与修复。phase5_best_model_selection.json：当前最高模型及各分支预处理位置。</p></section></main></html>'''
    (HERE / "先进架构优化_第五阶段报告.html").write_text(document, encoding="utf-8")
    (HERE / "先进架构优化_第五阶段报告.md").write_text("# 第五阶段：协同兴趣\n\n" + table.to_string(index=False) + "\n\n" + blend_table.to_string(index=False) + "\n\n" + "\n\n".join(notes), encoding="utf-8")
    if best_kind == "blend":
        baseline_directory = str(dcn_directory) if best_record["combination"] == "replace_tabm32" else str(HERE / "results_architecture_ensembles")
        definition = {**best_record, "members": {"baseline": baseline_directory, "collaborative": best_record["model_dir"]},
                      "baseline_kind": "DCN" if best_record["combination"] == "replace_tabm32" else "original DCN/TabM32 equal ensemble",
                      "member_cache_dirs": {"baseline_DCN": load(dcn_directory / "metrics.json")["cache_dir"],
                                            "collaborative": lookup[best_record["model_dir"]]["cache_dir"]}}
    else:
        definition = best_record
    (HERE / "phase5_best_model_selection.json").write_text(json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8")
    state = load(HERE / "optimization_state.json")
    state.update(phase="phase5_collaborative_attention_verified_target_not_reached", goal_status="active", goal_achieved=False,
        last_goal_turn_classification="progress", live_tool_sessions={}, phase5_verified_models=5, phase5_fixed_ensembles=8,
        phase5_behavior_tests_total=64, best_random_validation_auc=best_auc,
        best_single_modern_validation_auc=max(state.get("best_single_modern_validation_auc", 0.), best_single["auc"]),
        phase5_single_results={name(row): row["auc"] for row in records},
        next_actions=["Test contrastive clicked-versus-never-clicked collaborative neighborhoods using training labels only",
                      "Preserve query-edge and degree exclusion for both positive and negative relations",
                      "Compare signed collaborative information with the verified positive-neighbor controls",
                      "Verify any actual AUC>0.85 winner on a fresh split or seed before completion"])
    if best_kind == "blend":
        state.update(best_random_results_dir=str(HERE / "results_collaborative_ensembles"), best_ensemble_definition=definition)
    elif best_kind == "single":
        state["best_random_results_dir"] = best_record["results_dir"]
    (HERE / "optimization_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    plan = load(HERE / "phase5_collaborative_attention_plan.json")
    plan.update(status="implemented_trained_full_raw_validation_verified", current_goal_achieved=False,
                single_model_best_auc=best_single["auc"], fixed_blend_best_auc=best_blend["auc"], verified_models=5)
    (HERE / "phase5_collaborative_attention_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_auc": best_auc, "previous_auc": previous["auc"], "gain": gain,
                      "best_single": best_single["auc"], "attention_minus_degree": attention["auc"] - degree["auc"],
                      "attention_minus_mean": attention["auc"] - mean["auc"], "goal_achieved": False}), flush=True)


if __name__ == "__main__":
    main()
