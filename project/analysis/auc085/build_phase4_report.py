"""Report hierarchical-ID controls and protected strong-backbone adaptation."""
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


MODE_NAMES = {"raw": "直接学习编号", "metadata": "共性信息＋编号", "shrink": "共性信息＋降低少见编号权重", "shared": "当前编号仅用共性信息"}


def main():
    proof = load(HERE / "phase4_full_verification.json")
    assert proof["status"] == "passed"
    records = load(HERE / "phase4_full_verified_results.json")
    assert len(records) >= 5
    assert all(row["replay_status"] == "full_precision_matched_replay_passed" for row in records)
    tests = load(HERE / "phase4_test_verification.json")
    assert tests["status"] == "passed" and tests["tests_run"] == 49
    for name, digest in tests["source_sha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest
    screen = {row["model_options"]["hierarchy_mode"]: row for row in records if row["experiment"].startswith("phase4_hierarchy8_")}
    assert set(screen) == set(MODE_NAMES)
    adapter = next(row for row in records if row["experiment"] == "phase4_protected_adapter")
    initial = load(Path(adapter["results_dir"]) / "protected_initialization.json")
    assert load(Path(adapter["results_dir"]) / "protected_base_audit.json")["status"] == "passed"
    teacher = load(Path(adapter["teacher_dir"]) / "metrics.json")
    cache = Path(adapter["cache_dir"])
    valid = np.load(cache / "valid_indices.npy")
    target = np.load(cache / "label.npy", mmap_mode="r")[valid]
    for row in records:
        np.testing.assert_array_equal(valid, np.load(Path(row["cache_dir"]) / "valid_indices.npy"))
    previous = next(row for row in load(HERE / "results_architecture_ensembles/metrics.json") if row["name"] == "dcn_tabm32")
    previous_prediction = np.load(HERE / "results_architecture_ensembles/dcn_tabm32_validation_prediction.npy")
    assert abs(metrics(target, previous_prediction)["auc"] - previous["auc"]) < 1e-10
    dcn_directory = HERE / "results/random20_rich_oof_statistics/bounded/seed42"
    dcn_record = load(dcn_directory / "metrics.json")
    np.testing.assert_array_equal(valid, np.load(Path(dcn_record["cache_dir"]) / "valid_indices.npy"))
    dcn_prediction = np.load(dcn_directory / "validation_prediction.npy").astype(np.float64)
    adapter_prediction = np.load(Path(adapter["results_dir"]) / "validation_prediction.npy").astype(np.float64)
    combinations = {
        "replace_tabm32_member": (.5 * dcn_prediction + .5 * adapter_prediction, "DCN＋保留主干的改造模型"),
        "combine_with_previous_best": (.5 * previous_prediction + .5 * adapter_prediction, "原最佳＋保留主干的改造模型"),
    }
    ensembles = []
    for name, (prediction, display) in combinations.items():
        scores = metrics(target, prediction)
        np.save(HERE / ("phase4_" + name + "_validation_prediction.npy"), prediction)
        ensembles.append({"name": name, "display_name": display, **scores, "weight_fitting": "none; 0.5/0.5 fixed before adapter training",
                          "validation_rows": len(valid), "prediction_file": str(HERE / ("phase4_" + name + "_validation_prediction.npy"))})
    (HERE / "phase4_fixed_ensemble_metrics.json").write_text(json.dumps(ensembles, indent=2, ensure_ascii=False), encoding="utf-8")
    table = []
    for row in records:
        mode = row["model_options"]["hierarchy_mode"]
        protected = row.get("training_regime") == "protected_backbone_shared_input_adapters"
        reference_auc = teacher["auc"] if protected else screen["raw"]["auc"]
        name = "保留TabM32主干，仅训练共性信息入口" if protected else MODE_NAMES[mode]
        row["display_name"] = name
        history = pd.read_csv(Path(row["results_dir"]) / "training_history.csv")
        table.append({"模型处理方式": name, "成员数": row["ensemble_members"], "最佳验证AUC": f"{row['auc']:.6f}",
                      "相对各自对照": f"{row['auc'] - reference_auc:+.6f}", "最后一次验证AUC": f"{history.iloc[-1].auc:.6f}",
                      "最佳训练轮次": row["best_epoch"]})
    frame = pd.DataFrame(table)
    frame.to_csv(HERE / "phase4_architecture_comparison.csv", index=False, encoding="utf-8-sig")
    ensemble_frame = pd.DataFrame([{"固定等权组合": row["display_name"], "验证AUC": f"{row['auc']:.6f}", "相对原最佳": f"{row['auc'] - previous['auc']:+.6f}"} for row in ensembles])
    ensemble_frame.to_csv(HERE / "phase4_ensemble_comparison.csv", index=False, encoding="utf-8-sig")
    candidates = [{"kind": "previous", "auc": previous["auc"], "name": "dcn_tabm32"}]
    candidates += [{"kind": "model", "auc": row["auc"], "record": row} for row in records]
    candidates += [{"kind": "ensemble", "auc": row["auc"], "record": row} for row in ensembles]
    best = max(candidates, key=lambda row: row["auc"])
    audit = load(HERE / "phase4_support_audit.json")
    collaboration = load(HERE / "phase5_target_collaboration_audit.json")
    adapter_history = pd.read_csv(Path(adapter["results_dir"]) / "training_history.csv")
    sparse = pd.DataFrame([{"编号类型": {"user_id": "用户", "task_id": "广告任务", "adv_id": "广告素材"}[row["field"]],
                           "训练中出现的编号": row["seen_current_ids"], "每个编号曝光次数中位数": f"{row['exposure_quantiles']['q50']:.0f}",
                           "没有点击的编号占比": f"{row['ids_with_no_click_fraction']:.2%}", "只有0～2次点击的编号占比": f"{row['ids_with_at_most_two_clicks_fraction']:.2%}"} for row in audit["fields"]])
    retained = "最佳权重仍是添加入口前的第0轮，说明这次入口训练未带来提升。" if adapter["best_epoch"] == 0 else f"保留主干的模型在累计训练{adapter['best_epoch']}轮时最好。"
    notes = [
        f"本轮核验了{len(records)}组模型，当前所有保留方案的最高随机验证AUC为{best['auc']:.6f}。目标0.85仍未达到，最终测试评分继续暂缓。",
        "新结构把当前用户、广告任务、广告素材的表示拆成两部分：年龄/兴趣/应用/广告主等共性信息，以及该编号自己的小向量。出现次数少的编号可以降低自身向量的权重。",
        "四组8成员模型分配相同参数、使用相同种子、训练顺序和输入；唯一研究变量是当前编号的表示方式。共性信息网络的末层从0开始，直接学习编号与加入共性信息两组初始预测完全一致。",
        "控制编号权重的次数仅来自广告训练输入的曝光记录，没有使用验证输入计数，也没有使用点击标签。上面的点击次数仅用于诊断数据稀疏程度，不会传给模型初始化。PAD和UNK不被当成有证据支持的编号。",
        f"直接学习编号对照的AUC为{screen['raw']['auc']:.6f}；加入共性信息为{screen['metadata']['auc']:.6f}；降低少见编号权重为{screen['shrink']['auc']:.6f}；当前编号只用共性信息为{screen['shared']['auc']:.6f}。这些是同一划分上的观察，不能用一个很小的差异声称稳定提分。",
        "去掉当前三类编号的直接向量，并不等于完全去掉用户身份：原有历史列表、其他字段组合和交叉拟合统计仍能区分部分用户。若该组仍过拟合，只能说明本次改造不足以解决，不能证明所有编号记忆都无害。",
        f"另一组保留原有TabM32强模型，只训练新加入的共性信息入口，共有{initial['trainable_adapter_parameters']:,}个可训练参数。改造前AUC为{teacher['auc']:.6f}，保留的改造模型为{adapter['auc']:.6f}。{retained}",
        f"保留主干的改造在训练前先重放全部验证样本，和原模型概率的最大误差为{initial['initial_teacher_replay_max_error']:.3g}。原有嵌入、数值编码、主干和预测头保持原值，并在保存模型后逐项核验。",
        f"保留主干实验最后一次Logloss为{adapter_history.iloc[-1].logloss:.6f}，原模型为{teacher['logloss']:.6f}；最后一次AUC为{adapter_history.iloc[-1].auc:.6f}。概率误差与排序能力衡量的是不同事情，优化前者不保证后者提高。",
        "8成员从头训练与32成员保留主干属于不同设置，所以各自和匹配对照比较，不能把两者差异全部归因于新结构。初始第0轮也作为候选保存，避免继续训练后反而丢掉原模型。",
        "固定融合各取50%概率，权重在训练前已经声明。报告保留全部组合，未根据验证标签拟合融合权重，也不把未提分的组合替换为主模型。",
        "49项行为测试通过；所有本轮模型从保存权重重新推理全部505,264条验证样本，并核验保存概率、训练输入曝光次数和指标。此前阶段的权重与核验证据均保留。",
        "当前使用的是反复调参的80%/20%随机验证，约53.83%的验证行在训练中有相同原始特征。分数不能代替未来日期效果；达到0.85后还需要另一划分或种子确认。资讯输入仍沿用离线任务提供的反馈快照假设。",
        f"下一步数据已核对：训练标签能构成{collaboration['unique_positive_user_task_edges']:,}条用户—广告点击关系，其中{collaboration['positive_edges_whose_pair_also_has_negative_rows']:,}组也出现过未点击。也就是说，同一个人可能既点过又没点过同一广告，不能把图关系当成点击答案。",
        "后续应改变学到的信息与监督路径，而不只叠加网络层：优先考虑仅由广告训练点击构建的用户—广告协同表示，并在训练预测时排除自身点击边，再比较与资讯兴趣、广告元信息的联合融合。该方向在此报告中仅完成数据可用性审计，尚未训练或取得新分数。"
    ]
    font = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(), "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), layout="constrained", gridspec_kw={"width_ratios": [1.25, 1, 1]})
    names = [MODE_NAMES[mode] for mode in MODE_NAMES] + ["TabM32原模型", "TabM32保留主干改造", "原最佳架构组合"]
    scores = [screen[mode]["auc"] for mode in MODE_NAMES] + [teacher["auc"], adapter["auc"], previous["auc"]]
    axes[0].barh(np.arange(len(scores)), np.array(scores) - .7, left=.7, color=["#5b8790"] * 4 + ["#8c7aa5", "#8c7aa5", "#bd8540"])
    axes[0].set_yticks(np.arange(len(scores)), names)
    axes[0].invert_yaxis()
    axes[0].set(title="最高分与0.85目标的距离", xlabel="验证AUC（横轴从0.70开始）", xlim=(.7, .86))
    axes[0].axvline(.85, color="#aa4b3f", linestyle="--")
    for index, score in enumerate(scores):
        axes[0].text(score + .002, index, f"{score:.4f}", va="center", fontsize=8)
    for mode in MODE_NAMES:
        h = pd.read_csv(Path(screen[mode]["results_dir"]) / "training_history.csv")
        axes[1].plot(h.epoch, h.auc, marker="o", label=MODE_NAMES[mode])
    axes[1].set(title="四组表示方式：后续是否掉分", xlabel="训练轮次", ylabel="验证AUC")
    axes[1].legend(frameon=False, fontsize=7)
    h = pd.read_csv(Path(adapter["results_dir"]) / "training_history.csv")
    axes[2].plot(h.epoch, h.auc, marker="o", color="#8c7aa5", label="只训练共享入口")
    axes[2].axhline(teacher["auc"], color="#667984", linestyle=":", label="训练前的强模型")
    axes[2].set(title="保留强模型：是否改善峰值", xlabel="累计训练轮次", ylabel="验证AUC")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=.18)
    fig.suptitle(f"第四阶段：共性信息、编号残差与保留主干；当前最高 {best['auc']:.4f}", fontsize=17)
    figure = HERE / "phase4_层次编号与保护主干.png"
    fig.savefig(figure, dpi=155)
    plt.close(fig)
    encoded = base64.b64encode(figure.read_bytes()).decode("ascii")
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第四阶段：编号表示与保留主干实测</title>
<style>body{{margin:0;background:#f4f6f8;color:#243a48;font:15px/1.9 "Microsoft YaHei",sans-serif}}main{{max-width:1400px;margin:auto;padding:26px}}section{{background:#fff;border:1px solid #dce5e9;border-radius:12px;padding:24px;margin:20px 0;overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #e3e9ee}}th{{color:#1b6e74}}img{{width:100%;height:auto}}li{{margin:12px 0}}.lead{{font-size:20px;color:#1b6e74}}</style>
<main><h1>编号表示与保留主干：第四阶段实测</h1><p class="lead">当前最高验证AUC：{best['auc']:.6f}；0.85目标尚未达到。最终测试评分暂缓。</p>
<section><h2>模型处理方式与结果</h2>{frame.to_html(index=False,border=0)}<p>8成员组与32成员组分别使用各自对照；第0轮表示尚未训练新入口时的原模型。</p></section>
<section><h2>训练前固定的等权融合</h2>{ensemble_frame.to_html(index=False,border=0)}<p>原最佳架构组合AUC为{previous['auc']:.6f}。</p></section>
<section><h2>模型曲线</h2><img src="data:image/png;base64,{encoded}" alt="编号处理对比、每轮验证AUC与保留主干曲线"></section>
<section><h2>为什么要改编号：先看训练数据</h2>{sparse.to_html(index=False,border=0)}</section>
<section><h2>用大白话理解</h2><ul>{''.join('<li>'+html.escape(text)+'</li>' for text in notes)}</ul></section>
<section><h2>核验文件</h2><p>phase4_full_verification.json：全部验证样本重放。phase4_support_audit.json：训练数据稀疏程度。phase4_test_verification.json：行为测试与代码版本。各模型目录的support_initialization.json及protected_base_audit.json：曝光次数来源与主干参数保护。</p></section></main></html>'''
    (HERE / "先进架构优化_第四阶段报告.html").write_text(document, encoding="utf-8")
    (HERE / "先进架构优化_第四阶段报告.md").write_text("# 第四阶段：编号表示与保留主干\n\n" + frame.to_string(index=False) + "\n\n" + ensemble_frame.to_string(index=False) + "\n\n" + "\n\n".join(notes), encoding="utf-8")
    state = load(HERE / "optimization_state.json")
    state.update(phase="phase4_hierarchical_ids_and_protected_adapters_verified_target_not_reached", goal_achieved=False,
                 goal_status="active", live_tool_sessions={}, last_goal_turn_classification="progress", phase4_verified_models=len(records),
                 phase4_behavior_tests_total=49, phase4_adapter_best_epoch=adapter["best_epoch"],
                 phase5_target_collaboration_audit="phase5_target_collaboration_audit.json",
                 phase4_screen_auc={mode: screen[mode]["auc"] for mode in screen}, phase4_protected_adapter_auc=adapter["auc"],
                 best_random_validation_auc=best["auc"],
                 next_actions=["Test advertisement-training-click collaborative representation with self-edge exclusion for training queries",
                               "Fuse ad collaborative preference, supplied source interests and ad metadata on a modern backbone",
                               "Replay a genuine AUC>0.85 winner and confirm on a fresh split or seed before completion"])
    if best["kind"] == "ensemble":
        state.update(best_random_results_dir=str(HERE), best_ensemble_definition=best["record"])
    elif best["kind"] == "model":
        state["best_random_results_dir"] = best["record"]["results_dir"]
    else:
        state["best_random_results_dir"] = str(HERE / "results_architecture_ensembles")
    state["best_single_modern_validation_auc"] = max(state.get("best_single_modern_validation_auc", 0.), max(row["auc"] for row in records))
    (HERE / "optimization_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "phase4_best_model_selection.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_auc": best["auc"], "screen_auc": state["phase4_screen_auc"], "protected_adapter_auc": adapter["auc"],
                      "adapter_best_epoch": adapter["best_epoch"], "goal_achieved": False}), flush=True)


if __name__ == "__main__":
    main()
