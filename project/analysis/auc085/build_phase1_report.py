"""Report progress toward the original final-test target without redefining it."""
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
BENCH = HERE.parent / "model_comparison"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    checks = load(HERE / "phase1_verification.json")
    assert checks["status"] == "passed"
    records = load(HERE / "phase1_verified_results.json")
    lookup = {(r["experiment"], r["model"]): r for r in records}
    definitions = [("原DCN-v2", "random20_original_features", "dcn"),
                   ("编号向量约束", "random20_original_features", "bounded"),
                   ("再加入同日资讯画像", "random20_rich_same_day", "bounded"),
                   ("再加入交叉拟合点击率统计", "random20_rich_oof_statistics", "bounded")]
    random_rows = [{"方案": label, "AUC ↑": f"{lookup[exp, model]['auc']:.6f}",
                    "Logloss ↓": f"{lookup[exp, model]['logloss']:.6f}",
                    "最佳轮次": lookup[exp, model]["best_epoch"]} for label, exp, model in definitions]
    old = load(BENCH / "results_id_norm_caps/20191116/CUA20/seed42/metrics.json")
    temporal_records = [("原编号约束模型", old),
                        ("加入同日资讯画像", lookup["time16_rich_same_day", "bounded"]),
                        ("再加入交叉拟合点击率统计", lookup["time16_rich_oof_statistics", "bounded"])]
    time_rows = [{"方案": label, "AUC ↑": f"{record['auc']:.6f}", "Logloss ↓": f"{record['logloss']:.6f}"}
                 for label, record in temporal_records]
    best_random = max((r for r in records if r["validation_strategy"] == "random20"), key=lambda r: r["auc"])
    best_time = max((r for r in records if r["validation_strategy"] == "time16"), key=lambda r: r["auc"])
    duplicate = checks["random_validation_duplicate_predictor_fraction"]
    notes = [f"随机验证当前最佳AUC={best_random['auc']:.6f}；按日期验证当前最佳={best_time['auc']:.6f}。两者属于不同验证集，不能用差值冒充模型改进。",
             "原有训练代码每一轮都打乱样本顺序。本轮新增的是把不同日期的广告随机分成80%训练、20%验证，词表和缩放也在随机训练子集上重新拟合。",
             f"随机验证共505,264条，其中{duplicate:.2%}在训练子集中存在完全相同的原始预测特征（含日期，不含标签和log_id）。这是已审计的重复比例，不是样本行索引重叠。",
             "增强资讯版本加入3个类别、5组兴趣列表及19个数值特征，结合用户当天的长期兴趣、点击资讯类别、阅读行为和广告侧短期兴趣匹配。",
             "同日资讯特征依赖离线任务已经提供整日源域数据这一可用性假设。它不是在线广告展示前预估的验证设定；新增源域特征没有读取广告验证标签。",
             "点击率统计覆盖18组单字段和组合，例如广告位、任务、用户×广告位、广告主×广告位等。训练统计采用五折交叉拟合，相同原始特征记录留在同一折；每条训练记录的统计排除该折全部标签。验证统计仅使用广告训练子集标签。",
             "统计特征提高了本轮排序AUC，但16日Logloss比资讯画像版本更差，后续训练也仍然退步。AUC提高不等于概率校准和训练稳定性都已改善。",
             "目前本轮仅运行种子42，新增特征尚未完成另一天及其他种子的复核；此前原编号约束模型的两日期、两种子复核仍保留。",
             "原目标仍是final_test的真实AUC超过0.85。用户已暂缓最终评分；该目标尚未达到或验证，不能用随机验证结果替代最终成绩。"]
    font = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(), "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained")
    values = [[lookup[exp, model]["auc"] for _, exp, model in definitions], [r["auc"] for _, r in temporal_records]]
    labels = [[label for label, _, _ in definitions], [label for label, _ in temporal_records]]
    for ax, data, names, title in zip(axes, values, labels, ["随机20%验证：同一划分内比较", "2019-11-16验证：同一日期内比较"]):
        positions = np.arange(len(data))
        ax.barh(positions, np.array(data) - .5, left=.5, color=["#8a9eac"] * (len(data) - 1) + ["#147e80"])
        ax.set_yticks(positions, names)
        ax.invert_yaxis()
        ax.set(xlim=(.5, .87), xlabel="验证AUC（图轴从随机水平0.5起）", title=title)
        ax.axvline(.85, color="#b24b3f", linestyle="--", linewidth=1.2)
        for y, value in zip(positions, data):
            ax.text(value + .004, y, f"{value:.4f}", va="center")
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("继续向0.85优化：当前尚未达到目标", fontsize=18)
    figure = HERE / "phase1_AUC对比.png"
    fig.savefig(figure, dpi=170)
    plt.close(fig)
    image = base64.b64encode(figure.read_bytes()).decode("ascii")
    random_table, time_table = pd.DataFrame(random_rows), pd.DataFrame(time_rows)
    random_table.to_csv(HERE / "phase1_random_comparison.csv", index=False, encoding="utf-8-sig")
    time_table.to_csv(HERE / "phase1_temporal_comparison.csv", index=False, encoding="utf-8-sig")
    bullets = "".join("<li>" + html.escape(text) + "</li>" for text in notes)
    headline = f"随机验证最高{best_random['auc']:.4f}；16日验证最高{best_time['auc']:.4f}。0.85目标尚未达到。"
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>向0.85优化：第一阶段实测</title>
<style>body{{margin:0;background:#f3f6f8;color:#213547;font:15px/1.9 "Microsoft YaHei",sans-serif}}main{{max-width:1200px;margin:auto;padding:28px}}section{{background:white;border:1px solid #dce5e9;border-radius:12px;padding:24px;margin:22px 0;overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}td,th{{padding:10px;border-bottom:1px solid #e1e8ed;text-align:left}}th{{color:#147e80}}li{{margin:10px 0}}img{{width:100%}}.lead{{font-size:20px;color:#147e80}}</style>
<main><h1>向0.85优化：第一阶段实测</h1><p class="lead">{headline}</p><section><h2>随机划分：80%训练，20%验证</h2>{random_table.to_html(index=False,border=0)}</section>
<section><h2>保留16日验证作对照</h2>{time_table.to_html(index=False,border=0)}</section><section><img src="data:image/png;base64,{image}" alt="当前验证成绩与目标差距"></section>
<section><h2>大白话说明与核验范围</h2><ul>{bullets}</ul></section><section><h2>后续继续方向</h2><p>尝试广告与资讯的直接联合训练，共享用户和兴趣表示；再对照排序目标与负样本策略，并复验其他日期、种子。所有新方案继续与同一划分的对照比较。</p></section>
<p>本轮六个检查点的预测已复算、权重已加载复核；目标统计还按原始训练标签抽样重建，见phase1_verification.json。原始四个CSV与此前模型保留。</p></main></html>'''
    (HERE / "向085优化_第一阶段报告.html").write_text(document, encoding="utf-8")
    (HERE / "向085优化_第一阶段报告.md").write_text("# 向0.85优化：第一阶段\n\n" + headline + "\n\n" + random_table.to_string(index=False) + "\n\n" + time_table.to_string(index=False) + "\n\n" + "\n".join("- " + text for text in notes), encoding="utf-8")
    state = {"objective": "Train a model whose real final_test AUC exceeds 0.85", "goal_achieved": False,
             "goal_status": "active", "final_scoring": "temporarily_deferred_by_user; continue optimization",
             "last_goal_turn_classification": "progress", "phase": "phase1_validation_comparison_and_source_features_completed",
             "best_random_validation_auc": best_random["auc"], "best_temporal_validation_auc": best_time["auc"],
             "best_random_results_dir": best_random["results_dir"], "best_temporal_results_dir": best_time["results_dir"],
             "verified_models": len(records), "verification": str(HERE / "phase1_verification.json"),
             "missing_completion_evidence": "No real final-test AUC; no offline AUC above 0.85 yet",
             "next_actions": ["Try direct source/ad multi-task learning with shared user and news-interest embeddings; exclude pro and cillabel from source-click prediction inputs",
                              "Compare pairwise ranking loss or controlled negative sampling against identical data/split controls",
                              "Recheck new candidates on another date and seed; do not equate random-holdout AUC with final-test AUC",
                              "Prepare all-data training and result.csv with original log_id order and six-decimal probabilities when the selected model is ready"]}
    (HERE / "optimization_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(headline)
    print(HERE / "向085优化_第一阶段报告.html")


if __name__ == "__main__":
    main()
