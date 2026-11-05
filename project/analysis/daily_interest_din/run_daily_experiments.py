"""Run matched daily-interest models sequentially and freeze the development choice."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"user_personalization"))
from evaluation import paired_user_bootstrap


def read(path):return json.loads(path.read_text(encoding="utf-8"))


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def passes(candidate,control):
    return candidate["auc"]-control["auc"]>=.0005 and candidate["logloss"]-control["logloss"]<=.0001


def main(args):
    protocol=read(HERE/"protocol.json");records=[];started=time.perf_counter()
    def run(day,arm,seed=42):
        folder=HERE/"results"/f"day{str(day)[-2:]}"/arm/f"seed{seed}";path=folder/"metrics.json"
        cache=HERE.parent/"user_personalization/cache_day15_stats" if day==20191115 else HERE.parent/"auc085/cache_rich_stats_time16"
        features=HERE/f"cache_day{str(day)[-2:]}"
        if not path.exists():
            subprocess.run([sys.executable,str(HERE/"train_daily.py"),"--day",str(day),"--arm",arm,
                            "--cache",str(cache),"--features",str(features),"--seed",str(seed)],check=True)
        record=read(path)
        if record.get("status")!="complete_and_verified":raise ValueError("Incomplete experiment")
        if record["day"]!=day or record["arm"]!=arm or record["seed"]!=seed:raise ValueError("Run identity mismatch")
        provenance=read(folder/"provenance.json")
        if provenance["protocol"]!=protocol:raise ValueError("Cannot mix frozen protocols")
        from train_daily import digest
        for name,value in provenance["source_sha256"].items():
            if digest(Path(name))!=value:raise ValueError("Run source changed: "+name)
        if digest(folder/"model.pt")!=record["model_sha256"]:raise ValueError("Checkpoint changed")
        paired=[r for r in records if r["day"]==day and r["seed"]==seed]
        if any(r["initial_backbone_sha256"]!=record["initial_backbone_sha256"] for r in paired):raise ValueError("Unmatched backbone initialization")
        records.append(record);write(HERE/"progress.json",{"status":"running","completed":len(records),"last":[day,arm,seed]})
        return record
    development={arm:run(20191115,arm) for arm in protocol["arms"]}
    winner=max([a for a in development if a!="control"],key=lambda a:development[a]["auc"])
    selected={"candidate":winner,"frozen_before_day16":True,"day15_passed":passes(development[winner],development["control"])}
    write(HERE/"selection_frozen.json",selected)
    control16=run(20191116,"control");candidate16=run(20191116,winner)
    selected["day16_passed"]=passes(candidate16,control16)
    passed=selected["day15_passed"] and selected["day16_passed"]
    if passed:
        for day in [20191115,20191116]:
            run(day,"control",2026);run(day,winner,2026)
        for day in [20191115,20191116]:
            pair={r["arm"]:r for r in records if r["day"]==day and r["seed"]==2026}
            passed=passed and passes(pair[winner],pair["control"])
    selected["recommendation"]=winner if passed else "control; new daily interest not confirmed"
    comparisons=[]
    for day in [20191115,20191116]:
        for seed in [42,2026]:
            pair={r["arm"]:r for r in records if r["day"]==day and r["seed"]==seed}
            if "control" not in pair or winner not in pair:continue
            cache=Path(pair["control"]["cache"]);valid=np.load(cache/"valid_indices.npy");labels=np.load(cache/"label.npy",mmap_mode="r")[valid]
            manifest=read(cache/"manifest.json");users=np.load(cache/"categories.npy",mmap_mode="r")[valid,manifest["categorical_fields"].index("user_id")]
            folder=HERE/"results"/f"day{str(day)[-2:]}"
            a=np.load(folder/"control"/f"seed{seed}"/"validation_prediction.npy");b=np.load(folder/winner/f"seed{seed}"/"validation_prediction.npy")
            comparisons.append({"day":day,"seed":seed,**paired_user_bootstrap(labels,a,b,users,draws=args.bootstrap_draws)})
    result={"status":"complete_and_verified","selection":selected,"runs":records,"comparisons":comparisons,
            "protocol":protocol,"elapsed_seconds":time.perf_counter()-started,"final_labels_available":False,
            "goal_final_auc_above085_achieved":False}
    write(HERE/"results.json",result)
    table=[]
    for r in records:
        table.append({"day":r["day"],"arm":r["arm"],"seed":r["seed"],"auc":r["auc"],"logloss":r["logloss"],
            "within_user_auc":r["diagnostics"]["within_user"]["pair_weighted_auc"],
            "new_task_auc":r["diagnostics"]["slices"]["new_task"]["auc"],
            "zero_click_user_logloss":r["diagnostics"]["slices"]["zero_train_clicks"]["logloss"],
            "best_fractional_epoch":r["best_fractional_epoch"],"reload_error":r["maximum_reload_error"]})
    pd.DataFrame(table).to_csv(HERE/"comparison.csv",index=False,encoding="utf-8-sig")
    lines=["# 当天兴趣与DIN跨天实验","", "目标是预测新一天广告；本轮不以随机划分AUC作为达标依据。", "",
           "所有组保留27类别、12个短列表及39个既有数值，后36广告标签统计统一替换为每行严格此前日期的曝光/平滑倾向。源域同日点击和阅读是作业已提供输入；没有把未知当日广告标签放入输入。", "",
           "新增完整主题分布按过去3自然日/当天、曝光/点击/阅读深度/明确负反馈分开。四组共享BoundedIDDCN；control无新增内容，mean加分布均值，din按候选广告加权两个日期窗口，today_din只使用新分支当天分布。前39既有输入和原列表仍含当天画像，因此today_din不是仅用当天所有信息、control也不是完全没有当天信息。", "",
           "DIN分支为分别编码新闻主题与广告属性后学习跨域关联的适配模型，并非原论文完整复现。日内没有位置编码；注意力采用2*sigmoid幅度门，不做强制和为1的softmax。不同反馈各自归一化，并额外提供支持数量。", "",
           "|日期|种子|方案|AUC|Logloss|同用户配对AUC|新任务AUC|", "|---|---|---|---|---|---|---|"]
    for r in table:lines.append(f"|{r['day']}|{r['seed']}|{r['arm']}|{r['auc']:.8f}|{r['logloss']:.8f}|{r['within_user_auc']:.8f}|{r['new_task_auc']:.8f}|")
    lines += ["", "冻结候选："+winner+"；结论："+selected["recommendation"]+"。", "",
              "同用户AUC仅在验证日同时有正负标签的用户中定义，不能直接与整体AUC相减。两日期曾用于反复开发，用户bootstrap属于探索性不确定性。", "",
              "模型均在完整验证集合重载检查；最大误差见comparison.csv。统计日级边界和源域无序不变性由合成测试检查。", "",
              "本轮没有访问17日隐藏标签，不能确认17日AUC超过0.85；原有提交不替换。详细曲线、画像分布和模型哈希见各run，完整比较及区间见results.json。"]
    (HERE/"当天兴趣DIN实验报告.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write(HERE/"progress.json",{"status":"complete_and_verified","completed":len(records),"selection":selected})
    print("ALL_DAILY_COMPLETE "+json.dumps(selected),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--bootstrap-draws",type=int,default=200);main(parser.parse_args())
