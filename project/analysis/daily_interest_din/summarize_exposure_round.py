"""Summarize genuine future-day improvements, failed candidates and next submission."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    rows=[]
    for kind,pattern in [('tree','tree_results/*/metrics.json'),('dense','dense_results/*/metrics.json')]:
        for path in sorted(HERE.glob(pattern)):
            d=json.loads(path.read_text(encoding='utf-8'))
            rows.append({'kind':kind,'experiment':path.parent.name,'day':d['day'],'auc':d['auc'],'logloss':d['logloss'],
                         'input_fields':d.get('features',d.get('numeric_dim')),'reload_error':d['maximum_reload_error'],
                         'recent_days':d.get('recent_days',0)})
    blends=[]
    for day in [15,16]:
        cache=HERE.parent/'user_personalization/cache_day15_stats' if day==15 else HERE.parent/'auc085/cache_rich_stats_time16'
        valid=np.load(cache/'valid_indices.npy');y=np.load(cache/'label.npy',mmap_mode='r')[valid]
        a=np.load(HERE/f'dense_results/day{day}_dcn_exposure_snapshot/validation_prediction.npy')
        b=np.load(HERE/f'tree_results/day{day}_snapshot_exposure_d6/validation_prediction.npy')
        p=.5*a+.5*b
        blends.append({'day':20191100+day,'weights':{'dcn_exposure':.5,'tree_exposure':.5},'auc':float(roc_auc_score(y,p)),
                       'logloss':float(log_loss(y,p,labels=[0,1]))})
    pd.DataFrame(rows).to_csv(HERE/'跨天曝光实验汇总.csv',index=False,encoding='utf-8-sig')
    output={'status':'development_experiments_complete','experiments':rows,'equal_blends':blends,
            'final_auc_goal':.85,'final_auc_verified':False,'previous_random_model_external_score':.70590,
            'evidence':'Held-out complete15/16 days; target stats strict past, full same-day input-only exposures and source snapshots',
            'selection_bias_note':'Dates repeatedly used for development; many candidate comparisons; not an untouched evaluation'}
    (HERE/'exposure_round_results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 跨天当天曝光与兴趣快照实验','',
       '**17日实际AUC>0.85仍未被验证。本轮在完整跨天开发验证上取得明确进展：曝光DCN与曝光树等权融合，15日0.785863、16日0.791786。**','',
       '## 关键发现','',
       '用户当天资讯兴趣可以提取，但完整主题分布/DIN分支在已有画像主干上只带来约0.0001的小变化，16日不稳定。直接从当天已提供的全部广告输入计算曝光量、重复比例和历史列表变化，比继续加注意力更有效。','',
       '当天曝光和历史快照不使用当天广告标签。训练与验证都使用各日已提供的完整输入；广告点击统计严格早于每行日期。该设定适用于本作业整日输入可用的离线预测，不能当作广告曝光前在线因果评估。','',
       '## 对照结果','', '|日期|方案|AUC|Logloss|','|---|---|---|---|']
    for r in rows:lines.append(f"|{r['day']}|{r['experiment']}|{r['auc']:.8f}|{r['logloss']:.8f}|")
    for r in blends:lines.append(f"|{r['day']}|曝光DCN＋曝光树各50%|{r['auc']:.8f}|{r['logloss']:.8f}|")
    lines+=['','## 采用与不采用','',
       '- 采用作为新候选：严格历史数值＋当天曝光43维＋当天历史快照48维；DCN一轮与深6树约2000棵，等权。','- 近期3天深8树只在15日更好，16日退步，不作为固定最终方案。','- 细粒度曝光、历史曝光变化、历史点击预算、全历史深8树未超过主要开发对照，保留记录但不合并。','- 丰富164维源域聚合在15日只有小增益，尚无配对16日确认，不纳入最终方案。','',
       '## 最终重训与边界','',
       '新候选以全部2526320条有标签广告重新拟合词表、缩放及严格历史统计，训练预算在final推理前冻结。final404182行保持label=-1，不用其结果调参。曝光及历史快照训练部分与既有开发输入逐行一致。','',
       '目标提交位于output/chronological_exposure_v1/submission.csv，列为id,pctr，六位小数；生成和独立核验完成后方可用于评分。没有本地final标签，不能宣布AUC0.85。旧submission与随机模型成品保留。','',
       '26项时间、同日源、无序集合、输入频次、快照和模型测试通过；DIN6组完整模型独立重放误差0。树/曝光DCN训练器也已重载其完整验证集，误差0。全量最终模型的独立预测重放与原始历史统计探针见output内verification.json。','',
       '两天已被反复用于开发，融合也在这两天比较过，表中结果是开发证据，不是未触碰独立测试。后续最有价值的判断是新候选的17日真实评分。']
    (HERE/'跨天曝光优化报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(blends),flush=True)


if __name__=='__main__':main()
