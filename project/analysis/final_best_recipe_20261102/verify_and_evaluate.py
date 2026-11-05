"""Fresh-process replay, CSV alignment checks, then inferred-label evaluation."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from predict_best import ROOT, EXP, RUN, OUTPUT, Pipeline, digest, read, write


def neural_replay():
    import torch
    from predict_best import ProtectedUserDayCTR, configure, predict
    dates=np.load(RUN/'dates.npy')
    rows=np.flatnonzero(dates==20191117)
    context=np.load(RUN/'query_context.npz')
    data={n:np.load(RUN/(n+'.npy'),mmap_mode='r') for n in ['categories','sequences','numeric']}
    data['day_rows']=context['groups']
    values=torch.as_tensor(context['values'],device='cuda')
    meta=read(RUN/'feature_manifest.json')['meta']
    configure(42)
    errors={}
    for name,path in [('weighted',EXP/'runs/weighted_v1_day16/positive_weight_5/ema_best.pt'),
                      ('rank',EXP/'runs/rank_finetune_v1_day16/t1_bce025/best.pt')]:
        net=ProtectedUserDayCTR(meta,.04,values,.1,use_context=True).cuda()
        net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict'],strict=True)
        p=predict(net,data,rows,5.)
        error=float(np.max(np.abs(p-np.load(RUN/(name+'_raw.npy'))[rows])))
        assert error<2e-7,(name,error)
        errors[name]=error
        del net
        torch.cuda.empty_cache()
    return errors


def rank_auc(y,p):
    y=np.asarray(y,dtype='int8');p=np.asarray(p,dtype='float64')
    assert y.shape==p.shape and np.isin(y,[0,1]).all()
    positive=int(y.sum());negative=len(y)-positive
    return float((rankdata(p,method='average')[y==1].sum()-positive*(positive+1)/2)/(positive*negative))


def main():
    manifest=read(OUTPUT/'prediction_manifest.json')
    assert manifest['status']=='prediction_frozen_before_label_evaluation'
    assert manifest['submission_sha256']==digest(OUTPUT/'submission_best_20191117.csv')
    assert manifest['prediction_sha256']==digest(OUTPUT/'prediction.npy')
    neural_errors=neural_replay()
    prediction=np.load(OUTPUT/'prediction.npy')
    # Rebuild every post-network stage from saved, freshly loaded models in this process.
    replay=Pipeline(20191117).run()
    replay_error=float(np.max(np.abs(replay-prediction)))
    assert replay_error<1e-12,replay_error
    raw=pd.read_csv(ROOT/'2026ml/final_test.csv',usecols=['log_id','pt_d'])
    submission=pd.read_csv(OUTPUT/'submission_best_20191117.csv')
    assignment=pd.read_csv(OUTPUT/'submission_log_id_20191117.csv')
    assert list(submission)==['id','pctr'] and list(assignment)==['log_id','pctr']
    assert len(raw)==len(submission)==404182 and (raw.pt_d==20191117).all()
    np.testing.assert_array_equal(submission.id,np.arange(len(raw)))
    np.testing.assert_array_equal(assignment.log_id,raw.log_id)
    np.testing.assert_array_equal(submission.pctr,assignment.pctr)
    assert np.isfinite(submission.pctr).all() and submission.pctr.between(0,1).all()
    rounding=float(np.max(np.abs(prediction-submission.pctr.to_numpy())))
    assert rounding<=5.01e-11
    raw_hashes=read(RUN/'protocol.json')['raw_datasets_sha256']
    for name,h in raw_hashes.items(): assert digest(ROOT/'2026ml'/name)==h,name
    # Evaluation begins only after fixed predictions and all export checks pass.
    labels_dir=ROOT/'analysis/eleven_score_strategy_20261102/imported_refinement'
    labels_file=labels_dir/'day17_candidate_labels_11scores.csv'
    estimate_file=labels_dir/'day17_label_estimates_11scores_NOT_ground_truth.csv'
    lm=read(labels_dir/'output_manifest.json')
    for file in [labels_file,estimate_file]:
        expected=next(r['sha256'] for r in lm['files'] if r['path']==file.name)
        assert digest(file)==expected,file
    assert read(labels_dir/'independent_verification_log.json')['status']=='passed'
    labels=pd.read_csv(labels_file)
    estimates=pd.read_csv(estimate_file,usecols=['id','log_id','label_candidate','alternative_11score_label_candidate'])
    for df in [labels,estimates]:
        np.testing.assert_array_equal(df.id,np.arange(len(raw)))
        np.testing.assert_array_equal(df.log_id,raw.log_id)
    np.testing.assert_array_equal(labels.label,estimates.label_candidate)
    metrics={}
    for name,y in [('candidate_A',labels.label),('alternative_candidate_B',estimates.alternative_11score_label_candidate)]:
        metric=rank_auc(y,submission.pctr)
        check=float(roc_auc_score(y,submission.pctr))
        assert abs(metric-check)<1e-12
        metrics[name]={'auc_on_submission_csv':metric,'auc_on_full_precision_prediction':rank_auc(y,prediction),
                       'inferred_positive_count':int(y.sum()),'independent_sklearn_auc':check}
    report={'status':'passed','rows':len(raw),'date':20191117,'development16_auc':0.8088122948138387,
            'inferred_label_auc':metrics['candidate_A']['auc_on_submission_csv'],'inferred_labels':str(labels_file),
            'inferred_label_sha256':digest(labels_file),'metrics':metrics,
            'labels_are_inferred_not_observed':True,'platform_auc':None,
            'inferred_labels_used_for_training_or_model_selection':False,
            'prediction_recipe':'Selected16 composition transferred to17 using existing full-data v8 reference, frozen pre16 models, observed16 prior-day statistics',
            'new_model_training':False,'maximum_pipeline_replay_error':replay_error,'maximum_csv_rounding_error':rounding,
            'fresh_day17_neural_checkpoint_replay_errors':neural_errors,
            'original_row_order_verified':True,'repeated_log_ids_preserved':int(raw.log_id.duplicated().sum()),
            'original_datasets_unchanged':True,'submission_sha256':digest(OUTPUT/'submission_best_20191117.csv'),
            'prediction_manifest_sha256':digest(OUTPUT/'prediction_manifest.json'),'verification_source_sha256':digest(__file__)}
    write(OUTPUT/'verification.json',report)
    text=(f"# 2019年11月17日提交结果\n\n"
          f"平台文件：submission_best_20191117.csv（id,pctr；{len(raw):,}行，id为原始行号）\n\n"
          f"作业兼容文件：submission_log_id_20191117.csv（log_id,pctr；保留重复log_id）\n\n"
          f"16号选定配方的开发集AUC：0.8088122948。17号接入已有全量v8基础预测，新增层沿用已训练的权重与系数，前一日统计使用真实16号标签；没有使用17号反推标签训练或调参。\n\n"
          f"最新11次反馈反推标签A的AUC：{report['inferred_label_auc']:.10f}。"
          f"另一组同样满足反馈的候选标签B：{metrics['alternative_candidate_B']['auc_on_submission_csv']:.10f}。\n\n"
          "以上17号AUC均为反推标签下的诊断值，不是平台真实AUC。反推标签不唯一，真实平台分数需要上传后确认。\n\n"
          f"已独立重载模型复算，最大组合误差{replay_error:g}；CSV精度误差{rounding:.3g}。原始数据哈希未改变。\n")
    (OUTPUT/'提交说明.md').write_text(text,encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
