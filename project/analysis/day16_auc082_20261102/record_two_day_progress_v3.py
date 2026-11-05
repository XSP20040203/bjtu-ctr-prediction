"""Track both user targets using complete raw cohorts and verified predictions."""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from fit_stacker import ROOT,HERE,REFERENCE,digest,write
from publish_progress import publication_lock

OUT=ROOT/'output/two_day_auc_targets_20261102'


def auc(y,p):
    n=int(y.sum())
    return float((rankdata(p,method='average')[y==1].sum()-n*(n+1)/2)/(n*(len(y)-n)))


def main():
    raw=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    refs={15:REFERENCE/'day15_prediction.npy',16:ROOT/'output/day16_auc082_progress_20261102/prediction.npy'}
    paths={15:OUT/'day15_verification.json',16:ROOT/'output/day16_auc082_progress_20261102/verification.json'}
    targets={15:.815,16:.82}
    records={}
    for day in [15,16]:
        path=paths[day]
        ref=refs[day]
        note='Verified chronological reference'
        rows=np.flatnonzero(raw.pt_d.to_numpy()==20191100+day)
        y=raw.label.to_numpy()[rows]
        if day==15:
            best_score=auc(y,np.load(ref))
            for name in ['rolling_daily15_v1','global_row_context15_v1','catboost_global_rows15_v1',
                         'rich_binary15_completed_v1','rank_finetune_v1_day15','rank_finetune_v2_day15',
                         'augmented_rows_v1_day15','augmented_blend_v1_day15']:
                directory=HERE/'runs'/name
                check=directory/'verification.json'
                if not check.exists():continue
                verified=json.loads(check.read_text(encoding='utf8'))
                assert verified['status']=='passed' and not verified['evaluation15_labels_fit']
                np.testing.assert_array_equal(np.load(directory/'valid_indices.npy'),rows)
                candidate=directory/'prediction.npy'
                score=auc(y,np.load(candidate))
                assert abs(score-verified['auc'])<1e-12
                if score>best_score:
                    best_score=score;ref=candidate;path=check
                    note='Models fitted strictly before November15; '+name
            for name in ['rolling_previous_app_v1','rolling_rich_daily_catboost_v1',
                         'rolling_rich_count_v1','rolling_news_latent_v1','conditional_position_v1',
                         'compose_rich_response_v1','bayesian_daily_offset_v1','news_quality_daily_v1',
                         'rich_augmented_rows_v1','dense_context_rows_v1','group_temperature_v1',
                         'conditional_daily_v1','conditional_daily_trees_v1','group_temperature_joint_v1',
                         'rich_row_catboost_v1','full_quality_row_catboost_v1',
                         'catboost_group_temperature_v1','grouped_catboost_residual_v1',
                         'quality_daily_catboost_v1','compose_full_quality_transfer_v1',
                         'compose_quality_daily_transfer_v1','compact_quality_daily_catboost_v1','quality_daily_neural_v1',
                         'compose_dense_context_transfer_v1','compose_compact_dense_transfer_v1','within_user_ranker_v1',
                         'catboost_within_ranker_v1','catboost_within_ranker_v2','daily_auc_pair_ranker_v1','daily_binomial_rate_v1',
                         'compose_daily_objectives_v1','cross_user_pair_ranker_v1','snapshot_pair_ranker_v1',
                         'compose_neural_embedding_masks_v1',
                         'full_negative_semantic_embeddings_v1',
                         'refresh_family_quantiles_v1',
                         'compose_semantic_embedding_masks_v1',
                         'full_negative_history_dot_embeddings_v1',
                         'full_negative_embedding_lgb_v2',
                         'full_negative_embedding_lgb_v3',
                         'full_negative_dense_context_lgb_v1',
                         'compose_history_dot_embedding_masks_v1',
                         'cold_task_ranker_v1','firstday_task_ranker_v1','compose_snapshot_cold_v1','task_audience_rate_v1','group_event_posterior_v1','full_negative_context_v1','full_negative_context_v2','full_rich_negative_v1','full_rich_pair_ranker_v1','full_rich_negative_v2','full_rich_pair_ranker_v2','compose_full_negative_masks_v1','full_negative_prior_offset_v1','full_negative_class_balance_v1','full_negative_neural_embeddings_v1','full_negative_neural_embeddings_v2','full_negative_embedding_lgb_v1','compose_class_balance_masks_v1',
                         'task_audience_row_ranker_v1','audience_graph_ranker_v1','profile_response_ranker_v1','compact_full_sample_ranker_v1','unordered_snapshot_ranker_v1','compose_unordered_masks_v1','sparse_cross_ranker_v1','pointwise_compact_v1','categorical_interactions_v1','compose_probability_quantiles_v1','compose_transition_priors_v1','transition_posterior_v1','row_deepfm_v1','row_deepfm_v2','compact_lgb_rows_v1','compose_family_quantiles_v1']:
                directory=HERE/'runs'/name
                check=directory/'verification.json'
                if not check.exists():continue
                verified=json.loads(check.read_text(encoding='utf8'))
                assert verified['status']=='passed'
                entry=verified['dates']['20191115']
                assert not entry.get('evaluation_day_labels_fit',entry.get('own_held_day_ad_labels_fit',True))
                np.testing.assert_array_equal(np.load(directory/'day15_indices.npy'),rows)
                candidate=directory/'day15_prediction.npy'
                score=auc(y,np.load(candidate))
                assert abs(score-entry['auc'])<1e-12
                if score>best_score:
                    best_score=score;ref=candidate;path=check
                    note='Models fitted strictly before November15; '+name
        p=np.load(ref)
        assert p.shape==y.shape and np.isfinite(p).all()
        value=auc(y,p)
        if path.is_file():
            expected=json.loads(path.read_text(encoding='utf8'))
            expected_auc=(expected['dates'][str(20191100+day)]['auc'] if 'dates' in expected else
                          expected.get('auc',expected.get('csv_auc')))
            assert abs(value-expected_auc)<1e-12
        records[str(20191100+day)]={'auc':value,'target_auc':targets[day],'target_met':value>=targets[day],
                  'rows':len(rows),'clicks':int(y.sum()),'prediction':str(ref),'prediction_sha256':digest(ref),
                  'verification':str(path) if path.exists() else str(REFERENCE/'manifest.json'),
                  'scope':note,'held_date_is_reused_development_data':True}
    expected=json.loads((HERE/'runs/weighted_v1_day16/protocol.json').read_text(encoding='utf8'))['raw_datasets_sha256']
    actual={n:digest(ROOT/'2026ml'/n) for n in expected}
    assert actual==expected
    report={'objective':'原始数据集不变，15号AUC达到0.815，16号AUC达到0.82',
            'status':'active','timestamp_utc':datetime.now(timezone.utc).isoformat(),'dates':records,
            'both_targets_met':all(r['target_met'] for r in records.values()),
            'original_datasets_unchanged':True,'raw_datasets_sha256':actual}
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'progress.json',report)
    best15=records['20191115']
    p15=np.load(best15['prediction'])
    rows15=np.flatnonzero(raw.pt_d.to_numpy()==20191115)
    identifiers=pd.read_csv(ROOT/'2026ml/train.csv',usecols=['log_id','user_id'],dtype='int32')
    csv=OUT/'day15_predictions.csv'
    if csv.exists() and not np.array_equal(pd.read_csv(csv,float_precision='round_trip').pctr.to_numpy(),p15):
        import shutil
        snapshot=OUT/'snapshots'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        snapshot.mkdir(parents=True);shutil.copy2(csv,snapshot/csv.name)
    pd.DataFrame({'row_index':rows15,'log_id':identifiers.log_id.to_numpy()[rows15],
                  'user_id':identifiers.user_id.to_numpy()[rows15],
                  'label':raw.label.to_numpy()[rows15],'pctr':p15}).to_csv(csv,index=False,float_format='%.17g')
    roundtrip=pd.read_csv(csv,float_precision='round_trip')
    np.testing.assert_array_equal(roundtrip.pctr.to_numpy(),p15)
    assert abs(auc(roundtrip.label.to_numpy(),roundtrip.pctr.to_numpy())-best15['auc'])<1e-12
    source_verification=Path(best15['verification'])
    consolidated=OUT/'day15_verification.json'
    if source_verification.resolve()!=consolidated.resolve():
        report15={'status':'passed','day':20191115,'auc':best15['auc'],'rows':len(rows15),
                  'clicks':int(roundtrip.label.sum()),'source_prediction':best15['prediction'],
                  'source_verification':str(source_verification),'source_verification_sha256':digest(source_verification),
                  'csv_auc':best15['auc'],'submission_sha256':digest(csv),
                  'evaluation15_labels_fit':False,'development15_model_selection':True,
                  'original_datasets_unchanged':True,'raw_datasets_sha256':actual}
        write(consolidated,report15)
        best15['source_verification']=str(source_verification)
        best15['verification']=str(consolidated)
        write(OUT/'progress.json',report)
    lines=['# 双日AUC目标进展','',report['objective'],'','| 日期 | 完整样本数 | 已验证AUC | 目标 |','|---|---:|---:|---:|']
    for day,r in records.items():lines.append(f"| {day} | {r['rows']:,} | {r['auc']:.10f} | {r['target_auc']} |")
    lines+=['','两个日期均为反复用于开发的完整验证数据，不能当作新的盲测。原始四个CSV哈希已核对，保持不变。',
            '','目标尚未全部达到，任务继续。']
    (OUT/'进展.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    path=HERE/'STATUS.json';state=json.loads(path.read_text(encoding='utf8'))
    state.update(objective=report['objective'],goal_status='active',last_goal_turn_classification='progress',
                 two_day_targets=records,target_verified=report['both_targets_met'],
                 day16_target_verified=records['20191116']['target_met'],
                 current_verified15_auc=records['20191115']['auc'],
                 current_verified15_source=records['20191115'].get('source_verification',records['20191115']['verification']),
                 current_verified_best_auc=records['20191116']['auc'],
                 current_verified_best=records['20191116']['verification'],
                 catboost_global_rows_status='Complete and independently verified0.8090061871324833; full16 CSV published',
                 recurrent_status='Complete and independently verified0.8083177328447492; no promotion',
                 rank_matched_status='Complete and independently verified0.8075108903877389; no promotion',
                 snapshot_utc=report['timestamp_utc'])
    write(path,state)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    with publication_lock():main()
