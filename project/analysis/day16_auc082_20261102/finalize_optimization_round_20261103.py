"""Collect verified outputs; choose by real16 AUC, never inferred17 AUC."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from fit_stacker import HERE, ROOT, digest, write
from conditional_daily import read

DESTINATION = ROOT/'output/optimization_round_20261103'
EXPORTS = [
    'best_recipe_cross_user_refit17_20191117_20261102',
    'best_recipe_snapshot_pair_20191117_20261102',
    'best_recipe_cold_task_20191117_20261103',
    'best_recipe_firstday_task_20191117_20261103',
    'best_recipe_snapshot_cold_20191117_20261103',
]
EXPERIMENTS = ['snapshot_pair_ranker_v1', 'cold_task_ranker_v1',
               'firstday_task_ranker_v1', 'compose_snapshot_cold_v1']


def main():
    assert not DESTINATION.exists(), 'Preserve the previous delivery'
    audit_path = ROOT/'output/two_day_auc_targets_20261102/independent_target_audit.json'
    audit = read(audit_path)
    assert audit['audit_status'] == 'passed' and audit['original_datasets_unchanged']
    experiments = {}
    for name in EXPERIMENTS:
        folder = HERE/'runs'/name
        result = read(folder/'verification.json')
        assert result['status'] == 'passed' and result['original_datasets_unchanged']
        experiments[name] = dict(verification=str(folder/'verification.json'),
            verification_sha256=digest(folder/'verification.json'),
            protocol_sha256=digest(folder/'protocol.json'),
            dates={date:dict(auc=record['auc'],best=record['best'])
                   for date,record in result['dates'].items()})
    exports = []
    for name in EXPORTS:
        folder = ROOT/'output'/name
        path = folder/'verification.json'
        if path.is_file():
            result = read(path)
            assert result['status'] == 'passed'
            assert not result['inferred_labels_used_for_training_tuning_feature_or_model_selection']
            exports.append((folder,result))
    chosen, report = max(exports,key=lambda item:(item[1]['development16_auc'],
                        item[1].get('correction_training_max_date',0)))
    assert abs(report['development16_auc']-audit['dates']['20191116']['auc']) < 1e-12
    source = chosen/'submission_best_20191117.csv'
    assert digest(source) == report['submission_sha256']
    feedback_path=HERE/'platform_feedback_0814_ranked_v1.json'
    feedback=read(feedback_path)
    previous=read(ROOT/Path(feedback['submission']).parent/'verification.json')
    assert previous['submission_sha256']==feedback['submission_sha256']
    frame = pd.read_csv(source,float_precision='round_trip')
    assert list(frame) == ['id','pctr'] and len(frame) == 404182
    np.testing.assert_array_equal(frame.id,np.arange(len(frame)))
    np.testing.assert_array_equal(frame.pctr,np.load(chosen/'day17_prediction.npy'))
    assert np.isfinite(frame.pctr).all() and frame.pctr.between(0,1).all()
    for name,expected in audit['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    DESTINATION.mkdir()
    target = DESTINATION/'submission_20191117.csv'
    shutil.copy2(source,target)
    assert digest(target) == report['submission_sha256']
    manifest = dict(status='passed',timestamp_utc=datetime.now(timezone.utc).isoformat(),
        submission=str(target),source_submission=str(source),rows=len(frame),
        submission_sha256=digest(target),source_verification=str(chosen/'verification.json'),
        selection_basis='Highest independently verified full real16 development AUC, then latest training cutoff',
        source_verification_sha256=digest(chosen/'verification.json'),
        full_cohort_auc={date:r['auc'] for date,r in audit['dates'].items()},
        inferred17_label_auc_evaluation_only=report['inferred_label_auc'],
        inferred17_labels_used_for_training_tuning_feature_or_model_selection=False,
        development_dates_repeatedly_reused=True,platform_auc_for_this_file=None,
        user_reported_previous_platform_auc=.814,
        previous_platform_feedback=str(feedback_path),
        previous_file_development16_auc=previous['development16_auc'],
        development16_gain_vs_previous_platform_file=report['development16_auc']-previous['development16_auc'],
        original_datasets_unchanged=True,raw_datasets_sha256=audit['raw_datasets_sha256'],
        experiments=experiments,targets_met=audit['goal_achieved'],
        goal_status='complete' if audit['goal_achieved'] else 'active')
    write(DESTINATION/'manifest.json',manifest)
    lines = ['# 2019-11-17 优化提交文件', '',
        '提交 `submission_20191117.csv`，格式 `id,pctr`，共 404,182 行，保留原始行顺序。', '',
        f'本轮选择：`{chosen.name}`。依据完整真实16号开发集的AUC选择，未使用反推17号标签选型。', '',
        '|评估数据|AUC|', '|---|---:|',
        f"|2019-11-15 完整开发集|{audit['dates']['20191115']['auc']:.12f}|",
        f"|2019-11-16 完整开发集|{audit['dates']['20191116']['auc']:.12f}|",
        f"|2019-11-17 反推标签，仅最终评估|{report['inferred_label_auc']:.12f}|", '',
        f"相较此前平台0.814对应文件，16号离线开发集AUC从 {previous['development16_auc']:.12f} 提高至 {report['development16_auc']:.12f}。", '',
        '15/16号被多次用于模型开发，这些结果不是新盲测分数。平台0.814属于此前提交，本文件平台分数尚未知。', '',
        '原始四份CSV的SHA256保持不变；已检查预测复算、CSV精度、概率范围和完整行顺序。', '',
        ('已达到15号0.815、16号0.82的目标。' if audit['goal_achieved'] else
         '当前仍未达到15号0.815、16号0.82的目标。')+'详见 manifest.json 和各实验的独立核验文件。']
    (DESTINATION/'结果说明.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print({key:manifest[key] for key in ['submission','rows','submission_sha256','full_cohort_auc',
        'inferred17_label_auc_evaluation_only','targets_met']},flush=True)


if __name__ == '__main__':
    main()
