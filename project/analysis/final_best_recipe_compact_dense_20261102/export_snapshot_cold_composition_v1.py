"""Export a fixed composition selected on real16, then evaluate inferred17."""
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import export_best_v2 as base
from fit_stacker import digest, write
from conditional_daily import read, emit
from compose_snapshot_cold_v1 import predict

HERE = Path(__file__).resolve().parent
RUN = HERE/'snapshot_cold_composition_run_v1'
OUTPUT = base.ROOT/'output/best_recipe_snapshot_cold_20191117_20261103'
COMPOSITION = base.EXP/'runs/compose_snapshot_cold_v1'
PARENTS = dict(base=base.ROOT/'output/best_recipe_cross_user_refit17_20191117_20261102',
               snapshot=base.ROOT/'output/best_recipe_snapshot_pair_20191117_20261102',
               cold=base.ROOT/'output/best_recipe_cold_task_20191117_20261103')
PARENT_RUNS = dict(base=HERE/'cross_user_refit17_run_v1',
                   snapshot=HERE/'snapshot_pair_run_v1', cold=HERE/'cold_task_run_v1')


def check():
    protocol = read(RUN/'protocol.json')
    for filename, expected in protocol['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in protocol['raw_datasets_sha256'].items():
        assert digest(base.ROOT/'2026ml'/filename) == expected, filename
    return protocol


def freeze():
    assert not RUN.exists() and not OUTPUT.exists(), 'Preserve previous exports'
    verification = read(COMPOSITION/'verification.json')
    assert verification['status'] == 'passed'
    selected = verification['dates']['20191116']['best']
    audit = read(base.ROOT/'output/two_day_auc_targets_20261102/independent_target_audit.json')
    assert audit['audit_status'] == 'passed'
    if selected['auc'] < audit['dates']['20191116']['auc']-1e-12:
        emit('snapshot_cold_export_superseded', selected=selected)
        return False
    reports = {key:read(folder/'verification.json') for key,folder in PARENTS.items()}
    assert all(report['status'] == 'passed' for report in reports.values())
    if selected['auc'] <= max(report['development16_auc'] for report in reports.values()):
        emit('snapshot_cold_export_unnecessary', selected=selected)
        return False
    hashes = reports['base']['raw_datasets_sha256']
    bindings = {}
    paths = [Path(__file__), COMPOSITION/'protocol.json', COMPOSITION/'verification.json',
             COMPOSITION/'predictions_locked.json']
    for key,folder in PARENTS.items():
        parent = read(PARENT_RUNS[key]/'protocol.json')
        assert parent['raw_datasets_sha256'] == hashes
        for filename, expected in parent['sources_sha256'].items():
            assert filename not in bindings or bindings[filename] == expected
            bindings[filename] = expected
        paths += [PARENT_RUNS[key]/'protocol.json', folder/'verification.json',
                  folder/'prediction_manifest.json']
        paths += [folder/f'day{day}_prediction.npy' for day in [16,17]]
    for filename,expected in read(COMPOSITION/'protocol.json')['sources_sha256'].items():
        assert filename not in bindings or bindings[filename] == expected
        bindings[filename] = expected
    paths += [COMPOSITION/f'day{day}_prediction.npy' for day in [15,16]]
    bindings.update({str(path):digest(path) for path in paths})
    for filename,expected in bindings.items():
        assert digest(filename) == expected, filename
    RUN.mkdir()
    write(RUN/'protocol.json', dict(status='frozen_before_forecast_and_inferred17_evaluation',
        timestamp_utc=datetime.now(timezone.utc).isoformat(), selected=selected,
        development16_auc=selected['auc'], sources_sha256=bindings, raw_datasets_sha256=hashes,
        selection_basis='Highest independently verified real16 development AUC',
        development_dates_repeatedly_reused=True,
        inferred17_labels_used_for_training_tuning_feature_or_model_selection=False,
        correction_training_max_date=20191115, inherited_reference_training_max_date=20191116))
    return True


def forecast(day, independent=False):
    protocol = read(RUN/'protocol.json')
    values = {key:np.load(folder/f'day{day}_prediction.npy') for key,folder in PARENTS.items()}
    p = predict(values, protocol['selected'], independent)
    assert p.shape == ({16:421983,17:404182}[day],)
    assert np.isfinite(p).all() and np.all((p > 0) & (p < 1))
    if day == 16:
        np.testing.assert_allclose(p, np.load(COMPOSITION/'day16_prediction.npy'), rtol=0, atol=2e-14)
    return p


def main():
    if not freeze():
        return
    protocol = check()
    OUTPUT.mkdir()
    for day in [16,17]:
        np.save(OUTPUT/f'day{day}_prediction.npy', forecast(day))
    raw = base.query_raw(20191117)
    p = np.load(OUTPUT/'day17_prediction.npy')
    pd.DataFrame(dict(id=np.arange(len(raw)),pctr=p)).to_csv(
        OUTPUT/'submission_best_20191117.csv', index=False, float_format='%.17g')
    write(OUTPUT/'prediction_manifest.json', dict(status='locked_before_inferred17_evaluation',
        protocol_sha256=digest(RUN/'protocol.json'),
        files_sha256={path.name:digest(path) for path in OUTPUT.iterdir() if path.is_file()}))
    maximum = 0.
    for day in [16,17]:
        a = forecast(day, True)
        error = float(np.max(np.abs(a-np.load(OUTPUT/f'day{day}_prediction.npy'))))
        assert error < 2e-14
        maximum = max(maximum,error)
    manifest = read(OUTPUT/'prediction_manifest.json')
    for filename,expected in manifest['files_sha256'].items():
        assert digest(OUTPUT/filename) == expected
    submission = pd.read_csv(OUTPUT/'submission_best_20191117.csv',float_precision='round_trip')
    assert list(submission) == ['id','pctr'] and len(submission) == 404182
    np.testing.assert_array_equal(submission.id,np.arange(len(raw)))
    np.testing.assert_array_equal(submission.pctr,p)
    actual = pd.read_csv(base.ROOT/'2026ml/train.csv',usecols=['pt_d','label'],dtype='int32')
    auc16 = float(roc_auc_score(actual.loc[actual.pt_d.eq(20191116),'label'],np.load(OUTPUT/'day16_prediction.npy')))
    assert abs(auc16-protocol['development16_auc']) < 1e-12
    # All parameters and output hashes were fixed and replayed before this read.
    assert digest(base.LABEL_FILE) == base.LABEL_SHA
    inferred = pd.read_csv(base.LABEL_FILE)
    np.testing.assert_array_equal(inferred.id,submission.id)
    np.testing.assert_array_equal(inferred.log_id,raw.log_id)
    assert inferred.label.isin([0,1]).all()
    auc17 = float(roc_auc_score(inferred.label,p))
    n = int(inferred.label.sum())
    rank_auc = float((rankdata(p)[inferred.label.eq(1)].sum()-n*(n+1)/2)/(n*(len(p)-n)))
    assert abs(auc17-rank_auc) < 1e-12
    check()
    report = dict(status='passed',date=20191117,rows=len(raw),development16_auc=auc16,
        selected_model=protocol['selected'],inferred_label_auc=auc17,independent_rank_auc=rank_auc,
        inferred_positive_count=n,labels_are_inferred_not_observed=True,platform_auc=None,
        new_model_training=False,development_dates_repeatedly_reused=True,
        inferred_labels_used_for_training_tuning_feature_or_model_selection=False,
        correction_training_max_date=20191115,inherited_reference_training_max_date=20191116,
        parent_exports_independently_verified=True,maximum_prediction_replay_error=maximum,
        csv_roundtrip_exact=True,original_row_order_verified=True,original_datasets_unchanged=True,
        raw_datasets_sha256=protocol['raw_datasets_sha256'],inferred_label_sha256=base.LABEL_SHA,
        protocol_sha256=digest(RUN/'protocol.json'),submission_sha256=digest(OUTPUT/'submission_best_20191117.csv'))
    write(OUTPUT/'verification.json',report)
    (OUTPUT/'提交说明.md').write_text(
        f'2019-11-17 提交文件：submission_best_20191117.csv；{len(p):,} 行，id,pctr。\n'
        f'真实16号开发集AUC：{auc16:.12f}；15/16号曾反复用于开发。\n'
        f'17号反推标签评估AUC：{auc17:.12f}，仅最终评估，未用于训练、调参或选型。\n'
        '原始四个CSV未修改；平台真实AUC未知。\n',encoding='utf8')
    emit('snapshot_cold17_verified',**report)


if __name__ == '__main__':
    main()
