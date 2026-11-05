"""Export the audited group-event winner using the verified day17 parent cache."""
from __future__ import annotations

import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / 'analysis/day16_auc082_20261102'
sys.path.insert(0, str(EXP))
from conditional_daily import read, emit
from fit_stacker import digest, write
from rich_userday_activity import array_hash
from group_event_posterior_core_v1 import FIELDS, events, deltas, predict

RUN = HERE / 'current_best_17_run_v3'
OUTPUT = ROOT / 'output/current_best_20191117_20261103_v3'
PARENT = ROOT / 'output/current_best_20191117_20261103_v2'
PARENT_RUN = HERE / 'current_best_17_run_v2'
WINNER = EXP / 'runs/group_event_posterior_v1'
FAMILY = EXP / 'runs/compose_family_quantiles_v1'
AUDIT = ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json'


def freeze():
    assert not RUN.exists() and not OUTPUT.exists(), 'Preserve previous exports'
    audit = read(AUDIT)
    assert audit['audit_status'] == 'passed'
    verified = read(WINNER / 'verification.json')
    assert verified['status'] == 'passed'
    selected = verified['dates']['20191116']['best']
    assert selected['name'] == 'recent_s20'
    assert selected['channel'] == 'ad_click_list_v003' and selected['weight'] == .25
    assert abs(selected['auc'] - audit['dates']['20191116']['auc']) < 1e-12
    assert digest(WINNER / 'day16_prediction.npy') == audit['dates']['20191116']['prediction_sha256']
    models = read(WINNER / 'models_locked.json')
    assert models['protocol_sha256'] == digest(WINNER / 'protocol.json')
    rate = models['rates']['16'][selected['name']]
    assert rate['max_training_date'] == 20191115
    assert read(WINNER / 'protocol.json')['inferred17_labels_read'] is False
    parent = read(PARENT / 'verification.json')
    assert parent['status'] == 'passed' and parent['inferred17_labels_read'] is False
    manifest = read(PARENT / 'prediction_manifest.json')
    assert manifest['protocol_sha256'] == digest(PARENT_RUN / 'protocol.json')
    assert manifest['input_contracts_sha256'] == digest(PARENT_RUN / 'input_contracts.json')
    assert digest(PARENT / 'day17_prediction.npy') == manifest['files_sha256']['day17_prediction.npy']
    assert digest(PARENT / 'submission_best_20191117.csv') == parent['submission_sha256']
    np.testing.assert_allclose(np.load(PARENT_RUN / 'day16_prediction.npy'),
                               np.load(FAMILY / 'day16_prediction.npy'), rtol=0, atol=2e-13)
    for filename, expected in read(WINNER / 'protocol.json')['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in audit['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    paths = [Path(__file__), AUDIT, WINNER / 'protocol.json', WINNER / 'verification.json',
             WINNER / 'models_locked.json', WINNER / 'day16_prediction.npy',
             WINNER / 'day16_indices.npy', FAMILY / 'day16_prediction.npy',
             PARENT / 'verification.json', PARENT / 'prediction_manifest.json',
             PARENT / 'day17_prediction.npy', PARENT_RUN / 'protocol.json',
             PARENT_RUN / 'input_contracts.json', PARENT_RUN / 'day16_prediction.npy',
             EXP / 'group_event_posterior_core_v1.py', EXP / 'transition_posterior_core_v1.py']
    RUN.mkdir()
    write(RUN / 'selection_audit_snapshot.json', audit)
    # The mutable audit is captured above; later completed jobs cannot change this export.
    paths.remove(AUDIT)
    paths.append(RUN / 'selection_audit_snapshot.json')
    write(RUN / 'protocol.json', dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(), selected=selected,
        development16_auc=selected['auc'], development_dates_repeatedly_reused=True,
        selection_basis='Best completed independently audited full-cohort day16 recipe at export freeze',
        inherited_base=str(PARENT), correction_model=str(WINNER), new_model_training=False,
        frozen_group_rates=rate, correction_rates_max_training_date=20191115,
        inherited_base_and_prediction_history_max_date=20191116,
        transfer='Apply the saved day16 group-event correction and weight to the verified day17 parent',
        inferred17_labels_read=False, day17_advertisement_labels_used=False,
        raw_datasets_sha256=audit['raw_datasets_sha256'],
        sources_sha256={str(path): digest(path) for path in paths}))
    emit('best17_v3_frozen', auc16=selected['auc'], selected=selected)


def check(raw=False):
    protocol = read(RUN / 'protocol.json')
    for filename, expected in protocol['sources_sha256'].items():
        assert digest(filename) == expected, filename
    if raw:
        for filename, expected in protocol['raw_datasets_sha256'].items():
            assert digest(ROOT / '2026ml' / filename) == expected, filename
    return protocol


def query(day):
    filename = 'train.csv' if day == 16 else 'final_test.csv'
    columns = FIELDS + ['log_id']
    parts = []
    for block in pd.read_csv(ROOT / '2026ml' / filename, usecols=columns,
                             dtype={name: str for name in FIELDS if 'list' in name}, chunksize=100000):
        parts.append(block.loc[block.pt_d.eq(20191100 + day), columns])
    raw = pd.concat(parts, ignore_index=True)
    assert len(raw) == {16: 421983, 17: 404182}[day]
    return raw


def group_hashes(grouped):
    return {channel: {key: array_hash(value) for key, value in groups.items()}
            for channel, groups in grouped.items()}


def forecast(day, independent=False):
    protocol = read(RUN / 'protocol.json')
    raw = query(day)
    parent_contract = read(PARENT_RUN / 'input_contracts.json')[str(20191100 + day)]
    assert array_hash(raw.log_id.to_numpy(dtype='int64')) == parent_contract['log_ids_sha256']
    base = np.load(FAMILY / 'day16_prediction.npy' if day == 16 else PARENT / 'day17_prediction.npy')
    assert len(base) == len(raw)
    grouped = events(raw[FIELDS], independent)
    if day == 16:
        assert group_hashes(grouped) == read(WINNER / 'models_locked.json')['query_groups_sha256']['16']
    selected = protocol['selected']
    delta = deltas(base, grouped, protocol['frozen_group_rates'], independent)[selected['channel']]
    prediction = predict(base, delta, selected['weight'], independent)
    assert np.isfinite(prediction).all() and np.all((prediction > 0) & (prediction < 1))
    if day == 16:
        np.testing.assert_allclose(prediction, np.load(WINNER / 'day16_prediction.npy'), rtol=0, atol=2e-13)
    contract = dict(day=20191100 + day, rows=len(raw),
                    log_ids_sha256=array_hash(raw.log_id.to_numpy(dtype='int64')),
                    base_sha256=array_hash(base), groups_sha256=group_hashes(grouped),
                    delta_sha256=array_hash(delta), changed_rows=int(np.count_nonzero(delta)),
                    inferred17_labels_read=False)
    emit('best17_v3_forecast', day=day, independent=independent,
         rows=len(raw), changed_rows=contract['changed_rows'])
    return prediction, contract, raw.log_id.to_numpy(dtype='int64')


def export():
    protocol = check()
    assert not OUTPUT.exists()
    contracts = {}
    for day in (16, 17):
        prediction, contract, log_ids = forecast(day)
        np.save(RUN / f'day{day}_prediction.npy', prediction)
        contracts[str(20191100 + day)] = contract
        write(RUN / 'input_contracts.json', contracts)
    OUTPUT.mkdir()
    np.save(OUTPUT / 'day17_prediction.npy', prediction)
    np.save(RUN / 'day17_log_ids.npy', log_ids)
    pd.DataFrame({'id': np.arange(len(prediction)), 'pctr': prediction}).to_csv(
        OUTPUT / 'submission_best_20191117.csv', index=False, float_format='%.17g')
    write(OUTPUT / 'prediction_manifest.json', dict(protocol_sha256=digest(RUN / 'protocol.json'),
        input_contracts_sha256=digest(RUN / 'input_contracts.json'),
        development16_auc=protocol['development16_auc'], inferred17_labels_read=False,
        files_sha256={path.name: digest(path) for path in OUTPUT.iterdir() if path.is_file()}))
    emit('best17_v3_submission_locked', path=str(OUTPUT / 'submission_best_20191117.csv'))


def verify():
    protocol = check()
    assert not (OUTPUT / 'verification.json').exists()
    contracts = read(RUN / 'input_contracts.json')
    maximum = 0.
    for day in (16, 17):
        prediction, contract, log_ids = forecast(day, True)
        # Independent summation may differ in its last floating-point bit.
        assert {k: v for k, v in contract.items() if k != 'delta_sha256'} == {
            k: v for k, v in contracts[str(20191100 + day)].items() if k != 'delta_sha256'}
        expected = np.load(RUN / f'day{day}_prediction.npy')
        error = float(np.max(np.abs(expected - prediction)))
        assert error < 2e-13
        maximum = max(maximum, error)
    np.testing.assert_array_equal(log_ids, np.load(RUN / 'day17_log_ids.npy'))
    saved = np.load(OUTPUT / 'day17_prediction.npy')
    np.testing.assert_array_equal(saved, np.load(RUN / 'day17_prediction.npy'))
    submission = pd.read_csv(OUTPUT / 'submission_best_20191117.csv', float_precision='round_trip')
    assert list(submission) == ['id', 'pctr'] and len(submission) == 404182
    np.testing.assert_array_equal(submission.id.to_numpy(), np.arange(len(saved)))
    np.testing.assert_array_equal(submission.pctr.to_numpy(), saved)
    assert np.isfinite(submission.pctr).all() and submission.pctr.between(0, 1).all()
    labels = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    rows = np.flatnonzero(labels.pt_d.to_numpy() == 20191116)
    np.testing.assert_array_equal(rows, np.load(WINNER / 'day16_indices.npy'))
    auc16 = float(roc_auc_score(labels.label.to_numpy()[rows], np.load(RUN / 'day16_prediction.npy')))
    assert abs(auc16 - protocol['development16_auc']) < 1e-12
    manifest = read(OUTPUT / 'prediction_manifest.json')
    assert manifest['protocol_sha256'] == digest(RUN / 'protocol.json')
    assert manifest['input_contracts_sha256'] == digest(RUN / 'input_contracts.json')
    for filename, expected in manifest['files_sha256'].items():
        assert digest(OUTPUT / filename) == expected
    check(raw=True)
    report = dict(status='passed', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        prediction_date=20191117, rows=len(saved), columns=['id', 'pctr'],
        development16_auc=auc16, development_dates_repeatedly_reused=True,
        selected=protocol['selected'], inferred17_labels_read=False, platform_auc=None,
        complete_day16_winner_replayed=True, independent_day17_event_groups_and_marginals_replayed=True,
        verified_parent_prediction_cache_reused=True,
        maximum_prediction_replay_error=maximum, original_row_order_verified=True,
        repeated_log_ids_preserved=int(pd.Series(log_ids).duplicated().sum()), csv_roundtrip_exact=True,
        original_datasets_unchanged=True, raw_datasets_sha256=protocol['raw_datasets_sha256'],
        correction_rates_max_training_date=20191115,
        inherited_base_and_prediction_history_max_date=20191116,
        changed_rows=contracts['20191117']['changed_rows'],
        submission_sha256=digest(OUTPUT / 'submission_best_20191117.csv'))
    write(OUTPUT / 'verification.json', report)
    (OUTPUT / '提交说明.md').write_text(
        '提交文件：submission_best_20191117.csv\n格式：id,pctr；404182行；id按原始行序从0开始。\n'
        f'导出冻结时已完成独立验证的16号开发集最佳方案，AUC={auc16:.12f}。\n'
        '15/16号反复用于开发，该AUC不是17号成绩。未读取17号反推标签。\n'
        '使用已验证的上一版17号底层预测，加上冻结的最优分组事件修正。\n'
        '分组修正参数训练截至15号，继承底层模型和历史特征截至16号。\n'
        '已通过16号最优预测复算、17号分组及条件概率独立复算、行序和CSV精度检查。\n',
        encoding='utf8')
    emit('best17_v3_verified', auc16=auc16, rows=len(saved), maximum_replay_error=maximum,
         path=str(OUTPUT / 'submission_best_20191117.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['freeze', 'export', 'verify', 'all'])
    phase = parser.parse_args().phase
    if phase == 'all':
        freeze()
        export()
        verify()
    else:
        {'freeze': freeze, 'export': export, 'verify': verify}[phase]()
