"""Export the real-day16 winner, then evaluate the frozen day17 submission."""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT/'analysis/day16_auc082_20261102'
sys.path.insert(0, str(EXP))
from group_temperature import FIELDS, GROUPS, group_codes, transform

RUN = HERE/'run_v1'
OUTPUT = ROOT/'output/best_recipe_group_temperature_20191117_20261102'
WINNER = EXP/'runs/group_temperature_v1'
QUALITY = EXP/'runs/news_quality_daily_v1'
PARENT = ROOT/'output/best_recipe_news_quality_20191117_20261102'
PARENT_PROTOCOL = ROOT/'analysis/final_best_recipe_news_quality_20261102/run_v2/protocol.json'
LABEL_FILE = ROOT/'analysis/eleven_score_strategy_20261102/imported_refinement/day17_candidate_labels_11scores.csv'
LABEL_SHA256 = 'ab29c227c97efcc4fbef4dd57d92a8657ce5c078f1cbd03f037d8183f0ca7e24'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def emit(event, **values):
    print(json.dumps(dict(event=event, **values), ensure_ascii=False), flush=True)


def freeze():
    assert not RUN.exists() and not OUTPUT.exists(), 'Preserve all prior exports and failed runs'
    published = read(ROOT/'output/day16_auc082_progress_20261102/verification.json')
    report = read(WINNER/'verification.json')
    parent = read(PARENT/'verification.json')
    manifest = read(PARENT/'prediction_manifest.json')
    assert published['best']['name'] == 'group_temperature_v1'
    assert report['status'] == parent['status'] == 'passed'
    record = report['dates']['20191116']['best']
    assert abs(published['best']['auc']-record['auc']) < 1e-12
    assert record['name'] == 'user_app_preserve_hazard_mass_t0.75'
    assert not parent['inferred_labels_used_for_training_tuning_or_selection']
    assert manifest['protocol_sha256'] == parent['protocol_sha256'] == digest(PARENT_PROTOCOL)
    for name, expected in manifest['files_sha256'].items():
        assert digest(PARENT/name) == expected, name
    winner_protocol = read(WINNER/'protocol.json')
    for name, expected in winner_protocol['sources_sha256'].items():
        assert digest(name) == expected, name
    for name, expected in winner_protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    paths = [Path(__file__), EXP/'group_temperature.py', EXP/'test_group_temperature.py',
             WINNER/'verification.json', WINNER/'protocol.json', WINNER/'predictions_locked.json',
             WINNER/'day16_prediction.npy', WINNER/'day16_indices.npy',
             QUALITY/'day16_prediction.npy', QUALITY/'verification.json',
             PARENT/'verification.json', PARENT/'prediction_manifest.json', PARENT_PROTOCOL,
             PARENT/'day16_prediction.npy', PARENT/'day17_prediction.npy']
    protocol = {
        'status': 'frozen_before_new_day17_prediction_and_inferred_label_evaluation',
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'selected': record, 'development16_auc': record['auc'],
        'selection_basis': 'Highest independently verified real-day16 development AUC at freeze time',
        'development_dates_reused_for_selection': True,
        'new_predictive_model_training': False,
        'parent_day17_forecast': str(PARENT/'day17_prediction.npy'),
        'parent_day17_forecast_independently_replayed': True,
        'parent_tree_training_max_date': 20191115,
        'parent_neural_and_ad_history_max_date': 20191116,
        'same_day_supplied_unlabeled_ad_groups_used': True,
        'same_day_supplied_news_feedback_used_by_parent': True,
        'inferred17_labels_used_for_training_tuning_or_selection': False,
        'inferred17_labels_previously_evaluated_on_other_frozen_exports': True,
        'raw_datasets_sha256': winner_protocol['raw_datasets_sha256'],
        'sources_sha256': {str(path): digest(path) for path in paths},
    }
    RUN.mkdir(parents=True)
    write(RUN/'protocol.json', protocol)
    emit('recipe_frozen', selected=record)
    return protocol


def check_sources():
    protocol = read(RUN/'protocol.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT/'2026ml'/name) == expected, name
    return protocol


def inputs(day):
    if day == 20191116:
        raw = pd.read_csv(ROOT/'2026ml/train.csv', usecols=FIELDS+['pt_d'], dtype='int32')
        rows = np.flatnonzero(raw.pt_d.to_numpy() == day)
        np.testing.assert_array_equal(rows, np.load(WINNER/'day16_indices.npy'))
        frame = raw.iloc[rows][FIELDS].reset_index(drop=True)
        base = np.load(QUALITY/'day16_prediction.npy')
        # The independently rebuilt parent export differs only at floating point precision.
        error = float(np.max(np.abs(base-np.load(PARENT/'day16_prediction.npy'))))
        assert error < 2e-7
    else:
        raw = pd.read_csv(ROOT/'2026ml/final_test.csv', usecols=FIELDS+['pt_d', 'log_id'], dtype='int32')
        assert raw.pt_d.eq(day).all() and len(raw) == 404182
        frame = raw[FIELDS]
        base = np.load(PARENT/'day17_prediction.npy')
    assert len(base) == len(frame)
    return raw, frame, base


def independent_transform(frame, base, record):
    """Pandas aggregation independently checks the frozen NumPy implementation."""
    values = frame[GROUPS[record['group']]].copy()
    values['z'] = logit(base.clip(1e-10, 1-1e-10))
    group_fields = GROUPS[record['group']]
    mean = values.groupby(group_fields, sort=False, observed=True).z.transform('mean').to_numpy()
    q = expit(mean+record['temperature']*(values.z.to_numpy()-mean))
    values['old_hazard'] = -np.log1p(-base)
    values['new_hazard'] = -np.log1p(-q.clip(1e-12, 1-1e-12))
    grouped = values.groupby(group_fields, sort=False, observed=True)
    old_mass = grouped.old_hazard.transform('sum').to_numpy()
    new_mass = grouped.new_hazard.transform('sum').to_numpy()
    return -np.expm1(-values.new_hazard.to_numpy()*old_mass/new_mass.clip(1e-15))


def predict():
    protocol = freeze()
    record = protocol['selected']
    OUTPUT.mkdir(parents=True)
    for day in [20191116, 20191117]:
        raw, frame, base = inputs(day)
        codes = group_codes(frame, GROUPS[record['group']])
        prediction = transform(base, codes, record['temperature'], record['mode'])
        if day == 20191116:
            np.testing.assert_array_equal(prediction, np.load(WINNER/'day16_prediction.npy'))
        np.save(OUTPUT/f'day{day%100}_prediction.npy', prediction)
        if day == 20191117:
            pd.DataFrame({'id': np.arange(len(raw)), 'pctr': prediction}).to_csv(
                OUTPUT/'submission_best_20191117.csv', index=False, float_format='%.10f')
            pd.DataFrame({'log_id': raw.log_id, 'pctr': prediction}).to_csv(
                OUTPUT/'submission_log_id_20191117.csv', index=False, float_format='%.10f')
        emit('prediction_frozen', day=day, rows=len(prediction))
    write(OUTPUT/'prediction_manifest.json', {
        'status': 'frozen_before_new_inferred17_evaluation',
        'protocol_sha256': digest(RUN/'protocol.json'),
        'files_sha256': {path.name: digest(path) for path in OUTPUT.iterdir() if path.is_file()},
        'inferred17_labels_used_for_prediction_or_selection': False,
    })


def verify():
    assert not (OUTPUT/'verification.json').exists(), 'Preserve completed verification'
    protocol = check_sources()
    record = protocol['selected']
    manifest = read(OUTPUT/'prediction_manifest.json')
    assert manifest['protocol_sha256'] == digest(RUN/'protocol.json')
    for name, expected in manifest['files_sha256'].items():
        assert digest(OUTPUT/name) == expected, name
    errors = {}
    hazard_errors = {}
    for day in [20191116, 20191117]:
        raw, frame, base = inputs(day)
        prediction = np.load(OUTPUT/f'day{day%100}_prediction.npy')
        replay = independent_transform(frame, base, record)
        error = float(np.max(np.abs(prediction-replay)))
        assert error < 2e-14
        errors[str(day)] = error
        codes = group_codes(frame, GROUPS[record['group']])
        mass_error = float(np.max(np.abs(np.bincount(codes, weights=-np.log1p(-base))-
                                        np.bincount(codes, weights=-np.log1p(-prediction)))))
        assert mass_error < 1e-10
        hazard_errors[str(day)] = mass_error
        if day == 20191116:
            np.testing.assert_array_equal(prediction, np.load(WINNER/'day16_prediction.npy'))
            labels16 = pd.read_csv(ROOT/'2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
            auc16 = float(roc_auc_score(labels16.loc[labels16.pt_d.eq(day), 'label'], prediction))
            assert abs(auc16-protocol['development16_auc']) < 1e-12
        emit('independent_transform_replayed', day=day, maximum_error=error)
    raw = pd.read_csv(ROOT/'2026ml/final_test.csv', usecols=['log_id', 'pt_d'], dtype='int32')
    submission = pd.read_csv(OUTPUT/'submission_best_20191117.csv', float_precision='round_trip')
    alternate = pd.read_csv(OUTPUT/'submission_log_id_20191117.csv', float_precision='round_trip')
    assert list(submission) == ['id', 'pctr'] and list(alternate) == ['log_id', 'pctr']
    assert len(raw) == len(submission) == len(alternate) == 404182
    np.testing.assert_array_equal(submission.id, np.arange(len(raw)))
    np.testing.assert_array_equal(alternate.log_id, raw.log_id)
    np.testing.assert_array_equal(submission.pctr, alternate.pctr)
    assert np.isfinite(submission.pctr).all() and submission.pctr.between(0, 1).all()
    rounding = float(np.max(np.abs(submission.pctr.to_numpy()-np.load(OUTPUT/'day17_prediction.npy'))))
    assert rounding < 5.01e-11
    check_sources()
    # First read of inferred labels in this export: recipe and both CSVs are already locked.
    assert digest(LABEL_FILE) == LABEL_SHA256
    inferred = pd.read_csv(LABEL_FILE)
    np.testing.assert_array_equal(inferred.id, submission.id)
    np.testing.assert_array_equal(inferred.log_id, raw.log_id)
    assert inferred.label.isin([0, 1]).all()
    auc = float(roc_auc_score(inferred.label, submission.pctr))
    positives = int(inferred.label.sum())
    rank_auc = float((rankdata(submission.pctr)[inferred.label.eq(1)].sum()-positives*(positives+1)/2)/
                     (positives*(len(inferred)-positives)))
    assert abs(auc-rank_auc) < 1e-12
    report = {
        'status': 'passed', 'date': 20191117, 'rows': len(raw), 'selected': record,
        'development16_auc': auc16, 'inferred_label_auc': auc, 'independent_rank_auc': rank_auc,
        'inferred_positive_count': positives, 'labels_are_inferred_not_observed': True,
        'platform_auc': None, 'inferred_labels_used_for_training_tuning_or_selection': False,
        'independent_transform_maximum_errors': errors, 'group_hazard_mass_maximum_errors': hazard_errors,
        'day16_verified_winner_replay_exact': True,
        'parent_day17_forecast_previously_independently_replayed': True,
        'parent_forecast_retrained_or_rebuilt_this_export': False,
        'maximum_csv_rounding_error': rounding, 'original_row_order_verified': True,
        'repeated_log_ids_preserved': int(raw.log_id.duplicated().sum()),
        'original_datasets_unchanged': True, 'raw_datasets_sha256': protocol['raw_datasets_sha256'],
        'inferred_label_sha256': LABEL_SHA256,
        'submission_sha256': digest(OUTPUT/'submission_best_20191117.csv'),
        'protocol_sha256': digest(RUN/'protocol.json'),
    }
    write(OUTPUT/'verification.json', report)
    (OUTPUT/'提交说明.md').write_text(
        f'2019年11月17日提交文件：submission_best_20191117.csv\n'
        f'格式 id,pctr，共{len(raw):,}行，id为final_test原始行序的零起始行号。\n'
        f'兼容文件：submission_log_id_20191117.csv，格式log_id,pctr，保留重复log_id。\n\n'
        f'依据16号真实标签开发集选择：{record["name"]}，AUC={auc16:.10f}。\n'
        f'采用已独立验证的新闻质量基础预测，在用户与应用分组内应用温度0.75并保持风险总量。\n'
        f'反推标签AUC={auc:.10f}；它不是官方真实标签，平台真实AUC未知。\n'
        f'反推标签未用于模型训练、调参或选择提交文件。\n'
        f'已独立复算分组变换、检查行序和CSV格式，四份原始数据文件未修改。\n', encoding='utf8')
    emit('verified_and_evaluated', **report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['predict', 'verify'], required=True)
    args = parser.parse_args()
    predict() if args.phase == 'predict' else verify()
