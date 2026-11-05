"""Deploy the frozen real16 audience winner; evaluate inferred17 only after replay."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import Pool
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import export_best_v2 as base
from catboost_userday_residual import load_model
from conditional_daily import read, emit
from fit_stacker import digest, write
from rich_row_catboost import frame_hash
from rich_userday_activity import array_hash
import task_audience_rate_v1 as audience
from task_audience_core import FIELDS, build
from task_audience_export_core_v1 import frozen_schema
from news_quality_features import READ, build_quality

HERE = Path(__file__).resolve().parent
RUN = HERE / 'task_audience_run_v1'
OUTPUT = base.ROOT / 'output/best_recipe_task_audience_20191117_20261103'
PARENT = base.ROOT / 'output/best_recipe_cold_task_20191117_20261103'
PARENT_RUN = HERE / 'cold_task_run_v1'
SOURCE = audience.OUT


def check_sources():
    protocol = read(RUN / 'protocol.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    for name, expected in protocol['raw_datasets_sha256'].items():
        assert digest(base.ROOT / '2026ml' / name) == expected, name
    return protocol


def freeze():
    assert not RUN.exists() and not OUTPUT.exists(), 'Preserve previous exports'
    verified = read(SOURCE / 'verification.json')
    parent_verified = read(PARENT / 'verification.json')
    assert verified['status'] == parent_verified['status'] == 'passed'
    selected = verified['dates']['20191116']['best']
    assert selected['mode'] in ['replace_task', 'add_task'] and selected['weight'] > 0
    assert selected['auc'] > parent_verified['development16_auc']
    assert not verified['dates']['20191116']['evaluation_day_labels_fit']
    training = read(SOURCE / 'input_day16/contract.json')
    assert training['maximum_training_date'] == 20191115
    parent = read(PARENT_RUN / 'protocol.json')
    source = read(SOURCE / 'protocol.json')
    assert parent['raw_datasets_sha256'] == source['raw_datasets_sha256']
    bindings = dict(parent['sources_sha256'])
    for path, expected in source['sources_sha256'].items():
        assert path not in bindings or bindings[path] == expected, path
        bindings[path] = expected
    paths = [Path(__file__), base.EXP / 'task_audience_export_core_v1.py',
             base.EXP / 'test_task_audience_export_core_v1.py',
             PARENT_RUN / 'protocol.json', PARENT / 'verification.json',
             PARENT / 'prediction_manifest.json', SOURCE / 'protocol.json',
             SOURCE / 'verification.json', SOURCE / 'models_locked.json',
             SOURCE / 'predictions_locked.json', SOURCE / 'input_day16/contract.json',
             SOURCE / 'day16_prediction.npy', SOURCE / 'day16_indices.npy',
             SOURCE / (selected['name'] + '.cbm'),
             SOURCE / (selected['name'] + '_margin.npy')]
    paths += [PARENT / f'day{day}_prediction.npy' for day in [16, 17]]
    paths += [base.PRIOR_EXPORT / 'run_v2' / phase / 'daily_quality.npz'
              for phase in ['input_day17', 'independent_day17']]
    for path in paths:
        filename, expected = str(path), digest(path)
        assert filename not in bindings or bindings[filename] == expected, filename
        bindings[filename] = expected
    for path, expected in bindings.items():
        assert digest(path) == expected, path
    RUN.mkdir()
    write(RUN / 'protocol.json', dict(
        status='frozen_before_final17_forecast_and_inferred_evaluation',
        timestamp_utc=datetime.now(timezone.utc).isoformat(), selected=selected,
        development16_auc=selected['auc'],
        selection_basis='Independently verified real16 audience winner',
        development_dates_repeatedly_reused=True,
        new_model_training=False, correction_training_max_date=20191115,
        inherited_reference_training_max_date=20191116,
        final17_ad_history_max_date=20191116,
        histogram_vocabulary='Frozen to the fitted real16 schema; new categories remain unknown',
        inferred17_labels_used_for_training_tuning_feature_or_model_selection=False,
        raw_datasets_sha256=source['raw_datasets_sha256'], sources_sha256=bindings))
    check_sources()
    emit('task_audience17_recipe_frozen', selected=selected)


def inputs(day, independent=False):
    if day == 20191116:
        frame, meta, contract = audience.inputs(16, independent)
        assert contract == read(SOURCE / 'input_day16/contract.json')
        return frame, meta, contract
    assert day == 20191117
    train = pd.read_csv(base.ROOT / '2026ml/train.csv', usecols=FIELDS + ['label'], dtype='int32')
    final = pd.read_csv(base.ROOT / '2026ml/final_test.csv', usecols=FIELDS, dtype='int32')
    assert train.pt_d.max() == 20191116 and final.pt_d.eq(day).all()
    labels = np.r_[train.label.to_numpy(), np.full(len(final), -1, dtype='int32')]
    ads = pd.concat([train[FIELDS], final[FIELDS]], ignore_index=True)
    offset = len(train)
    del train, final
    quality16 = np.load(audience.QUALITY / 'quality16.npz')
    phase = 'independent_day17' if independent else 'input_day17'
    quality17 = np.load(base.PRIOR_EXPORT / 'run_v2' / phase / 'daily_quality.npz')
    assert np.all(quality16['keys'][:, 0] <= 20191116)
    assert np.all(quality17['keys'][:, 0] == 20191117)
    keys = np.concatenate([quality16['keys'], quality17['keys']])
    values = np.concatenate([quality16['values'], quality17['values']])
    assert values.shape[1] == 337
    if independent:
        news = pd.concat([pd.read_csv(base.ROOT / '2026ml' / name, usecols=READ,
                                     dtype={'i_docId': str, 'i_s_sourceId': str})
                          for name in ['train_feeds.csv', 'test_feeds.csv']], ignore_index=True)
        replay, names = build_quality(news, keys, day)
        np.testing.assert_array_equal(replay, values)
        assert names == read(audience.QUALITY / 'input_contract16.json')['quality_names']
        values = replay
        del news
        gc.collect()
    frame, cats, meta = build(ads[FIELDS], labels, keys, values, day,
                              'sparse' if independent else 'bincount')
    fitted = read(SOURCE / 'input_day16/contract.json')
    frame = frozen_schema(frame, ads, meta, fitted['feature_names'], independent)
    assert cats == fitted['categories']
    np.testing.assert_array_equal(meta['rows'], np.arange(offset, offset + len(meta['rows'])))
    assert np.all(meta['counts'][meta['keys'][:, 0] == day] == -1)
    contract = dict(day=day, shape=list(frame.shape), feature_names=list(frame), categories=cats,
                    frame_sha256=frame_hash(frame),
                    metadata_sha256={key: array_hash(value) for key, value in meta.items()},
                    quality_keys_sha256=array_hash(keys), quality_values_sha256=array_hash(values),
                    maximum_ad_history_date=20191116, held_ad_counts_masked=True,
                    model_histogram_vocabulary_frozen=True)
    del ads, labels, values, quality16, quality17
    gc.collect()
    emit('task_audience17_inputs_ready', independent=independent, shape=contract['shape'])
    return frame, meta, contract


def forecast(day, independent=False):
    selected = read(RUN / 'protocol.json')['selected']
    frame, meta, contract = inputs(day, independent)
    model = load_model(SOURCE / (selected['name'] + '.cbm'))
    assert model.feature_names_ == list(frame)
    margin = model.predict(Pool(frame.iloc[meta['held_groups']], cat_features=contract['categories']),
                           prediction_type='RawFormulaVal', thread_count=2)
    parent = np.load(PARENT / f'day{day % 100}_prediction.npy')
    prediction = audience.prediction(parent, margin, meta, selected['weight'],
                                     selected['mode'], independent)
    assert len(prediction) == len(meta['rows']) and np.isfinite(prediction).all()
    assert np.all((prediction > 0) & (prediction < 1))
    if day == 20191116:
        np.testing.assert_array_equal(margin, np.load(SOURCE / (selected['name'] + '_margin.npy')))
        np.testing.assert_allclose(prediction, np.load(SOURCE / 'day16_prediction.npy'), rtol=0, atol=2e-14)
    contract['margin_sha256'] = array_hash(margin)
    del frame, model, parent, meta
    gc.collect()
    return prediction, contract


def predict():
    check_sources()
    assert not OUTPUT.exists(), 'Preserve previous outputs'
    contracts = {}
    for day in [20191116, 20191117]:
        prediction, contract = forecast(day)
        np.save(RUN / f'day{day % 100}_prediction.npy', prediction)
        contracts[str(day)] = contract
    write(RUN / 'input_contracts.json', contracts)
    OUTPUT.mkdir()
    raw = base.query_raw(20191117)
    prediction = np.load(RUN / 'day17_prediction.npy')
    assert len(raw) == len(prediction) == 404182
    pd.DataFrame(dict(id=np.arange(len(raw)), pctr=prediction)).to_csv(
        OUTPUT / 'submission_best_20191117.csv', index=False, float_format='%.17g')
    pd.DataFrame(dict(log_id=raw.log_id, pctr=prediction)).to_csv(
        OUTPUT / 'submission_log_id_20191117.csv', index=False, float_format='%.17g')
    for day in [16, 17]:
        np.save(OUTPUT / f'day{day}_prediction.npy', np.load(RUN / f'day{day}_prediction.npy'))
    write(OUTPUT / 'prediction_manifest.json', dict(
        status='locked_before_inferred17_label_evaluation', protocol_sha256=digest(RUN / 'protocol.json'),
        input_contracts_sha256=digest(RUN / 'input_contracts.json'),
        inferred17_labels_used_for_prediction_or_selection=False,
        files_sha256={path.name: digest(path) for path in OUTPUT.iterdir() if path.is_file()}))
    emit('task_audience17_submission_locked', path=str(OUTPUT / 'submission_best_20191117.csv'))


def verify():
    protocol = check_sources()
    assert not (OUTPUT / 'verification.json').exists(), 'Preserve previous verification'
    manifest = read(OUTPUT / 'prediction_manifest.json')
    assert manifest['protocol_sha256'] == digest(RUN / 'protocol.json')
    assert manifest['input_contracts_sha256'] == digest(RUN / 'input_contracts.json')
    for name, expected in manifest['files_sha256'].items():
        assert digest(OUTPUT / name) == expected, name
    replay, contract = forecast(20191117, True)
    assert contract == read(RUN / 'input_contracts.json')['20191117']
    prediction = np.load(OUTPUT / 'day17_prediction.npy')
    error = float(np.max(np.abs(replay - prediction)))
    assert error < 2e-14
    raw = base.query_raw(20191117)
    csv = pd.read_csv(OUTPUT / 'submission_best_20191117.csv', float_precision='round_trip')
    alternate = pd.read_csv(OUTPUT / 'submission_log_id_20191117.csv', float_precision='round_trip')
    assert list(csv) == ['id', 'pctr'] and list(alternate) == ['log_id', 'pctr']
    np.testing.assert_array_equal(csv.id, np.arange(404182))
    np.testing.assert_array_equal(alternate.log_id, raw.log_id)
    np.testing.assert_array_equal(csv.pctr, prediction)
    np.testing.assert_array_equal(alternate.pctr, prediction)
    actual = pd.read_csv(base.ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    auc16 = float(roc_auc_score(actual.loc[actual.pt_d.eq(20191116), 'label'],
                              np.load(OUTPUT / 'day16_prediction.npy')))
    assert abs(auc16 - protocol['development16_auc']) < 1e-12
    check_sources()
    # All recipes and output are frozen and independently replayed before reading these labels.
    assert digest(base.LABEL_FILE) == base.LABEL_SHA
    inferred = pd.read_csv(base.LABEL_FILE)
    np.testing.assert_array_equal(inferred.id, csv.id)
    np.testing.assert_array_equal(inferred.log_id, raw.log_id)
    assert inferred.label.isin([0, 1]).all()
    auc17 = float(roc_auc_score(inferred.label, prediction))
    positives = int(inferred.label.sum())
    rank_auc = float((rankdata(prediction)[inferred.label.eq(1)].sum() - positives * (positives + 1) / 2)
                     / (positives * (len(inferred) - positives)))
    assert abs(auc17 - rank_auc) < 1e-12
    report = dict(status='passed', date=20191117, rows=len(raw), development16_auc=auc16,
        selected_model=protocol['selected'], inferred_label_auc=auc17, independent_rank_auc=rank_auc,
        inferred_positive_count=positives, labels_are_inferred_not_observed=True, platform_auc=None,
        inferred_labels_used_for_training_tuning_feature_or_model_selection=False,
        development_dates_repeatedly_reused=True, new_model_training=False,
        correction_training_max_date=20191115, inherited_reference_training_max_date=20191116,
        final17_ad_history_max_date=20191116, model_histogram_vocabulary_frozen=True,
        day16_winning_forecasts_replayed=True, inherited_base_and_features_independently_verified=True,
        full_audience_features_and_model_margins_independently_replayed=True,
        raw_news_quality_independently_rebuilt=True, maximum_prediction_replay_error=error,
        csv_roundtrip_exact=True, original_row_order_verified=True, original_datasets_unchanged=True,
        raw_datasets_sha256=protocol['raw_datasets_sha256'], inferred_label_sha256=base.LABEL_SHA,
        protocol_sha256=digest(RUN / 'protocol.json'),
        submission_sha256=digest(OUTPUT / 'submission_best_20191117.csv'))
    write(OUTPUT / 'verification.json', report)
    (OUTPUT / '提交说明.md').write_text(
        f'提交文件：submission_best_20191117.csv，id,pctr，共{len(raw):,}行。\n'
        f'16号真实开发集AUC：{auc16:.12f}；15/16号已反复用于开发。\n'
        f'17号反推标签AUC：{auc17:.12f}，仅独立评估，不用于模型选择；平台AUC未知。\n'
        '受众模型沿用16号已验证的模型及类别词表；广告历史仅用到16号。\n'
        '四份原始CSV、行序、浮点精度、特征及预测独立复算均已验证。\n', encoding='utf8')
    write(HERE / 'delivery_task_audience_v1.json', dict(report,
        submission=str(OUTPUT / 'submission_best_20191117.csv'),
        original_target_goal_remains_active=True))
    emit('task_audience17_verified_and_evaluated', **report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['freeze', 'predict', 'verify', 'all'])
    phase = parser.parse_args().phase
    if phase == 'all':
        freeze()
        predict()
        verify()
    else:
        {'freeze': freeze, 'predict': predict, 'verify': verify}[phase]()
