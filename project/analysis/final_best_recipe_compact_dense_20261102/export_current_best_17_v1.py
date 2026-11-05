"""Export the verified day16 winner to day17 without opening inferred labels."""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'

import argparse
import gc
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.metrics import roc_auc_score

import export_best_v2 as base
import export_task_audience_v1 as audience_export
import task_audience_rate_v1 as audience
from catboost_userday_residual import load_model
from compact_full_sample_core_v1 import compact_frame, select_indices
from compose_unordered_masks_v1 import masked_blend
from conditional_daily import read, emit
from cross_user_pair_ranker_v1 import prediction as row_prediction
from fit_stacker import digest, write
from full_quality_row_catboost import quality_rows
from rich_row_catboost import frame_hash
from rich_userday_activity import array_hash
from task_audience_row_core import aligned
from unordered_snapshot_core_v1 import FIELDS, NAMES, METRICS, build

HERE = Path(__file__).resolve().parent
RUN = HERE / 'current_best_17_run_v1'
OUTPUT = base.ROOT / 'output/current_best_20191117_20261103'
EXP = base.EXP
COLD = base.ROOT / 'output/best_recipe_cold_task_20191117_20261103'
AUDIENCE = EXP / 'runs/task_audience_rate_v1'
COMPACT = EXP / 'runs/compact_full_sample_ranker_v1'
UNORDERED = EXP / 'runs/unordered_snapshot_ranker_v1'
COMPOSITION = EXP / 'runs/compose_unordered_masks_v1'
SELECTOR = EXP / 'runs/task_audience_row_ranker_v1'
NATIVE = EXP / 'runs/rich_augmented_rows_v1'


def checked_model(folder, name):
    record = next(r for r in read(folder / 'models_locked.json')['models'] if r['name'] == name)
    path = folder / (name + '.cbm')
    assert digest(path) == record['model_sha256'], path
    return path


def freeze():
    assert not RUN.exists() and not OUTPUT.exists(), 'Preserve previous work'
    audit_path = base.ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json'
    audit = read(audit_path)
    assert audit['audit_status'] == 'passed'
    selected = {}
    paths = [Path(__file__), audit_path]
    bindings = {}
    for key, folder in [('audience', AUDIENCE), ('compact', COMPACT), ('unordered', COMPOSITION)]:
        report = read(folder / 'verification.json')
        assert report['status'] == 'passed'
        selected[key] = report['dates']['20191116']['best']
        for filename in ['verification.json', 'protocol.json', 'day16_prediction.npy']:
            paths.append(folder / filename)
        for filename, expected in read(folder / 'protocol.json')['sources_sha256'].items():
            # Bind the actual source implementations; cache/model hashes are checked separately.
            if Path(filename).suffix == '.py':
                assert digest(filename) == expected, filename
                bindings[filename] = expected
    assert selected['audience']['name'] == 'day16_audience_d6_trees200'
    assert selected['audience']['mode'] == 'replace_task' and selected['audience']['weight'] == .2
    assert selected['compact']['name'] == 'day16_full_sample_d6_trees2400'
    assert selected['compact']['mode'] == 'replace_global' and selected['compact']['weight'] == .05
    assert selected['unordered']['name'] == 'day16_unordered_d8_trees600'
    assert selected['unordered']['mask'] == 'changed_click' and selected['unordered']['weight'] == .5
    assert abs(selected['unordered']['auc'] - audit['dates']['20191116']['auc']) < 1e-12
    cold_report = read(COLD / 'verification.json')
    assert cold_report['status'] == 'passed'
    cold_manifest = read(COLD / 'prediction_manifest.json')
    for day in [16, 17]:
        path = COLD / f'day{day}_prediction.npy'
        assert digest(path) == cold_manifest['files_sha256'][path.name]
        paths.append(path)
    paths += [COLD / 'verification.json', COLD / 'prediction_manifest.json']
    np.testing.assert_array_equal(np.load(COLD / 'day16_prediction.npy'),
                                  np.load(EXP / 'runs/cold_task_ranker_v1/day16_prediction.npy'))
    for key, folder in [('audience', AUDIENCE), ('compact', COMPACT), ('unordered', UNORDERED)]:
        path = checked_model(folder, selected[key]['name'])
        paths += [path, folder / 'models_locked.json']
    compact_contract = read(COMPACT / 'input_contract16.json')
    assert compact_contract['max_training_date'] == compact_contract['selector_max_training_date'] == 20191115
    assert read(AUDIENCE / 'input_day16/contract.json')['maximum_training_date'] == 20191115
    assert read(UNORDERED / 'input_contract16.json')['max_training_date'] == 20191115
    for folder in [COMPACT, UNORDERED]:
        paths += [folder / 'input_contract16.json', folder / 'held_contract16.json']
    paths += [SELECTOR / 'input_contract16.json', SELECTOR / 'day16_audience_row_d6_trees1600.cbm',
              NATIVE / 'input_contract16.json', AUDIENCE / 'input_day16/contract.json',
              audience_export.RUN / 'protocol.json', audience_export.RUN / 'input_contracts.json',
              HERE / 'export_task_audience_v1.py', HERE / 'export_best_v2.py', HERE / 'export_best.py']
    # Bind all cached native and news inputs used by this export, with their prior audit trail.
    for phase in ['input_day16', 'input_day17', 'independent_day17']:
        folder = base.PRIOR_EXPORT / 'run_v2' / phase
        contract = read(folder / 'contract.json')
        assert contract['own_query_ad_labels_masked'] and contract['max_ad_history_date'] < 20191100 + int(phase[-2:])
        assert digest(folder / 'features.npy') == contract['matrix_sha256']
        paths += [folder / name for name in ['contract.json', 'features.npy', 'daily_quality.npz']]
    paths += [base.PRIOR_EXPORT / 'run_v2/protocol.json',
              audience.QUALITY / 'quality16.npz', audience.QUALITY / 'input_contract16.json',
              audience.QUALITY / 'verification.json']
    assert read(audience.QUALITY / 'verification.json')['status'] == 'passed'
    raw_hashes = read(COMPACT / 'protocol.json')['raw_datasets_sha256']
    for filename, expected in raw_hashes.items():
        assert digest(base.ROOT / '2026ml' / filename) == expected, filename
    for path in paths:
        bindings[str(path)] = digest(path)
    RUN.mkdir()
    write(RUN / 'protocol.json', dict(
        status='frozen_before_day17_prediction', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        selected=selected, development16_auc=selected['unordered']['auc'],
        selection_basis='Best completed independently verified real day16 recipe',
        development_dates_repeatedly_reused=True, new_model_training=False,
        correction_models_training_max_date=20191115, inherited_base_training_max_date=20191116,
        day17_advertisement_labels_used=False, inferred17_labels_read=False,
        historical_advertisement_labels_max_date=20191116,
        raw_datasets_sha256=raw_hashes, sources_sha256=bindings))
    emit('current_best17_recipe_frozen', selected=selected)


def check(raw=True):
    protocol = read(RUN / 'protocol.json')
    for path, expected in protocol['sources_sha256'].items():
        assert digest(path) == expected, path
    if raw:
        for filename, expected in protocol['raw_datasets_sha256'].items():
            assert digest(base.ROOT / '2026ml' / filename) == expected, filename
    return protocol


def forecast(day, independent=False):
    protocol = read(RUN / 'protocol.json')
    selected = protocol['selected']
    emit('current_best17_forecast_start', day=day, independent=independent)
    raw = base.query_raw(day)
    # This function masks final-day advertisement labels and, on independent replay,
    # reconstructs news features and uses the sparse audience aggregation implementation.
    if day == 20191116:
        auxiliary, meta, audience_contract = audience.load_inputs(AUDIENCE / 'input_day16')
    else:
        auxiliary, meta, audience_contract = audience_export.inputs(day, independent)
    audience_model = load_model(checked_model(AUDIENCE, selected['audience']['name']))
    assert audience_model.feature_names_ == list(auxiliary)
    audience_margin = audience_model.predict(
        Pool(auxiliary.iloc[meta['held_groups']], cat_features=audience_contract['categories']),
        prediction_type='RawFormulaVal', thread_count=2)
    cold_prediction = np.load(COLD / f'day{day % 100}_prediction.npy')
    current = audience.prediction(cold_prediction, audience_margin, meta,
                                 selected['audience']['weight'], selected['audience']['mode'], independent)
    if day == 20191116:
        np.testing.assert_array_equal(audience_margin, np.load(AUDIENCE / (selected['audience']['name'] + '_margin.npy')))
        np.testing.assert_allclose(current, np.load(AUDIENCE / 'day16_prediction.npy'), rtol=0, atol=2e-14)
    else:
        # A preserved partial export has the same frozen recipe; compare without changing it.
        np.testing.assert_allclose(current, np.load(audience_export.OUTPUT / 'day17_prediction.npy'), rtol=0, atol=2e-14)
    audience_names = read(SELECTOR / 'input_contract16.json')['audience_columns']
    extra = aligned(meta['keys'], auxiliary[audience_names].to_numpy(dtype='float32'),
                    raw[['pt_d', 'task_id']].to_numpy(), day)
    audience_frame_sha = frame_hash(auxiliary)
    del auxiliary, meta, audience_model, cold_prediction
    gc.collect()
    path, native_contract = base.matrix_path(day, independent)
    matrix = np.load(path, mmap_mode='r')
    assert matrix.shape == (len(raw), 744)
    native = read(NATIVE / 'input_contract16.json')
    native_keep = [i for i, name in enumerate(native['feature_names'])
                   if not name.startswith('rich_daily__quality__')]
    assert len(native_keep) == 680
    names = [f'f{i}' for i in range(1017)] + ['task_audience__' + name for name in audience_names]
    categories = [f'f{native_keep.index(i)}' for i in native['categorical_columns']]
    selector = load_model(SELECTOR / 'day16_audience_row_d6_trees1600.cbm')
    assert selector.feature_names_ == names
    importance = selector.get_feature_importance(type='PredictionValuesChange', thread_count=2)
    columns = select_indices(names, categories, importance, 384)
    fitted = read(COMPACT / 'input_contract16.json')
    assert [names[i] for i in columns] == fitted['selected_columns']
    assert array_hash(np.round(importance, 10)) == fitted['selector_importance_sha256']
    phase = 'independent_day17' if independent else f'input_day{day % 100}'
    with np.load(base.PRIOR_EXPORT / 'run_v2' / phase / 'daily_quality.npz') as news:
        quality = quality_rows(news['keys'], news['values'], raw[['pt_d', 'user_id']].to_numpy())
    frame, cats = compact_frame(matrix, np.arange(len(raw)), native_keep, quality, extra,
                                columns, names, categories, independent=independent)
    assert cats == fitted['categories'] and frame.shape == (len(raw), 423)
    compact_sha = frame_hash(frame)
    if day == 20191116:
        held = read(COMPACT / 'held_contract16.json')
        assert compact_sha == held['frame_sha256']
        assert array_hash(quality) == held['quality_sha256']
        assert array_hash(extra) == held['audience_features_sha256']
    compact_model = load_model(checked_model(COMPACT, selected['compact']['name']))
    assert compact_model.feature_names_ == list(frame)
    compact_margin = compact_model.predict(Pool(frame, cat_features=cats),
                                           prediction_type='RawFormulaVal', thread_count=2)
    groups, _ = pd.factorize(raw.user_id.to_numpy(), sort=True)
    current = row_prediction(current, compact_margin, groups,
                             selected['compact']['weight'], selected['compact']['mode'], independent)
    if day == 20191116:
        np.testing.assert_array_equal(compact_margin, np.load(COMPACT / (selected['compact']['name'] + '_held_margin.npy')))
        np.testing.assert_allclose(current, np.load(COMPACT / 'day16_prediction.npy'), rtol=0, atol=2e-14)
    del matrix, quality, extra, selector, compact_model
    gc.collect()
    ad_path = base.ROOT / '2026ml' / ('train.csv' if day == 20191116 else 'final_test.csv')
    states = pd.read_csv(ad_path, usecols=FIELDS, dtype={name: str for name in FIELDS if 'list' in name})
    states = states.loc[states.pt_d.eq(day)].reset_index(drop=True)
    np.testing.assert_array_equal(states[['pt_d', 'user_id']].to_numpy(), raw[['pt_d', 'user_id']].to_numpy())
    values, state_names = build(states, independent=independent)
    assert state_names == NAMES and values.shape == (len(raw), 72)
    frame = pd.concat([frame, pd.DataFrame(values, columns=state_names)], axis=1)
    unordered_sha = frame_hash(frame)
    if day == 20191116:
        assert unordered_sha == read(UNORDERED / 'held_contract16.json')['frame_sha256']
        assert array_hash(values) == read(UNORDERED / 'held_contract16.json')['unordered_features_sha256']
    model = load_model(checked_model(UNORDERED, selected['unordered']['name']))
    assert model.feature_names_ == list(frame)
    unordered_margin = model.predict(Pool(frame, cat_features=cats),
                                     prediction_type='RawFormulaVal', thread_count=2)
    mask = np.any(values[:, :3 * len(METRICS)] != 0, axis=1)
    prediction = masked_blend(current, unordered_margin, mask, selected['unordered']['weight'], independent)
    if day == 20191116:
        np.testing.assert_array_equal(unordered_margin, np.load(UNORDERED / (selected['unordered']['name'] + '_held_margin.npy')))
        np.testing.assert_allclose(prediction, np.load(COMPOSITION / 'day16_prediction.npy'), rtol=0, atol=2e-14)
    assert prediction.shape == (len(raw),) and np.isfinite(prediction).all()
    assert np.all((prediction > 0) & (prediction < 1))
    contract = dict(day=day, rows=len(raw), log_ids_sha256=array_hash(raw.log_id.to_numpy()),
                    audience_frame_sha256=audience_frame_sha,
                    audience_margin_sha256=array_hash(audience_margin),
                    compact_frame_sha256=compact_sha, compact_margin_sha256=array_hash(compact_margin),
                    unordered_frame_sha256=unordered_sha, unordered_margin_sha256=array_hash(unordered_margin),
                    mask_sha256=array_hash(mask), mask_count=int(mask.sum()),
                    maximum_advertisement_history_date=day - 1, inferred17_labels_read=False)
    del frame, states, values, model, raw
    gc.collect()
    emit('current_best17_forecast_complete', day=day, independent=independent, mask_count=contract['mask_count'])
    return prediction, contract


def predict():
    protocol = check()
    contracts = {}
    for day in [20191116, 20191117]:
        prediction, contract = forecast(day)
        np.save(RUN / f'day{day % 100}_prediction.npy', prediction)
        contracts[str(day)] = contract
        write(RUN / 'input_contracts.json', contracts)
    assert not OUTPUT.exists()
    OUTPUT.mkdir()
    raw = base.query_raw(20191117)
    prediction = np.load(RUN / 'day17_prediction.npy')
    assert len(raw) == len(prediction) == 404182
    pd.DataFrame({'id': np.arange(len(raw)), 'pctr': prediction}).to_csv(
        OUTPUT / 'submission_best_20191117.csv', index=False, float_format='%.17g')
    pd.DataFrame({'log_id': raw.log_id, 'pctr': prediction}).to_csv(
        OUTPUT / 'submission_log_id_20191117.csv', index=False, float_format='%.17g')
    np.save(OUTPUT / 'day17_prediction.npy', prediction)
    write(OUTPUT / 'prediction_manifest.json', dict(
        status='locked_before_independent_replay', protocol_sha256=digest(RUN / 'protocol.json'),
        input_contracts_sha256=digest(RUN / 'input_contracts.json'),
        development16_auc=protocol['development16_auc'], inferred17_labels_read=False,
        files_sha256={path.name: digest(path) for path in OUTPUT.iterdir() if path.is_file()}))
    emit('current_best17_submission_locked', path=str(OUTPUT / 'submission_best_20191117.csv'))


def verify():
    protocol = check()
    assert not (OUTPUT / 'verification.json').exists()
    manifest = read(OUTPUT / 'prediction_manifest.json')
    assert manifest['protocol_sha256'] == digest(RUN / 'protocol.json')
    assert manifest['input_contracts_sha256'] == digest(RUN / 'input_contracts.json')
    for name, expected in manifest['files_sha256'].items():
        assert digest(OUTPUT / name) == expected, name
    replay, contract = forecast(20191117, independent=True)
    assert contract == read(RUN / 'input_contracts.json')['20191117']
    prediction = np.load(OUTPUT / 'day17_prediction.npy')
    error = float(np.max(np.abs(replay - prediction)))
    assert error < 2e-14
    raw = base.query_raw(20191117)
    submission = pd.read_csv(OUTPUT / 'submission_best_20191117.csv', float_precision='round_trip')
    alternate = pd.read_csv(OUTPUT / 'submission_log_id_20191117.csv', float_precision='round_trip')
    assert list(submission) == ['id', 'pctr'] and list(alternate) == ['log_id', 'pctr']
    assert len(submission) == len(raw) == 404182
    np.testing.assert_array_equal(submission.id, np.arange(len(raw)))
    np.testing.assert_array_equal(alternate.log_id, raw.log_id)
    np.testing.assert_array_equal(submission.pctr, prediction)
    np.testing.assert_array_equal(alternate.pctr, prediction)
    assert submission.pctr.between(0, 1).all() and np.isfinite(submission.pctr).all()
    labels = pd.read_csv(base.ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    auc16 = float(roc_auc_score(labels.loc[labels.pt_d.eq(20191116), 'label'],
                               np.load(RUN / 'day16_prediction.npy')))
    assert abs(auc16 - protocol['development16_auc']) < 1e-12
    check()
    report = dict(status='passed', timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  prediction_date=20191117, rows=len(raw), development16_auc=auc16,
                  development_dates_repeatedly_reused=True, selected=protocol['selected'],
                  inferred17_labels_read=False, platform_auc=None,
                  independent_raw_news_and_sparse_audience_rebuild=True,
                  compact_features_independent_column_rebuild=True,
                  unordered_features_independent_enumeration=True,
                  all_day16_model_margins_and_winning_prediction_replayed=True,
                  maximum_prediction_replay_error=error, original_row_order_verified=True,
                  csv_roundtrip_exact=True, repeated_log_ids_preserved=int(raw.log_id.duplicated().sum()),
                  original_datasets_unchanged=True, raw_datasets_sha256=protocol['raw_datasets_sha256'],
                  correction_models_training_max_date=20191115, inherited_base_training_max_date=20191116,
                  submission_sha256=digest(OUTPUT / 'submission_best_20191117.csv'))
    write(OUTPUT / 'verification.json', report)
    (OUTPUT / '提交说明.md').write_text(
        '提交文件：submission_best_20191117.csv\n'
        '格式：id,pctr；404182行，id为原始行序从0开始。\n'
        f'采用当前已完成且独立验证的最佳16号开发集方案，AUC={auc16:.12f}。\n'
        '15/16号曾反复用于开发；此分数不是17号测试成绩。\n'
        '未读取17号反推标签。使用已保存模型，新增修正模型训练截至15号，继承底层模型和历史特征截至16号。\n'
        '已核对独立特征重建、预测复算、原始行序、CSV精度和原始文件哈希。\n', encoding='utf8')
    emit('current_best17_verified', rows=len(raw), auc16=auc16, maximum_replay_error=error,
         submission=str(OUTPUT / 'submission_best_20191117.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['freeze', 'predict', 'verify'])
    {'freeze': freeze, 'predict': predict, 'verify': verify}[parser.parse_args().phase]()
