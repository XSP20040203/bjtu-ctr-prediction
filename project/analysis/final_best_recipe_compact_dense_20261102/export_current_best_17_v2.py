"""Export the verified family-quantile winner without reading inferred targets."""
from __future__ import annotations

import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'

import argparse
import gc
import hashlib
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoost, Pool
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / 'analysis/day16_auc082_20261102'
sys.path.insert(0, str(EXP))
from conditional_daily import read, emit
from fit_stacker import digest, write
from rich_userday_activity import array_hash
from full_quality_row_catboost import full_frame, quality_rows
from cold_task_core import identity_columns
from unordered_snapshot_core_v1 import FIELDS, METRICS, build
from transition_posterior_core_v1 import FIELDS as BIN_FIELDS, task_bins, predict as posterior
from quantile_blend_core_v1 import matched_logit, blend

RUN = HERE / 'current_best_17_run_v2'
OUTPUT = ROOT / 'output/current_best_20191117_20261103_v2'
PARENT = ROOT / 'output/current_best_20191117_20261103'
PARENT_RUN = HERE / 'current_best_17_run_v1'
COLD = EXP / 'runs/cold_task_ranker_v1'
TRANSITION = EXP / 'runs/compose_transition_priors_v1'
FAMILY = EXP / 'runs/compose_family_quantiles_v1'
POINTWISE = EXP / 'runs/pointwise_compact_v1'
RATES = EXP / 'runs/transition_posterior_v1/rates_locked.json'
NATIVE = EXP / 'runs/rich_augmented_rows_v1/input_contract16.json'
CACHE = ROOT / 'analysis/final_best_recipe_quality_transfer_20261102/run_v2'


def freeze():
    assert not RUN.exists() and not OUTPUT.exists(), 'Preserve previous exports'
    audit = read(ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json')
    assert audit['audit_status'] == 'passed'
    selected = {}
    paths = [Path(__file__), NATIVE, RATES, PARENT_RUN / 'protocol.json',
             PARENT_RUN / 'input_contracts.json', HERE / 'cold_task_run_v1/input_contracts.json',
             PARENT / 'verification.json', PARENT / 'prediction_manifest.json',
             PARENT_RUN / 'day16_prediction.npy', PARENT / 'day17_prediction.npy',
             COLD / 'models_locked.json', COLD / 'input_contract16.json', COLD / 'held_contract16.json']
    for folder in (TRANSITION, FAMILY, POINTWISE, COLD):
        report = read(folder / 'verification.json')
        assert report['status'] == 'passed'
        if folder != COLD:
            selected[folder.name] = report['dates']['20191116']['best']
        paths += [folder / 'protocol.json', folder / 'verification.json', folder / 'day16_prediction.npy']
        for filename, expected in read(folder / 'protocol.json')['sources_sha256'].items():
            if Path(filename).suffix == '.py':
                assert digest(filename) == expected, filename
                paths.append(Path(filename))
    assert selected['pointwise_compact_v1']['name'] == 'base'
    assert selected['pointwise_compact_v1']['weight'] == 0
    assert selected['compose_transition_priors_v1']['model'] == 'recent_shrink5'
    assert selected['compose_transition_priors_v1']['threshold'] == 1
    assert selected['compose_transition_priors_v1']['weight'] == .25
    winner = selected['compose_family_quantiles_v1']
    assert winner['name'] == 'cold_task_ranker_v1__late_mean'
    assert winner['mask'] == 'changed_click' and winner['weight'] == .5
    assert abs(winner['auc'] - audit['dates']['20191116']['auc']) < 1e-12
    assert digest(FAMILY / 'day16_prediction.npy') == audit['dates']['20191116']['prediction_sha256']
    parent = read(PARENT / 'verification.json')
    assert parent['status'] == 'passed' and parent['inferred17_labels_read'] is False
    manifest = read(PARENT / 'prediction_manifest.json')
    assert digest(PARENT / 'day17_prediction.npy') == manifest['files_sha256']['day17_prediction.npy']
    np.testing.assert_allclose(np.load(PARENT_RUN / 'day16_prediction.npy'),
                               np.load(POINTWISE / 'day16_prediction.npy'), rtol=0, atol=2e-14)
    records = [r for r in read(COLD / 'models_locked.json')['models'] if r['day'] == 16]
    cutoff = sorted({r['trees'] for r in records})[-2]
    records = [r for r in records if r['trees'] >= cutoff]
    assert len(records) == 4 and read(COLD / 'input_contract16.json')['max_training_date'] == 20191115
    for record in records:
        path = COLD / (record['name'] + '.cbm')
        assert digest(path) == record['model_sha256']
        paths += [path, COLD / (record['name'] + '_held_margin.npy')]
    for phase in ('input_day16', 'input_day17', 'independent_day17'):
        folder = CACHE / phase
        contract = read(folder / 'contract.json')
        assert contract['own_query_ad_labels_masked']
        assert contract['max_ad_history_date'] < 20191100 + int(phase[-2:])
        assert digest(folder / 'features.npy') == contract['matrix_sha256']
        paths += [folder / name for name in ('features.npy', 'daily_quality.npz', 'contract.json')]
    for filename, expected in audit['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    RUN.mkdir()
    write(RUN / 'selection_audit_snapshot.json', audit)
    paths.append(RUN / 'selection_audit_snapshot.json')
    paths += [EXP / name for name in ('quantile_blend_core_v1.py', 'transition_posterior_core_v1.py',
              'unordered_snapshot_core_v1.py', 'cold_task_core.py', 'full_quality_row_catboost.py')]
    write(RUN / 'protocol.json', dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(), selected=selected, models=records,
        development16_auc=winner['auc'], development_dates_repeatedly_reused=True,
        selection_basis='Highest completed independently audited full-cohort day16 recipe at export freeze',
        inherited_base=str(PARENT), new_model_training=False,
        correction_models_and_posterior_rates_training_max_date=20191115,
        inherited_base_and_prediction_history_max_date=20191116,
        transfer='Apply the saved day16 correction models and frozen day16 posterior rates to day17',
        inferred17_labels_read=False, day17_advertisement_labels_used=False,
        raw_datasets_sha256=audit['raw_datasets_sha256'],
        sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)}))
    emit('best17_v2_frozen', auc16=winner['auc'], models=[r['name'] for r in records])


def check():
    protocol = read(RUN / 'protocol.json')
    for filename, expected in protocol['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    return protocol


def query(day):
    filename = 'train.csv' if day == 16 else 'final_test.csv'
    parts = []
    for block in pd.read_csv(ROOT / '2026ml' / filename, usecols=FIELDS + ['log_id'],
                             dtype={name: str for name in FIELDS if 'list' in name}, chunksize=100000):
        parts.append(block.loc[block.pt_d.eq(20191100 + day)])
    raw = pd.concat(parts, ignore_index=True)
    assert len(raw) == {16: 421983, 17: 404182}[day]
    return raw


def load_model(path):
    model = CatBoost()
    with tempfile.TemporaryDirectory(prefix='best17_cb_') as folder:
        temporary = Path(folder) / 'model.cbm'
        temporary.write_bytes(path.read_bytes())
        model.load_model(str(temporary))
    return model


def margins(day, raw, independent=False):
    protocol = read(RUN / 'protocol.json')
    phase = 'independent_day17' if independent else f'input_day{day}'
    folder = CACHE / phase
    native = read(NATIVE)
    fitted = read(COLD / 'input_contract16.json')
    masked, _ = identity_columns(native['feature_names'])
    matrix = np.load(folder / 'features.npy', mmap_mode='r')
    assert matrix.shape == (len(raw), 744)
    with np.load(folder / 'daily_quality.npz') as source:
        keys, values = source['keys'], source['values']
    models = {r['name']: load_model(COLD / (r['name'] + '.cbm')) for r in protocol['models']}
    outputs = {name: np.empty(len(raw), dtype='float64') for name in models}
    frame_digest = hashlib.sha256()
    keep = [i for i, name in enumerate(native['feature_names']) if not name.startswith('rich_daily__quality__')]
    categories = [f'f{keep.index(i)}' for i in native['categorical_columns']]
    assert categories == fitted['categories']
    for start in range(0, len(raw), 20000):
        stop = min(len(raw), start + 20000)
        block = raw.iloc[start:stop]
        quality = quality_rows(keys, values, block[['pt_d', 'user_id']].to_numpy())
        if independent:
            # Rebuild each column directly from independently reconstructed source caches.
            columns = {f'f{j}': np.asarray(matrix[start:stop, i], dtype='float32') for j, i in enumerate(keep)}
            columns.update({f'f{len(keep) + j}': quality[:, j] for j in range(337)})
            frame = pd.DataFrame(columns)
            for name in categories:
                frame[name] = frame[name].to_numpy(dtype='int64')
        else:
            frame, cats, _ = full_frame(matrix, np.arange(start, stop), native['feature_names'],
                                        native['categorical_columns'], quality)
            assert cats == categories
        for column in masked:
            frame[f'f{column}'] = 0
        assert frame.shape == (stop - start, 1017)
        frame_digest.update(np.ascontiguousarray(frame.to_numpy(dtype='float32')).tobytes())
        pool = Pool(frame, cat_features=categories)
        for name, model in models.items():
            assert model.feature_names_ == list(frame)
            outputs[name][start:stop] = model.predict(pool, prediction_type='RawFormulaVal', thread_count=2)
        if start % 100000 == 0:
            emit('best17_v2_chunk', day=day, independent=independent, rows=stop, total=len(raw))
        del frame, pool, quality
    frame_sha = frame_digest.hexdigest()
    expected = read(HERE / 'cold_task_run_v1/input_contracts.json')[str(20191100 + day)]['full_frame_sha256']
    assert frame_sha == expected, (day, frame_sha, expected)
    if day == 16:
        for name, margin in outputs.items():
            np.testing.assert_array_equal(margin, np.load(COLD / (name + '_held_margin.npy')))
    del matrix, values, keys, models
    gc.collect()
    return outputs, frame_sha


def forecast(day, independent=False):
    protocol = read(RUN / 'protocol.json')
    raw = query(day)
    base = np.load(POINTWISE / 'day16_prediction.npy' if day == 16 else PARENT / 'day17_prediction.npy')
    assert len(raw) == len(base)
    bins = task_bins(raw[BIN_FIELDS], independent)
    transition = protocol['selected']['compose_transition_priors_v1']
    rates = read(RATES)['rates']['16'][transition['model']]
    assert rates['max_training_date'] == 20191115
    current = posterior(base, bins, rates, transition['threshold'], transition['weight'], independent)
    if day == 16:
        np.testing.assert_allclose(current, np.load(TRANSITION / 'day16_prediction.npy'), rtol=0, atol=2e-14)
    emit('best17_v2_states_start', day=day, independent=independent)
    states, _ = build(raw[FIELDS], independent)
    mask = np.any(states[:, :3 * len(METRICS)] != 0, axis=1)
    inherited = read(PARENT_RUN / 'input_contracts.json')[str(20191100 + day)]
    assert array_hash(mask) == inherited['mask_sha256'] and int(mask.sum()) == inherited['mask_count']
    del states
    gc.collect()
    model_margins, frame_sha = margins(day, raw, independent)
    target = np.mean([matched_logit(current, model_margins[r['name']], independent)
                      for r in protocol['models']], axis=0)
    result = blend(current, target, mask, protocol['selected']['compose_family_quantiles_v1']['weight'], independent)
    assert np.isfinite(result).all() and np.all((result > 0) & (result < 1))
    if day == 16:
        np.testing.assert_allclose(result, np.load(FAMILY / 'day16_prediction.npy'), rtol=0, atol=2e-13)
    contract = dict(day=20191100 + day, rows=len(raw), frame_sha256=frame_sha,
                    log_ids_sha256=array_hash(raw.log_id.to_numpy(dtype='int64')),
                    mask_sha256=array_hash(mask), mask_count=int(mask.sum()), bins_sha256=array_hash(bins),
                    margins_sha256={name: array_hash(value) for name, value in model_margins.items()},
                    inferred17_labels_read=False)
    emit('best17_v2_forecast_complete', day=day, independent=independent)
    return result, contract, raw.log_id.to_numpy()


def predict():
    protocol = check()
    assert not OUTPUT.exists()
    contracts = {}
    for day in (16, 17):
        prediction, contract, log_ids = forecast(day)
        np.save(RUN / f'day{day}_prediction.npy', prediction)
        contracts[str(20191100 + day)] = contract
        write(RUN / 'input_contracts.json', contracts)
    OUTPUT.mkdir()
    pd.DataFrame({'id': np.arange(len(prediction)), 'pctr': prediction}).to_csv(
        OUTPUT / 'submission_best_20191117.csv', index=False, float_format='%.17g')
    np.save(OUTPUT / 'day17_prediction.npy', prediction)
    np.save(RUN / 'day17_log_ids.npy', log_ids)
    write(OUTPUT / 'prediction_manifest.json', dict(protocol_sha256=digest(RUN / 'protocol.json'),
        input_contracts_sha256=digest(RUN / 'input_contracts.json'),
        development16_auc=protocol['development16_auc'], inferred17_labels_read=False,
        files_sha256={path.name: digest(path) for path in OUTPUT.iterdir() if path.is_file()}))
    emit('best17_v2_submission_locked', path=str(OUTPUT / 'submission_best_20191117.csv'))


def verify():
    protocol = check()
    assert not (OUTPUT / 'verification.json').exists()
    manifest = read(OUTPUT / 'prediction_manifest.json')
    assert manifest['protocol_sha256'] == digest(RUN / 'protocol.json')
    assert manifest['input_contracts_sha256'] == digest(RUN / 'input_contracts.json')
    for name, expected in manifest['files_sha256'].items():
        assert digest(OUTPUT / name) == expected, name
    replay, contract, log_ids = forecast(17, True)
    assert contract == read(RUN / 'input_contracts.json')['20191117']
    np.testing.assert_array_equal(log_ids, np.load(RUN / 'day17_log_ids.npy'))
    prediction = np.load(OUTPUT / 'day17_prediction.npy')
    maximum = float(np.max(np.abs(prediction - replay)))
    assert maximum < 2e-13
    submission = pd.read_csv(OUTPUT / 'submission_best_20191117.csv', float_precision='round_trip')
    assert list(submission) == ['id', 'pctr'] and len(submission) == 404182
    np.testing.assert_array_equal(submission.id, np.arange(len(submission)))
    np.testing.assert_array_equal(submission.pctr, prediction)
    assert np.isfinite(submission.pctr).all() and submission.pctr.between(0, 1).all()
    labels = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
    auc16 = float(roc_auc_score(labels.loc[labels.pt_d.eq(20191116), 'label'], np.load(RUN / 'day16_prediction.npy')))
    assert abs(auc16 - protocol['development16_auc']) < 1e-12
    check()
    report = dict(status='passed', timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  prediction_date=20191117, rows=len(prediction), columns=['id', 'pctr'],
                  development16_auc=auc16, development_dates_repeatedly_reused=True,
                  selected=protocol['selected'], inferred17_labels_read=False, platform_auc=None,
                  all_four_day16_model_margins_replayed_exactly=True,
                  complete_day16_winner_replayed=True, independent_day17_source_caches_and_columns_rebuilt=True,
                  independent_day17_bins_snapshot_and_rank_mapping_rebuilt=True,
                  maximum_prediction_replay_error=maximum, original_row_order_verified=True,
                  repeated_log_ids_preserved=int(pd.Series(log_ids).duplicated().sum()), csv_roundtrip_exact=True,
                  original_datasets_unchanged=True, raw_datasets_sha256=protocol['raw_datasets_sha256'],
                  submission_sha256=digest(OUTPUT / 'submission_best_20191117.csv'))
    write(OUTPUT / 'verification.json', report)
    (OUTPUT / '提交说明.md').write_text(
        '提交文件：submission_best_20191117.csv\n格式：id,pctr；404182行；id为原始行序，从0开始。\n'
        f'依据当前已完成且独立验证的16号开发集最佳融合方案导出，AUC={auc16:.12f}。\n'
        '15/16号反复用于开发，该AUC不是17号成绩。未读取17号反推标签。\n'
        '沿用已保存模型和后验参数，修正模型及后验参数训练截至15号，继承底层模型和历史特征截至16号。\n'
        '已验证模型预测复算、独立缓存和特征重建、原始行序、CSV精度及原始四份CSV哈希。\n', encoding='utf8')
    emit('best17_v2_verified', auc16=auc16, rows=len(prediction), maximum_replay_error=maximum,
         path=str(OUTPUT / 'submission_best_20191117.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['freeze', 'predict', 'verify', 'all'], required=True)
    phase = parser.parse_args().phase
    if phase == 'all':
        freeze()
        predict()
        verify()
    else:
        {'freeze': freeze, 'predict': predict, 'verify': verify}[phase]()
