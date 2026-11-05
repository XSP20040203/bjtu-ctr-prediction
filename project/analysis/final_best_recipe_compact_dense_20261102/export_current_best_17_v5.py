"""Export the audited complete-negative/cohort winner without day17 labels."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'

import gc
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / 'analysis/day16_auc082_20261102'
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT / 'analysis/daily_interest_din'))
from conditional_daily import read, emit
from fit_stacker import digest, write
from rich_userday_activity import array_hash
from full_negative_context_inputs_v1 import schema
from full_quality_row_catboost import quality_rows
from catboost_userday_residual import load_model
from cross_user_pair_ranker_v1 import prediction as row_prediction
from quantile_blend_core_v1 import matched_logit, blend
from unordered_snapshot_core_v1 import FIELDS, build as snapshots
from prepare_exposure_features import SINGLE, build_exposure_features
from full_negative_mask_core_v1 import masks

RUN = HERE / 'current_best_17_run_v5'
OUTPUT = ROOT / 'output/current_best_20191117_20261103_v5'
PARENT = ROOT / 'output/current_best_20191117_20261103_v3'
CACHE = ROOT / 'analysis/final_best_recipe_quality_transfer_20261102/run_v2'
CONTEXT = EXP / 'runs/full_negative_context_v2'
WINNER = EXP / 'runs/compose_full_negative_masks_v1'
ANCHOR = EXP / 'runs/group_event_posterior_v1'
INPUT = EXP / 'runs/full_negative_context_v1/input_day16'
AUDIT = ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json'
MODEL_NAMES = ['day16_all_negative_d6_trees2400', 'day16_all_negative_d8_recent2_trees2400']


def freeze():
    assert not RUN.exists() and not OUTPUT.exists(), 'Preserve previous exports'
    audit = read(AUDIT)
    assert audit['audit_status'] == 'passed'
    verified = read(WINNER / 'verification.json')
    assert verified['status'] == 'passed'
    selected = verified['dates']['20191116']['best']
    assert selected['name'] == 'all_negative_d6' and selected['mask'] == 'cold_task'
    assert selected['mode'] == 'rank_global' and selected['weight'] == .2
    assert abs(selected['auc'] - audit['dates']['20191116']['auc']) < 1e-12
    assert digest(WINNER / 'day16_prediction.npy') == audit['dates']['20191116']['prediction_sha256']
    context_selected = read(CONTEXT / 'verification.json')['dates']['20191116']['best']
    assert context_selected['name'] == MODEL_NAMES[1]
    assert context_selected['mode'] == 'replace_global' and context_selected['weight'] == .05
    parent = read(PARENT / 'verification.json')
    assert parent['status'] == 'passed' and parent['inferred17_labels_read'] is False
    assert digest(PARENT / 'submission_best_20191117.csv') == parent['submission_sha256']
    parent_manifest = read(PARENT / 'prediction_manifest.json')
    assert digest(PARENT / 'day17_prediction.npy') == parent_manifest['files_sha256']['day17_prediction.npy']
    declared = schema(16)
    paths = [Path(__file__), PARENT / 'verification.json', PARENT / 'prediction_manifest.json',
             PARENT / 'day17_prediction.npy', CONTEXT / 'models_locked.json',
             INPUT / 'contract.json', EXP / 'runs/rich_augmented_rows_v1/input_contract16.json']
    for folder in (CONTEXT, WINNER):
        assert read(folder / 'verification.json')['status'] == 'passed'
        protocol = read(folder / 'protocol.json')
        assert protocol['inferred17_labels_read'] is False
        for filename, expected in protocol['sources_sha256'].items():
            assert digest(filename) == expected, filename
            if Path(filename).suffix == '.py':
                paths.append(Path(filename))
        paths += [folder / 'protocol.json', folder / 'verification.json', folder / 'day16_prediction.npy']
    records = [r for r in read(CONTEXT / 'models_locked.json')['models'] if r['name'] in MODEL_NAMES]
    assert len(records) == 2 and all(r['max_training_date'] == 20191115 for r in records)
    for record in records:
        path = CONTEXT / (record['name'] + '.cbm')
        assert digest(path) == record['model_sha256']
        paths += [path, CONTEXT / (record['name'] + '_held_margin.npy')]
    for phase in ('input_day16', 'input_day17', 'independent_day17'):
        folder = CACHE / phase
        contract = read(folder / 'contract.json')
        assert contract['own_query_ad_labels_masked']
        assert contract['max_ad_history_date'] < contract['day']
        assert digest(folder / 'features.npy') == contract['matrix_sha256']
        assert digest(folder / 'daily_quality.npz') == contract['quality_file_sha256']
        paths += [folder / name for name in ('contract.json', 'features.npy', 'daily_quality.npz')]
    for filename, expected in audit['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    RUN.mkdir()
    write(RUN / 'selection_audit_snapshot.json', audit)
    paths.append(RUN / 'selection_audit_snapshot.json')
    protocol = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), selected=selected,
        context_selected=context_selected, schema=declared, models=records,
        development16_auc=selected['auc'], development_dates_repeatedly_reused=True,
        selection_basis='Highest completed independently audited full day16 development AUC at export freeze',
        new_model_training=False, saved_correction_models_max_training_date=20191115,
        prediction_history_max_date=20191116, inherited_base=str(PARENT),
        inferred17_labels_read=False, raw_datasets_sha256=audit['raw_datasets_sha256'],
        sources_sha256={str(p): digest(p) for p in dict.fromkeys(paths)})
    write(RUN / 'protocol.json', protocol)
    emit('best17_v5_frozen', auc16=selected['auc'], selected=selected)


def query(day):
    filename = 'train.csv' if day == 16 else 'final_test.csv'
    columns = list(dict.fromkeys(FIELDS + SINGLE + ['log_id']))
    parts = []
    for block in pd.read_csv(ROOT / '2026ml' / filename, usecols=columns,
                             dtype={n: str for n in FIELDS if 'list' in n}, chunksize=100000):
        parts.append(block.loc[block.pt_d.eq(20191100 + day), columns])
    raw = pd.concat(parts, ignore_index=True)
    assert len(raw) == {16: 421983, 17: 404182}[day]
    return raw


def build_inputs(day, independent=False):
    declared = read(RUN / 'protocol.json')['schema']
    raw = query(day)
    phase = 'independent_day17' if independent else f'input_day{day}'
    folder = CACHE / phase
    cache_contract = read(folder / 'contract.json')
    assert array_hash(raw.log_id.to_numpy(dtype='int32')) == cache_contract['original_row_log_ids_sha256']
    native = read(EXP / 'runs/rich_augmented_rows_v1/input_contract16.json')['feature_names']
    quality_names = declared['quality_names']
    source = np.load(folder / 'features.npy', mmap_mode='r')
    assert source.shape == (len(raw), len(native))
    with np.load(folder / 'daily_quality.npz') as values:
        keys, quality = values['keys'], values['values']
    assert array_hash(quality) == cache_contract['quality_sha256']
    target = RUN / f'day{day}_{"independent" if independent else "primary"}_features.npy'
    assert not target.exists()
    matrix = np.lib.format.open_memmap(target, mode='w+', dtype='float32', shape=(len(raw), len(declared['columns'])))
    common = declared['common']
    for start in range(0, len(raw), 20000):
        stop = min(start + 20000, len(raw))
        aligned = quality_rows(keys, quality, raw.iloc[start:stop][['pt_d', 'user_id']].to_numpy())
        for name in common:
            matrix[start:stop, declared['columns'].index(name)] = (
                source[start:stop, native.index(name)] if name in native else aligned[:, quality_names.index(name)])
    del source, keys, quality, aligned
    gc.collect()
    for builder, expected in ((build_exposure_features, declared['exposure_names']), (snapshots, declared['columns'][-72:])):
        values, names = builder(raw) if builder is build_exposure_features else builder(raw, independent)
        assert names == expected
        for index, name in enumerate(names):
            matrix[:, declared['columns'].index(name)] = values[:, index]
        del values
        gc.collect()
    matrix.flush()
    if day == 16:
        original = np.load(EXP / 'runs/full_negative_context_v2/independent_day16/features.npy', mmap_mode='r')
        held = np.load(INPUT / 'held.npy')
        for start in range(0, len(raw), 10000):
            stop = min(start + 10000, len(raw))
            np.testing.assert_array_equal(matrix[start:stop], original[held[start:stop]])
    if independent:
        primary = np.load(RUN / 'day17_primary_features.npy', mmap_mode='r')
        for start in range(0, len(raw), 10000):
            np.testing.assert_array_equal(matrix[start:start+10000], primary[start:start+10000])
    emit('best17_v5_features_verified', day=day, independent=independent, rows=len(raw), columns=matrix.shape[1])
    return matrix, raw


def forecast(day, independent=False):
    protocol = read(RUN / 'protocol.json')
    matrix, raw = build_inputs(day, independent)
    names, cats = protocol['schema']['columns'], protocol['schema']['categories']
    margins = {}
    for record in protocol['models']:
        model = load_model(CONTEXT / (record['name'] + '.cbm'))
        assert model.feature_names_ == names
        margin = np.empty(len(raw), dtype='float64')
        for start in range(0, len(raw), 20000):
            stop = min(start + 20000, len(raw))
            frame = pd.DataFrame(np.array(matrix[start:stop]), columns=names)
            for name in cats:
                values = frame[name].to_numpy()
                np.testing.assert_array_equal(values, values.astype('int64'))
                frame[name] = values.astype('int64')
            assert np.isfinite(frame.to_numpy()).all()
            margin[start:stop] = model.predict(Pool(frame, cat_features=cats), prediction_type='RawFormulaVal', thread_count=2)
        if day == 16:
            np.testing.assert_array_equal(margin, np.load(CONTEXT / (record['name'] + '_held_margin.npy')))
        margins[record['name']] = margin
        del model
        emit('best17_v5_model_replayed', day=day, independent=independent, name=record['name'])
    anchor = np.load(ANCHOR / 'day16_prediction.npy' if day == 16 else PARENT / 'day17_prediction.npy')
    groups, _ = pd.factorize(raw.user_id.to_numpy(), sort=True)
    current = row_prediction(anchor, margins[MODEL_NAMES[1]], groups, .05, 'replace_global', independent)
    if day == 16:
        np.testing.assert_allclose(current, np.load(CONTEXT / 'day16_prediction.npy'), rtol=0, atol=2e-13)
    prior = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'task_id', 'user_id', 'label'], dtype='int32')
    earlier = prior.loc[prior.pt_d.lt(20191100 + day)]
    metadata = pd.concat([earlier[['pt_d', 'task_id', 'user_id']], raw[['pt_d', 'task_id', 'user_id']]], ignore_index=True)
    cohort = masks(metadata.pt_d.to_numpy(), metadata.user_id.to_numpy(), metadata.task_id.to_numpy(),
        np.arange(len(earlier)), earlier.label.to_numpy(), np.arange(len(earlier), len(metadata)), 20191100+day, independent)['cold_task']
    target = matched_logit(anchor, margins[MODEL_NAMES[0]], independent)
    adjusted = blend(anchor, target, np.ones(len(raw), dtype=bool), .2, independent)
    result = np.where(cohort, adjusted, current)
    assert np.isfinite(result).all() and np.all((result > 0) & (result < 1))
    if day == 16:
        np.testing.assert_allclose(result, np.load(WINNER / 'day16_prediction.npy'), rtol=0, atol=2e-13)
        auc = roc_auc_score(prior.loc[prior.pt_d.eq(20191116), 'label'], result)
        assert abs(auc - protocol['development16_auc']) < 1e-12
    emit('best17_v5_forecast_verified', day=day, independent=independent, cold_task_rows=int(cohort.sum()))
    return result, dict(rows=len(raw), feature_matrix_sha256=digest(Path(matrix.filename)),
        log_ids_sha256=array_hash(raw.log_id.to_numpy(dtype='int64')), cold_task_rows=int(cohort.sum()),
        cold_task_mask_sha256=array_hash(cohort), margins_sha256={k:array_hash(v) for k,v in margins.items()})


def main():
    freeze()
    protocol = read(RUN / 'protocol.json')
    development, dev_contract = forecast(16)
    np.save(RUN / 'day16_prediction.npy', development)
    prediction, contract = forecast(17)
    np.save(RUN / 'day17_prediction.npy', prediction)
    replay, independent = forecast(17, True)
    np.testing.assert_allclose(prediction, replay, rtol=0, atol=2e-13)
    assert contract == independent
    for filename, expected in protocol['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    OUTPUT.mkdir()
    np.save(OUTPUT / 'day17_prediction.npy', prediction)
    csv = OUTPUT / 'submission_best_20191117.csv'
    pd.DataFrame({'id':np.arange(len(prediction)), 'pctr':prediction}).to_csv(csv, index=False, float_format='%.17g')
    saved = pd.read_csv(csv, float_precision='round_trip')
    assert list(saved) == ['id','pctr'] and len(saved) == 404182
    np.testing.assert_array_equal(saved.id, np.arange(len(prediction)))
    np.testing.assert_array_equal(saved.pctr, prediction)
    write(RUN / 'input_contracts.json', {'20191116':dev_contract, '20191117':contract})
    report = dict(status='passed', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        prediction_date=20191117, rows=len(saved), columns=['id','pctr'],
        development16_auc=protocol['development16_auc'], selected=protocol['selected'],
        development_dates_repeatedly_reused=True, inferred17_labels_read=False, platform_auc=None,
        complete_day16_winner_features_models_and_prediction_replayed=True,
        independent_day17_features_models_masks_and_prediction_replayed=True,
        maximum_prediction_replay_error=float(np.max(np.abs(prediction-replay))),
        saved_correction_models_max_training_date=20191115, prediction_history_max_date=20191116,
        original_row_order_verified=True, csv_roundtrip_exact=True, original_datasets_unchanged=True,
        raw_datasets_sha256=protocol['raw_datasets_sha256'], submission_sha256=digest(csv))
    write(OUTPUT / 'prediction_manifest.json', dict(protocol_sha256=digest(RUN/'protocol.json'),
        input_contracts_sha256=digest(RUN/'input_contracts.json'), development16_auc=protocol['development16_auc'],
        inferred17_labels_read=False, files_sha256={p.name:digest(p) for p in OUTPUT.iterdir() if p.is_file()}))
    write(OUTPUT / 'verification.json', report)
    (OUTPUT/'提交说明.md').write_text('提交文件：submission_best_20191117.csv\n'
        '格式：id,pctr；404182行；id按原始行序从0开始。\n'
        f'使用导出冻结时已完成独立核验的最优方案，16号开发集AUC={protocol["development16_auc"]:.12f}。\n'
        '15/16号反复用于开发；该AUC不是17号平台成绩。未读取17号反推标签。\n'
        '使用已保存的最优模型：全量负样本CatBoost融合及冷广告分群修正。\n'
        '修正模型训练截至15号，底层预测和历史特征截至16号。\n'
        '16号特征及最优预测已复算；17号特征、模型、分群、行序及CSV精度已核验。\n', encoding='utf8')
    emit('best17_v5_submission_verified', path=str(csv), rows=len(saved), auc16=protocol['development16_auc'], sha256=digest(csv))


if __name__ == '__main__':
    main()
