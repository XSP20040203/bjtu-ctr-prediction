"""Export the audited semantic/cohort winner without any inferred day17 labels."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'

import gc
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import Pool
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / 'analysis/day16_auc082_20261102'
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT / 'analysis/model_comparison'))
from conditional_daily import read, emit
from fit_stacker import digest, write
from rich_userday_activity import array_hash
from catboost_userday_residual import load_model
from cross_user_pair_ranker_v1 import prediction as row_prediction
from full_negative_mask_core_v1 import masks
from frozen_neural_embedding_core_v1 import FIELDS, SEQUENCES, NAMES, encode, design

RUN = HERE / 'current_best_17_run_v9'
OUTPUT = ROOT / 'output/current_best_20191117_20261103_v9'
PARENT = ROOT / 'output/current_best_20191117_20261103_v8'
PARENT_RUN = HERE / 'current_best_17_run_v8'
CONTEXT_RUN = HERE / 'current_best_17_run_v5'
NEURAL_RUN = HERE / 'current_best_17_run_v6'
FULL = EXP / 'runs/full_negative_neural_embeddings_v2'
SEMANTIC = EXP / 'runs/full_negative_semantic_embeddings_v1'
WINNER = EXP / 'runs/compose_semantic_embedding_masks_v1'
BASE = EXP / 'runs/refresh_family_quantiles_v1'
AUDIT = ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json'
VOCABS = ROOT / 'analysis/auc085/cache_rich_stats_time16/vocabs.json'
MODEL = 'day16_semantic_d6_trees2400'
BATCH = 10000


def context_path(day, independent=False):
    phase = 'independent' if independent else 'primary'
    return CONTEXT_RUN / f'day{day}_{phase}_features.npy'


def freeze():
    assert not RUN.exists() and not OUTPUT.exists(), 'Preserve previous exports and failures'
    audit = read(AUDIT)
    assert audit['audit_status'] == 'passed' and audit['original_datasets_unchanged']
    report = read(WINNER / 'verification.json')
    assert report['status'] == 'passed'
    selected = report['dates']['20191116']['best']
    assert {k: v for k, v in selected.items() if k != 'auc'} == dict(
        name='semantic_d6', mask='user_no_past_ad_click', mode='add_within', weight=.1)
    assert abs(selected['auc'] - audit['dates']['20191116']['auc']) < 1e-12
    assert digest(WINNER / 'day16_prediction.npy') == audit['dates']['20191116']['prediction_sha256']
    np.testing.assert_array_equal(np.load(PARENT_RUN / 'day16_prediction.npy'),
                                  np.load(BASE / 'day16_prediction.npy'))
    parent = read(PARENT / 'verification.json')
    manifest = read(PARENT / 'prediction_manifest.json')
    assert parent['status'] == 'passed' and parent['inferred17_labels_read'] is False
    assert digest(PARENT_RUN / 'protocol.json') == manifest['protocol_sha256']
    assert digest(PARENT_RUN / 'input_contracts.json') == manifest['input_contracts_sha256']
    for name, expected in manifest['files_sha256'].items():
        assert digest(PARENT / name) == expected, name
    assert digest(PARENT / 'submission_best_20191117.csv') == parent['submission_sha256']
    paths = [Path(__file__), AUDIT, VOCABS]
    paths += [EXP / name for name in ('frozen_neural_embedding_core_v1.py',
        'semantic_embedding_inputs_v1.py', 'full_negative_mask_core_v1.py',
        'cross_user_pair_ranker_v1.py', 'within_user_ranking_core.py',
        'catboost_userday_residual.py', 'conditional_daily.py', 'fit_stacker.py',
        'rich_userday_activity.py', 'benchmark_core.py') if (EXP / name).exists()]
    for folder in (WINNER, SEMANTIC, FULL, BASE):
        verification = read(folder / 'verification.json')
        assert verification['status'] == 'passed'
        assert verification['dates']['20191116']['evaluation_day_labels_fit'] is False
        protocol = read(folder / 'protocol.json')
        assert protocol['inferred17_labels_read'] is False
        assert protocol['raw_datasets_sha256'] == audit['raw_datasets_sha256']
        for filename, expected in protocol['sources_sha256'].items():
            if Path(filename).suffix == '.py':
                assert digest(filename) == expected, filename
                paths.append(Path(filename))
        paths += [folder / name for name in ('protocol.json', 'verification.json', 'day16_prediction.npy')]
    model_lock = read(SEMANTIC / 'models_locked.json')
    assert model_lock['protocol_sha256'] == digest(SEMANTIC / 'protocol.json')
    assert model_lock['inputs_locked_sha256'] == digest(SEMANTIC / 'inputs_locked.json')
    record = next(value for value in model_lock['models'] if value['name'] == MODEL)
    assert record['day'] == 16 and record['trees'] == 2400 and record['max_training_date'] == 20191115
    assert digest(SEMANTIC / (MODEL + '.cbm')) == record['model_sha256']
    schema = read(SEMANTIC / 'protocol.json')['feature_designs']['16']
    source = read(FULL / 'input_day16/contract.json')
    teacher = read(FULL / 'neural_teacher_day16.json')
    assert read(SEMANTIC / 'inputs_locked.json')['16'] == source
    assert source['teacher'] == teacher
    assert teacher['held_day_labels_fit'] is False and teacher['inferred17_labels_read'] is False
    assert digest(teacher['checkpoint']) == teacher['checkpoint_sha256']
    assert digest(FULL / 'input_day16/features.npy') == source['matrix_sha256']
    neural_bindings = read(NEURAL_RUN / 'protocol.json')['sources_sha256']
    for path in (VOCABS, FULL / 'neural_tables_day16.npz', FULL / 'neural_teacher_day16.json'):
        assert digest(path) == neural_bindings[str(path)], path
    contracts = read(NEURAL_RUN / 'input_contracts.json')
    for day, independent in ((16, False), (17, False), (17, True)):
        path = context_path(day, independent)
        assert digest(path) == contracts[str(20191100 + day)]['context_features_sha256']
        assert digest(path) == neural_bindings[str(path)]
        paths.append(path)
    paths += [PARENT / name for name in ('verification.json', 'prediction_manifest.json', 'day17_prediction.npy')]
    paths += [PARENT_RUN / name for name in ('protocol.json', 'input_contracts.json', 'day16_prediction.npy')]
    paths += [NEURAL_RUN / name for name in ('protocol.json', 'input_contracts.json')]
    paths += [SEMANTIC / name for name in ('models_locked.json', 'inputs_locked.json',
        'full_projection_verification.json', MODEL + '.cbm', MODEL + '_held_margin.npy')]
    paths += [FULL / name for name in ('neural_tables_day16.npz', 'neural_teacher_day16.json',
        'input_day16/contract.json', 'input_day16/features.npy', 'input_day16/held.npy')]
    paths += [WINNER / name for name in ('inputs_locked.json', 'predictions_locked.json', 'day16_indices.npy')]
    paths.append(Path(teacher['checkpoint']))
    for filename, expected in audit['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    RUN.mkdir()
    write(RUN / 'selection_audit_snapshot.json', audit)
    paths.append(RUN / 'selection_audit_snapshot.json')
    write(RUN / 'protocol.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        selected=selected, model=record, schema=schema, development16_auc=audit['dates']['20191116']['auc'],
        development_dates_repeatedly_reused=True,
        selection_basis='Highest completed independently audited full-day16 AUC at export freeze',
        new_model_training=False, saved_correction_model_max_training_date=20191115,
        prediction_history_max_date=20191116, inherited_base=str(PARENT),
        independent_replay='Independent cached context, raw query encoding, NumPy/Torch embeddings, column gather, strict-past cohorts and within-user arithmetic',
        inferred17_labels_read=False, raw_datasets_sha256=audit['raw_datasets_sha256'],
        sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)}))
    emit('best17_v9_frozen', auc16=audit['dates']['20191116']['auc'], selected=selected)


def raw_query(day):
    filename = 'train.csv' if day == 16 else 'final_test.csv'
    columns = list(dict.fromkeys(FIELDS + ['log_id']))
    types = {name: 'str' if name in SEQUENCES else 'int32' for name in FIELDS}
    types['log_id'] = 'int64'
    pieces = []
    for block in pd.read_csv(ROOT / '2026ml' / filename, usecols=columns, dtype=types, chunksize=60000):
        pieces.append(block.loc[block.pt_d.eq(20191100 + day), columns])
    raw = pd.concat(pieces, ignore_index=True)
    assert len(raw) == {16: 421983, 17: 404182}[day]
    expected = read(NEURAL_RUN / 'input_contracts.json')[str(20191100 + day)]
    assert array_hash(raw.log_id.to_numpy(dtype='int64')) == expected['log_ids_sha256']
    return raw


def neural_features(raw, day, independent=False):
    teacher = read(FULL / 'neural_teacher_day16.json')
    vocabulary = read(VOCABS)
    with np.load(FULL / 'neural_tables_day16.npz') as saved:
        tables = {name: saved[name] for name in saved.files}
    for name, value in tables.items():
        assert array_hash(value) == teacher['tables_sha256'][name]
    phase = 'independent' if independent else 'primary'
    path = RUN / f'day{day}_{phase}_neural.npy'
    assert not path.exists()
    result = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=(len(raw), len(NAMES)))
    reference = np.load(FULL / 'input_day16/features.npy' if day == 16 else
                        NEURAL_RUN / 'day17_primary_neural.npy', mmap_mode='r')
    held = np.load(FULL / 'input_day16/held.npy') if day == 16 else None
    for start in range(0, len(raw), BATCH):
        stop = min(start + BATCH, len(raw))
        encoded = encode(raw.iloc[start:stop][FIELDS], teacher['meta'], vocabulary,
                         20191100 + day, independent)
        values = design(encoded, tables, teacher['meta'], independent)
        np.testing.assert_array_equal(values, reference[held[start:stop]] if day == 16 else reference[start:stop])
        result[start:stop] = values
    result.flush()
    expected = read(NEURAL_RUN / 'input_contracts.json')[str(20191100 + day)]['neural_features_sha256']
    assert digest(path) == expected
    emit('best17_v9_neural_verified', day=day, independent=independent, rows=len(raw))
    return result


def forecast(day, independent=False):
    protocol = read(RUN / 'protocol.json')
    raw = raw_query(day)
    neural = neural_features(raw, day, independent)
    context = np.load(context_path(day, independent), mmap_mode='r')
    assert context.shape == (len(raw), 395)
    names, categories = protocol['schema']['columns'], protocol['schema']['categories']
    neural_columns = [NAMES.index(name) for name in names[395:]]
    assert len(names) == 489 and len(neural_columns) == 94
    model = load_model(SEMANTIC / (MODEL + '.cbm'))
    assert model.feature_names_ == names
    margin = np.empty(len(raw), dtype='float64')
    frame_hash = hashlib.sha256()
    for start in range(0, len(raw), BATCH):
        stop = min(start + BATCH, len(raw))
        if independent:
            data = {name: np.asarray(context[start:stop, index] if index < 395 else
                neural[start:stop, neural_columns[index - 395]], dtype='float32')
                for index, name in enumerate(names)}
            frame = pd.DataFrame(data, copy=False)
        else:
            values = np.column_stack((context[start:stop], neural[start:stop][:, neural_columns]))
            frame = pd.DataFrame(values, columns=names, copy=False)
        for name in categories:
            numbers = frame[name].to_numpy()
            integers = numbers.astype('int64')
            np.testing.assert_array_equal(numbers, integers)
            frame[name] = integers
        gathered = frame.to_numpy(dtype='float32')
        assert np.isfinite(gathered).all()
        frame_hash.update(gathered.tobytes(order='C'))
        margin[start:stop] = model.predict(Pool(frame, cat_features=categories),
            prediction_type='RawFormulaVal', thread_count=2)
        if start % 100000 == 0:
            emit('best17_v9_batch', day=day, independent=independent, rows=stop, total=len(raw))
    del model, context, neural, frame, gathered
    gc.collect()
    if day == 16:
        np.testing.assert_array_equal(margin, np.load(SEMANTIC / (MODEL + '_held_margin.npy')))
        projection = read(SEMANTIC / 'full_projection_verification.json')['dates']['16']['held']
        assert frame_hash.hexdigest() == projection['row_major_float32_sha256']
    prior = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'user_id', 'task_id', 'label'], dtype='int32')
    earlier = prior.loc[prior.pt_d.lt(20191100 + day)]
    assert int(earlier.pt_d.max()) == 20191100 + day - 1
    metadata = pd.concat((earlier[['pt_d', 'user_id', 'task_id']], raw[['pt_d', 'user_id', 'task_id']]), ignore_index=True)
    cohort = masks(metadata.pt_d.to_numpy(), metadata.user_id.to_numpy(), metadata.task_id.to_numpy(),
        np.arange(len(earlier)), earlier.label.to_numpy(), np.arange(len(earlier), len(metadata)),
        20191100 + day, independent)
    mask = cohort[protocol['selected']['mask']]
    if independent:
        _, groups = np.unique(raw.user_id.to_numpy(), return_inverse=True)
    else:
        groups, _ = pd.factorize(raw.user_id.to_numpy(), sort=True)
    base = np.load(PARENT_RUN / 'day16_prediction.npy' if day == 16 else PARENT / 'day17_prediction.npy')
    adjusted = row_prediction(base, margin, groups, protocol['selected']['weight'],
                             protocol['selected']['mode'], independent)
    prediction = np.where(mask, adjusted, base)
    np.testing.assert_array_equal(prediction[~mask], base[~mask])
    assert np.isfinite(prediction).all() and np.all((prediction > 0) & (prediction < 1))
    if day == 16:
        inputs = read(WINNER / 'inputs_locked.json')['16']
        assert array_hash(mask) == inputs['masks_sha256']['user_no_past_ad_click']
        assert array_hash(groups) == inputs['groups_sha256']
        assert array_hash(margin) == inputs['margins_sha256']['semantic_d6']
        np.testing.assert_allclose(prediction, np.load(WINNER / 'day16_prediction.npy'), rtol=0, atol=2e-13)
        auc = roc_auc_score(prior.loc[prior.pt_d.eq(20191116), 'label'], prediction)
        assert abs(auc - protocol['development16_auc']) < 1e-12
    phase = 'independent' if independent else 'primary'
    np.save(RUN / f'day{day}_{phase}_margin.npy', margin)
    contract = dict(rows=len(raw), columns=489,
        log_ids_sha256=array_hash(raw.log_id.to_numpy(dtype='int64')),
        context_features_sha256=digest(context_path(day, independent)),
        neural_features_sha256=digest(RUN / f'day{day}_{phase}_neural.npy'),
        row_major_float32_frame_sha256=frame_hash.hexdigest(), margin_sha256=array_hash(margin),
        groups_sha256=array_hash(groups), selected_mask_sha256=array_hash(mask),
        selected_mask_rows=int(mask.sum()), mask_history_max_date=int(earlier.pt_d.max()),
        inferred17_labels_read=False)
    emit('best17_v9_forecast_verified', day=day, independent=independent, rows=len(raw), cohort_rows=int(mask.sum()))
    return prediction, contract


def main():
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    freeze()
    protocol = read(RUN / 'protocol.json')
    development, development_contract = forecast(16)
    np.save(RUN / 'day16_prediction.npy', development)
    prediction, contract = forecast(17)
    np.save(RUN / 'day17_prediction.npy', prediction)
    replay, independent = forecast(17, True)
    assert contract == independent
    np.testing.assert_allclose(prediction, replay, rtol=0, atol=2e-13)
    # Check frozen inputs again before publishing the completed artifact.
    for filename, expected in protocol['sources_sha256'].items():
        assert digest(filename) == expected, filename
    for filename, expected in protocol['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    OUTPUT.mkdir()
    np.save(OUTPUT / 'day17_prediction.npy', prediction)
    csv = OUTPUT / 'submission_best_20191117.csv'
    pd.DataFrame({'id': np.arange(len(prediction)), 'pctr': prediction}).to_csv(csv, index=False, float_format='%.17g')
    saved = pd.read_csv(csv, float_precision='round_trip')
    assert list(saved) == ['id', 'pctr'] and len(saved) == 404182
    np.testing.assert_array_equal(saved.id.to_numpy(), np.arange(len(prediction)))
    np.testing.assert_array_equal(saved.pctr.to_numpy(), prediction)
    write(RUN / 'input_contracts.json', {'20191116': development_contract, '20191117': contract})
    report = dict(status='passed', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        prediction_date=20191117, rows=len(saved), columns=['id', 'pctr'],
        development16_auc=protocol['development16_auc'], selected=protocol['selected'],
        development_dates_repeatedly_reused=True, inferred17_labels_read=False, platform_auc=None,
        complete_day16_selected_model_features_mask_and_prediction_replayed=True,
        independent_day17_neural_encoding_features_model_cohort_and_prediction_replayed=True,
        independently_verified_parent_context_and_baseline_reused=True,
        maximum_prediction_replay_error=float(np.max(np.abs(prediction - replay))),
        saved_correction_model_max_training_date=20191115, prediction_history_max_date=20191116,
        original_row_order_verified=True, csv_roundtrip_exact=True, original_datasets_unchanged=True,
        raw_datasets_sha256=protocol['raw_datasets_sha256'], submission_sha256=digest(csv))
    write(OUTPUT / 'prediction_manifest.json', dict(protocol_sha256=digest(RUN / 'protocol.json'),
        input_contracts_sha256=digest(RUN / 'input_contracts.json'),
        development16_auc=protocol['development16_auc'], inferred17_labels_read=False,
        files_sha256={path.name: digest(path) for path in OUTPUT.iterdir() if path.is_file()}))
    write(OUTPUT / 'verification.json', report)
    (OUTPUT / '提交说明.md').write_text('提交文件：submission_best_20191117.csv\n'
        '格式：id,pctr；404182行；id按原始行序从0开始。\n'
        f'本次冻结时已完成独立核验的最优方案，16号开发集AUC={protocol["development16_auc"]:.12f}。\n'
        '15/16号反复用于开发；该AUC不是17号平台成绩。未读取17号反推标签。\n'
        '沿用v8已验证底层预测，对无历史广告点击的用户叠加最优语义特征模型的用户内排序修正。\n'
        '使用保存的2400棵树模型（训练截至15号），17号历史统计使用截至16号的真实标签。\n'
        '16号完整预测已复算；17号独立编码、模型和分群复算、原始行序和CSV精度已核验。\n', encoding='utf8')
    assert not torch.cuda.is_initialized()
    emit('best17_v9_submission_verified', path=str(csv), rows=len(saved),
         auc16=protocol['development16_auc'], sha256=digest(csv))


if __name__ == '__main__':
    main()
