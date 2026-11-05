"""Export the latest audited neural/cohort recipe using observed history only."""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'

import gc
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
from quantile_blend_core_v1 import matched_logit, blend
from full_negative_mask_core_v1 import masks
from frozen_neural_embedding_core_v1 import FIELDS, SEQUENCES, NAMES, encode, design

RUN = HERE / 'current_best_17_run_v6'
OUTPUT = ROOT / 'output/current_best_20191117_20261103_v6'
PARENT = ROOT / 'output/current_best_20191117_20261103_v5'
PARENT_RUN = HERE / 'current_best_17_run_v5'
CLASS = EXP / 'runs/full_negative_class_balance_v1'
CLASS_MASK = EXP / 'runs/compose_class_balance_masks_v1'
NEURAL = EXP / 'runs/full_negative_neural_embeddings_v2'
WINNER = EXP / 'runs/compose_neural_embedding_masks_v1'
AUDIT = ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json'
MODELS = [(CLASS, 'day16_positive25_d6_trees3200'),
          (NEURAL, 'day16_neural_embedding_d6_trees2400')]
VOCABS = ROOT / 'analysis/auc085/cache_rich_stats_time16/vocabs.json'


def freeze():
    assert not RUN.exists() and not OUTPUT.exists(), 'Preserve previous exports'
    audit = read(AUDIT)
    assert audit['audit_status'] == 'passed' and audit['original_datasets_unchanged']
    selected = read(WINNER / 'verification.json')['dates']['20191116']['best']
    assert selected == dict(name='neural_embedding_d6', mask='user_with_past_ad_click',
                            mode='rank_global', weight=.05, auc=audit['dates']['20191116']['auc'])
    assert digest(WINNER / 'day16_prediction.npy') == audit['dates']['20191116']['prediction_sha256']
    class_selected = read(CLASS / 'verification.json')['dates']['20191116']['best']
    assert class_selected['name'] == MODELS[0][1]
    assert class_selected['mode'] == 'add_within' and class_selected['weight'] == .025
    class_mask_selected = read(CLASS_MASK / 'verification.json')['dates']['20191116']['best']
    assert class_mask_selected['name'] == 'positive25_d6'
    assert class_mask_selected['mask'] == 'user_no_past_ad_click'
    assert class_mask_selected['mode'] == 'add_within' and class_mask_selected['weight'] == .1
    parent = read(PARENT / 'verification.json')
    assert parent['status'] == 'passed' and parent['inferred17_labels_read'] is False
    assert digest(PARENT / 'submission_best_20191117.csv') == parent['submission_sha256']
    manifest = read(PARENT / 'prediction_manifest.json')
    assert digest(PARENT / 'day17_prediction.npy') == manifest['files_sha256']['day17_prediction.npy']
    assert digest(PARENT_RUN / 'protocol.json') == manifest['protocol_sha256']
    assert digest(PARENT_RUN / 'input_contracts.json') == manifest['input_contracts_sha256']
    np.testing.assert_array_equal(np.load(PARENT_RUN / 'day16_prediction.npy'),
        np.load(EXP / 'runs/compose_full_negative_masks_v1/day16_prediction.npy'))
    paths = [Path(__file__), PARENT / 'verification.json', PARENT / 'prediction_manifest.json',
        PARENT / 'day17_prediction.npy', PARENT_RUN / 'protocol.json',
        PARENT_RUN / 'input_contracts.json', PARENT_RUN / 'day16_prediction.npy', VOCABS,
        NEURAL / 'neural_tables_day16.npz', NEURAL / 'neural_teacher_day16.json',
        NEURAL / 'input_day16/contract.json', NEURAL / 'input_day16/held.npy']
    for folder in (CLASS, CLASS_MASK, NEURAL, WINNER):
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
    records = []
    for folder, name in MODELS:
        lock = read(folder / 'models_locked.json')
        assert lock['protocol_sha256'] == digest(folder / 'protocol.json')
        assert lock['inputs_locked_sha256'] == digest(folder / 'inputs_locked.json')
        record = next(record for record in lock['models'] if record['name'] == name)
        assert record['day'] == 16 and record['max_training_date'] == 20191115
        assert digest(folder / (name + '.cbm')) == record['model_sha256']
        records.append(dict(record, folder=str(folder)))
        paths += [folder / filename for filename in ('models_locked.json', 'inputs_locked.json',
            name + '.cbm', name + '_held_margin.npy')]
    contracts = read(PARENT_RUN / 'input_contracts.json')
    for day, independent in ((16, False), (17, False), (17, True)):
        path = context_path(day, independent)
        assert digest(path) == contracts[str(20191100 + day)]['feature_matrix_sha256']
        paths.append(path)
    neural_contract = read(NEURAL / 'input_day16/contract.json')
    assert neural_contract == read(NEURAL / 'inputs_locked.json')['16']
    assert neural_contract['teacher'] == read(NEURAL / 'neural_teacher_day16.json')
    assert neural_contract['teacher']['checkpoint_sha256'] == digest(neural_contract['teacher']['checkpoint'])
    neural_sources = read(NEURAL / 'protocol.json')['sources_sha256']
    assert digest(VOCABS) == neural_sources[str(VOCABS)]
    assert digest(NEURAL / 'input_day16/features.npy') == neural_contract['matrix_sha256']
    paths.append(NEURAL / 'input_day16/features.npy')
    for filename, expected in audit['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    RUN.mkdir()
    write(RUN / 'selection_audit_snapshot.json', audit)
    paths.append(RUN / 'selection_audit_snapshot.json')
    protocol = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), selected=selected,
        class_selected=class_selected, class_mask_selected=class_mask_selected,
        models=records, schema=neural_contract['schema'],
        development16_auc=selected['auc'], development_dates_repeatedly_reused=True,
        selection_basis='Highest completed independently audited full day16 development AUC at export freeze',
        new_model_training=False, saved_correction_models_max_training_date=20191115,
        prediction_history_max_date=20191116, inherited_base=str(PARENT),
        context_features='Reuse byte-verified independently rebuilt v5 context matrices',
        neural_features='Fixed day16 teacher tables and vocabulary; query encoding and NumPy/Torch aggregation independently replayed',
        inferred17_labels_read=False, raw_datasets_sha256=audit['raw_datasets_sha256'],
        sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)})
    write(RUN / 'protocol.json', protocol)
    emit('best17_v6_frozen', auc16=selected['auc'], selected=selected)


def context_path(day, independent=False):
    phase = 'independent' if independent else 'primary'
    return PARENT_RUN / f'day{day}_{phase}_features.npy'


def raw_query(day):
    filename = 'train.csv' if day == 16 else 'final_test.csv'
    dtype = {name: 'str' if name in SEQUENCES else 'int32' for name in FIELDS}
    columns = list(dict.fromkeys(FIELDS + ['log_id']))
    pieces = []
    for block in pd.read_csv(ROOT / '2026ml' / filename, usecols=columns, dtype=dtype, chunksize=60000):
        pieces.append(block.loc[block.pt_d.eq(20191100 + day), columns])
    raw = pd.concat(pieces, ignore_index=True)
    assert len(raw) == {16: 421983, 17: 404182}[day]
    contract = read(PARENT_RUN / 'input_contracts.json')[str(20191100 + day)]
    assert array_hash(raw.log_id.to_numpy(dtype='int64')) == contract['log_ids_sha256']
    return raw


def neural_features(raw, day, independent=False):
    teacher = read(NEURAL / 'neural_teacher_day16.json')
    assert teacher['held_day_labels_fit'] is False and teacher['inferred17_labels_read'] is False
    assert digest(teacher['checkpoint']) == teacher['checkpoint_sha256']
    vocabulary = read(VOCABS)
    with np.load(NEURAL / 'neural_tables_day16.npz') as values:
        tables = {name: values[name] for name in values.files}
    for name, value in tables.items():
        assert array_hash(value) == teacher['tables_sha256'][name]
    phase = 'independent' if independent else 'primary'
    path = RUN / f'day{day}_{phase}_neural.npy'
    assert not path.exists()
    matrix = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=(len(raw), len(NAMES)))
    reference = None
    if day == 16:
        reference = np.load(NEURAL / 'input_day16/features.npy', mmap_mode='r')
        held = np.load(NEURAL / 'input_day16/held.npy')
    elif independent:
        reference = np.load(RUN / 'day17_primary_neural.npy', mmap_mode='r')
    for start in range(0, len(raw), 20000):
        stop = min(start + 20000, len(raw))
        encoded = encode(raw.iloc[start:stop][FIELDS], teacher['meta'], vocabulary,
                         20191100 + day, independent)
        values = design(encoded, tables, teacher['meta'], independent)
        matrix[start:stop] = values
        if reference is not None:
            expected = reference[held[start:stop]] if day == 16 else reference[start:stop]
            np.testing.assert_array_equal(values, expected)
    matrix.flush()
    emit('best17_v6_neural_features_verified', day=day, independent=independent, rows=len(raw))
    return matrix


def forecast(day, independent=False):
    protocol = read(RUN / 'protocol.json')
    raw = raw_query(day)
    neural = neural_features(raw, day, independent)
    context = np.load(context_path(day, independent), mmap_mode='r')
    assert context.shape == (len(raw), 395)
    if independent:
        primary = np.load(context_path(day), mmap_mode='r')
        for start in range(0, len(raw), 20000):
            np.testing.assert_array_equal(context[start:start+20000], primary[start:start+20000])
    margins = []
    for record in protocol['models']:
        path = Path(record['folder']) / (record['name'] + '.cbm')
        assert digest(path) == record['model_sha256']
        model = load_model(path)
        is_neural = record['name'] == MODELS[1][1]
        names = protocol['schema']['columns'] if is_neural else protocol['schema']['columns'][:395]
        cats = protocol['schema']['categories']
        assert model.feature_names_ == names
        margin = np.empty(len(raw), dtype='float64')
        for start in range(0, len(raw), 20000):
            stop = min(start + 20000, len(raw))
            values = (np.column_stack([context[start:stop], neural[start:stop]]) if is_neural
                      else np.array(context[start:stop]))
            frame = pd.DataFrame(values, columns=names)
            for name in cats:
                numbers = frame[name].to_numpy()
                np.testing.assert_array_equal(numbers, numbers.astype('int64'))
                frame[name] = numbers.astype('int64')
            assert np.isfinite(frame.to_numpy()).all()
            margin[start:stop] = model.predict(Pool(frame, cat_features=cats),
                prediction_type='RawFormulaVal', thread_count=2)
        if day == 16:
            np.testing.assert_array_equal(margin, np.load(Path(record['folder']) / (record['name'] + '_held_margin.npy')))
        margins.append(margin)
        del model, frame, values
        gc.collect()
        emit('best17_v6_model_replayed', day=day, independent=independent, name=record['name'])
    prior = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'user_id', 'task_id', 'label'], dtype='int32')
    earlier = prior.loc[prior.pt_d.lt(20191100 + day)]
    metadata = pd.concat([earlier[['pt_d', 'user_id', 'task_id']],
                          raw[['pt_d', 'user_id', 'task_id']]], ignore_index=True)
    cohort = masks(metadata.pt_d.to_numpy(), metadata.user_id.to_numpy(), metadata.task_id.to_numpy(),
        np.arange(len(earlier)), earlier.label.to_numpy(), np.arange(len(earlier), len(metadata)),
        20191100 + day, independent)
    groups, _ = pd.factorize(raw.user_id.to_numpy(), sort=True)
    base = np.load(PARENT_RUN / 'day16_prediction.npy' if day == 16 else PARENT / 'day17_prediction.npy')
    global_adjusted = row_prediction(base, margins[0], groups, .025, 'add_within', independent)
    mask_adjusted = row_prediction(base, margins[0], groups, .1, 'add_within', independent)
    class_prediction = np.where(cohort['user_no_past_ad_click'], mask_adjusted, global_adjusted)
    target = matched_logit(class_prediction, margins[1], independent)
    neural_adjusted = blend(class_prediction, target, np.ones(len(raw), dtype=bool), .05, independent)
    prediction = np.where(cohort['user_with_past_ad_click'], neural_adjusted, class_prediction)
    assert np.isfinite(prediction).all() and np.all((prediction > 0) & (prediction < 1))
    if day == 16:
        for values, folder in ((global_adjusted, CLASS), (class_prediction, CLASS_MASK), (prediction, WINNER)):
            np.testing.assert_allclose(values, np.load(folder / 'day16_prediction.npy'), rtol=0, atol=2e-13)
        auc = roc_auc_score(prior.loc[prior.pt_d.eq(20191116), 'label'], prediction)
        assert abs(auc - protocol['development16_auc']) < 1e-12
    contract = dict(rows=len(raw), context_features_sha256=digest(context_path(day, independent)),
        neural_features_sha256=digest(Path(neural.filename)),
        log_ids_sha256=array_hash(raw.log_id.to_numpy(dtype='int64')),
        class_margin_sha256=array_hash(margins[0]), neural_margin_sha256=array_hash(margins[1]),
        user_no_past_ad_click_mask_sha256=array_hash(cohort['user_no_past_ad_click']),
        user_with_past_ad_click_mask_sha256=array_hash(cohort['user_with_past_ad_click']),
        user_with_past_ad_click_rows=int(cohort['user_with_past_ad_click'].sum()),
        mask_history_max_date=int(earlier.pt_d.max()), inferred17_labels_read=False)
    emit('best17_v6_forecast_verified', day=day, independent=independent, rows=len(raw))
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
    np.testing.assert_allclose(prediction, replay, rtol=0, atol=2e-13)
    assert contract == independent
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
    np.testing.assert_array_equal(saved.id, np.arange(len(prediction)))
    np.testing.assert_array_equal(saved.pctr, prediction)
    write(RUN / 'input_contracts.json', {'20191116': development_contract, '20191117': contract})
    report = dict(status='passed', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        prediction_date=20191117, rows=len(saved), columns=['id', 'pctr'],
        development16_auc=protocol['development16_auc'], selected=protocol['selected'],
        development_dates_repeatedly_reused=True, inferred17_labels_read=False, platform_auc=None,
        complete_day16_selected_models_layers_and_prediction_replayed=True,
        independent_day17_neural_encoding_models_masks_and_prediction_replayed=True,
        context_inputs_reused_from_previous_independently_verified_export=True,
        maximum_prediction_replay_error=float(np.max(np.abs(prediction - replay))),
        saved_correction_models_max_training_date=20191115, prediction_history_max_date=20191116,
        original_row_order_verified=True, csv_roundtrip_exact=True, original_datasets_unchanged=True,
        raw_datasets_sha256=protocol['raw_datasets_sha256'], submission_sha256=digest(csv))
    write(OUTPUT / 'prediction_manifest.json', dict(protocol_sha256=digest(RUN / 'protocol.json'),
        input_contracts_sha256=digest(RUN / 'input_contracts.json'), development16_auc=protocol['development16_auc'],
        inferred17_labels_read=False, files_sha256={p.name: digest(p) for p in OUTPUT.iterdir() if p.is_file()}))
    write(OUTPUT / 'verification.json', report)
    (OUTPUT / '提交说明.md').write_text('提交文件：submission_best_20191117.csv\n'
        '格式：id,pctr；404182行；id按原始行序从0开始。\n'
        f'本次冻结时已完成独立核验的最优方案，16号开发集AUC={protocol["development16_auc"]:.12f}。\n'
        '15/16号反复用于开发；该AUC不是17号平台成绩。未读取17号反推标签。\n'
        '沿用已验证底层预测，叠加全量负样本类别权重模型和神经特征分群融合。\n'
        '使用保存的最优修正模型（训练截至15号），17号历史统计使用截至16号的真实标签。\n'
        '16号完整最优预测已复算；17号独立复算、行序及CSV精度已核验。\n', encoding='utf8')
    emit('best17_v6_submission_verified', path=str(csv), rows=len(saved), auc16=protocol['development16_auc'], sha256=digest(csv))


if __name__ == '__main__':
    main()
