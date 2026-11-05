"""Export the audited refreshed family winner in bounded-memory batches.

Only original observed labels through November 16 are opened. Query-only
audience aggregation is checked against the previously frozen complete compact
feature hashes for both November 16 and November 17.
"""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'

import gc
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoost, Pool
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / 'analysis/day16_auc082_20261102'
sys.path.insert(0, str(EXP))
from compact_full_sample_core_v1 import compact_frame
from quantile_blend_core_v1 import matched_logit, blend
from task_audience_core import FIELDS, HISTOGRAMS

RUN = HERE / 'current_best_17_run_v8'
OUTPUT = ROOT / 'output/current_best_20191117_20261103_v8'
PARENT = ROOT / 'output/current_best_20191117_20261103_v6'
PARENT_RUN = HERE / 'current_best_17_run_v6'
REFERENCE = HERE / 'current_best_17_run_v1'
WINNER = EXP / 'runs/refresh_family_quantiles_v1'
COMPACT = EXP / 'runs/compact_full_sample_ranker_v1'
BASE = EXP / 'runs/compose_neural_embedding_masks_v1'
CACHE = ROOT / 'analysis/final_best_recipe_quality_transfer_20261102/run_v2'
AUDIT = ROOT / 'output/two_day_auc_targets_20261102/independent_target_audit.json'
BATCH = 5000


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def cache_path(day, independent=False):
    return CACHE / ('independent_day17' if day == 17 and independent else f'input_day{day}')


def freeze():
    assert not RUN.exists() and not OUTPUT.exists(), 'Preserve previous exports and failures'
    audit = read(AUDIT)
    assert audit['audit_status'] == 'passed' and audit['original_datasets_unchanged']
    result = read(WINNER / 'verification.json')
    assert result['status'] == 'passed'
    selected = result['dates']['20191116']['best']
    assert selected == dict(name='compact_full_sample_ranker_v1__late_mean', mask='cold_task',
                            weight=.1, auc=audit['dates']['20191116']['auc'])
    assert digest(WINNER / 'day16_prediction.npy') == audit['dates']['20191116']['prediction_sha256']
    winner_protocol = read(WINNER / 'protocol.json')
    assert winner_protocol['inferred17_labels_read'] is False
    assert winner_protocol['raw_datasets_sha256'] == audit['raw_datasets_sha256']
    assert read(COMPACT / 'verification.json')['status'] == 'passed'
    model_lock = read(COMPACT / 'models_locked.json')
    assert model_lock['protocol_sha256'] == digest(COMPACT / 'protocol.json')
    all_records = [record for record in model_lock['models'] if record['day'] == 16]
    late_cut = sorted({record['trees'] for record in all_records})[-2]
    records = [record for record in all_records if record['trees'] >= late_cut]
    assert len(records) == 4 and late_cut == 1600
    fitted = read(COMPACT / 'input_contract16.json')
    assert fitted['max_training_date'] == 20191115
    for record in records:
        assert record['input_contract_sha256'] == digest(COMPACT / 'input_contract16.json')
        assert digest(COMPACT / (record['name'] + '.cbm')) == record['model_sha256']
    parent = read(PARENT / 'verification.json')
    manifest = read(PARENT / 'prediction_manifest.json')
    assert parent['status'] == 'passed' and parent['inferred17_labels_read'] is False
    assert manifest['protocol_sha256'] == digest(PARENT_RUN / 'protocol.json')
    assert manifest['input_contracts_sha256'] == digest(PARENT_RUN / 'input_contracts.json')
    for name, expected in manifest['files_sha256'].items():
        assert digest(PARENT / name) == expected, name
    assert digest(PARENT / 'submission_best_20191117.csv') == parent['submission_sha256']
    np.testing.assert_array_equal(np.load(PARENT_RUN / 'day16_prediction.npy'), np.load(BASE / 'day16_prediction.npy'))
    assert read(BASE / 'verification.json')['dates']['20191116']['auc'] == parent['development16_auc']
    paths = [Path(__file__), EXP / 'compact_full_sample_core_v1.py', EXP / 'quantile_blend_core_v1.py',
             EXP / 'task_audience_core.py', WINNER / 'protocol.json', WINNER / 'verification.json',
             WINNER / 'inputs_locked.json', WINNER / 'predictions_locked.json',
             WINNER / 'day16_prediction.npy', WINNER / 'day16_indices.npy', BASE / 'day16_prediction.npy',
             PARENT_RUN / 'protocol.json', PARENT_RUN / 'input_contracts.json', PARENT_RUN / 'day16_prediction.npy',
             PARENT / 'verification.json', PARENT / 'prediction_manifest.json', PARENT / 'day17_prediction.npy',
             REFERENCE / 'protocol.json', REFERENCE / 'input_contracts.json',
             COMPACT / 'protocol.json', COMPACT / 'verification.json', COMPACT / 'models_locked.json',
             COMPACT / 'input_contract16.json', COMPACT / 'held_contract16.json',
             EXP / 'runs/rich_augmented_rows_v1/input_contract16.json',
             EXP / 'runs/task_audience_row_ranker_v1/input_contract16.json',
             EXP / 'runs/task_audience_rate_v1/input_day16/contract.json']
    for record in records:
        paths += [COMPACT / (record['name'] + suffix) for suffix in ('.cbm', '_held_margin.npy')]
    prior_bindings = read(REFERENCE / 'protocol.json')['sources_sha256']
    for day, independent in ((16, False), (17, False), (17, True)):
        folder = cache_path(day, independent)
        contract = read(folder / 'contract.json')
        assert digest(folder / 'features.npy') == contract['matrix_sha256']
        assert digest(folder / 'daily_quality.npz') == contract['quality_file_sha256']
        for name in ('features.npy', 'daily_quality.npz', 'contract.json'):
            path = folder / name
            assert digest(path) == prior_bindings[str(path)], path
            paths.append(path)
    for name in ('compact_full_sample_core_v1.py', 'task_audience_core.py'):
        path = EXP / name
        assert digest(path) == prior_bindings[str(path)]
    for filename, expected in audit['raw_datasets_sha256'].items():
        assert digest(ROOT / '2026ml' / filename) == expected, filename
    RUN.mkdir()
    write(RUN / 'selection_audit_snapshot.json', audit)
    paths.append(RUN / 'selection_audit_snapshot.json')
    protocol = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), selected=selected, models=records,
        development16_auc=selected['auc'], development_dates_repeatedly_reused=True,
        selection_basis='Highest completed independently audited full day16 AUC at export freeze',
        new_model_training=False, saved_correction_models_max_training_date=20191115,
        prediction_history_max_date=20191116, inherited_base=str(PARENT),
        feature_verification='Query-only compact features must match complete frozen 16/17 feature hashes',
        independent_replay='Independent 17 source cache, sparse audience aggregation, column gather, rank mapping and odds',
        inferred17_labels_read=False, raw_datasets_sha256=audit['raw_datasets_sha256'],
        sources_sha256={str(path): digest(path) for path in dict.fromkeys(paths)})
    write(RUN / 'protocol.json', protocol)
    emit('best17_v8_frozen', auc16=selected['auc'], selected=selected)


def query(day):
    path = ROOT / '2026ml' / ('train.csv' if day == 16 else 'final_test.csv')
    pieces = []
    for block in pd.read_csv(path, usecols=FIELDS + ['log_id'], dtype={**{name: 'int32' for name in FIELDS}, 'log_id': 'int64'}, chunksize=60000):
        pieces.append(block.loc[block.pt_d.eq(20191100 + day)].copy())
    raw = pd.concat(pieces, ignore_index=True)
    assert len(raw) == {16: 421983, 17: 404182}[day]
    assert array_hash(raw.log_id.to_numpy(dtype='int64')) == read(PARENT_RUN / 'input_contracts.json')[str(20191100 + day)]['log_ids_sha256']
    return raw


def audience_query(raw, day, quality_keys, quality, independent):
    """Calculate only requested-date audience features, with bounded quality memory."""
    wanted = read(EXP / 'runs/task_audience_row_ranker_v1/input_contract16.json')['audience_columns']
    wanted_set = set(wanted)
    groups, task_index = pd.factorize(raw.task_id, sort=True)
    tasks = task_index.to_numpy()
    number = len(tasks)
    sizes = np.bincount(groups, minlength=number).astype('float64')
    pair = raw.groupby(['task_id', 'user_id'], sort=True).size().rename('weight').reset_index()
    pg = np.searchsorted(tasks, pair.task_id.to_numpy())
    unique = np.bincount(pg, minlength=number).astype('float64')
    matrix = csr_matrix((np.ones(len(groups)), (groups, np.arange(len(groups)))), shape=(number, len(groups))) if independent else None

    def aggregate(value):
        value = np.asarray(value, dtype='float64')
        return np.asarray(matrix @ value).reshape(-1) if independent else np.bincount(groups, weights=value, minlength=number)

    columns = dict(log_exposures=np.log1p(sizes), log_users=np.log1p(unique), exposures_per_user=sizes/unique)
    for name in wanted:
        if name.startswith('fraction__'):
            _, field, value = name.split('__')
            if value == 'unknown':
                schema = read(EXP / 'runs/task_audience_rate_v1/input_day16/contract.json')['feature_names']
                prefix = f'fraction__{field}__'
                known = [int(item[len(prefix):]) for item in schema if item.startswith(prefix) and not item.endswith('__unknown')]
                rows = ~raw[field].isin(known).to_numpy()
            else:
                rows = raw[field].to_numpy() == int(value)
            columns[name] = aggregate(rows) / sizes
        elif name.startswith('log_unique__'):
            columns[name] = np.log1p(raw.groupby('task_id', sort=True)[name.split('__')[1]].nunique().to_numpy())
    for field in ('u_refreshTimes', 'u_feedLifeCycle', 'app_score'):
        values = raw[field].to_numpy(dtype='float64')
        mean = aggregate(values) / sizes
        columns[field + '__mean'] = mean
        columns[field + '__std'] = np.sqrt(np.maximum(aggregate(values**2)/sizes - mean**2, 0))
    fields = ['user_id', 'task_id', 'adv_prim_id', 'spread_app_id']
    prior = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'] + fields, dtype='int32')
    prior = prior.loc[prior.pt_d.lt(20191100 + day)].copy()
    assert int(prior.pt_d.max()) == 20191100 + day - 1
    cold = ~raw.task_id.isin(prior.task_id.unique()).to_numpy()
    global_rate = (prior.label.sum() + .5) / (len(prior) + 50.)
    for field in fields:
        counts = prior.groupby(field, sort=True).label.agg(['size', 'sum'])
        n = raw[field].map(counts['size']).fillna(0).to_numpy(dtype='float64')
        c = raw[field].map(counts['sum']).fillna(0).to_numpy(dtype='float64')
        values = [np.log1p(n), np.log1p(c), (c+20*global_rate)/(n+20), (c+200*global_rate)/(n+200), n == 0]
        for title, value in zip(('log_n', 'log_c', 'ctr20', 'ctr200', 'new'), values):
            prefix = f'past__{field}__{title}'
            if prefix + '__mean' not in wanted_set and prefix + '__std' not in wanted_set:
                continue
            value = np.asarray(value, dtype='float64')
            mean = aggregate(value) / sizes
            columns[prefix + '__mean'] = mean
            columns[prefix + '__std'] = np.sqrt(np.maximum(aggregate(value**2)/sizes - mean**2, 0))
    del prior, counts, n, c, values, matrix
    gc.collect()
    assert np.all(quality_keys[:, 0] == 20191100 + day)
    qindex = pd.Index(quality_keys[:, 1])
    assert qindex.is_unique
    qi = qindex.get_indexer(pair.user_id)
    assert np.all(qi >= 0)
    weights = pair.weight.to_numpy(dtype='float64')
    weighted = csr_matrix((weights, (pg, qi)), shape=(number, len(quality))) if independent else None
    unweighted = csr_matrix((np.ones(len(pg)), (pg, qi)), shape=(number, len(quality))) if independent else None
    quality_columns = sorted({int(name.split('__')[0].split('_')[1]) for name in wanted if name.startswith('quality_')})
    for column in quality_columns:
        q = np.asarray(quality[:, column], dtype='float64')
        if independent:
            mean = np.asarray(weighted @ q).reshape(-1) / sizes
            second = np.asarray(weighted @ (q*q)).reshape(-1) / sizes
            user = np.asarray(unweighted @ q).reshape(-1) / unique
        else:
            q = q[qi]
            mean = np.bincount(pg, weights=q*weights, minlength=number) / sizes
            second = np.bincount(pg, weights=q*q*weights, minlength=number) / sizes
            user = np.bincount(pg, weights=q, minlength=number) / unique
        prefix = f'quality_{column}'
        columns[prefix + '__exposure_mean'] = mean
        columns[prefix + '__exposure_std'] = np.sqrt(np.maximum(second-mean**2, 0))
        columns[prefix + '__user_mean'] = user
    assert wanted_set.issubset(columns), sorted(wanted_set-set(columns))
    values = np.column_stack([np.round(columns[name], 5).astype('float32') for name in wanted])
    assert np.isfinite(values).all()
    return values, groups, cold


def load_model(path):
    model = CatBoost()
    with tempfile.TemporaryDirectory(prefix='best17_v8_') as folder:
        temporary = Path(folder) / 'model.cbm'
        temporary.write_bytes(path.read_bytes())
        model.load_model(str(temporary))
    return model


def forecast(day, independent=False):
    protocol = read(RUN / 'protocol.json')
    raw = query(day)
    folder = cache_path(day, independent)
    contract = read(folder / 'contract.json')
    assert digest(folder / 'features.npy') == contract['matrix_sha256']
    assert digest(folder / 'daily_quality.npz') == contract['quality_file_sha256']
    with np.load(folder / 'daily_quality.npz') as saved:
        keys, quality = saved['keys'], saved['values']
    audience, groups, cold = audience_query(raw, day, keys, quality, independent)
    native = read(EXP / 'runs/rich_augmented_rows_v1/input_contract16.json')
    fitted = read(COMPACT / 'input_contract16.json')
    native_keep = [i for i, name in enumerate(native['feature_names']) if not name.startswith('rich_daily__quality__')]
    audience_names = read(EXP / 'runs/task_audience_row_ranker_v1/input_contract16.json')['audience_columns']
    names = [f'f{i}' for i in range(1017)] + ['task_audience__' + name for name in audience_names]
    categories = [f'f{native_keep.index(i)}' for i in native['categorical_columns']]
    selected = np.array([names.index(name) for name in fitted['selected_columns']], dtype='int64')
    matrix = np.load(folder / 'features.npy', mmap_mode='r')
    assert matrix.shape == (len(raw), 744)
    positions = pd.MultiIndex.from_arrays(keys.T).get_indexer(pd.MultiIndex.from_frame(raw[['pt_d', 'user_id']]))
    assert np.all(positions >= 0)
    models = [load_model(COMPACT / (record['name'] + '.cbm')) for record in protocol['models']]
    for model in models:
        assert model.feature_names_ == fitted['selected_columns']
    margins = [np.empty(len(raw), dtype='float64') for _ in models]
    frame_hash = hashlib.sha256()
    quality_hash, extra_hash = hashlib.sha256(), hashlib.sha256()
    for start in range(0, len(raw), BATCH):
        stop = min(start + BATCH, len(raw))
        news = quality[positions[start:stop]]
        extra = audience[groups[start:stop]]
        frame, cats = compact_frame(matrix, np.arange(start, stop), native_keep, news, extra,
                                     selected, names, categories, independent=independent)
        assert cats == fitted['categories']
        frame_hash.update(np.ascontiguousarray(frame.to_numpy(dtype='float32')).tobytes())
        quality_hash.update(np.ascontiguousarray(news).tobytes())
        extra_hash.update(np.ascontiguousarray(extra).tobytes())
        pool = Pool(frame, cat_features=cats)
        for model, margin in zip(models, margins):
            margin[start:stop] = model.predict(pool, prediction_type='RawFormulaVal', thread_count=2)
        if start % 100000 == 0:
            emit('best17_v8_batch', day=day, independent=independent, rows=stop, total=len(raw))
        del frame, pool, news, extra
    expected = read(REFERENCE / 'input_contracts.json')[str(20191100 + day)]
    assert frame_hash.hexdigest() == expected['compact_frame_sha256'], (day, frame_hash.hexdigest(), expected['compact_frame_sha256'])
    if day == 16:
        held = read(COMPACT / 'held_contract16.json')
        assert quality_hash.hexdigest() == held['quality_sha256']
        assert extra_hash.hexdigest() == held['audience_features_sha256']
        assert array_hash(cold) == read(WINNER / 'inputs_locked.json')['16']['masks_sha256']['cold_task']
        for record, margin in zip(protocol['models'], margins):
            np.testing.assert_array_equal(margin, np.load(COMPACT / (record['name'] + '_held_margin.npy')))
    del matrix, quality, models, audience
    gc.collect()
    base = np.load(BASE / 'day16_prediction.npy' if day == 16 else PARENT / 'day17_prediction.npy')
    target = np.mean([matched_logit(base, margin, independent) for margin in margins], axis=0)
    prediction = blend(base, target, cold, protocol['selected']['weight'], independent)
    assert np.isfinite(prediction).all() and np.all((prediction > 0) & (prediction < 1))
    if day == 16:
        np.testing.assert_allclose(prediction, np.load(WINNER / 'day16_prediction.npy'), rtol=0, atol=2e-13)
        labels = pd.read_csv(ROOT / '2026ml/train.csv', usecols=['pt_d', 'label'], dtype='int32')
        y = labels.loc[labels.pt_d.eq(20191116), 'label'].to_numpy()
        assert abs(roc_auc_score(y, prediction) - protocol['development16_auc']) < 1e-12
    result = dict(rows=len(raw), log_ids_sha256=array_hash(raw.log_id.to_numpy(dtype='int64')),
        compact_frame_sha256=frame_hash.hexdigest(), quality_sha256=quality_hash.hexdigest(),
        audience_features_sha256=extra_hash.hexdigest(), cold_task_mask_sha256=array_hash(cold),
        cold_task_rows=int(cold.sum()), margins_sha256={record['name']: array_hash(margin)
            for record, margin in zip(protocol['models'], margins)},
        mask_history_max_date=20191100 + day - 1, inferred17_labels_read=False)
    emit('best17_v8_forecast_verified', day=day, independent=independent, rows=len(raw), cold_task_rows=int(cold.sum()))
    return prediction, result


def main():
    freeze()
    protocol = read(RUN / 'protocol.json')
    development, development_contract = forecast(16)
    np.save(RUN / 'day16_prediction.npy', development)
    prediction, contract = forecast(17)
    np.save(RUN / 'day17_prediction.npy', prediction)
    replay, independent = forecast(17, True)
    assert contract == independent
    np.testing.assert_allclose(prediction, replay, rtol=0, atol=2e-13)
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
        complete_day16_four_models_features_mask_and_selected_prediction_replayed=True,
        independent_day17_compact_features_four_models_masks_and_prediction_replayed=True,
        previous_frozen_complete_compact_feature_hashes_matched_on_both_dates=True,
        verified_parent_prediction_cache_reused=True,
        maximum_prediction_replay_error=float(np.max(np.abs(prediction-replay))),
        saved_correction_models_max_training_date=20191115, prediction_history_max_date=20191116,
        original_row_order_verified=True, csv_roundtrip_exact=True, original_datasets_unchanged=True,
        raw_datasets_sha256=protocol['raw_datasets_sha256'], submission_sha256=digest(csv))
    write(OUTPUT / 'prediction_manifest.json', dict(protocol_sha256=digest(RUN / 'protocol.json'),
        input_contracts_sha256=digest(RUN / 'input_contracts.json'), development16_auc=protocol['development16_auc'],
        inferred17_labels_read=False, files_sha256={path.name: digest(path) for path in OUTPUT.iterdir() if path.is_file()}))
    write(OUTPUT / 'verification.json', report)
    (OUTPUT / '提交说明.md').write_text('提交文件：submission_best_20191117.csv\n'
        '格式：id,pctr；404182行；id按原始行序从0开始。\n'
        f'本次冻结时已完成独立核验的最优方案，16号开发集AUC={protocol["development16_auc"]:.12f}。\n'
        '15/16号反复用于开发；该AUC不是17号平台成绩。未读取17号反推标签。\n'
        '沿用v6已验证底层预测，在冷启动任务上叠加四个历史排序模型检查点的分位数融合。\n'
        '四个修正模型训练截至15号，17号历史统计使用截至16号的真实标签。\n'
        '16号完整预测已复算；17号独立复算、原始行序和CSV精度已核验。\n', encoding='utf8')
    emit('best17_v8_submission_verified', path=str(csv), rows=len(saved), auc16=protocol['development16_auc'], sha256=digest(csv))


if __name__ == '__main__':
    main()
