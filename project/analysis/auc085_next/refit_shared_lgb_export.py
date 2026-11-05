"""Fixed 378-round shared LightGBM full refit and independently verified v4.

Only a new output/chronological_shared_lgb_v4 directory is written. Existing
outputs/checkpoints remain read-only; final labels are never loaded. The shared
tree omits raw user_id/task_id/adv_id categories and every sequence-token ID,
while retaining the exact validated 349-column feature order.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('MKL_NUM_THREADS', '3')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '3')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DAILY = HERE.parent / 'daily_interest_din'
CACHE = DAILY / 'full_refit_cache'
BASE_MODELS = DAILY / 'full_refit_models'
V2 = ROOT / 'output' / 'chronological_lgb_v2'
V3 = ROOT / 'output' / 'chronological_multiscale_v3'
EXPORT = ROOT / 'output' / 'chronological_shared_lgb_v4'
sys.path.insert(0, str(HERE / 'deps'))

import lightgbm as lgb
import numpy as np
import pandas as pd

ROUNDS = 378
BATCH_SIZE = 25000
WEIGHTS = {'v3': .8, 'shared_lgb': .2}
PARAMS = {'objective': 'binary', 'metric': ['auc', 'binary_logloss'],
          'learning_rate': .03, 'num_leaves': 15, 'max_depth': -1,
          'min_data_in_leaf': 500, 'max_bin': 63, 'feature_fraction': .9,
          'bagging_fraction': .9, 'bagging_freq': 1, 'lambda_l2': 10.,
          'cat_l2': 20., 'cat_smooth': 20., 'max_cat_threshold': 32,
          'verbosity': -1, 'num_threads': 3, 'seed': 42, 'deterministic': True,
          'force_col_wise': True, 'feature_pre_filter': False}
DEVELOPMENT = {
    '20191115': {'shared_lgb_auc': .7767474673043586, 'shared_best_iteration': 320,
                 'v3_auc': .7876002210086661, 'v4_auc': .7880938883633664},
    '20191116': {'shared_lgb_auc': .7813616401925747, 'shared_best_iteration': 435,
                 'v3_auc': .7948963507440203, 'v4_auc': .7957095722599832},
}
WARNINGS = [
    '开发日期15/16反复用于模型和固定融合权重选型，不是独立盲测；开发AUC不代表final AUC。',
    '源域新闻数据、当前日曝光和广告历史快照采用所提供整日离线输入假设，不代表逐条在线曝光前可用。',
    'final标签不读取且强制为-1；final_auc=None，不能声称final达标或AUC达到0.85。',
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}


def log(event, **fields):
    text = json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'pid': os.getpid(),
                       'event': event, **fields}, ensure_ascii=True)
    with (EXPORT / 'run.log').open('a', encoding='utf-8') as stream:
        stream.write(text + '\n')
    print(text, flush=True)


def probability(values, rows, name):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (rows,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise AssertionError(name + ': invalid probabilities')
    return values


def inputs():
    manifest = read(CACHE / 'manifest.json')
    exposure = read(DAILY / 'exposure_features.json')
    snapshot = read(DAILY / 'snapshot_features.json')
    raw_meta = read(V3 / 'multiscale_raw.json')
    if manifest['final_labels_available'] is not False or any(meta['advertisement_labels_loaded'] is not False
                                                             for meta in [exposure, snapshot]):
        raise AssertionError('Existing input label boundary differs')
    if raw_meta['validation_day'] != 20191117 or raw_meta['advertisement_validation_labels_used'] is not False:
        raise AssertionError('Full multiscale input must use only past labels')
    if raw_meta['final_labels_read'] is not False:
        raise AssertionError('Full multiscale final target boundary differs')
    arrays = {name: np.load(CACHE / (name + '.npy'), mmap_mode='r') for name in
              ['categories', 'numeric', 'exposure_raw', 'snapshot_raw', 'date', 'log_id', 'user_id']}
    arrays['multiscale_raw'] = np.load(V3 / 'multiscale_raw.npy', mmap_mode='r')
    ntrain, nfinal = manifest['train_rows'], manifest['final_rows']
    if manifest['rows'] != ntrain + nfinal or any(len(values) != manifest['rows'] for values in arrays.values()):
        raise AssertionError('Full input row counts differ')
    if arrays['numeric'].shape[1] != 166 or arrays['multiscale_raw'].shape[1] != 159:
        raise AssertionError('Full base/multiscale dimensions differ')
    if not arrays['multiscale_raw'].flags.f_contiguous:
        raise AssertionError('Expected existing F-order raw multiscale storage')
    if arrays['date'][:ntrain].max() != 20191116 or not np.all(arrays['date'][:ntrain] < 20191117):
        raise AssertionError('Training chronology differs')
    if not np.all(arrays['date'][ntrain:] == 20191117):
        raise AssertionError('Final chronology differs')
    if manifest['numeric_fields'][75:] != exposure['features'] + snapshot['features']:
        raise AssertionError('Full base raw-extra schema differs')
    cats = [name for name in manifest['categorical_fields'] if name not in ['user_id', 'task_id', 'adv_id']]
    features = cats + manifest['numeric_fields'][:75] + exposure['features'] + snapshot['features'] + raw_meta['features']
    if len(features) != 349 or len(cats) != 24 or len(set(features)) != 349:
        raise AssertionError('Expected 349 fields and 24 categorical fields')
    if any(name in cats for name in ['user_id', 'task_id', 'adv_id']) or any(re.search(r'_token\d+$', name) for name in features):
        raise AssertionError('Unexpected raw identity/sequence category')
    return manifest, arrays, features, cats


def development_parameters(features, cats):
    files = {}
    for day, expected in DEVELOPMENT.items():
        folder = DAILY / 'lgb_results' / ('goal085_day' + day[-2:] + '_lgb_shared_multi')
        metrics, protocol = read(folder / 'metrics.json'), read(folder / 'protocol.json')
        if metrics['params'] != {**PARAMS, 'num_threads': 6} or metrics['fields'] != 349:
            raise AssertionError('Shared LightGBM development parameters differ')
        if metrics['best_iteration'] != expected['shared_best_iteration'] or abs(metrics['auc'] - expected['shared_lgb_auc']) > 1e-14:
            raise AssertionError('Shared LightGBM development evidence differs')
        if protocol['features'] != features or protocol['categorical_fields'] != cats:
            raise AssertionError('Full features/order must equal both development protocols')
        if protocol['no_identity'] is not True or protocol['no_sequence_ids'] is not True:
            raise AssertionError('Expected shared no-identity/no-sequence tree')
        files[day + '_metrics'] = file_record(folder / 'metrics.json')
        files[day + '_protocol'] = file_record(folder / 'protocol.json')
    if ROUNDS != round(np.mean([row['shared_best_iteration'] for row in DEVELOPMENT.values()])):
        raise AssertionError('Expected round(mean([320,435]))=378')
    return files


def shared_frame(start, stop, arrays, manifest, features, cats):
    """Single float32 F allocation, with unchanged cached category IDs.

    Native category values are explicitly named categorical in LightGBM; no
    pandas per-chunk category remapping is permitted. Only the requested train
    or final chunk is materialized. Extra 91 base features always come from raw
    exposure/snapshot arrays, and all 159 multiscale values remain unscaled.
    """
    matrix = np.empty((stop - start, 349), dtype=np.float32, order='F')
    position = 0
    for col, name in enumerate(manifest['categorical_fields']):
        if name not in cats:
            continue
        values = arrays['categories'][start:stop, col]
        if np.any(values < 0) or np.any(values >= 2 ** 24):
            raise AssertionError('Category ID not exactly representable as float32')
        matrix[:, position] = values
        position += 1
    for key, width in [('numeric', 75), ('exposure_raw', 43), ('snapshot_raw', 48), ('multiscale_raw', 159)]:
        values = arrays[key]
        if values.shape[1] < width:
            raise AssertionError('Source feature width differs: ' + key)
        for col in range(width):
            matrix[:, position] = values[start:stop, col]
            if not np.isfinite(matrix[:, position]).all():
                raise AssertionError('Nonfinite shared feature: ' + features[position])
            position += 1
    if position != len(features):
        raise AssertionError('Constructed shared frame width differs')
    return pd.DataFrame(matrix, columns=features, copy=False)


def final_ids(manifest, arrays):
    ids = pd.read_csv(ROOT / '2026ml' / 'final_test.csv', usecols=['log_id', 'user_id', 'pt_d'], dtype='int64')
    if len(ids) != manifest['final_rows']:
        raise AssertionError('Original final row count differs')
    for field, cached in [('log_id', 'log_id'), ('user_id', 'user_id'), ('pt_d', 'date')]:
        np.testing.assert_array_equal(ids[field].to_numpy(), arrays[cached][manifest['train_rows']:])
    return ids


def alignment_probes(manifest, arrays, features, cats):
    rows = [0, manifest['train_rows'] - 1, manifest['train_rows'], manifest['rows'] - 1]
    checks = 0
    for row in rows:
        frame = shared_frame(row, row + 1, arrays, manifest, features, cats)
        reference = [arrays['categories'][row, col] for col, name in enumerate(manifest['categorical_fields']) if name in cats]
        reference += list(arrays['numeric'][row, :75])
        reference += list(arrays['exposure_raw'][row])
        reference += list(arrays['snapshot_raw'][row])
        reference += list(arrays['multiscale_raw'][row])
        np.testing.assert_array_equal(frame.iloc[0].to_numpy(), np.asarray(reference, dtype=np.float32))
        checks += len(reference)
    return {'probe_rows': rows, 'feature_value_checks': checks, 'maximum_error': 0.,
            'features_match_both_development_protocols': True, 'categorical_ids_unchanged': True,
            'sequence_arrays_not_loaded': True, 'raw_exposure_snapshot_multiscale_used': True}


def infer(model, manifest, arrays, features, cats, prefix):
    if model.feature_name() != features or model.current_iteration() != ROUNDS:
        raise AssertionError('Full shared model schema/budget differs')
    values = np.empty(manifest['final_rows'], dtype=np.float64)
    for offset in range(0, len(values), BATCH_SIZE):
        stop = min(offset + BATCH_SIZE, len(values))
        frame = shared_frame(manifest['train_rows'] + offset, manifest['train_rows'] + stop,
                             arrays, manifest, features, cats)
        values[offset:stop] = model.predict(frame, num_iteration=ROUNDS, num_threads=3)
        del frame
        log(prefix, rows=stop, total=len(values))
    return probability(values, len(values), prefix)


def v3_prediction(nfinal):
    return probability(np.load(V3 / 'ensemble_final_prediction.npy', mmap_mode='r'), nfinal, 'v3')


def main():
    started = time.perf_counter()
    if EXPORT.exists():
        raise FileExistsError('Preserving existing v4 output: ' + str(EXPORT))
    manifest, arrays, features, cats = inputs()
    evidence = development_parameters(features, cats)
    if read(V3 / 'manifest.json')['status'] != 'complete_and_verified' or read(V3 / 'verification.json')['status'] != 'passed':
        raise AssertionError('Requires independently verified v3 base blend')
    baseline = v3_prediction(manifest['final_rows'])
    ids = final_ids(manifest, arrays)
    protected_paths = sorted(path for folder in [V2, V3] for path in folder.iterdir() if path.is_file())
    protected_paths += [BASE_MODELS / name for name in ['dcn.pt', 'catboost.cbm', 'dcn_final_prediction.npy', 'tree_final_prediction.npy']]
    protected = {str(path.resolve()): file_record(path) for path in protected_paths}
    sources = {str(path.resolve()): file_record(path) for path in [Path(__file__), DAILY / 'train_lightgbm.py',
                                                                 DAILY / 'exposure_features.json', DAILY / 'snapshot_features.json']}
    probes = alignment_probes(manifest, arrays, features, cats)
    protocol = {'status': 'fixed_before_training_and_final_inference', 'params': PARAMS, 'rounds': ROUNDS,
                'budget_source': 'round(mean([320,435]))=378 from day15/day16 shared-tree best iterations',
                'early_stopping_rounds': None, 'full_training_refit': True, 'full_train_rows': manifest['train_rows'],
                'final_rows': manifest['final_rows'], 'ensemble': WEIGHTS,
                'expanded_weights': {'dcn': .24, 'tree': .24, 'lgb_v2': .12, 'multiscale_dcn': .20, 'shared_lgb': .20},
                'features': features, 'categorical_fields': cats, 'fields': 349, 'categorical_count': 24,
                'no_identity': True, 'no_sequence_ids': True, 'category_ids': 'Native full-cache integer IDs, exactly stored as float32 and explicitly marked categorical',
                'numeric_inputs': 'full numeric[:,:75] + exposure_raw43 + snapshot_raw48 + v3 multiscale_raw159, no extra-feature standardization',
                'matrix_storage': 'Single float32 column-major training matrix, released before 25000-row final inference',
                'device': 'CPU', 'num_threads': 3, 'inference_batch_size': BATCH_SIZE,
                'parameter_source': str((DAILY / 'train_lightgbm.py').resolve()),
                'parameter_deviation': {'num_threads': {'development': 6, 'full_refit': 3}},
                'development_auc': DEVELOPMENT, 'development_evidence_files': evidence,
                'alignment_probes': probes, 'protected_files': protected, 'source_files': sources,
                'cache_manifest': file_record(CACHE / 'manifest.json'), 'input_sha256': manifest['input_sha256'],
                'weights_source': 'Fixed 20% shared tree added to v3; both repeatedly used development dates improved',
                'advertisement_statistics': 'Strictly earlier calendar-date advertisement outcomes',
                'same_day_input_assumption': 'Supplied source-day news profiles, full daily exposures and ad-history snapshots available offline',
                'final_labels_read': False, 'final_labels_available': False, 'final_label_value': -1,
                'final_auc': None, 'warnings': WARNINGS}
    EXPORT.mkdir(parents=True, exist_ok=False)
    write(EXPORT / 'protocol.json', protocol)
    log('phase_start', phase='protocol_committed', rows=manifest['train_rows'], rounds=ROUNDS,
        features=len(features), categorical_count=len(cats), num_threads=3)
    log('phase_start', phase='single_matrix_training_features')
    labels = np.load(CACHE / 'label.npy', mmap_mode='r')
    train_labels = np.array(labels[:manifest['train_rows']], copy=True)
    del labels  # Never access the final suffix of label.npy.
    if not np.isin(train_labels, [0, 1]).all() or int(train_labels.sum()) != manifest['train_clicks']:
        raise AssertionError('Full training labels differ from cache manifest')
    frame = shared_frame(0, manifest['train_rows'], arrays, manifest, features, cats)
    log('training_matrix_ready', bytes=int(frame.values.nbytes), shape=list(frame.shape))
    fit = lgb.Dataset(frame, label=train_labels, categorical_feature=cats, free_raw_data=True, params=PARAMS)
    fit.construct()
    del frame, train_labels
    gc.collect()
    log('phase_start', phase='fixed_378_round_cpu_training')
    training_started = time.perf_counter()
    def progress(env):
        iteration = env.iteration + 1
        if iteration == 1 or iteration % 50 == 0 or iteration == ROUNDS:
            log('training_progress', iteration=iteration, rounds=ROUNDS,
                seconds=time.perf_counter() - training_started)
    model = lgb.train(PARAMS, fit, num_boost_round=ROUNDS, callbacks=[progress])
    training_seconds = time.perf_counter() - training_started
    if model.current_iteration() != ROUNDS:
        raise AssertionError('Full model did not use exactly 378 rounds')
    model_path = EXPORT / 'shared_lgb_model.txt'
    model_path.write_text(model.model_to_string(num_iteration=ROUNDS), encoding='utf-8')
    model.free_dataset()
    del fit
    gc.collect()
    log('phase_start', phase='full_final_shared_tree_inference')
    shared = infer(model, manifest, arrays, features, cats, 'shared_final_prediction')
    del model
    gc.collect()
    np.save(EXPORT / 'shared_lgb_final_prediction.npy', shared)
    blend = probability(.8 * baseline + .2 * shared, manifest['final_rows'], 'v4 ensemble')
    np.save(EXPORT / 'ensemble_final_prediction.npy', blend)
    log('phase_start', phase='fixed_v4_csv_export')
    pd.DataFrame({'log_id': ids.log_id.to_numpy(), 'pctr': blend}).to_csv(
        EXPORT / 'result.csv', index=False, float_format='%.6f', encoding='utf-8')
    pd.DataFrame({'id': np.arange(len(ids)), 'pctr': blend}).to_csv(
        EXPORT / 'submission.csv', index=False, float_format='%.6f', encoding='utf-8')
    detail = pd.DataFrame({'id': np.arange(len(ids))})
    for name in ['log_id', 'user_id', 'pt_d']:
        detail[name] = ids[name].to_numpy()
    detail['label'] = np.full(len(ids), -1, dtype=np.int8)
    detail['pctr'] = blend
    detail['v3_pctr'] = baseline
    detail['shared_lgb_pctr'] = shared
    v3_detail = pd.read_csv(V3 / 'prediction_details.csv', usecols=['id', 'log_id', 'dcn_pctr', 'tree_pctr', 'lgb_pctr', 'multiscale_dcn_pctr'], float_precision='round_trip')
    np.testing.assert_array_equal(v3_detail.id.to_numpy(), detail.id.to_numpy())
    np.testing.assert_array_equal(v3_detail.log_id.to_numpy(), detail.log_id.to_numpy())
    for name in ['dcn_pctr', 'tree_pctr', 'lgb_pctr', 'multiscale_dcn_pctr']:
        detail[name] = v3_detail[name].to_numpy()
    detail.to_csv(EXPORT / 'prediction_details.csv', index=False, float_format='%.17g', encoding='utf-8')
    names = ['protocol.json', 'shared_lgb_model.txt', 'shared_lgb_final_prediction.npy',
             'ensemble_final_prediction.npy', 'result.csv', 'submission.csv', 'prediction_details.csv']
    record = {**protocol, 'status': 'predicted_awaiting_independent_verification',
              'train_clicks': manifest['train_clicks'], 'models': {'shared_lgb': file_record(model_path),
              'v3_components': read(V3 / 'manifest.json')['models']},
              'files': {name: file_record(EXPORT / name) for name in names},
              'training_seconds': training_seconds, 'lightgbm_version': lgb.__version__,
              'duplicate_log_id_rows': int(ids.log_id.duplicated().sum()),
              'probability': {'minimum': float(blend.min()), 'maximum': float(blend.max()), 'mean': float(blend.mean())},
              'final_id_rule': 'result.log_id retains all original final_test.csv rows and duplicates; submission.id is zero-based row position'}
    write(EXPORT / 'manifest.json', record)
    del arrays, baseline, shared, blend, detail, ids, v3_detail
    gc.collect()
    log('phase_start', phase='independent_process_full_model_reload_verification')
    subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--verify-only'], check=True)
    if read(EXPORT / 'verification.json')['status'] != 'passed':
        raise AssertionError('Independent verification failed')
    log('complete', rows=manifest['final_rows'], full_train_rows=manifest['train_rows'], rounds=ROUNDS,
        training_seconds=training_seconds, total_seconds=time.perf_counter() - started, final_auc=None)
    record = read(EXPORT / 'manifest.json')
    record['status'] = 'complete_and_verified'
    record['elapsed_seconds'] = time.perf_counter() - started
    record['files']['verification.json'] = file_record(EXPORT / 'verification.json')
    record['files']['run.log'] = file_record(EXPORT / 'run.log')
    write(EXPORT / 'manifest.json', record)


def verify():
    started = time.perf_counter()
    if (EXPORT / 'verification.json').exists():
        raise FileExistsError('Preserving existing v4 verification')
    log('phase_start', phase='independent_old_output_and_new_export_hash_verification')
    record = read(EXPORT / 'manifest.json')
    if record['params'] != PARAMS or record['rounds'] != ROUNDS or record['ensemble'] != WEIGHTS:
        raise AssertionError('Fixed shared tree refit protocol differs')
    for item in (list(record['files'].values()) + list(record['protected_files'].values())
                 + list(record['source_files'].values()) + list(record['development_evidence_files'].values())
                 + [record['cache_manifest']]):
        if digest(item['path']) != item['sha256']:
            raise AssertionError('Input/output/protected file changed: ' + item['path'])
    if digest(ROOT / '2026ml' / 'final_test.csv') != record['input_sha256']['final_test.csv']:
        raise AssertionError('Original final input changed')
    manifest, arrays, features, cats = inputs()
    development_parameters(features, cats)
    probes = alignment_probes(manifest, arrays, features, cats)
    ids = final_ids(manifest, arrays)
    log('phase_start', phase='independent_378_round_model_all_final_replay')
    restored = lgb.Booster(model_str=(EXPORT / 'shared_lgb_model.txt').read_text(encoding='utf-8'))
    replay = infer(restored, manifest, arrays, features, cats, 'independent_shared_prediction')
    saved = probability(np.load(EXPORT / 'shared_lgb_final_prediction.npy'), len(ids), 'saved shared tree')
    reload_error = float(np.max(np.abs(replay - saved)))
    if reload_error > 1e-12:
        raise AssertionError('Independent whole-final shared model replay differs')
    baseline = v3_prediction(len(ids))
    blend = .8 * baseline + .2 * replay
    saved_blend = probability(np.load(EXPORT / 'ensemble_final_prediction.npy'), len(ids), 'saved v4')
    blend_error = float(np.max(np.abs(blend - saved_blend)))
    if blend_error > 1e-12:
        raise AssertionError('Independent v4 ensemble differs')
    result = pd.read_csv(EXPORT / 'result.csv', float_precision='round_trip')
    submission = pd.read_csv(EXPORT / 'submission.csv', float_precision='round_trip')
    detail = pd.read_csv(EXPORT / 'prediction_details.csv', float_precision='round_trip')
    if result.columns.tolist() != ['log_id', 'pctr'] or submission.columns.tolist() != ['id', 'pctr']:
        raise AssertionError('Export submission columns differ')
    if any(len(frame) != len(ids) for frame in [result, submission, detail]):
        raise AssertionError('Export row count differs')
    np.testing.assert_array_equal(result.log_id.to_numpy(), ids.log_id.to_numpy())
    np.testing.assert_array_equal(submission.id.to_numpy(), np.arange(len(ids)))
    np.testing.assert_array_equal(detail.id.to_numpy(), np.arange(len(ids)))
    for name in ['log_id', 'user_id', 'pt_d']:
        np.testing.assert_array_equal(detail[name].to_numpy(), ids[name].to_numpy())
    if not np.all(detail.label.to_numpy() == -1):
        raise AssertionError('Final labels must be synthesized as -1')
    rounding = {}
    for name, frame in [('result.csv', result), ('submission.csv', submission)]:
        probability(frame.pctr.to_numpy(), len(ids), name)
        rounding[name] = float(np.max(np.abs(frame.pctr.to_numpy() - blend)))
        if rounding[name] > 5.000001e-7:
            raise AssertionError('CSV six-digit probability rounding differs')
        with (EXPORT / name).open('rb') as stream:
            if stream.read(3) == b'\xef\xbb\xbf':
                raise AssertionError('Unexpected UTF8 BOM')
        with (EXPORT / name).open(encoding='utf-8', newline='') as stream:
            reader = csv.reader(stream)
            next(reader)
            if any(len(row) != 2 or not re.fullmatch(r'[01]\.\d{6}', row[1]) for row in reader):
                raise AssertionError('Expected exactly six decimals')
    np.testing.assert_array_equal(result.pctr.to_numpy(), submission.pctr.to_numpy())
    detail_errors = {'pctr': float(np.max(np.abs(detail.pctr.to_numpy() - blend))),
                     'v3_pctr': float(np.max(np.abs(detail.v3_pctr.to_numpy() - baseline))),
                     'shared_lgb_pctr': float(np.max(np.abs(detail.shared_lgb_pctr.to_numpy() - replay)))}
    v3_detail = pd.read_csv(V3 / 'prediction_details.csv', usecols=['id', 'log_id', 'dcn_pctr', 'tree_pctr', 'lgb_pctr', 'multiscale_dcn_pctr'], float_precision='round_trip')
    np.testing.assert_array_equal(v3_detail.id.to_numpy(), detail.id.to_numpy())
    np.testing.assert_array_equal(v3_detail.log_id.to_numpy(), detail.log_id.to_numpy())
    for name in ['dcn_pctr', 'tree_pctr', 'lgb_pctr', 'multiscale_dcn_pctr']:
        detail_errors[name] = float(np.max(np.abs(detail[name].to_numpy() - v3_detail[name].to_numpy())))
    if max(detail_errors.values()) > 1e-14:
        raise AssertionError('Prediction details differ from independently restored components')
    v3_from_parts = (.3 * detail.dcn_pctr.to_numpy() + .3 * detail.tree_pctr.to_numpy()
                     + .15 * detail.lgb_pctr.to_numpy() + .25 * detail.multiscale_dcn_pctr.to_numpy())
    v3_part_error = float(np.max(np.abs(v3_from_parts - baseline)))
    if v3_part_error > 1e-14:
        raise AssertionError('Base v3 component details do not reproduce v3 ensemble')
    output = {'status': 'passed', 'independent_process': True, 'verification_pid': os.getpid(),
              'rows': len(ids), 'full_train_rows': manifest['train_rows'], 'fields': 349,
              'categorical_count': 24, 'budget_rounds': ROUNDS, 'num_threads': 3,
              'full_final_model_reload': True, 'maximum_full_shared_lgb_reload_error': reload_error,
              'maximum_ensemble_replay_error': blend_error, 'maximum_v3_component_replay_error': v3_part_error,
              'maximum_csv_rounding_error': rounding, 'maximum_detail_error': detail_errors,
              'original_ids_and_row_order': True, 'duplicate_log_id_rows_preserved': int(ids.log_id.duplicated().sum()),
              'submission_ids_are_zero_based_original_row_positions': True,
              'features_equal_both_development_protocols': True, 'alignment_probes': probes,
              'no_identity_categories': True, 'no_sequence_token_ids': True,
              'utf8_no_bom': True, 'six_decimals': True, 'all_final_labels_forced_to_minus_one': True,
              'input_model_source_and_export_hashes_match': True, 'entire_v2_v3_and_old_models_unchanged': True,
              'final_labels_read': False, 'final_labels_available': False, 'final_auc': None,
              'development_auc': DEVELOPMENT, 'warnings': WARNINGS,
              'verification_seconds': time.perf_counter() - started}
    write(EXPORT / 'verification.json', output)
    log('independent_verification_passed', rows=len(ids), maximum_reload_error=reload_error,
        maximum_ensemble_replay_error=blend_error, seconds=time.perf_counter() - started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    verify() if args.verify_only else main()
