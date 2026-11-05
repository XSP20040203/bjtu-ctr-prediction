"""Fixed 112-round CPU LightGBM refit and independently verified final export.

Only this script's new output/chronological_lgb_v2 directory is written. Existing
DCN/CatBoost checkpoints, caches and exports are read-only. Final advertisement
labels are never read: only the training prefix of label.npy is accessed, and
all final labels are explicitly synthesized as -1.
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
EXPORT = ROOT / 'output' / 'chronological_lgb_v2'
sys.path.insert(0, str(HERE / 'deps'))

import lightgbm as lgb
import numpy as np
import pandas as pd

ROUNDS = 112
BATCH_SIZE = 25000
WEIGHTS = {'dcn': 0.4, 'tree': 0.4, 'lgb': 0.2}
PARAMS = {
    'objective': 'binary', 'metric': ['auc', 'binary_logloss'],
    'learning_rate': 0.03, 'num_leaves': 31, 'max_depth': -1,
    'min_data_in_leaf': 200, 'max_bin': 63, 'feature_fraction': 0.9,
    'bagging_fraction': 0.9, 'bagging_freq': 1, 'lambda_l2': 10.,
    'cat_l2': 20., 'cat_smooth': 20., 'max_cat_threshold': 32,
    'verbosity': -1, 'num_threads': 3, 'seed': 42, 'deterministic': True,
    'force_col_wise': True, 'feature_pre_filter': False,
}
DEVELOPMENT = {
    '20191115': {'lgb_auc': 0.7667657472867364,
                 'dcn_tree_equal_auc': 0.7858631196467175,
                 'fixed_40_40_20_auc': 0.786473152710307,
                 'lgb_best_iteration': 117},
    '20191116': {'lgb_auc': 0.7698131086471949,
                 'dcn_tree_equal_auc': 0.7917860844136609,
                 'fixed_40_40_20_auc': 0.79243244179162,
                 'lgb_best_iteration': 107},
}
WARNINGS = [
    '开发日期15/16已反复用于模型和权重选型，不是独立盲测；开发AUC不代表final AUC。',
    '源域新闻数据、当前日曝光和广告历史快照采用所提供整日离线输入；不代表逐条在线曝光前可用。',
    'final广告标签不可用且不读取，所有final标签强制为-1；final_auc=None，不能声称final达标或AUC达到0.85。',
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size}


def probabilities(values, rows, name):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (rows,) or not np.isfinite(values).all():
        raise AssertionError(name + ': invalid prediction shape or nonfinite values')
    if np.any((values < 0) | (values > 1)):
        raise AssertionError(name + ': probability outside [0, 1]')
    return values


def inputs():
    manifest = read(CACHE / 'manifest.json')
    exposure = read(DAILY / 'exposure_features.json')
    snapshot = read(DAILY / 'snapshot_features.json')
    if manifest['final_labels_available'] is not False:
        raise AssertionError('Full cache must declare unavailable final labels')
    if any(meta['advertisement_labels_loaded'] is not False for meta in [exposure, snapshot]):
        raise AssertionError('Raw exposure/snapshot inputs must be label-free')
    arrays = {name: np.load(CACHE / (name + '.npy'), mmap_mode='r') for name in
              ['categories', 'sequences', 'numeric', 'exposure_raw', 'snapshot_raw',
               'date', 'log_id', 'user_id']}
    ntrain, nfinal = manifest['train_rows'], manifest['final_rows']
    total = ntrain + nfinal
    if manifest['rows'] != total or any(len(value) != total for value in arrays.values()):
        raise AssertionError('Cache row counts differ')
    if arrays['numeric'].shape[1] != 166 or arrays['sequences'].shape[2] < 5:
        raise AssertionError('Unexpected full-refit feature schema')
    if arrays['date'][:ntrain].max() != 20191116 or not np.all(arrays['date'][ntrain:] == 20191117):
        raise AssertionError('Chronological train/final boundary differs')
    if not np.all(arrays['date'][:ntrain] < 20191117):
        raise AssertionError('Training contains final-date rows')
    if arrays['categories'].shape[1] != len(manifest['categorical_fields']):
        raise AssertionError('Categorical schema differs')
    if arrays['sequences'].shape[1] != len(manifest['sequence_fields']):
        raise AssertionError('Sequence schema differs')
    if arrays['exposure_raw'].shape[1] != len(exposure['features']):
        raise AssertionError('Exposure schema differs')
    if arrays['snapshot_raw'].shape[1] != len(snapshot['features']):
        raise AssertionError('Snapshot schema differs')
    if manifest['numeric_fields'][75:] != exposure['features'] + snapshot['features']:
        raise AssertionError('Raw extra-feature names/order differ')
    categorical = list(manifest['categorical_fields'])
    tokens = [field + '_token' + str(position) for field in manifest['sequence_fields']
              for position in range(5)]
    categorical += tokens
    # Exactly refit_predict_full.tree_frame's column order. LightGBM receives
    # original cache category IDs, as train_lightgbm.py does, marked categorical
    # by name. Do not infer/remap pandas categories independently in final chunks.
    features = (manifest['categorical_fields'] + manifest['numeric_fields'][:75]
                + tokens + exposure['features'] + snapshot['features'])
    if len(features) != 253 or len(categorical) != 87 or len(set(features)) != len(features):
        raise AssertionError('Expected 253 features and 87 categorical fields')
    return manifest, arrays, features, categorical


def tree_frame(start, stop, arrays, features):
    """Use only the requested rows and one float32, column-major allocation.

    All cached integer category IDs fit exactly into float32; LightGBM casts
    explicitly designated categorical columns to int32. This preserves native
    IDs without pandas' potentially different per-chunk category code mapping.
    Numeric[:75] matches the tree base; extra numeric[75:] is deliberately never
    used, since exposure_raw/snapshot_raw contain the unscaled tree inputs.
    """
    matrix = np.empty((stop - start, len(features)), dtype=np.float32, order='F')
    position = 0
    sources = [arrays['categories'][start:stop], arrays['numeric'][start:stop, :75],
               arrays['sequences'][start:stop, :, :5].reshape(stop - start, -1),
               arrays['exposure_raw'][start:stop], arrays['snapshot_raw'][start:stop]]
    for source in sources:
        for col in range(source.shape[1]):
            values = source[:, col]
            if source.dtype.kind in 'iu' and (np.any(values < 0) or np.any(values >= 2 ** 24)):
                raise AssertionError('Category ID cannot be represented exactly as float32')
            matrix[:, position] = values
            if not np.isfinite(matrix[:, position]).all():
                raise AssertionError('Nonfinite feature: ' + features[position])
            position += 1
    if position != len(features):
        raise AssertionError('Frame column count differs')
    return pd.DataFrame(matrix, columns=features, copy=False)


def final_ids(manifest, arrays):
    # A usecols allowlist prevents loading any final target column, even if one
    # happens to be present in the supplied CSV.
    ids = pd.read_csv(ROOT / '2026ml' / 'final_test.csv',
                      usecols=['log_id', 'user_id', 'pt_d'], dtype='int64')
    ntrain, nfinal = manifest['train_rows'], manifest['final_rows']
    if len(ids) != nfinal:
        raise AssertionError('Original final row count differs')
    for field, cached in [('log_id', 'log_id'), ('user_id', 'user_id'), ('pt_d', 'date')]:
        np.testing.assert_array_equal(ids[field].to_numpy(), arrays[cached][ntrain:])
    final_label = np.full(nfinal, -1, dtype=np.int8)
    if not np.all(final_label == -1):
        raise AssertionError('Final labels must be synthesized as -1')
    return ids, final_label


def infer(model, manifest, arrays, features, prefix):
    ntrain, nfinal = manifest['train_rows'], manifest['final_rows']
    if model.feature_name() != features or model.current_iteration() != ROUNDS:
        raise AssertionError('Restored model feature order or budget differs')
    values = np.empty(nfinal, dtype=np.float64)
    for offset in range(0, nfinal, BATCH_SIZE):
        stop = min(offset + BATCH_SIZE, nfinal)
        frame = tree_frame(ntrain + offset, ntrain + stop, arrays, features)
        values[offset:stop] = model.predict(frame, num_iteration=ROUNDS, num_threads=3)
        del frame
        print(f'{prefix} {stop}/{nfinal}', flush=True)
    return probabilities(values, nfinal, 'lgb')


def baseline_predictions(nfinal):
    result = {}
    for name in ['dcn', 'tree']:
        path = BASE_MODELS / (name + '_final_prediction.npy')
        if not path.is_file():
            raise FileNotFoundError('Required existing component prediction: ' + str(path))
        result[name] = probabilities(np.load(path, mmap_mode='r'), nfinal, name)
    return result


def check_development_parameters():
    evidence = {}
    for day, tag in [('20191115', 'goal085_day15_lgb31_utf8'),
                     ('20191116', 'goal085_day16_lgb31')]:
        path = DAILY / 'lgb_results' / tag / 'metrics.json'
        metrics = read(path)
        expected = {**PARAMS, 'num_threads': 6}
        if metrics['params'] != expected or metrics['best_iteration'] != DEVELOPMENT[day]['lgb_best_iteration']:
            raise AssertionError('Development LightGBM parameters/budget source differ')
        if abs(metrics['auc'] - DEVELOPMENT[day]['lgb_auc']) > 1e-14 or metrics['fields'] != 253:
            raise AssertionError('Development LightGBM evidence differs')
        evidence[day] = file_record(path)
    if ROUNDS != round(np.mean([row['lgb_best_iteration'] for row in DEVELOPMENT.values()])):
        raise AssertionError('Refit budget must be rounded mean of 117 and 107')
    return evidence


def main():
    started = time.perf_counter()
    if EXPORT.exists():
        raise FileExistsError('Preserving existing output: ' + str(EXPORT))
    manifest, arrays, features, categorical = inputs()
    evidence_files = check_development_parameters()
    predictions = baseline_predictions(manifest['final_rows'])
    ids, final_label = final_ids(manifest, arrays)
    base_files = {name: file_record(BASE_MODELS / filename) for name, filename in
                  [('dcn', 'dcn.pt'), ('tree', 'catboost.cbm')]}
    source_files = {str(path.resolve()): file_record(path) for path in
                    [Path(__file__), DAILY / 'train_lightgbm.py', DAILY / 'refit_predict_full.py',
                     DAILY / 'exposure_features.json', DAILY / 'snapshot_features.json']}
    component_files = {name: file_record(BASE_MODELS / (name + '_final_prediction.npy'))
                       for name in ['dcn', 'tree']}
    # Both parameters and budget are committed to disk before training or any
    # final LightGBM inference. This is a fixed full-data fit, without early stop.
    protocol = {
        'status': 'fixed_before_training_and_final_inference', 'params': PARAMS,
        'rounds': ROUNDS, 'early_stopping_rounds': None, 'full_training_refit': True,
        'full_train_rows': manifest['train_rows'], 'final_rows': manifest['final_rows'],
        'budget_source': 'round(mean([117,107]))=112; no final-label selection',
        'parameter_source': str((DAILY / 'train_lightgbm.py').resolve()),
        'parameter_deviation': {'num_threads': {'development': 6, 'full_refit': 3}},
        'ensemble': WEIGHTS, 'features': features, 'categorical_fields': categorical,
        'categorical_encoding': 'Original cached integer IDs; explicit LightGBM categorical names; exact float32 storage',
        'numeric_inputs': 'numeric.npy[:,:75] + exposure_raw.npy + snapshot_raw.npy; no scaled numeric extras',
        'inference_batch_size': BATCH_SIZE, 'device': 'CPU',
        'development_auc': DEVELOPMENT, 'development_evidence_files': evidence_files,
        'final_labels_available': False, 'final_labels_read': False,
        'final_label_value': -1, 'final_auc': None, 'warnings': WARNINGS,
        'advertisement_statistics': manifest['advertisement_statistics'],
        'same_day_inputs': 'Supplied full source-day snapshots, unlabeled daily exposures and ad-history snapshots',
        'source_files': source_files, 'baseline_models': base_files,
        'baseline_prediction_files': component_files,
        'cache_manifest': file_record(CACHE / 'manifest.json'),
    }
    EXPORT.mkdir(parents=True, exist_ok=False)
    write(EXPORT / 'protocol.json', protocol)
    labels = np.load(CACHE / 'label.npy', mmap_mode='r')
    # Never index the final suffix of the label cache.
    train_labels = np.array(labels[:manifest['train_rows']], copy=True)
    del labels
    if not np.isin(train_labels, [0, 1]).all() or int(train_labels.sum()) != manifest['train_clicks']:
        raise AssertionError('Full training labels disagree with cache manifest')
    print(f'LGB_REFIT rows={len(train_labels)} fields={len(features)} rounds={ROUNDS} threads=3', flush=True)
    frame = tree_frame(0, manifest['train_rows'], arrays, features)
    fit = lgb.Dataset(frame, label=train_labels, categorical_feature=categorical,
                      free_raw_data=True, params=PARAMS)
    fit.construct()
    del frame, train_labels
    gc.collect()
    training_started = time.perf_counter()
    model = lgb.train(PARAMS, fit, num_boost_round=ROUNDS, callbacks=[lgb.log_evaluation(25)])
    training_seconds = time.perf_counter() - training_started
    if model.current_iteration() != ROUNDS:
        raise AssertionError('Full refit did not produce exactly 112 rounds')
    # Python text I/O avoids LightGBM's Unicode Windows filename issue.
    model_path = EXPORT / 'lgb_model.txt'
    model_path.write_text(model.model_to_string(num_iteration=ROUNDS), encoding='utf-8')
    model.free_dataset()
    del fit
    gc.collect()
    predictions['lgb'] = infer(model, manifest, arrays, features, 'LGB_FINAL')
    del model
    gc.collect()
    probability = probabilities(sum(WEIGHTS[name] * predictions[name] for name in WEIGHTS),
                                manifest['final_rows'], 'ensemble')
    np.save(EXPORT / 'lgb_final_prediction.npy', predictions['lgb'])
    np.save(EXPORT / 'ensemble_final_prediction.npy', probability)
    pd.DataFrame({'log_id': ids.log_id.to_numpy(), 'pctr': probability}).to_csv(
        EXPORT / 'result.csv', index=False, float_format='%.6f', encoding='utf-8')
    pd.DataFrame({'id': np.arange(len(ids)), 'pctr': probability}).to_csv(
        EXPORT / 'submission.csv', index=False, float_format='%.6f', encoding='utf-8')
    detail = pd.DataFrame({'id': np.arange(len(ids))})
    for name in ['log_id', 'user_id', 'pt_d']:
        detail[name] = ids[name].to_numpy()
    detail['label'] = final_label
    detail['pctr'] = probability
    for name in WEIGHTS:
        detail[name + '_pctr'] = predictions[name]
    detail.to_csv(EXPORT / 'prediction_details.csv', index=False, float_format='%.17g', encoding='utf-8')
    names = ['protocol.json', 'lgb_model.txt', 'lgb_final_prediction.npy',
             'ensemble_final_prediction.npy', 'result.csv', 'submission.csv', 'prediction_details.csv']
    record = {
        **protocol, 'status': 'predicted_awaiting_independent_verification',
        'train_clicks': manifest['train_clicks'], 'train_dates': sorted(np.unique(arrays['date'][:manifest['train_rows']]).tolist()),
        'models': {**base_files, 'lgb': file_record(model_path)},
        'files': {name: file_record(EXPORT / name) for name in names},
        'input_sha256': manifest['input_sha256'],
        'final_id_rule': 'result.log_id retains original final_test.csv row order, including repeated log_id; submission.id is zero-based row position',
        'lightgbm_version': lgb.__version__, 'numpy_version': np.__version__, 'pandas_version': pd.__version__,
        'training_seconds': training_seconds, 'elapsed_seconds_before_verification': time.perf_counter() - started,
        'probability': {'minimum': float(probability.min()), 'maximum': float(probability.max()),
                        'mean': float(probability.mean())},
        'duplicate_final_log_id_rows': int(ids.log_id.duplicated().sum()),
    }
    write(EXPORT / 'manifest.json', record)
    del arrays, predictions, probability, detail, ids, final_label
    gc.collect()
    # A separate interpreter must reload the serialized model and re-predict
    # every final row. No training model object or in-memory predictions survive.
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--verify-only'], check=True)
    verified = read(EXPORT / 'verification.json')
    if verified['status'] != 'passed':
        raise AssertionError('Independent verification did not pass')
    print('LGB_EXPORT_COMPLETE ' + json.dumps({'output': str(EXPORT), 'rows': manifest['final_rows'],
          'training_rows': manifest['train_rows'], 'rounds': ROUNDS, 'final_auc': None,
          'elapsed_seconds': time.perf_counter() - started, 'verification': verified['status']}), flush=True)


def verify():
    started = time.perf_counter()
    if (EXPORT / 'verification.json').exists():
        raise FileExistsError('Preserving existing independent verification')
    record = read(EXPORT / 'manifest.json')
    protocol = read(EXPORT / 'protocol.json')
    if record['params'] != PARAMS or protocol['params'] != PARAMS or protocol['rounds'] != ROUNDS:
        raise AssertionError('Recorded parameters differ from fixed refit')
    if record['ensemble'] != WEIGHTS or record['final_auc'] is not None:
        raise AssertionError('Fixed blend/final score boundary differs')
    for item in list(record['files'].values()) + list(record['models'].values()) + list(record['source_files'].values()) + list(record['baseline_prediction_files'].values()) + list(record['development_evidence_files'].values()) + [record['cache_manifest']]:
        if digest(item['path']) != item['sha256']:
            raise AssertionError('File/model/source hash changed: ' + item['path'])
    # Hashing the original input checks byte identity; CSV reads still use only
    # the identifier/date allowlist and never read final label values.
    if digest(ROOT / '2026ml' / 'final_test.csv') != record['input_sha256']['final_test.csv']:
        raise AssertionError('Original final input changed')
    manifest, arrays, features, categorical = inputs()
    if protocol['features'] != features or protocol['categorical_fields'] != categorical:
        raise AssertionError('Recorded feature schema differs')
    ids, final_label = final_ids(manifest, arrays)
    restored = lgb.Booster(model_str=(EXPORT / 'lgb_model.txt').read_text(encoding='utf-8'))
    replay = infer(restored, manifest, arrays, features, 'LGB_INDEPENDENT_REPLAY')
    saved_lgb = probabilities(np.load(EXPORT / 'lgb_final_prediction.npy'), len(ids), 'saved lgb')
    reload_error = float(np.max(np.abs(replay - saved_lgb)))
    if reload_error > 1e-12:
        raise AssertionError('Independent full-final model replay differs')
    components = baseline_predictions(len(ids))
    ensemble = probabilities(WEIGHTS['dcn'] * components['dcn'] + WEIGHTS['tree'] * components['tree'] + WEIGHTS['lgb'] * replay,
                             len(ids), 'independent ensemble')
    saved_ensemble = probabilities(np.load(EXPORT / 'ensemble_final_prediction.npy'), len(ids), 'saved ensemble')
    ensemble_error = float(np.max(np.abs(ensemble - saved_ensemble)))
    if ensemble_error > 1e-12:
        raise AssertionError('Independent ensemble differs')
    result = pd.read_csv(EXPORT / 'result.csv', float_precision='round_trip')
    submission = pd.read_csv(EXPORT / 'submission.csv', float_precision='round_trip')
    detail = pd.read_csv(EXPORT / 'prediction_details.csv', float_precision='round_trip')
    if result.columns.tolist() != ['log_id', 'pctr'] or submission.columns.tolist() != ['id', 'pctr']:
        raise AssertionError('Submission columns differ')
    if any(len(frame) != len(ids) for frame in [result, submission, detail]):
        raise AssertionError('Export row count differs')
    np.testing.assert_array_equal(result.log_id.to_numpy(), ids.log_id.to_numpy())
    np.testing.assert_array_equal(submission.id.to_numpy(), np.arange(len(ids)))
    np.testing.assert_array_equal(detail.id.to_numpy(), np.arange(len(ids)))
    for name in ['log_id', 'user_id', 'pt_d']:
        np.testing.assert_array_equal(detail[name].to_numpy(), ids[name].to_numpy())
    np.testing.assert_array_equal(detail.label.to_numpy(), final_label)
    rounding = {}
    for name, frame in [('result.csv', result), ('submission.csv', submission)]:
        probabilities(frame.pctr.to_numpy(), len(ids), name)
        rounding[name] = float(np.max(np.abs(frame.pctr.to_numpy() - ensemble)))
        if rounding[name] > 5.000001e-7:
            raise AssertionError('Export rounding differs: ' + name)
        path = EXPORT / name
        with path.open('rb') as stream:
            if stream.read(3) == b'\xef\xbb\xbf':
                raise AssertionError('Unexpected UTF-8 BOM')
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.reader(stream)
            next(reader)
            for row in reader:
                if len(row) != 2 or not re.fullmatch(r'[01]\.\d{6}', row[1]):
                    raise AssertionError('Expected exactly six decimal places')
    np.testing.assert_array_equal(result.pctr.to_numpy(), submission.pctr.to_numpy())
    detail_errors = {'pctr': float(np.max(np.abs(detail.pctr.to_numpy() - ensemble)))}
    for name, expected in {**components, 'lgb': replay}.items():
        detail_errors[name + '_pctr'] = float(np.max(np.abs(detail[name + '_pctr'].to_numpy() - expected)))
    if max(detail_errors.values()) > 1e-14:
        raise AssertionError('Prediction details differ from independent components')
    output = {
        'status': 'passed', 'independent_process': True, 'verification_pid': os.getpid(),
        'full_final_model_reload': True, 'rows': len(ids), 'full_train_rows': manifest['train_rows'],
        'budget_rounds': ROUNDS, 'num_threads': 3, 'inference_batch_size': BATCH_SIZE,
        'maximum_full_lgb_reload_error': reload_error, 'maximum_ensemble_replay_error': ensemble_error,
        'maximum_csv_rounding_error': rounding, 'maximum_detail_error': detail_errors,
        'original_ids_and_row_order': True, 'duplicate_log_id_rows_preserved': int(ids.log_id.duplicated().sum()),
        'submission_ids_are_zero_based_original_row_positions': True,
        'utf8_no_bom': True, 'six_decimals': True, 'all_final_labels_forced_to_minus_one': True,
        'final_labels_read': False, 'final_labels_available': False, 'final_auc': None,
        'input_model_source_and_export_hashes_match': True,
        'existing_dcn_tree_models_and_predictions_unchanged': True,
        'development_auc': DEVELOPMENT, 'warnings': WARNINGS,
        'verification_seconds': time.perf_counter() - started,
    }
    write(EXPORT / 'verification.json', output)
    record['status'] = 'complete_and_verified'
    record['files']['verification.json'] = file_record(EXPORT / 'verification.json')
    record['verification'] = {'status': 'passed', 'independent_process': True,
                              'maximum_full_lgb_reload_error': reload_error}
    write(EXPORT / 'manifest.json', record)
    print('LGB_VERIFICATION ' + json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true',
                        help='Independently verify the newly created export; never trains models.')
    args = parser.parse_args()
    verify() if args.verify_only else main()
